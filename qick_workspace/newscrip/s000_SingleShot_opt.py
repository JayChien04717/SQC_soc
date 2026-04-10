import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import skew as _skew

# ----- Qick package ----- #
from qick import *

# ----- User Library ----- #
from .base_program import BaseProgram
from .singleshot_utils import hist


##################
# Define Program #
##################


class SingleShotOptProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")
        if cfg.get("shot_f", False):
            self.setup_qubit_gen(cfg, "ef")
            self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # 1. Ground state readout
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        self.delay_auto(cfg["relax_delay"], tag="relax_wait")

        # 2. Excited state readout
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        # 3. Optional f-state readout (ge pi + ef pi → |f⟩)
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


#####################
# Define Experiment #
#####################


class SingleShot_ge_opt:
    """
    Grid search optimization for single-shot readout (length, gain, freq).
    Note: Does not inherit from BaseExperiment because it utilizes a custom
    multi-dimensional software loop rather than liveplotfun.
    """

    def __init__(self, config):
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    def run(self, SHOTS, sweep_para: dict, shot_f=False):
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f
        self._shot_f = shot_f

        raw_length = sweep_para.get("length")
        self.length_sweep = (
            raw_length
            if isinstance(raw_length, (list, tuple, np.ndarray))
            else [raw_length]
        )

        raw_gain = sweep_para.get("gain")
        self.gain_sweep = (
            raw_gain if isinstance(raw_gain, (list, tuple, np.ndarray)) else [raw_gain]
        )

        raw_freq = sweep_para.get("freq")
        self.freq_sweep = (
            raw_freq if isinstance(raw_freq, (list, tuple, np.ndarray)) else [raw_freq]
        )

        final_shape = (
            len(self.length_sweep),
            len(self.gain_sweep),
            len(self.freq_sweep),
            SHOTS,
        )

        self.I_g_array = np.full(final_shape, np.nan)
        self.Q_g_array = np.full(final_shape, np.nan)
        self.I_e_array = np.full(final_shape, np.nan)
        self.Q_e_array = np.full(final_shape, np.nan)
        if shot_f:
            self.I_f_array = np.full(final_shape, np.nan)
            self.Q_f_array = np.full(final_shape, np.nan)

        # --- TQDM Dynamic Setup ---
        is_l_sweep = len(self.length_sweep) > 1
        is_g_sweep = len(self.gain_sweep) > 1
        is_f_sweep = len(self.freq_sweep) > 1

        outermost_real_sweep = None
        if is_l_sweep:
            outermost_real_sweep = "l"
        elif is_g_sweep:
            outermost_real_sweep = "g"
        elif is_f_sweep:
            outermost_real_sweep = "f"

        l_iter = self.length_sweep
        if "l" == outermost_real_sweep:
            l_iter = tqdm(self.length_sweep, desc="Length loop")

        for l_idx, l_val in enumerate(l_iter):
            g_iter = self.gain_sweep
            if "g" == outermost_real_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop")
            elif is_g_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop", leave=False)

            for g_idx, g_val in enumerate(g_iter):
                f_iter = self.freq_sweep
                if "f" == outermost_real_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop")
                elif is_f_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop", leave=False)

                for f_idx, f_val in enumerate(f_iter):
                    cfg_update = {"steps": SHOTS}
                    if l_val is not None:
                        cfg_update["ro_length"] = l_val
                    if g_val is not None:
                        cfg_update["res_gain_ge"] = g_val
                    if f_val is not None:
                        cfg_update["res_freq_ge"] = f_val

                    self.cfg.update(cfg_update)

                    # Single program handles both g and e states
                    ssp = SingleShotOptProgram(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list = ssp.acquire(self.soc, rounds=1, progress=False)

                    # Extract triggers from the single readout channel
                    I_g = iq_list[0][0, :, 0]
                    Q_g = iq_list[0][0, :, 1]
                    I_e = iq_list[0][1, :, 0]
                    Q_e = iq_list[0][1, :, 1]

                    self.I_g_array[l_idx, g_idx, f_idx, :] = I_g
                    self.Q_g_array[l_idx, g_idx, f_idx, :] = Q_g
                    self.I_e_array[l_idx, g_idx, f_idx, :] = I_e
                    self.Q_e_array[l_idx, g_idx, f_idx, :] = Q_e

                    if shot_f:
                        self.I_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 0]
                        self.Q_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 1]

        self.data = {
            "Ig": self.I_g_array,
            "Qg": self.Q_g_array,
            "Ie": self.I_e_array,
            "Qe": self.Q_e_array,
        }
        if shot_f:
            self.data["If"] = self.I_f_array
            self.data["Qf"] = self.Q_f_array

    def analyze(self):
        try:
            from scipy.interpolate import RegularGridInterpolator
            from scipy.optimize import minimize

            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False

        try:
            len_L = len(self.length_sweep)
            len_G = len(self.gain_sweep)
            len_F = len(self.freq_sweep)
        except AttributeError:
            print("Error: 'run' method must be called first to define sweep axes.")
            return

        fid_Array = np.zeros((len_L, len_G, len_F))
        sep_array = np.zeros((len_L, len_G, len_F))
        sk_g_array = np.zeros((len_L, len_G, len_F))
        sk_e_array = np.zeros((len_L, len_G, len_F))

        shot_f = getattr(self, "_shot_f", False)
        metric_label = "GMM fidelity (gef)" if shot_f else "GMM fidelity (ge)"

        I_g_data = self.data["Ig"]
        Q_g_data = self.data["Qg"]
        I_e_data = self.data["Ie"]
        Q_e_data = self.data["Qe"]
        if shot_f:
            I_f_data = self.data["If"]
            Q_f_data = self.data["Qf"]

        for l_idx in tqdm(range(len_L), desc=f"Analyze {metric_label}"):
            for g_idx in range(len_G):
                for f_idx in range(len_F):
                    I_g = I_g_data[l_idx, g_idx, f_idx]
                    Q_g = Q_g_data[l_idx, g_idx, f_idx]
                    I_e = I_e_data[l_idx, g_idx, f_idx]
                    Q_e = Q_e_data[l_idx, g_idx, f_idx]

                    data_slice = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}
                    if shot_f:
                        data_slice["If"] = I_f_data[l_idx, g_idx, f_idx]
                        data_slice["Qf"] = Q_f_data[l_idx, g_idx, f_idx]

                    result = hist(data_slice, plot=False, verbose=False)
                    fid_Array[l_idx, g_idx, f_idx] = result[0][0]

                    # LDA projection for quality metrics (lightweight, no extra GMM)
                    mg = np.array([I_g.mean(), Q_g.mean()])
                    me = np.array([I_e.mean(), Q_e.mean()])
                    v = me - mg
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        pg = ((I_g - mg[0]) * v[0] + (Q_g - mg[1]) * v[1]) / n
                        pe = ((I_e - mg[0]) * v[0] + (Q_e - mg[1]) * v[1]) / n
                        sep_array[l_idx, g_idx, f_idx] = n
                        sk_g_array[l_idx, g_idx, f_idx] = float(_skew(pg))
                        sk_e_array[l_idx, g_idx, f_idx] = float(_skew(pe))

        max_idx = np.unravel_index(np.argmax(fid_Array), fid_Array.shape)
        max_l_idx, max_g_idx, max_f_idx = max_idx

        max_fid_grid = fid_Array[max_idx]
        max_length_grid = self.length_sweep[max_l_idx]
        max_gain_grid = self.gain_sweep[max_g_idx]
        max_freq_grid = self.freq_sweep[max_f_idx]

        print(f"\n--- Grid Search Result ---")
        print(f"Max fidelity (on grid): {max_fid_grid:.4f}")

        length_str_grid = (
            f"{max_length_grid:.3f} us" if max_length_grid is not None else "default"
        )
        gain_str_grid = (
            f"{max_gain_grid:.5f} DAC" if max_gain_grid is not None else "default"
        )
        freq_str_grid = (
            f"{max_freq_grid:.5f} MHz" if max_freq_grid is not None else "default"
        )

        print(
            f"At length={length_str_grid}, gain={gain_str_grid}, freq={freq_str_grid}"
        )

        max_length, max_gain, max_freq = max_length_grid, max_gain_grid, max_freq_grid

        if not SCIPY_AVAILABLE:
            print("\nWarning: `scipy` not found. Skipping interpolation.")
            print("To enable, run: pip install scipy")
        else:
            try:
                all_axes_data = [
                    (self.length_sweep, len_L, max_length_grid),
                    (self.gain_sweep, len_G, max_gain_grid),
                    (self.freq_sweep, len_F, max_freq_grid),
                ]

                opt_param_indices = []
                opt_axes = []
                opt_initial_guess = []
                opt_bounds = []
                fixed_params = {}

                for i, (axis, length, initial_val) in enumerate(all_axes_data):
                    if length > 1 and axis[0] is not None:
                        opt_param_indices.append(i)
                        opt_axes.append(np.array(axis))
                        opt_initial_guess.append(initial_val)
                        opt_bounds.append((np.min(axis), np.max(axis)))
                    else:
                        fixed_params[i] = initial_val

                if not opt_param_indices:
                    print(
                        "\nNo parameters to optimize (all sweeps have length 1 or are None)."
                    )
                    print("Returning grid search result.")
                else:
                    squeezed_fid_Array = np.squeeze(fid_Array)
                    if squeezed_fid_Array.ndim == 0:
                        squeezed_fid_Array = squeezed_fid_Array.reshape(
                            (1,) * len(opt_param_indices)
                        )

                    try:
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="cubic"
                        )
                    except ValueError:
                        print(
                            "Warning: Not enough data for cubic interpolation. Falling back to linear."
                        )
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="linear"
                        )

                    def objective_func(opt_params):
                        return -interpolator(opt_params)[0]

                    result = minimize(
                        objective_func,
                        opt_initial_guess,
                        method="L-BFGS-B",
                        bounds=opt_bounds,
                    )

                    # L-BFGS-B can fail on noisy/oscillating cubic splines
                    # (ABNORMAL_TERMINATION_IN_LNSRCH). Retry with Nelder-Mead
                    # which is gradient-free and more robust on such surfaces.
                    if not result.success:
                        result = minimize(
                            objective_func,
                            opt_initial_guess,
                            method="Nelder-Mead",
                            bounds=opt_bounds,
                            options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000},
                        )

                    if not result.success:
                        print(
                            f"\nWarning: Interpolation optimization failed. {result.message}"
                        )
                        print("Returning grid search result.")
                    else:
                        # Cap: interpolation cannot exceed grid max.
                        # Cubic splines can overshoot (Runge phenomenon), giving
                        # falsely inflated fidelity outside the measured grid.
                        max_fid_interp = min(-result.fun, max_fid_grid)
                        final_params = [None, None, None]

                        for i, param_val in enumerate(result.x):
                            final_params[opt_param_indices[i]] = param_val

                        for i, param_val in fixed_params.items():
                            final_params[i] = param_val

                        max_length, max_gain, max_freq = final_params

                        print(f"\n--- Interpolated Result ---")
                        print(f"Max fidelity (interpolated): {max_fid_interp:.4f}")

                        length_str = (
                            f"{max_length:.3f} us"
                            if max_length is not None
                            else "default"
                        )
                        gain_str = (
                            f"{max_gain:.5f} DAC" if max_gain is not None else "default"
                        )
                        freq_str = (
                            f"{max_freq:.5f} MHz" if max_freq is not None else "default"
                        )

                        print(
                            f"At length={length_str}, gain={gain_str}, freq={freq_str}"
                        )

            except Exception as e:
                print(f"\nAn error occurred during interpolation: {e}")
                print("Returning grid search result.")

        self.fid_Array  = fid_Array
        self.sep_array  = sep_array
        self.sk_g_array = sk_g_array
        self.sk_e_array = sk_e_array

        return_L = round(max_length, 3) if max_length is not None else None
        return_G = round(max_gain, 6) if max_gain is not None else None
        return_F = round(max_freq, 6) if max_freq is not None else None

        return return_L, return_G, return_F

    def plot_grid_analysis(self, skew_thr=0.5):
        """
        Six-panel grid overview + full hist() for the best clean point.

        Panels
        ------
        Top row:
          1. Constrained GMM fidelity heatmap (all grid points)
          2. IQ separation heatmap
          3. Clean fidelity  — points with |sk_g|, |sk_e| < skew_thr shown;
                               polluted points greyed out
        Bottom row:
          4. |g⟩ skewness   (positive = RITS pulling g toward |f⟩)
          5. |e⟩ skewness   (positive = T1 decay pulling e toward |g⟩)
          6. Quality score  (fid − 0.05 × (|sk_g| + |sk_e|))

        After the heatmaps, a full 4-panel hist() is shown for the best clean
        grid point so you can visually inspect the distribution.

        Parameters
        ----------
        skew_thr : float
            Skewness threshold for 'clean' classification (default 0.5).
        """
        if not hasattr(self, "fid_Array"):
            print("Running analyze() first ...")
            self.analyze()
        if not hasattr(self, "sk_g_array"):
            print("Error: quality metrics not found. Re-run analyze().")
            return

        fid_arr = self.fid_Array        # (len_L, len_G, len_F)
        sep_arr = self.sep_array
        skg_arr = self.sk_g_array
        ske_arr = self.sk_e_array
        len_L, len_G, len_F = fid_arr.shape

        # Collapse freq dimension: for each (L, G) take the freq with best fidelity
        best_f = np.argmax(fid_arr, axis=2)                      # (len_L, len_G)
        def _take(arr):
            return np.take_along_axis(arr, best_f[:, :, None], axis=2)[:, :, 0]

        fid_2d     = _take(fid_arr)
        sep_2d     = _take(sep_arr)
        skg_2d     = _take(skg_arr)
        ske_2d     = _take(ske_arr)
        quality_2d = fid_2d - 0.05 * (np.abs(skg_2d) + np.abs(ske_2d))
        clean_mask = (np.abs(skg_2d) < skew_thr) & (np.abs(ske_2d) < skew_thr)
        clean_fid  = np.where(clean_mask, fid_2d, np.nan)

        # Axis tick labels (actual parameter values when available)
        def _labels(sweep):
            if sweep[0] is None:
                return [str(i) for i in range(len(sweep))]
            return [f"{v:.3g}" for v in sweep]

        l_labels = _labels(self.length_sweep)
        g_labels = _labels(self.gain_sweep)
        font_sz  = max(4, 7 - max(len_L, len_G) // 5)

        def _imshow(ax, data, title, cmap, vmin=None, vmax=None,
                    fmt=".3f", cbar_label="", bad_color="0.80"):
            cmap_obj = plt.cm.get_cmap(cmap).copy()
            cmap_obj.set_bad(color=bad_color)
            im = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                           origin="upper", aspect="auto")
            plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Gain", fontsize=9)
            ax.set_ylabel("Length", fontsize=9)
            ax.set_xticks(range(len_G))
            ax.set_xticklabels(g_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len_L))
            ax.set_yticklabels(l_labels, fontsize=7)
            for i in range(len_L):
                for j in range(len_G):
                    v = data[i, j]
                    if np.isnan(v):
                        continue
                    bg  = im.cmap(im.norm(v))
                    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    tc  = "white" if lum < 0.45 else "black"
                    ax.text(j, i, format(v, fmt), ha="center", va="center",
                            fontsize=font_sz, color=tc)

        sk_max = max(float(np.nanmax(np.abs(skg_2d))),
                     float(np.nanmax(np.abs(ske_2d))), 0.1)

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("SingleShot Optimization — Grid Analysis", fontsize=13)

        _imshow(axs[0, 0], fid_2d,
                "Constrained GMM Fidelity", "RdYlGn",
                vmin=0.5, vmax=1.0, fmt=".3f", cbar_label="Fidelity")
        _imshow(axs[0, 1], sep_2d,
                "IQ Separation (LDA)", "Blues",
                vmin=0, fmt=".3f", cbar_label="Separation [ADC]")
        _imshow(axs[0, 2], clean_fid,
                f"Clean Fidelity  (|sk_g|, |sk_e| < {skew_thr})", "RdYlGn",
                vmin=0.5, vmax=1.0, fmt=".3f",
                cbar_label="Fidelity", bad_color="0.80")
        _imshow(axs[1, 0], skg_2d,
                "|g⟩ Skewness  (RITS → positive)", "RdBu_r",
                vmin=-sk_max, vmax=sk_max, fmt="+.2f", cbar_label="Skewness")
        _imshow(axs[1, 1], ske_2d,
                "|e⟩ Skewness  (T1 decay → positive)", "RdBu_r",
                vmin=-sk_max, vmax=sk_max, fmt="+.2f", cbar_label="Skewness")
        _imshow(axs[1, 2], quality_2d,
                "Quality Score  (fid − 0.05·(|sk_g|+|sk_e|))", "RdYlGn",
                fmt=".3f", cbar_label="Score")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # ── Console output ──────────────────────────────────────────────────
        def _val_str(sweep, idx):
            v = sweep[idx]
            return f"{v:.4g}" if v is not None else str(idx)

        all_pts = [
            (l, g, int(best_f[l, g]),
             fid_2d[l, g], sep_2d[l, g], skg_2d[l, g], ske_2d[l, g])
            for l in range(len_L) for g in range(len_G)
        ]
        clean_pts = [(l, g, f, fid, sep, sk_g, sk_e)
                     for l, g, f, fid, sep, sk_g, sk_e in all_pts
                     if abs(sk_g) < skew_thr and abs(sk_e) < skew_thr]
        clean_pts.sort(key=lambda x: x[3], reverse=True)

        print(f"\n=== Top 5 clean points  (|sk_g|, |sk_e| < {skew_thr}) ===")
        print(f"  {'L':>8}  {'G':>8}  {'F':>8}  {'fid':>6}  {'sep':>6}  {'sk_g':>6}  {'sk_e':>6}")
        for l, g, f, fid, sep, sk_g, sk_e in clean_pts[:5]:
            print(f"  {_val_str(self.length_sweep,l):>8}"
                  f"  {_val_str(self.gain_sweep,g):>8}"
                  f"  {_val_str(self.freq_sweep,f):>8}"
                  f"  {fid:.4f}  {sep:.4f}  {sk_g:+.3f}  {sk_e:+.3f}")

        print(f"\n=== Pareto: best fidelity at each skewness budget ===")
        for thr in [0.3, 0.5, 0.7, 1.0]:
            cands = [(fid, sk_g, sk_e, l, g, f)
                     for l, g, f, fid, sep, sk_g, sk_e in all_pts
                     if abs(sk_g) < thr and abs(sk_e) < thr]
            if not cands:
                continue
            fid, sk_g, sk_e, l, g, f = max(cands, key=lambda x: x[0])
            print(f"  skew < {thr:.1f}:  best fid={fid:.4f}"
                  f"  L={_val_str(self.length_sweep,l)}"
                  f"  G={_val_str(self.gain_sweep,g)}"
                  f"  sk_g={sk_g:+.3f}, sk_e={sk_e:+.3f}")

        # ── Full hist for best clean point ──────────────────────────────────
        if clean_pts:
            l, g, f, fid, sep, sk_g, sk_e = clean_pts[0]
            l_v = _val_str(self.length_sweep, l)
            g_v = _val_str(self.gain_sweep, g)
            f_v = _val_str(self.freq_sweep, f)
            print(f"\n=== Full hist for best clean point: "
                  f"L={l_v}, G={g_v}, F={f_v}  (fid={fid:.4f}) ===")
            data_slice = {
                "Ig": self.data["Ig"][l, g, f],
                "Qg": self.data["Qg"][l, g, f],
                "Ie": self.data["Ie"][l, g, f],
                "Qe": self.data["Qe"][l, g, f],
            }
            if getattr(self, "_shot_f", False):
                data_slice["If"] = self.data["If"][l, g, f]
                data_slice["Qf"] = self.data["Qf"][l, g, f]
            hist(data_slice, plot=True, verbose=True,
                 title=(f"Best clean point  L={l_v}, G={g_v}, F={f_v}"
                        f"  —  fid={fid:.4f}  sk_g={sk_g:+.3f}  sk_e={sk_e:+.3f}"))
        else:
            print(f"\nNo clean points found with |sk_g|, |sk_e| < {skew_thr}.")
            print("Consider increasing skew_thr or checking the sweep range.")

        plt.show()

    def plot_top_fidelity_histograms(self, top_n=9):
        if not hasattr(self, "fid_Array"):
            print("Running analyze() to generate fidelity data...")
            self.analyze()
            if not hasattr(self, "fid_Array"):
                print(
                    "Error: Fidelity data not available after running analyze(). Cannot plot."
                )
                return

        fid_Array = self.fid_Array

        if fid_Array.ndim != 3:
            print("Error: fid_Array must be a 3D array (length, gain, freq).")
            return

        flat_fid_Array = fid_Array.flatten()
        top_n_flat_indices = np.argsort(flat_fid_Array)[-top_n:][::-1]
        top_n_indices = np.unravel_index(top_n_flat_indices, fid_Array.shape)

        top_n_I_g = [self.data["Ig"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_g = [self.data["Qg"][idx] for idx in zip(*top_n_indices)]
        top_n_I_e = [self.data["Ie"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_e = [self.data["Qe"][idx] for idx in zip(*top_n_indices)]

        all_I_data = np.concatenate(top_n_I_g + top_n_I_e)
        all_Q_data = np.concatenate(top_n_Q_g + top_n_Q_e)

        overall_min = min(np.min(all_I_data), np.min(all_Q_data))
        overall_max = max(np.max(all_I_data), np.max(all_Q_data))
        range_span = overall_max - overall_min
        padding_val = range_span * 0.05

        plot_min = overall_min - padding_val
        plot_max = overall_max + padding_val
        plot_extent = [plot_min, plot_max, plot_min, plot_max]

        hexbin_gridsize = 50
        grid_size = int(np.ceil(np.sqrt(top_n)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size)
        )
        axes = axes.flatten()

        print(f"\nPlotting top {top_n} fidelity points using hexbin...")
        for i in range(min(top_n, len(axes))):
            l_idx, g_idx, f_idx = (
                top_n_indices[0][i],
                top_n_indices[1][i],
                top_n_indices[2][i],
            )

            I_g = self.data["Ig"][l_idx, g_idx, f_idx]
            Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
            I_e = self.data["Ie"][l_idx, g_idx, f_idx]
            Q_e = self.data["Qe"][l_idx, g_idx, f_idx]

            current_fid = fid_Array[l_idx, g_idx, f_idx]
            length = self.length_sweep[l_idx]
            gain = self.gain_sweep[g_idx]
            freq = self.freq_sweep[f_idx]

            ax = axes[i]
            ax.hexbin(
                I_e,
                Q_e,
                gridsize=hexbin_gridsize,
                cmap="Reds",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )
            ax.hexbin(
                I_g,
                Q_g,
                gridsize=hexbin_gridsize,
                cmap="Blues",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )

            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect("equal", adjustable="box")

            title_str = (
                f"Fidelity: {current_fid:.4f}\n"
                f"L={length:.3f} us, G={gain:.5f} DAC, F={freq:.5f} MHz"
            )
            ax.set_title(title_str, fontsize=10)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

        for j in range(top_n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
