import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ----- Qick package ----- #
from qick import *

# ----- User Library ----- #
from .base_program import BaseProgram


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

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
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

    @staticmethod
    def _auc_fidelity(I_g, Q_g, I_e, Q_e):
        """
        Estimate readout fidelity as the AUC of the ROC curve on the
        LDA-projected 1-D data.

        Why AUC instead of per-state BIC-GMM
        -------------------------------------
        With ~1000 shots per state and heavily overlapping IQ clouds, per-
        state BIC-GMM overfits:
        - BIC selects 2 components for |g⟩ because the overlap region from
          |e⟩ shots looks like a second mode.
        - The resulting Bayes boundary is tuned to training-data noise and
          gives optimistic in-sample scores (87 % → actual 68 %).
        - Cubic interpolation then extrapolates beyond the grid, adding
          further inflation (87 % → 90 %).

        AUC is the Wilcoxon–Mann–Whitney statistic:
            AUC = P(proj_e > proj_g)  for a randomly chosen pair
        It equals the balanced accuracy at the empirically optimal threshold,
        computed WITHOUT any model fitting.

        Properties
        ----------
        - Zero free parameters  → no overfitting, reproducible across shot counts
        - Handles any distribution shape (T1 tails, bimodal |e⟩, etc.)
        - Variance ≈ AUC(1-AUC)/n  →  std < 1.5 % at n=1000 shots/state
        - Equivalent to the non-parametric threshold sweep (what humans do
          when reading a histogram): finds the empirically optimal cut

        Returns 0.5 (chance level) if data is degenerate.
        """
        from sklearn.metrics import roc_auc_score

        # ── Project 2D IQ onto the g→e mean direction (LDA axis) ──────────
        mean_g = np.array([np.mean(I_g), np.mean(Q_g)])
        mean_e = np.array([np.mean(I_e), np.mean(Q_e)])
        vec    = mean_e - mean_g
        norm   = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return 0.5

        proj_g = ((I_g - mean_g[0]) * vec[0] + (Q_g - mean_g[1]) * vec[1]) / norm
        proj_e = ((I_e - mean_g[0]) * vec[0] + (Q_e - mean_g[1]) * vec[1]) / norm

        # ── AUC = P(proj_e > proj_g) = balanced accuracy at optimal threshold
        scores = np.concatenate([proj_g, proj_e])
        labels = np.array([0] * len(proj_g) + [1] * len(proj_e))

        try:
            return float(roc_auc_score(labels, scores))
        except Exception:
            return 0.5

    @staticmethod
    def _auc_fidelity_gef(I_g, Q_g, I_e, Q_e, I_f, Q_f):
        """
        Average pairwise AUC for three-state (g/e/f) discrimination.
        Computes AUC for (g vs e), (g vs f), (e vs f) and returns their mean.
        """
        from sklearn.metrics import roc_auc_score

        pairs = [
            (I_g, Q_g, I_e, Q_e),
            (I_g, Q_g, I_f, Q_f),
            (I_e, Q_e, I_f, Q_f),
        ]
        aucs = []
        for I1, Q1, I2, Q2 in pairs:
            mean1 = np.array([np.mean(I1), np.mean(Q1)])
            mean2 = np.array([np.mean(I2), np.mean(Q2)])
            vec   = mean2 - mean1
            norm  = float(np.linalg.norm(vec))
            if norm < 1e-12:
                aucs.append(0.5)
                continue
            proj1 = ((I1 - mean1[0]) * vec[0] + (Q1 - mean1[1]) * vec[1]) / norm
            proj2 = ((I2 - mean1[0]) * vec[0] + (Q2 - mean1[1]) * vec[1]) / norm
            scores = np.concatenate([proj1, proj2])
            labels = np.array([0] * len(proj1) + [1] * len(proj2))
            try:
                aucs.append(float(roc_auc_score(labels, scores)))
            except Exception:
                aucs.append(0.5)
        return float(np.mean(aucs))

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

        shot_f = getattr(self, "_shot_f", False)
        metric_label = "AUC fidelity (gef avg pairwise)" if shot_f else "AUC fidelity (ge)"

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

                    if shot_f:
                        fid_Array[l_idx, g_idx, f_idx] = self._auc_fidelity_gef(
                            I_g, Q_g, I_e, Q_e,
                            I_f_data[l_idx, g_idx, f_idx],
                            Q_f_data[l_idx, g_idx, f_idx],
                        )
                    else:
                        fid_Array[l_idx, g_idx, f_idx] = self._auc_fidelity(
                            I_g, Q_g, I_e, Q_e
                        )

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

        self.fid_Array = fid_Array

        return_L = round(max_length, 3) if max_length is not None else None
        return_G = round(max_gain, 6) if max_gain is not None else None
        return_F = round(max_freq, 6) if max_freq is not None else None

        return return_L, return_G, return_F

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
