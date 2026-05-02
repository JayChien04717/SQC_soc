"""
Setup/single_shot — s000: Single-shot readout (g/e/f) and optimization.
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram
from ...config.system_cfg import DATA_PATH
from ...tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from .singleshot_utils import plot_hist, general_hist, hist, _fit_gmm


# ── Programs ──────────────────────────────────────────────────────────────────

class SingleShotProgram_gef(BaseProgram):
    """QICK program for g/e/f single-shot readout with multi-trigger body."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_ge_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotOptProgram(BaseProgram):
    """QICK program for single-shot readout optimization (g/e/f states)."""

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
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


# ── Experiment: SingleShot_gef ────────────────────────────────────────────────

class SingleShot_gef:
    """Single-shot readout for g/e/f state discrimination."""

    def __init__(self, config):
        from ...core.base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False):
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f
        prog = SingleShotProgram_gef(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        iq_list = prog.acquire(self.soc, rounds=1, progress=True)
        Ig = iq_list[0][0, :, 0]
        Qg = iq_list[0][0, :, 1]
        Ie = iq_list[0][1, :, 0]
        Qe = iq_list[0][1, :, 1]
        if shot_f:
            If = iq_list[0][2, :, 0]
            Qf = iq_list[0][2, :, 1]
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe, "If": If, "Qf": Qf}
        else:
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe}
        return self.data

    def plot(self, fid_avg=False, verbose=True):
        return hist(self.data, plot=True, verbose=verbose, fid_avg=fid_avg)

    def saveLabber(self, qb_idx, yoko_value=None):
        from ...core.base_experiment import BaseExperiment
        has_f = "If" in self.data
        expt_name = ("s000_singleshot_gef" if has_f else "s000_singleshot_ge") + f"_{qb_idx}"
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        print("Current data file: " + file_path)
        dict_val = config_to_yaml(self.cfg)
        shotdata = np.array([
            self.data["Ig"] + 1j * self.data["Qg"],
            self.data["Ie"] + 1j * self.data["Qe"],
        ] + ([self.data["If"] + 1j * self.data["Qf"]] if has_f else []))
        states = [0, 1, 2] if has_f else [0, 1]
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "# shot", "unit": "#", "values": np.arange(self.cfg["shots"])},
            y_info={"name": "State", "unit": "", "values": states},
            z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
            comment=f"{dict_val}", tag="SingleShot",
        )


# ── Experiment: SingleShot_ge_opt ─────────────────────────────────────────────

class SingleShot_ge_opt:
    """Grid search + GP optimization for single-shot readout parameters."""

    def __init__(self, config):
        from ...core.base_experiment import BaseExperiment
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
            raw_length if isinstance(raw_length, (list, tuple, np.ndarray)) else [raw_length]
        )
        raw_gain = sweep_para.get("gain")
        self.gain_sweep = (
            raw_gain if isinstance(raw_gain, (list, tuple, np.ndarray)) else [raw_gain]
        )
        raw_freq = sweep_para.get("freq")
        self.freq_sweep = (
            raw_freq if isinstance(raw_freq, (list, tuple, np.ndarray)) else [raw_freq]
        )

        final_shape = (len(self.length_sweep), len(self.gain_sweep), len(self.freq_sweep), SHOTS)
        self.I_g_array = np.full(final_shape, np.nan)
        self.Q_g_array = np.full(final_shape, np.nan)
        self.I_e_array = np.full(final_shape, np.nan)
        self.Q_e_array = np.full(final_shape, np.nan)
        if shot_f:
            self.I_f_array = np.full(final_shape, np.nan)
            self.Q_f_array = np.full(final_shape, np.nan)

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

                    ssp = SingleShotOptProgram(
                        self.soccfg, reps=1,
                        final_delay=self.cfg["relax_delay"], cfg=self.cfg,
                    )
                    iq_list = ssp.acquire(self.soc, rounds=1, progress=False)

                    self.I_g_array[l_idx, g_idx, f_idx, :] = iq_list[0][0, :, 0]
                    self.Q_g_array[l_idx, g_idx, f_idx, :] = iq_list[0][0, :, 1]
                    self.I_e_array[l_idx, g_idx, f_idx, :] = iq_list[0][1, :, 0]
                    self.Q_e_array[l_idx, g_idx, f_idx, :] = iq_list[0][1, :, 1]

                    if shot_f:
                        self.I_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 0]
                        self.Q_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 1]

        self.data = {"Ig": self.I_g_array, "Qg": self.Q_g_array,
                     "Ie": self.I_e_array, "Qe": self.Q_e_array}
        if shot_f:
            self.data["If"] = self.I_f_array
            self.data["Qf"] = self.Q_f_array

    @staticmethod
    def _compute_metrics(I_g, Q_g, I_e, Q_e):
        mg = np.array([I_g.mean(), Q_g.mean()])
        me = np.array([I_e.mean(), Q_e.mean()])
        v = me - mg
        n = float(np.linalg.norm(v))
        sep = 0.0
        snr = 0.0
        if n > 1e-12:
            pg = ((I_g - mg[0]) * v[0] + (Q_g - mg[1]) * v[1]) / n
            pe = ((I_e - mg[0]) * v[0] + (Q_e - mg[1]) * v[1]) / n
            sep = n
            snr = n**2 / (pg.var() + pe.var() + 1e-30)
        all_c = np.concatenate([I_g + 1j * Q_g, I_e + 1j * Q_e])
        theta_rad = -np.arctan2(me[1] - mg[1], me[0] - mg[0])
        def _rot_I(c):
            return c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)
        proj_g = _rot_I(I_g + 1j * Q_g)
        proj_e = _rot_I(I_e + 1j * Q_e)
        proj_all = _rot_I(all_c)
        span = (proj_all.max() - proj_all.min()) / 2
        mid = (proj_all.max() + proj_all.min()) / 2
        xlims = [mid - span, mid + span]
        (state_gmms, state_order, conf_matrix, thresholds,
         primary_means, primary_stds, primary_weights) = _fit_gmm([proj_g, proj_e], xlims)
        fid = float(np.mean(np.diag(conf_matrix)))
        soft_accs = []
        for i, proj in enumerate([proj_g, proj_e]):
            X = proj.reshape(-1, 1)
            ll = np.array([gmm.score_samples(X) for gmm in state_gmms])
            ll_shifted = ll - ll.max(axis=0)
            posteriors = np.exp(ll_shifted)
            posteriors /= posteriors.sum(axis=0)
            soft_accs.append(float(posteriors[i].mean()))
        soft_fid = float(np.mean(soft_accs))
        gmm_e = state_gmms[1]
        leakage = float(1.0 - np.max(gmm_e.weights_)) if gmm_e.n_components > 1 else 0.0
        gmm_g = state_gmms[0]
        thermal = float(1.0 - np.max(gmm_g.weights_)) if gmm_g.n_components > 1 else 0.0
        return dict(fid=fid, soft_fid=soft_fid, snr=snr, sep=sep, leakage=leakage, thermal=thermal)

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        is_eff = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                dominated = np.all(costs[is_eff] <= c, axis=1) & np.any(costs[is_eff] < c, axis=1)
                is_eff[is_eff] = ~dominated
                is_eff[i] = True
        return is_eff

    @staticmethod
    def _expected_improvement(gp, X_candidates: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        from scipy.stats import norm as sp_norm
        mu, sigma = gp.predict(X_candidates, return_std=True)
        sigma = sigma.reshape(-1)
        imp = mu - y_best - xi
        Z = np.where(sigma > 1e-9, imp / sigma, 0.0)
        ei = imp * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _acquire_single_point(self, length, gain, freq, SHOTS):
        cfg_update = {"steps": SHOTS}
        if length is not None:
            cfg_update["ro_length"] = length
        if gain is not None:
            cfg_update["res_gain_ge"] = gain
        if freq is not None:
            cfg_update["res_freq_ge"] = freq
        self.cfg.update(cfg_update)
        ssp = SingleShotOptProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )
        iq_list = ssp.acquire(self.soc, rounds=1, progress=False)
        I_g = iq_list[0][0, :, 0]
        Q_g = iq_list[0][0, :, 1]
        I_e = iq_list[0][1, :, 0]
        Q_e = iq_list[0][1, :, 1]
        data_slice = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}
        fid = hist(data_slice, plot=False, verbose=False)[0][0]
        metrics = self._compute_metrics(I_g, Q_g, I_e, Q_e)
        metrics["fid"] = fid
        return I_g, Q_g, I_e, Q_e, metrics

    def analyze(self, leakage_threshold=0.20, thermal_threshold=0.10,
                bo_n_iter=0, bo_xi=0.01, pareto=True):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            GP_AVAILABLE = True
        except ImportError:
            GP_AVAILABLE = False
            print("Warning: scikit-learn not found. GP interpolation disabled.")

        try:
            len_L = len(self.length_sweep)
            len_G = len(self.gain_sweep)
            len_F = len(self.freq_sweep)
        except AttributeError:
            print("Error: call run() first to define sweep axes.")
            return

        shape3 = (len_L, len_G, len_F)
        fid_Array = np.zeros(shape3)
        soft_fid_array = np.zeros(shape3)
        snr_array = np.zeros(shape3)
        sep_array = np.zeros(shape3)
        leakage_array = np.zeros(shape3)
        thermal_array = np.zeros(shape3)

        shot_f = getattr(self, "_shot_f", False)
        metric_label = "GMM fidelity (gef)" if shot_f else "GMM fidelity (ge)"

        for l_idx in tqdm(range(len_L), desc=f"Analyze [{metric_label}]"):
            for g_idx in range(len_G):
                for f_idx in range(len_F):
                    I_g = self.data["Ig"][l_idx, g_idx, f_idx]
                    Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
                    I_e = self.data["Ie"][l_idx, g_idx, f_idx]
                    Q_e = self.data["Qe"][l_idx, g_idx, f_idx]
                    data_slice = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}
                    if shot_f:
                        data_slice["If"] = self.data["If"][l_idx, g_idx, f_idx]
                        data_slice["Qf"] = self.data["Qf"][l_idx, g_idx, f_idx]
                    result = hist(data_slice, plot=False, verbose=False)
                    fid_Array[l_idx, g_idx, f_idx] = result[0][0]
                    m = self._compute_metrics(I_g, Q_g, I_e, Q_e)
                    soft_fid_array[l_idx, g_idx, f_idx] = m["soft_fid"]
                    snr_array[l_idx, g_idx, f_idx] = m["snr"]
                    sep_array[l_idx, g_idx, f_idx] = m["sep"]
                    leakage_array[l_idx, g_idx, f_idx] = m["leakage"]
                    thermal_array[l_idx, g_idx, f_idx] = m["thermal"]

        feasible_mask = (leakage_array <= leakage_threshold) & (thermal_array <= thermal_threshold)
        n_feasible = feasible_mask.sum()
        print(f"\n{n_feasible}/{feasible_mask.size} grid points pass physical constraints.")

        if n_feasible == 0:
            print("Warning: no feasible points. Relaxing constraints.")
            feasible_mask = leakage_array <= (leakage_array.min() + 0.10)

        fid_feasible = np.where(feasible_mask, fid_Array, -np.inf)
        max_idx = np.unravel_index(np.argmax(fid_feasible), fid_feasible.shape)
        max_l_idx, max_g_idx, max_f_idx = max_idx
        best_fid_grid = float(fid_Array[max_idx])
        best_length_grid = self.length_sweep[max_l_idx]
        best_gain_grid = self.gain_sweep[max_g_idx]
        best_freq_grid = self.freq_sweep[max_f_idx]
        max_length, max_gain, max_freq = best_length_grid, best_gain_grid, best_freq_grid

        print(f"\n--- Grid best (feasible) ---")
        print(f"  fid={best_fid_grid:.4f}  length={best_length_grid}  "
              f"gain={best_gain_grid}  freq={best_freq_grid}")

        self.fid_Array = fid_Array
        self.soft_fid_array = soft_fid_array
        self.snr_array = snr_array
        self.sep_array = sep_array
        self.leakage_array = leakage_array
        self.thermal_array = thermal_array
        self._feasible_mask = feasible_mask

        return_L = round(float(max_length), 3) if max_length is not None else None
        return_G = round(float(max_gain), 6) if max_gain is not None else None
        return_F = round(float(max_freq), 6) if max_freq is not None else None
        return return_L, return_G, return_F


__all__ = [
    "SingleShotProgram_gef",
    "SingleShot_gef",
    "SingleShotOptProgram",
    "SingleShot_ge_opt",
    "plot_hist",
    "general_hist",
    "hist",
]
