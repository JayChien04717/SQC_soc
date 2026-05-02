"""
AutoCalibrate: Rebuilt ge calibration pipeline using the composite framework.

Reproduces the full qick_workspace auto_calibrate.py logic with:
  - ExperimentData returns from every step
  - CalibrationStore updates after each step
  - GP-guided Ramsey frequency correction preserved
  - BatchExperiment-compatible interface
"""

from __future__ import annotations

import numpy as np

from ..core.experiment_data import ExperimentData
from .store import CalibrationStore


class AutoCalibrate:
    """
    Automated single-qubit ge calibration pipeline.

    Parameters
    ----------
    config_all : ExperimentConfig
        Live config store (``QickworkspaceV2.config.system_cfg.ExperimentConfig``).
    qubit : str
        Qubit label, e.g. ``"Q1"``.
    cal_store : CalibrationStore or None
        If provided, results are persisted after each step.
    """

    TARGET_PI_GAIN = 0.5
    MAX_RABI_RETRIES = 3
    MAX_RAMSEY_RETRIES = 3
    RAMSEY_DETUNE_LIMIT_MHZ = 0.05

    def __init__(self, config_all, qubit: str, cal_store: CalibrationStore | None = None):
        from ..core.base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.config_all = config_all
        self.qubit = qubit
        self.cal_store = cal_store
        self.results: dict = {}
        self._T2r: float | None = None
        self._T2e: float | None = None
        self._T1:  float | None = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cfg(self) -> dict:
        return self.config_all.get_qubit(self.qubit)

    def _update(self, key: str, value):
        self.config_all.update(key, value, q_index=self.qubit)
        if self.cal_store is not None:
            self.cal_store.set(self.qubit, key, value)

    def _log(self, step: str, msg: str):
        print(f"  [{step}] {msg}")

    # ── Pipeline runner ───────────────────────────────────────────────────────

    def run(self, skip: tuple = ()):
        PIPELINE = [
            ("res_spec",   self.step_res_spec),
            ("qubit_spec", self.step_qubit_spec),
            ("power_rabi", self.step_power_rabi),
            ("ramsey",     self.step_ramsey),
            ("spin_echo",  self.step_spin_echo),
            ("t1",         self.step_t1),
            ("ss_opt",     self.step_ss_opt),
        ]
        for name, fn in PIPELINE:
            if name in skip:
                print(f"[auto_cal] skip: {name}")
                continue
            print(f"\n{'='*60}")
            print(f"  AUTO-CAL  |  {name}  |  qubit = {self.qubit}")
            print(f"{'='*60}")
            fn()

    # ── Step 1 — Resonator spectroscopy ──────────────────────────────────────

    def step_res_spec(self, span=10.0, steps=101, py_avg=5):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.resonator.res_spec import ResonatorSpec
        center = self._cfg()["res_freq_ge"]
        for attempt, half_span in enumerate([span, span * 2, span * 4]):
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("res_freq_ge", QickSweep1D("freqloop", center - half_span, center + half_span)),
                ("relax_delay", 0),
            ])
            expt = ResonatorSpec(run_cfg)
            result: ExperimentData = expt.run(py_avg, solve_type="hm")
            if result.fit_result and result.fit_result.get("f0_Hz") is not None:
                freq_mhz = round(result.fit_result["f0_Hz"] / 1e6, 4)
                self._update("res_freq_ge", freq_mhz)
                self.results["res_freq_ge"] = freq_mhz
                self._log("res_spec", f"res_freq_ge = {freq_mhz} MHz")
                return freq_mhz
            self._log("res_spec", f"no fit (attempt {attempt+1}), expanding to ±{half_span*2:.0f} MHz")
        raise RuntimeError("res_spec: circle-fit failed after 3 attempts")

    # ── Step 2 — Qubit spectroscopy ───────────────────────────────────────────

    def step_qubit_spec(self, span=50.0, steps=101, py_avg=5):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.qubit_ge.qubit_spec import QubitSpec
        center = self._cfg()["qb_freq_ge"]
        for attempt, half_span in enumerate([span, span * 2, span * 4]):
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("qb_freq_ge", QickSweep1D("freqloop", center - half_span, center + half_span)),
                ("qb_mixer", center),
                ("qb_gain_ge", 0.1),
                ("qb_flat_top_length_ge", 1.0),
                ("relax_delay", 1),
            ])
            expt = QubitSpec(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            if result.scalar_result is not None:
                freq_mhz = round(float(result.scalar_result), 4)
                self._update("qb_freq_ge", freq_mhz)
                self._update("qb_mixer", freq_mhz)
                self.results["qb_freq_ge"] = freq_mhz
                if result.fit_params is not None and len(result.fit_params) > 1:
                    linewidth = abs(float(result.fit_params[1]))
                    if linewidth > 1e-6:
                        self._T2r = round(1.0 / (np.pi * linewidth), 3)
                        self._log("qubit_spec", f"linewidth={linewidth:.3f} MHz → T2* seed={self._T2r:.2f} µs")
                self._log("qubit_spec", f"qb_freq_ge = {freq_mhz} MHz")
                return freq_mhz
            self._log("qubit_spec", f"no fit (attempt {attempt+1}), expanding to ±{half_span*2:.0f} MHz")
        raise RuntimeError("qubit_spec: Lorentzian fit failed after 3 attempts")

    # ── Step 3 — Power Rabi ───────────────────────────────────────────────────

    def step_power_rabi(self, steps=100, py_avg=10):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.qubit_ge.rabi import PowerRabi
        for attempt in range(self.MAX_RABI_RETRIES):
            run_cfg = self._cfg()
            sigma = run_cfg.get("sigma_ge", 0.01)
            run_cfg.update([
                ("steps", steps),
                ("qb_gain_ge", QickSweep1D("gainloop", 0.0, 1.0)),
            ])
            expt = PowerRabi(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            if result.fit_result.get("pi_gain") is None:
                raise RuntimeError("power_rabi: fit returned None")
            pi_gain  = float(result.fit_result["pi_gain"])
            pi2_gain = float(result.fit_result["pi2_gain"])
            if 0.15 <= pi_gain <= 0.90:
                self._update("pi_gain_ge",  round(pi_gain,  6))
                self._update("pi2_gain_ge", round(pi2_gain, 6))
                self.results["pi_gain_ge"]  = pi_gain
                self.results["pi2_gain_ge"] = pi2_gain
                self._log("power_rabi", f"pi_gain_ge={pi_gain:.4f}, pi2_gain_ge={pi2_gain:.4f}")
                return pi_gain, pi2_gain
            new_sigma = round(sigma * (pi_gain / self.TARGET_PI_GAIN), 5)
            direction = "increase" if pi_gain > 0.90 else "decrease"
            self._log("power_rabi", f"attempt {attempt+1}: pi_gain={pi_gain:.4f} out of range → "
                      f"{direction} sigma_ge: {sigma:.5f} → {new_sigma:.5f}")
            self._update("sigma_ge", new_sigma)
        raise RuntimeError(f"power_rabi: pi_gain still out of range after {self.MAX_RABI_RETRIES} retries")

    # ── Step 4 — Ramsey ───────────────────────────────────────────────────────

    def step_ramsey(self, steps=100, py_avg=20, ramsey_freq=2.0):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.ramsey import Ramsey
        stop = max(2.0, 3.0 * self._T2r) if self._T2r else 5.0
        _freq_history:  list[float] = []
        _error_history: list[float] = []

        for attempt in range(self.MAX_RAMSEY_RETRIES + 2):
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
                ("ramsey_freq", ramsey_freq),
            ])
            expt = Ramsey(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            fp = result.fit_params
            T2r        = abs(float(fp[3]))
            signed_err = float(fp[1]) - ramsey_freq
            abs_err    = abs(signed_err)
            self._T2r  = T2r
            self.results["T2r_us"] = T2r
            _freq_history.append(self._cfg()["qb_freq_ge"])
            _error_history.append(signed_err)
            self._log("ramsey", f"attempt {attempt+1}: T2*={T2r:.2f} µs, detune error={signed_err:+.4f} MHz")

            if abs_err < self.RAMSEY_DETUNE_LIMIT_MHZ:
                corrected = expt.correct_detune()
                self._update("qb_freq_ge", corrected)
                self._log("ramsey", f"converged → qb_freq_ge={corrected:.4f} MHz")
                return T2r, corrected

            gp_freq = None
            if len(_freq_history) >= 2:
                gp_freq = self._gp_predict_zero_crossing(_freq_history, _error_history)
            if gp_freq is not None:
                self._update("qb_freq_ge", gp_freq)
                self._log("ramsey", f"  → GP correction: qb_freq_ge={gp_freq:.4f} MHz")
            else:
                corrected = expt.correct_detune()
                self._update("qb_freq_ge", corrected)
            stop = max(2.0, min(3.0 * T2r, 15.0))

        self._log("ramsey", f"did not converge in {self.MAX_RAMSEY_RETRIES+2} attempts, proceeding")
        return self._T2r, self._cfg()["qb_freq_ge"]

    # ── Step 5 — Spin Echo ────────────────────────────────────────────────────

    def step_spin_echo(self, steps=100, py_avg=100):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.spin_echo import SpinEcho
        T2r = self._T2r or 2.0
        T2e_est = 3.0 * T2r
        stop = max(20.0, 5.0 * T2e_est)
        ramsey_freq = round(min(0.5, 1.5 / stop), 4)
        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
            ("ramsey_freq", ramsey_freq),
        ])
        expt = SpinEcho(run_cfg)
        result: ExperimentData = expt.run(py_avg)
        fp = result.fit_params
        T2e = float(fp[3]) if ramsey_freq != 0 else float(fp[2])
        self._T2e = T2e
        self.results["T2e_us"] = T2e
        self._log("spin_echo", f"T2E={T2e:.2f} µs (stop={stop:.0f} µs, ramsey_freq={ramsey_freq} MHz)")
        return T2e

    # ── Step 6 — T1 ──────────────────────────────────────────────────────────

    def step_t1(self, steps=100, py_avg=50):
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.t1 import T1
        T_ref = self._T2e or self._T2r or 5.0
        stop  = max(50.0, 3.0 * T_ref)
        relax = max(float(self._cfg().get("relax_delay", 100)), 5.0 * T_ref)
        self._update("relax_delay", round(relax))
        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
        ])
        expt = T1(run_cfg)
        result: ExperimentData = expt.run(py_avg)
        T1_val = float(result.fit_params[2])
        self._T1 = T1_val
        self.results["T1_us"] = T1_val
        new_relax = max(100, round(5.0 * T1_val))
        self._update("relax_delay", new_relax)
        self._log("t1", f"T1={T1_val:.2f} µs → relax_delay={new_relax} µs")
        return T1_val

    # ── Step 7 — Single-shot optimisation ────────────────────────────────────

    def step_ss_opt(self, shots=1000, coarse_pts=(5, 3, 3), bo_n_iter=15, bo_xi=0.02):
        from ..experiments.setup.single_shot import SingleShot_ge_opt, SingleShot_gef, hist
        run_cfg = self._cfg()
        freq_centre = run_cfg["res_freq_ge"]
        n_freq, n_gain, n_len = coarse_pts
        sweep_para = {
            "freq":   np.linspace(freq_centre - 1.0, freq_centre + 8.0, n_freq),
            "gain":   np.linspace(0.07, 0.10, n_gain),
            "length": np.linspace(2.0, 4.0, n_len),
        }
        self._log("ss_opt", f"Phase 1: coarse grid {n_freq}×{n_gain}×{n_len} = {n_freq*n_gain*n_len} pts")
        ssh_opt = SingleShot_ge_opt(run_cfg)
        ssh_opt.run(shots, sweep_para=sweep_para)
        self._log("ss_opt", f"Phase 2: GP surrogate + {bo_n_iter} online BO steps")
        length, gain, freq = ssh_opt.analyze(bo_n_iter=bo_n_iter, bo_xi=bo_xi, pareto=True)

        self._update("ro_length",   length)
        self._update("res_gain_ge", gain)
        self._update("res_freq_ge", freq)
        self._update("res_phase", 0)

        run_cfg = self._cfg()
        ssh = SingleShot_gef(run_cfg)
        ssh.run(5000, shot_f=False)
        ssh_result = ssh.plot(fid_avg=True, verbose=True)
        phase    = float(ssh_result[2])
        fidelity = float(ssh_result[0][0]) if hasattr(ssh_result[0], "__len__") else float(ssh_result[0])
        self._update("res_phase", phase)

        self.results.update(dict(ro_length=length, res_gain_ge=gain, res_freq_ge=freq,
                                 res_phase_deg=phase, readout_fidelity=fidelity))
        self._log("ss_opt", f"Best: length={length:.3f} µs, gain={gain:.5f}, freq={freq:.4f} MHz")
        self._log("ss_opt", f"res_phase={phase:.2f} deg, fidelity={fidelity:.4f}")
        return length, gain, freq

    # ── GP zero-crossing predictor ────────────────────────────────────────────

    def _gp_predict_zero_crossing(self, freq_vals: list[float], error_vals: list[float]) -> float | None:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            return None
        try:
            X = np.array(freq_vals).reshape(-1, 1)
            y = np.array(error_vals)
            x_centre = float(X.mean())
            x_scale  = max(float(X.std()), 1e-3)
            Xn = (X - x_centre) / x_scale
            kernel = (
                RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(Xn, y)
            x_lo = freq_vals[-1] - 5.0 * x_scale
            x_hi = freq_vals[-1] + 5.0 * x_scale
            x_search = np.linspace(x_lo, x_hi, 2000)
            y_pred, y_std = gp.predict((x_search - x_centre).reshape(-1, 1) / x_scale, return_std=True)
            zero_idx   = int(np.argmin(np.abs(y_pred)))
            pred_freq  = round(float(x_search[zero_idx]), 4)
            pred_sigma = float(y_std[zero_idx])
            self._log("ramsey", f"  GP zero-crossing: {pred_freq:.4f} MHz (σ={pred_sigma:.4f} MHz)")
            if pred_sigma > 5.0 * self.RAMSEY_DETUNE_LIMIT_MHZ:
                self._log("ramsey", "  GP uncertainty too large — falling back to naive correction")
                return None
            return pred_freq
        except Exception as exc:
            self._log("ramsey", f"  GP prediction failed ({exc}) — using naive correction")
            return None

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  Auto-Calibration Summary  —  qubit {self.qubit}")
        print(f"{'='*60}")
        for key, val in self.results.items():
            print(f"  {key:<28s} = {val:.6g}" if isinstance(val, float) else f"  {key:<28s} = {val}")
        print(f"{'='*60}")
