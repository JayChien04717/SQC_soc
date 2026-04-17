"""
auto_calibrate.py — Automated single-qubit ge calibration pipeline.

Follows the procedure in autocal.md:
  1. Resonator spectroscopy      → res_freq_ge
  2. Qubit spectroscopy ge       → qb_freq_ge
  3. Power Rabi ge               → pi_gain_ge, pi2_gain_ge  (with sigma rescaling)
  4. Ramsey ge                   → qb_freq_ge (fine-tuned), T2*
  5. Spin Echo ge                → T2E
  6. T1 ge                       → T1, relax_delay
  7. Single-shot optimisation    → ro_length, res_gain_ge, res_freq_ge, res_phase

Usage
-----
    from qick_workspace.scrip.auto_calibrate import AutoCalibrate

    cal = AutoCalibrate(config_all, qubit="Q1")
    cal.run()                          # full pipeline
    cal.run(skip=("spin_echo", "t1"))  # skip coherence steps
    cal.summary()                      # print results table
"""

import numpy as np
from qick.asm_v2 import QickSweep1D

from .s002_res_spec_ge import ResonatorSpec
from .s003_qubit_spec_ge import QubitSpec
from .s005_power_rabi_ge import PowerRabi
from .s006_Ramsey_ge import Ramsey
from .s007_SpinEcho_ge import SpinEcho
from .s008_T1_ge import T1
from .s000_SingleShot_opt import SingleShot_ge_opt
from .s000_SingleShot_prog import SingleShot_gef


class AutoCalibrate:
    """
    Automated single-qubit ge calibration pipeline.

    Adaptive sweep parameters are computed from the evolving config_all state.
    Each step updates config_all in place before the next step runs.

    Parameters
    ----------
    config_all : ExperimentConfig
        Live config store for all qubits.
    qubit : str
        Qubit label, e.g. "Q1".
    """

    # --- Rabi sigma rescaling ---
    TARGET_PI_GAIN = 0.5       # aim to land pi_gain here after rescaling
    MAX_RABI_RETRIES = 3

    # --- Ramsey convergence ---
    MAX_RAMSEY_RETRIES = 3
    RAMSEY_DETUNE_LIMIT_MHZ = 0.05   # stop iterating when |error| < 50 kHz

    def __init__(self, config_all, qubit):
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.config_all = config_all
        self.qubit = qubit
        self.results = {}

        # Populated as the pipeline runs; used for adaptive time ranges
        self._T2r = None   # Ramsey T2*  [us]
        self._T2e = None   # Spin Echo T2E  [us]
        self._T1  = None   # T1  [us]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _cfg(self):
        """Return a fresh config dict for this qubit."""
        return self.config_all.get_qubit(self.qubit)

    def _update(self, key, value):
        self.config_all.update(key, value, q_index=self.qubit)

    def _log(self, step, msg):
        print(f"  [{step}] {msg}")

    # -------------------------------------------------------------------------
    # Top-level runner
    # -------------------------------------------------------------------------

    def run(self, skip=()):
        """
        Run the full calibration pipeline in order.

        Parameters
        ----------
        skip : tuple of str
            Names of steps to skip.
            Valid names: "res_spec", "qubit_spec", "power_rabi",
                         "ramsey", "spin_echo", "t1", "ss_opt".
        """
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

    # -------------------------------------------------------------------------
    # Step 1 — Resonator spectroscopy
    # -------------------------------------------------------------------------

    def step_res_spec(self, span=10.0, steps=101, py_avg=5):
        """
        Sweep resonator frequency around the current config value.
        Doubles the span up to x4 if the circle-fit fails.

        Parameters
        ----------
        span    : float — initial half-span [MHz], default ±10 MHz
        steps   : int
        py_avg  : int

        Returns
        -------
        res_freq_ge : float  [MHz]
        """
        center = self._cfg()["res_freq_ge"]

        for attempt, half_span in enumerate([span, span * 2, span * 4]):
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("res_freq_ge", QickSweep1D("freqloop", center - half_span, center + half_span)),
                ("relax_delay", 0),
            ])
            expt = ResonatorSpec(run_cfg)
            result = expt.run(py_avg=py_avg, solve_type="hm")

            if result is not None:
                freq_mhz = round(result[0] / 1e6, 4)
                self._update("res_freq_ge", freq_mhz)
                self.results["res_freq_ge"] = freq_mhz
                self._log("res_spec", f"res_freq_ge = {freq_mhz} MHz")
                return freq_mhz

            self._log("res_spec",
                      f"no fit (attempt {attempt + 1}), "
                      f"expanding to ±{half_span * 2:.0f} MHz")

        raise RuntimeError(
            "res_spec: circle-fit failed after 3 attempts — "
            "check probe power or widen the initial span"
        )

    # -------------------------------------------------------------------------
    # Step 2 — Qubit spectroscopy
    # -------------------------------------------------------------------------

    def step_qubit_spec(self, span=50.0, steps=101, py_avg=5):
        """
        Two-tone spectroscopy around the current qb_freq_ge config value.
        Doubles the span up to x4 if the Lorentzian fit fails.
        Also extracts linewidth to seed the Ramsey stop-time estimate.

        Parameters
        ----------
        span    : float — initial half-span [MHz], default ±50 MHz
        steps   : int
        py_avg  : int

        Returns
        -------
        qb_freq_ge : float  [MHz]
        """
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
            result = expt.run(py_avg)

            if result is not None:
                freq_mhz = round(float(result), 4)
                self._update("qb_freq_ge", freq_mhz)
                self._update("qb_mixer", freq_mhz)
                self.results["qb_freq_ge"] = freq_mhz

                # Seed T2* estimate from Lorentzian linewidth: T2* ≈ 1/(π × FWHM)
                if hasattr(expt, "fit_params") and expt.fit_params is not None:
                    linewidth = abs(float(expt.fit_params[1]))   # FWHM [MHz]
                    if linewidth > 1e-6:
                        self._T2r = round(1.0 / (np.pi * linewidth), 3)
                        self._log("qubit_spec",
                                  f"linewidth = {linewidth:.3f} MHz → "
                                  f"T2* seed = {self._T2r:.2f} µs")

                self._log("qubit_spec", f"qb_freq_ge = {freq_mhz} MHz")
                return freq_mhz

            self._log("qubit_spec",
                      f"no fit (attempt {attempt + 1}), "
                      f"expanding to ±{half_span * 2:.0f} MHz")

        raise RuntimeError(
            "qubit_spec: Lorentzian fit failed after 3 attempts — "
            "check drive power or widen the initial span"
        )

    # -------------------------------------------------------------------------
    # Step 3 — Power Rabi
    # -------------------------------------------------------------------------

    def step_power_rabi(self, steps=100, py_avg=10):
        """
        Sweep qubit drive gain 0 → 1.

        If pi_gain is outside [0.15, 0.90], sigma_ge is rescaled by the
        ratio pi_gain / TARGET_PI_GAIN so that the next scan lands near
        TARGET_PI_GAIN.  Repeats up to MAX_RABI_RETRIES times.

        Returns
        -------
        (pi_gain_ge, pi2_gain_ge) : (float, float)
        """
        for attempt in range(self.MAX_RABI_RETRIES):
            run_cfg = self._cfg()
            sigma = run_cfg.get("sigma_ge", 0.01)

            run_cfg.update([
                ("steps", steps),
                ("qb_gain_ge", QickSweep1D("gainloop", 0.0, 1.0)),
            ])
            expt = PowerRabi(run_cfg)
            result = expt.run(py_avg)

            if result is None:
                raise RuntimeError(
                    "power_rabi: fit returned None — check qubit frequency and drive channel"
                )

            pi_gain, pi2_gain = result

            if 0.15 <= pi_gain <= 0.90:
                self._update("pi_gain_ge", round(pi_gain, 6))
                self._update("pi2_gain_ge", round(pi2_gain, 6))
                self.results["pi_gain_ge"] = pi_gain
                self.results["pi2_gain_ge"] = pi2_gain
                self._log("power_rabi",
                          f"pi_gain_ge = {pi_gain:.4f}, pi2_gain_ge = {pi2_gain:.4f}")
                return pi_gain, pi2_gain

            # Rescale sigma: pi_gain ∝ 1/sigma, so one rescale is exact
            new_sigma = round(sigma * (pi_gain / self.TARGET_PI_GAIN), 5)
            direction = "increase" if pi_gain > 0.90 else "decrease"
            self._log("power_rabi",
                      f"attempt {attempt + 1}: pi_gain = {pi_gain:.4f} out of range → "
                      f"{direction} sigma_ge: {sigma:.5f} → {new_sigma:.5f}")
            self._update("sigma_ge", new_sigma)

        raise RuntimeError(
            f"power_rabi: pi_gain still out of [0.15, 0.90] after "
            f"{self.MAX_RABI_RETRIES} sigma rescales"
        )

    # -------------------------------------------------------------------------
    # Step 4 — Ramsey (frequency fine-tune + T2*)
    # -------------------------------------------------------------------------

    def step_ramsey(self, steps=100, py_avg=20, ramsey_freq=2.0):
        """
        GP-guided Ramsey fringe to fine-tune qb_freq_ge and measure T2*.

        On each retry the signed detuning error and the qubit frequency that
        produced it are recorded.  Once two or more observations are available,
        a 1-D sklearn GP is fitted to the (qb_freq_tried, signed_error) history
        and the predicted zero-crossing is used as the next qubit frequency
        guess instead of the naive ``correct_detune()`` result.  This makes
        convergence robust to noisy fits and can reduce the number of retries
        needed.

        Falls back to the original ``correct_detune()`` when
        - fewer than 2 observations are available, or
        - sklearn is not installed, or
        - the GP uncertainty exceeds ``5 × RAMSEY_DETUNE_LIMIT_MHZ``.

        Parameters
        ----------
        ramsey_freq : float — artificial detuning [MHz], default 2 MHz

        Returns
        -------
        (T2r_us, qb_freq_ge) : (float, float)
        """
        stop = max(2.0, 3.0 * self._T2r) if self._T2r else 5.0

        # GP history: lists of (qb_freq_tried [MHz], signed_detune_error [MHz])
        _freq_history  = []
        _error_history = []

        for attempt in range(self.MAX_RAMSEY_RETRIES + 2):   # +2 budget for GP
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
                ("ramsey_freq", ramsey_freq),
            ])
            expt = Ramsey(run_cfg)
            fit_params, _ = expt.run(py_avg)

            # fit_params from fitdecaysin: [amplitude, freq, phase, tau]
            T2r        = abs(float(fit_params[3]))
            signed_err = float(fit_params[1]) - ramsey_freq   # positive → freq too low
            abs_err    = abs(signed_err)
            self._T2r            = T2r
            self.results["T2r_us"] = T2r

            # Record (freq_tried, residual_error) for GP
            _freq_history.append(self._cfg()["qb_freq_ge"])
            _error_history.append(signed_err)

            self._log("ramsey",
                      f"attempt {attempt + 1}: T2* = {T2r:.2f} µs, "
                      f"detune error = {signed_err:+.4f} MHz")

            if abs_err < self.RAMSEY_DETUNE_LIMIT_MHZ:
                corrected_freq = expt.correct_detune()
                self._update("qb_freq_ge", corrected_freq)
                self._log("ramsey",
                          f"converged → qb_freq_ge = {corrected_freq:.4f} MHz")
                return T2r, corrected_freq

            # --- Choose next correction strategy ---
            gp_freq = None
            if len(_freq_history) >= 2:
                gp_freq = self._gp_predict_zero_crossing(_freq_history, _error_history)

            if gp_freq is not None:
                self._update("qb_freq_ge", gp_freq)
                self._log("ramsey", f"  → GP correction: qb_freq_ge = {gp_freq:.4f} MHz")
            else:
                corrected_freq = expt.correct_detune()
                self._update("qb_freq_ge", corrected_freq)

            # Narrow the stop window around measured T2* for next iteration
            stop = max(2.0, min(3.0 * T2r, 15.0))

        self._log("ramsey",
                  f"did not converge in {self.MAX_RAMSEY_RETRIES + 2} attempts, "
                  f"proceeding with last result")
        return self._T2r, self._cfg()["qb_freq_ge"]

    # -------------------------------------------------------------------------
    # Step 5 — Spin Echo
    # -------------------------------------------------------------------------

    def step_spin_echo(self, steps=100, py_avg=100):
        """
        Hahn echo to measure T2E.

        Adaptive ranges (from autocal.md):
          T2E_estimate = 3 × T2*
          stop         = max(20, 5 × T2E_estimate)
          ramsey_freq  = 1.5 / stop   → 2-3 oscillations in window

        Returns
        -------
        T2e_us : float
        """
        T2r = self._T2r or 2.0
        T2e_est = 3.0 * T2r
        stop = max(20.0, 5.0 * T2e_est)
        ramsey_freq = round(min(0.5, 1.5 / stop), 4)   # keep fringes visible

        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
            ("ramsey_freq", ramsey_freq),
        ])
        expt = SpinEcho(run_cfg)
        fit_params, _ = expt.run(py_avg)

        # tau index: 3 for decaysin (ramsey_freq≠0), 2 for expfunc (ramsey_freq==0)
        T2e = float(fit_params[3]) if ramsey_freq != 0 else float(fit_params[2])
        self._T2e = T2e
        self.results["T2e_us"] = T2e
        self._log("spin_echo",
                  f"T2E = {T2e:.2f} µs  "
                  f"(stop = {stop:.0f} µs, ramsey_freq = {ramsey_freq} MHz)")
        return T2e

    # -------------------------------------------------------------------------
    # Step 6 — T1
    # -------------------------------------------------------------------------

    def step_t1(self, steps=100, py_avg=50):
        """
        T1 decay measurement.

        Adaptive ranges (from autocal.md):
          T1_estimate = T2E   (T1 ≈ T2E for transmons; safe lower bound)
          stop        = max(50, 3 × T1_estimate)
          relax_delay is updated to 5 × T1 before and after the measurement.

        Returns
        -------
        T1_us : float
        """
        T_ref = self._T2e or self._T2r or 5.0
        stop = max(50.0, 3.0 * T_ref)

        # Pre-update relax_delay so the T1 scan itself is unbiased
        relax = max(float(self._cfg().get("relax_delay", 100)), 5.0 * T_ref)
        self._update("relax_delay", round(relax))

        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
        ])
        expt = T1(run_cfg)
        fit_params, _ = expt.run(py_avg)

        # fit_params from fitexp: [amplitude, offset, tau]
        T1_val = float(fit_params[2])
        self._T1 = T1_val
        self.results["T1_us"] = T1_val

        # Final relax_delay with the measured T1
        new_relax = max(100, round(5.0 * T1_val))
        self._update("relax_delay", new_relax)
        self._log("t1", f"T1 = {T1_val:.2f} µs → relax_delay = {new_relax} µs")
        return T1_val

    # -------------------------------------------------------------------------
    # Step 7 — Single-shot readout optimisation
    # -------------------------------------------------------------------------

    def step_ss_opt(
        self,
        shots       = 1000,
        coarse_pts  = (5, 3, 3),
        bo_n_iter   = 15,
        bo_xi       = 0.02,
    ):
        """
        Two-phase Bayesian-optimised single-shot readout parameter search.

        Phase 1 — Coarse grid  (hardware runs = product of coarse_pts)
        ----------------------------------------------------------------
        Runs a sparse Cartesian grid over (freq, gain, length) to map the
        fidelity landscape.  Default 5×3×3 = 45 points × 1000 shots each.
        ``SingleShot_ge_opt.analyze()`` fits a GP surrogate (Matern-2.5) to
        these 45 observations and finds the GP-predicted optimum offline,
        with physical constraints (leakage, thermal) applied automatically.

        Phase 2 — Online BO refinement  (``bo_n_iter`` extra hardware runs)
        --------------------------------------------------------------------
        The same GP is then used to compute Expected-Improvement (EI) and
        propose the next measurement point.  Each new hardware measurement
        is added to the training set and the GP is re-fitted.  Infeasible
        points (leakage / thermal too high) receive a fidelity penalty so
        the GP learns to avoid them.

        Total hardware acquisitions: 45 + 15 = 60 (vs. the old 220).
        Total shots: 60 × 1000 = 60,000 (vs. 220 × 2000 = 440,000). ~7× faster.

        Falls back to the original dense grid when sklearn is not installed
        (GP interpolation requires scikit-learn).

        Parameters
        ----------
        shots      : int   — IQ shots per grid point (default 1000).
        coarse_pts : tuple — (n_freq, n_gain, n_length) grid points
                             (default (5, 3, 3) → 45 combinations).
        bo_n_iter  : int   — online BO refinement iterations (default 15).
        bo_xi      : float — EI exploration factor; larger → more exploration.

        Returns
        -------
        (ro_length, res_gain_ge, res_freq_ge) : (float, float, float)
        """
        run_cfg     = self._cfg()
        freq_centre = run_cfg["res_freq_ge"]
        n_freq, n_gain, n_len = coarse_pts

        sweep_para = {
            "freq":   np.linspace(freq_centre - 1.0, freq_centre + 8.0, n_freq),
            "gain":   np.linspace(0.07, 0.10, n_gain),
            "length": np.linspace(2.0, 4.0, n_len),
        }

        total_pts = n_freq * n_gain * n_len
        self._log("ss_opt",
                  f"Phase 1: coarse grid {n_freq}×{n_gain}×{n_len} = {total_pts} pts "
                  f"({shots} shots each)")

        ssh_opt = SingleShot_ge_opt(run_cfg)
        ssh_opt.run(shots, sweep_para=sweep_para)

        self._log("ss_opt",
                  f"Phase 2: GP surrogate + {bo_n_iter} online BO refinement steps")
        length, gain, freq = ssh_opt.analyze(
            bo_n_iter = bo_n_iter,
            bo_xi     = bo_xi,
            pareto    = True,
        )

        self._update("ro_length",   length)
        self._update("res_gain_ge", gain)
        self._update("res_freq_ge", freq)

        # Reset phase then measure IQ rotation angle with full statistics
        self._update("res_phase", 0)
        run_cfg = self._cfg()
        ssh = SingleShot_gef(run_cfg)
        ssh.run(5000, shot_f=False)
        # hist() returns [fids, thresholds, angle_deg, conf_matrix_pct]
        ssh_result = ssh.plot(fid_avg=True, verbose=True)

        phase    = float(ssh_result[2])
        fidelity = (float(ssh_result[0][0])
                    if hasattr(ssh_result[0], "__len__")
                    else float(ssh_result[0]))
        self._update("res_phase", phase)

        self.results["ro_length"]        = length
        self.results["res_gain_ge"]      = gain
        self.results["res_freq_ge"]      = freq
        self.results["res_phase_deg"]    = phase
        self.results["readout_fidelity"] = fidelity

        self._log("ss_opt",
                  f"Best: length = {length:.3f} µs, gain = {gain:.5f}, "
                  f"freq = {freq:.4f} MHz")
        self._log("ss_opt",
                  f"res_phase = {phase:.2f} deg, fidelity = {fidelity:.4f}")
        return length, gain, freq

    # -------------------------------------------------------------------------
    # GP helpers
    # -------------------------------------------------------------------------

    def _gp_predict_zero_crossing(self, freq_vals, error_vals):
        """
        Fit a 1-D sklearn GP to the history of (qb_freq_tried, signed_detune_error)
        pairs and predict the qubit frequency where detune_error ≈ 0.

        Physics: to first order,
            signed_error(f) ≈ f_true − f
        so the correction function is linear.  The GP models this robustly
        against shot-noise in the Ramsey fit, using all previous measurements
        rather than just the most recent one.

        Parameters
        ----------
        freq_vals  : list[float] — qb_freq_ge values tried [MHz]
        error_vals : list[float] — signed detuning errors at each freq [MHz]

        Returns
        -------
        predicted_freq : float or None
            Optimal qb_freq_ge in MHz, or None if sklearn is unavailable
            or the GP uncertainty is too large to trust.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            return None

        try:
            X = np.array(freq_vals).reshape(-1, 1)
            y = np.array(error_vals)

            # Normalise inputs so GP length-scale optimisation is stable;
            # qb_freq is in GHz-scale MHz numbers, corrections are sub-MHz.
            x_centre = float(X.mean())
            x_scale  = max(float(X.std()), 1e-3)
            Xn = (X - x_centre) / x_scale

            kernel = (
                RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
            )
            gp.fit(Xn, y)

            # Search for zero crossing in a ± 5 × scatter window around the
            # latest estimate (avoids extrapolating far outside observations).
            x_lo = freq_vals[-1] - 5.0 * x_scale
            x_hi = freq_vals[-1] + 5.0 * x_scale
            x_search  = np.linspace(x_lo, x_hi, 2000)
            xn_search = (x_search - x_centre) / x_scale
            y_pred, y_std = gp.predict(xn_search.reshape(-1, 1), return_std=True)

            # Zero crossing: minimise |mean prediction|
            zero_idx   = int(np.argmin(np.abs(y_pred)))
            pred_freq  = round(float(x_search[zero_idx]), 4)
            pred_sigma = float(y_std[zero_idx])

            self._log("ramsey",
                      f"  GP zero-crossing: {pred_freq:.4f} MHz "
                      f"(σ = {pred_sigma:.4f} MHz)")

            # Reject if uncertainty is larger than 5× the convergence goal
            if pred_sigma > 5.0 * self.RAMSEY_DETUNE_LIMIT_MHZ:
                self._log("ramsey",
                          "  GP uncertainty too large — falling back to naive correction")
                return None

            return pred_freq

        except Exception as exc:
            self._log("ramsey", f"  GP prediction failed ({exc}) — using naive correction")
            return None

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self):
        """Print a table of all calibrated values from this run."""
        print(f"\n{'='*60}")
        print(f"  Auto-Calibration Summary  —  qubit {self.qubit}")
        print(f"{'='*60}")
        for key, val in self.results.items():
            if isinstance(val, float):
                print(f"  {key:<28s} = {val:.6g}")
            else:
                print(f"  {key:<28s} = {val}")
        print(f"{'='*60}")
