"""
s011 — Power Rabi (ef)
=======================
Amplitude Rabi on ef: ge pi pulse first, then sweep ef drive gain.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, fix_phase


class PowerRabiEfProgram(BaseProgram):
    """QICK program for ef Power Rabi used in qubit temperature measurement."""

    def _initialize(self, cfg):
        """Set up resonator, ge and ef generators, gain loop, and pi/probe pulses."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_pulse_ef")

    def _body(self, cfg):
        """Apply optional cooling, optional ge pi (temp_ref mode), ef sweep, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        if cfg.get("temp_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse_ef", t=0)
        self.delay_auto(0.02)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


# ── 物理常數 ──────────────────────────────────────────────────────────
_H = 6.62607015e-34  # J·s
_KB = 1.380649e-23  # J/K


class QubitTemperatureEf(BaseExperiment):
    """
    Temperature measurement using ef-transition Rabi.

    Runs two acquisition passes (meas and ref) and extracts the qubit
    temperature from the ratio of their Rabi amplitudes.

    Parameters
    ----------
    cfg : dict
        Experiment configuration.  Must include ``qb_freq_ge``, ``qb_freq_ef``,
        and standard Power Rabi ef keys.

    Examples
    --------
    ::

        exp = QubitTemperatureEf(cfg)
        T_K = exp.run(py_avg=500)
        T_K = exp.run(py_avg=500, full_model=True)
    """

    EXPT_NAME = "s013b_qubit_temp_ef"
    TAG = "Temperature"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Temperature (Rabi ef)"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the PowerRabiEfProgram."""
        return PowerRabiEfProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the ef gain sweep axis in DAC units."""
        self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)
        return self.gains

    def run(self, py_avg, full_model=False):
        """
        Execute meas run and ref run, then compute temperature.

        Parameters
        ----------
        py_avg : int
            Software averages per run.
        full_model : bool, optional
            Use the full three-level solver when ``True``; use the simple
            ``P_f ≈ 0`` approximation when ``False``.

        Returns
        -------
        T_K : float or None
            Estimated qubit temperature in Kelvin, or ``None`` if the
            calculation failed.
        """
        self._full_model = full_model

        # ── Meas run: ge π-pulse → ef Rabi ───────────────────────────
        print("[Temp] Ref run: ef Rabi only...")
        self.cfg["temp_ref"] = False
        super().run(py_avg)
        self._iq_meas = self.iqdata.copy()

        # ── Ref run: ef Rabi only ─────────────────────────────────────
        print("[Temp] Meas run: ge π + ef Rabi...")
        self.cfg["temp_ref"] = True
        super().run(py_avg)
        self._iq_ref = self.iqdata.copy()

        return self._compute_temperature()

    def _compute_temperature(self):
        """Fit Rabi amplitudes for meas and ref runs and solve for temperature."""
        x = self._sweep_vals_x

        # rotate IQ → magnitude
        mag_meas = np.abs(self._iq_meas)
        mag_ref = np.abs(self._iq_ref)

        # fit decaying sinusoid
        pOpt_meas, _, _ = fitdecaysin(x, mag_meas)
        pOpt_ref, _, _ = fitdecaysin(x, mag_ref)

        fit_meas = decaysin(x, *pOpt_meas)
        fit_ref = decaysin(x, *pOpt_ref)

        # peak-to-peak / 2 = true Rabi amplitude
        A_meas = (fit_meas.max() - fit_meas.min()) / 2
        A_ref = (fit_ref.max() - fit_ref.min()) / 2

        print(
            f"[Temp] A_meas={A_meas:.4f}  A_ref={A_ref:.4f}  ratio={A_ref / A_meas:.4f}"
        )

        if A_meas <= 0 or A_ref <= 0:
            print("[Temp] ERROR: non-positive amplitude — check data quality.")
            return None

        fge_Hz = self.cfg["qb_freq_ge"] * 1e6
        fef_Hz = self.cfg["qb_freq_ef"] * 1e6

        T_K = self._solve_temperature(A_meas, A_ref, fge_Hz, fef_Hz)

        if T_K is not None:
            print(f"[Temp] Estimated temperature: {T_K * 1e3:.2f} mK")
            self._plot_temperature(
                x, mag_meas, fit_meas, mag_ref, fit_ref, A_meas, A_ref, T_K
            )
        else:
            print("[Temp] Temperature calculation failed.")

        return T_K

    def _solve_temperature(self, A_meas, A_ref, fge_Hz, fef_Hz):
        """
        Solve for qubit temperature from the measured Rabi amplitude ratio.

        Parameters
        ----------
        A_meas : float
            Rabi amplitude from the meas run (ef only).
        A_ref : float
            Rabi amplitude from the ref run (ge pi + ef).
        fge_Hz : float
            ge transition frequency in Hz.
        fef_Hz : float
            ef transition frequency in Hz.

        Returns
        -------
        T : float or None
            Temperature in Kelvin, or ``None`` if the solver failed.
        """
        # Use Ref/Meas as the standard ratio (always < 1 for T > 0)
        measured_ratio = A_meas / A_ref

        if not self._full_model:
            # --- Simple Model (P_f ≈ 0) ---
            # ratio = P_e / P_g = exp(-hf_ge / kT)
            # T = -hf_ge / (kB * ln(ratio))
            if measured_ratio <= 0 or measured_ratio >= 1:
                print(f"[Temp] Unphysical ratio: {measured_ratio:.4f}")
                return None

            return -(_H * fge_Hz) / (_KB * np.log(measured_ratio))

        else:
            # --- Full Three-Level Solver ---
            E_ge = _H * fge_Hz
            E_gf = _H * (fge_Hz + fef_Hz)  # Energy of state f relative to g

            def residual(T):
                # Partition function Z = 1 + exp(-E_e/kT) + exp(-E_f/kT)
                beta = 1.0 / (_KB * T)
                e_e = np.exp(-E_ge * beta)
                e_f = np.exp(-E_gf * beta)
                Z = 1 + e_e + e_f

                p_g, p_e, p_f = 1 / Z, e_e / Z, e_f / Z

                # Theoretical ratio: (P_e - P_f) / (P_g - P_f)
                # This accounts for population already in 'f' reducing the Rabi contrast
                theory_ratio = (p_e - p_f) / (p_g - p_f)
                return theory_ratio - measured_ratio

            try:
                # bracket [5mK, 1K] is usually safe for dilution refrigerators
                sol = root_scalar(residual, bracket=[0.005, 1.0], method="brentq")
                return sol.root if sol.converged else None
            except ValueError as e:
                print(f"[Temp] Solver failed (likely ratio too high/low): {e}")
                return None

    def _post_fit(self, x_vals):
        """
        Override parent's ``_post_fit`` to return temperature instead of pi gain.

        ``_compute_temperature()`` already handles fitting and plotting,
        so this is a no-op placeholder to satisfy the BaseExperiment interface.

        Parameters
        ----------
        x_vals : ndarray
            Gain sweep axis (unused here).

        Returns
        -------
        T_K : float or None
            Last computed temperature in Kelvin.
        """
        return getattr(self, "_last_T_K", None)

    # ── Plot ─────────────────────────────────────────────────────────
    marker_style = {
        "marker": "o",
        "markersize": 5,
        "alpha": 0.7,
        "linestyle": "-",
    }

    def _plot_temperature(
        self, x, mag_meas, fit_meas, mag_ref, fit_ref, A_meas, A_ref, T_K
    ):
        """Plot meas and ref Rabi data with fits and annotate the computed temperature."""
        # 改為單一坐標軸
        fig, ax = plt.subplots(figsize=(8, 6))

        # 定義顏色
        colors = {"meas": "C0", "ref": "C1"}

        # 繪製 Meas 數據 (使用 marker_style)
        ax.plot(
            x, mag_meas, color=colors["meas"], label="Meas data", **self.marker_style
        )
        ax.plot(
            x,
            fit_meas,
            color=colors["meas"],
            linewidth=2,
            alpha=0.9,
            label=f"Meas fit (A={A_meas:.4f})",
        )

        # 繪製 Ref 數據 (使用 marker_style)
        ax.plot(x, mag_ref, color=colors["ref"], label="Ref data", **self.marker_style)
        ax.plot(
            x,
            fit_ref,
            color=colors["ref"],
            linewidth=2,
            alpha=0.9,
            label=f"Ref fit (A={A_ref:.4f})",
        )

        # 設定圖表屬性
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel("Magnitude (a.u.)")

        title_info = (
            f"{self.TITLE_PREFIX}\n"
            f"A_meas={A_meas:.4f}, A_ref={A_ref:.4f}, ratio={A_ref / A_meas:.4f}\n"
            f"T = {T_K * 1e3:.2f} mK"
        )
        ax.set_title(title_info, fontsize=11)
        ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        plt.show()

        self._last_T_K = T_K

    def _save_comment(self, dict_val):
        """Append the calculated temperature to the regular cfg parameter comment."""
        base_comment = super()._save_comment(dict_val)
        T_K = getattr(self, "_last_T_K", None)
        if T_K is not None:
            return f"Calculated Qubit Temperature: {T_K * 1e3:.2f} mK\n\n{base_comment}"
        return base_comment

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """
        Override saveLabber to save both meas and ref data as a 2D dataset.

        The y-axis is set to ``[0, 1]`` representing meas and ref runs
        respectively.  z-data is stacked as ``[self._iq_meas, self._iq_ref]``.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the experiment name.
        yoko_value : float or None, optional
            Yokogawa flux bias value embedded in the filename.
        config_all : object or None, optional
            Full config object with a ``to_yaml(q_id)`` method.
        title : str or None, optional
            Custom title string for the saved file name.
        """
        # Set temporary 2D y-axis info
        orig_y = getattr(self, "_sweep_vals_y", None)
        orig_iq = getattr(self, "iqdata", None)

        self.Y_SAVE_NAME = "Meas_Type"
        self.Y_SAVE_UNIT = "0:Meas, 1:Ref"
        self.Y_SAVE_SCALE = 1.0

        self._sweep_vals_y = np.array([0, 1])
        # Stack 1D data into 2D (shape: 2 x N)
        self.iqdata = np.array([self._iq_meas, self._iq_ref])

        try:
            super().saveLabber(qb_idx, yoko_value, config_all, title)
        finally:
            # Restore state
            self._sweep_vals_y = orig_y
            self.iqdata = orig_iq
