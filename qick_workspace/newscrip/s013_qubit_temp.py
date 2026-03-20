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
from .mock_signals import mock_decaysin
from ..tools.fitting import decaysin, fitdecaysin, fix_phase


class PowerRabiEfProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_pulse_ef")

    def _body(self, cfg):
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

    Usage
    -----
    exp = QubitTemperatureEf(soccfg, cfg)
    T_K = exp.run(py_avg=500)          # returns temperature in Kelvin
    T_K = exp.run(py_avg=500, full_model=True)  # full 3-level solver
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
        return PowerRabiEfProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)
        return self.gains

    # ── 兩次 run，覆寫父類 run() ──────────────────────────────────────
    def run(self, py_avg, simulate=False, full_model=False):
        """
        Execute meas run + ref run, then compute temperature.

        Parameters
        ----------
        py_avg     : int   — software averages per run
        simulate   : bool  — use mock data instead of hardware
        full_model : bool  — use full 3-level solver (default: P_f≈0 approx)

        Returns
        -------
        T_K : float | None  — temperature in Kelvin
        """
        self._full_model = full_model

        # ── Meas run: ge π-pulse → ef Rabi ───────────────────────────
        print("[Temp] Meas run: ge π + ef Rabi...")
        self.cfg["temp_ref"] = False
        super().run(py_avg, simulate=simulate)
        self._iq_meas = self.iqdata.copy()

        # ── Ref run: ef Rabi only ─────────────────────────────────────
        print("[Temp] Ref run: ef Rabi only...")
        self.cfg["temp_ref"] = True
        super().run(py_avg, simulate=simulate)
        self._iq_ref = self.iqdata.copy()

        return self._compute_temperature()

    # ── 溫度計算 ──────────────────────────────────────────────────────
    def _compute_temperature(self):
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
        # 修正：定義 ratio 為 Ref / Meas (通常 < 1)
        ratio = A_meas / A_ref

        if not self._full_model:
            # ── Closed-form (P_f ≈ 0) ────────────────────────────────
            # P_e / P_g = exp(-hf/kT)  =>  T = -hf / (kB * ln(ratio))
            if ratio >= 1.0:
                print(f"[Temp] WARNING: ratio={ratio:.4f} ≥ 1 is unphysical")
                return None
            return -(_H * fge_Hz) / (_KB * np.log(ratio))

        else:
            # ── Full three-level solver ───────────────────────────────
            E_ge = _H * fge_Hz
            E_gef = _H * (fge_Hz + fef_Hz)

            def residual(T):
                e1 = np.exp(-E_ge / (_KB * T))
                e2 = np.exp(-E_gef / (_KB * T))
                Z = 1 + e1 + e2
                P_g, P_e, P_f = 1 / Z, e1 / Z, e2 / Z
                # A_ref ∝ P_e - P_f
                # A_meas ∝ P_g - P_f (因為做了 pi_ge 脈衝，g, e 互換)
                denom = P_g - P_f
                if abs(denom) < 1e-14:
                    return np.inf
                return (P_e - P_f) / denom - ratio

            try:
                # 這裡的 bracket 可以根據稀釋製冷機的範圍調整 [10mK, 1K]
                sol = root_scalar(
                    residual, bracket=[0.005, 1.0], method="brentq", xtol=1e-7
                )
                return sol.root if sol.converged else None
            except ValueError as e:
                print(f"[Temp] root_scalar failed: {e}")
                return None

    # ── 覆寫 _post_fit，回傳溫度而非 π gain ──────────────────────────
    def _post_fit(self, x_vals):
        """
        Override parent's _post_fit.
        Parent returns (pi_gain, pi2_gain); here we return T_K.
        _compute_temperature() already handles fitting and plotting,
        so this is a no-op placeholder to satisfy BaseExperiment interface.
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
            + (" [full 3-level]" if self._full_model else " [P_f≈0]")
        )
        ax.set_title(title_info, fontsize=11)
        ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        plt.show()

        self._last_T_K = T_K

    # ── Simulation (覆寫，兩次 run 各自模擬對應的 amplitude) ─────────
    def _simulate(self, x_pts):
        """
        Mock data: ref amplitude ≈ ratio × meas amplitude.
        At 50 mK, ratio = exp(-h*5GHz / kB*50mK) ≈ 0.08.
        """
        is_ref = self.cfg.get("temp_ref", False)
        amp = 0.08 * 0.50 if is_ref else 0.50  # ref much smaller
        return mock_decaysin(x_pts, amp=amp, freq=2.0, decay=3.0, offset=0.5)

    def _save_comment(self, dict_val):
        """
        Append the calculated temperature to the regular cfg parameter comment.
        """
        base_comment = super()._save_comment(dict_val)
        T_K = getattr(self, "_last_T_K", None)
        if T_K is not None:
            return f"Calculated Qubit Temperature: {T_K * 1e3:.2f} mK\n\n{base_comment}"
        return base_comment

    # ── Override saveLabber ──────────────────────────────────────────
    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """
        Override saveLabber to save both Meas and Ref data as a 2D dataset.
        y-axis is forcefully set to [0, 1] representing Meas and Ref.
        z-data is stacked as [self._iq_meas, self._iq_ref].
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
