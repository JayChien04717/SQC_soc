"""
QubitGE/rabi — s004: Time Rabi + s005: Power Rabi + s005b: Power Rabi with reset.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import PowerRabiAnalysis, TimeRabiAnalysis
from ...tools.fitting import decaysin, fitdecaysin, fix_phase
from ...plotter.plot_utils import plot_final


# ── s004 — Time Rabi ──────────────────────────────────────────────────────────

class TimeRabiProgram(BaseProgram):
    """QICK program for time Rabi: sweeps flat-top pulse length."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("lenloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class TimeRabi(BaseExperiment):
    """Time Rabi (ge): sweeps flat-top pulse length, fits decaying sinusoid."""

    EXPT_NAME = "s004_time_rabi_ge"
    TAG = "Rabi"
    X_LABEL = "Pulse Length (us)"
    TITLE_PREFIX = "Qubit Time Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_flat_top_length_ge"]
    X_SAVE_NAME = "Pulse Length"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    Analysis = TimeRabiAnalysis

    def _create_program(self):
        return TimeRabiProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "length", as_array=True)

    def _post_fit(self, x_vals):
        self.fit_params, error, fig, ax = plot_final(
            x_vals, self.iqdata, "Pulse Length (us)", fitdecaysin, decaysin, return_ax=True,
        )
        self.fit_errors = error
        pi_len, pi2_len = fix_phase(self.fit_params)
        ax.axvline(pi_len, color="red", linestyle="--", label=r"$\pi$ Length")
        ax.axvline(pi2_len, color="red", linestyle="--", label=r"$\pi/2$ Length")
        fig.suptitle(f"Time Rabi ge, Rabi freq = {self.fit_params[1]:.2f} MHz")
        fig.tight_layout()
        return pi_len, pi2_len


# ── s005 — Power Rabi ─────────────────────────────────────────────────────────

class PowerRabiProgram(BaseProgram):
    """QICK program for power Rabi: sweeps qubit drive gain."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabi(BaseExperiment):
    """Power Rabi (ge): sweeps gain, fits decaying sinusoid → π and π/2 gains."""

    EXPT_NAME = "s005_power_rabi_ge"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Analysis = PowerRabiAnalysis

    def _create_program(self):
        return PowerRabiProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _post_fit(self, x_vals):
        self.fit_params, error, fig, ax = plot_final(
            x_vals, self.iqdata, "Dac Gain(a.u)", fitdecaysin, decaysin, return_ax=True,
        )
        self.fit_errors = error
        fig.suptitle("Power Rabi ge")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        ax.axvline(pi_gain, color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")
        return round(pi_gain, 6), round(pi2_gain, 6)


# ── s005b — Power Rabi with Reset ────────────────────────────────────────────

class PowerRabiResetProgram(BaseProgram):
    """Power Rabi with active-reset cooling before each shot."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiReset(PowerRabi):
    """Power Rabi with active reset (s005b)."""

    EXPT_NAME = "s005b_power_rabi_reset_ge"
    TAG = "Rabi"
    TITLE_PREFIX = "Qubit Power Rabi ge (Reset)"

    def _create_program(self):
        return PowerRabiResetProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )
