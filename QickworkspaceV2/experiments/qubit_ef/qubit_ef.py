"""
QubitEF/qubit_ef — s010-s013: EF transition spectroscopy, Rabi, Ramsey, T1, temp.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import (
    LorentzianAnalysis, PowerRabiAnalysis, RamseyAnalysis, T1Analysis, QubitTempAnalysis,
)
from ...tools.fitting import fitlor, lorfunc, fitdecaysin, decaysin, fitexp, expfunc
from ...plotter.plot_utils import plot_final


# ── s010 — Qubit Spec EF ──────────────────────────────────────────────────────

class QubitSpecEfProgram(BaseProgram):
    """EF spectroscopy: ge π pulse then sweeps ef drive frequency."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class QubitSpecEf(BaseExperiment):
    """Qubit spectroscopy (ef): ge π then sweep ef frequency."""

    EXPT_NAME = "s010_qubit_spec_ef"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ef Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = LorentzianAnalysis

    def _create_program(self):
        return QubitSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "freq", as_array=True)

    def _post_fit(self, x_vals):
        fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Frequency(MHz)", fitlor, lorfunc
        )
        fig.suptitle(f"Qubit ef Spectrum, freq = {fit_params[2]:.4f} MHz")
        fig.tight_layout()
        self.fit_params = fit_params
        self.fit_errors = error
        return round(fit_params[2], 6)


# ── s011 — Power Rabi EF ──────────────────────────────────────────────────────

class PowerRabiEfProgram(BaseProgram):
    """EF power Rabi: ge π pulse then sweep ef drive gain."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiEf(BaseExperiment):
    """EF power Rabi (s011): sweep ef gain → π_ef and π/2_ef gains."""

    EXPT_NAME = "s011_power_rabi_ef"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ef"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Analysis = PowerRabiAnalysis

    def _create_program(self):
        return PowerRabiEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "gain", as_array=True)

    def _post_fit(self, x_vals):
        from ...tools.fitting import fix_phase
        self.fit_params, error, fig, ax = plot_final(
            x_vals, self.iqdata, "Dac Gain(a.u)", fitdecaysin, decaysin, return_ax=True,
        )
        self.fit_errors = error
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        fig.suptitle("Power Rabi ef")
        fig.tight_layout()
        return round(pi_gain, 6), round(pi2_gain, 6)


# ── s012 — Ramsey EF ─────────────────────────────────────────────────────────

class RamseyEfProgram(BaseProgram):
    """EF Ramsey: ge π → wait → ef π/2 — wait — ef π/2."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse1", gain_key="pi2_gain_ef")
        ramsey_phase = cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse2", gain_key="pi2_gain_ef", phase=ramsey_phase)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse2", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class RamseyEf(BaseExperiment):
    """EF Ramsey (s012): extract T2* and detuning for ef transition."""

    EXPT_NAME = "s012_Ramsey_ef"
    TAG = "Ramsey"
    X_LABEL = "Ramsey Times (us)"
    TITLE_PREFIX = "Qubit Ramsey ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = RamseyAnalysis

    def _create_program(self):
        return RamseyEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        if self.cfg["ramsey_freq"] != 0:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitdecaysin, decaysin
            )
            fig.suptitle(f"T2 Ramsey ef = {self.fit_params[3]:.2f} us", fontsize=15)
        else:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitexp, expfunc
            )
            fig.suptitle(f"T2 Ramsey ef = {self.fit_params[2]:.2f} us", fontsize=15)
        self.fit_errors = error
        fig.tight_layout()
        return self.fit_params, error


# ── s013 — T1 EF ─────────────────────────────────────────────────────────────

class T1EfProgram(BaseProgram):
    """EF T1: ge π → ef π → wait → readout."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pi", gain_key="pi_gain_ef")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pi", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        self.measure(cfg)


class T1Ef(BaseExperiment):
    """EF T1 (s013): energy relaxation from |f⟩ to |e⟩."""

    EXPT_NAME = "s013_T1_ef"
    TAG = "T1"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit T1 ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = T1Analysis

    def _create_program(self):
        return T1EfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Times(us)", fitexp, expfunc
        )
        self.fit_errors = error
        fig.suptitle(f"T1 ef = {self.fit_params[2]:.2f} ±{error[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params, error


# ── s013 — Qubit Temperature ──────────────────────────────────────────────────

class QubitTempProgram(BaseProgram):
    """Qubit temperature: acquire shots for ground and excited states."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("shotloop", cfg.get("shots", 1000))
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class QubitTemp(BaseExperiment):
    """Qubit temperature measurement (s013_qubit_temp) via population ratio."""

    EXPT_NAME = "s013_qubit_temp"
    TAG = "Temperature"
    X_LABEL = "State"
    TITLE_PREFIX = "Qubit Temperature"
    X_SAVE_NAME = "State"
    X_SAVE_UNIT = ""
    X_SAVE_SCALE = 1.0

    Analysis = QubitTempAnalysis

    def _create_program(self):
        return QubitTempProgram(
            self.soccfg, reps=1,
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        import numpy as np
        return np.array([0, 1])  # ground, excited
