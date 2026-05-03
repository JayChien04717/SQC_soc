"""
QubitEF/qubit_ef — s010-s011: EF transition spectroscopy, Rabi, and temperature.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import (
    LorentzianAnalysis, PowerRabiAnalysis, QubitTempAnalysis,
)


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
