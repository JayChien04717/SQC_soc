"""
Ramsey EF (s012) — T2* and detuning for the ef transition.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import RamseyAnalysis


class RamseyEfProgram(BaseProgram):
    """EF Ramsey: ge π → ef π/2 — wait — ef π/2."""

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
