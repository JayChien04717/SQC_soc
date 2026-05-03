"""
T1 EF (s013) — energy relaxation from |f⟩ to |e⟩.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import T1Analysis


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
