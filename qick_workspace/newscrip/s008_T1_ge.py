"""
s008 — T1 (ge)
================
T1 decay: pi pulse followed by variable wait, then readout.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import expfunc, fitexp
from ..plotter.plot_utils import plot_final


class T1Program(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        self.measure(cfg)


class T1(BaseExperiment):
    EXPT_NAME = "s008_T1_ge"
    TAG = "T1"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit T1 ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return T1Program(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Times(us)", fitexp, expfunc
        )
        fig.suptitle(f"T1 = {self.fit_params[2]:.2f} +-{error[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params, error

    def _save_comment(self, dict_val):
        return f"T1 = {self.fit_params[2]:.2f} us \n{dict_val}"
