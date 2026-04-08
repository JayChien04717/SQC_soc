"""
s006 — Ramsey (ge)
===================
Ramsey fringe / T2*: two pi/2 pulses separated by variable delay.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, expfunc, fitexp
from ..plotter.plot_utils import plot_final


class RamseyProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])

        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        ramsey_phase = (
            cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        )
        self.setup_qb_pulse(
            cfg,
            "ge",
            name="qb_pulse2",
            gain_key="pi2_gain_ge",
            phase=ramsey_phase,
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class Ramsey(BaseExperiment):
    EXPT_NAME = "s006_Ramsey_ge"
    TAG = "Ramsey"
    X_LABEL = "Ramsey Times (us)"
    TITLE_PREFIX = "Qubit Ramsey ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return RamseyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        if self.cfg["ramsey_freq"] != 0:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitdecaysin, decaysin
            )
            fig.suptitle(
                f"T2 Ramsey = {self.fit_params[3]:.2f} us, "
                f"detune = {self.fit_params[1]:.5f}MHz "
                f"\u00b1 {error[1] * 1e3:.3f}kHz",
                fontsize=15,
            )
        else:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitexp, expfunc
            )
            fig.suptitle(f"T2 Ramsey = {self.fit_params[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params, error

    def correct_detune(self):
        if abs(self.fit_params[1] - self.cfg["ramsey_freq"]) > 0.005:
            self.cfg["qb_freq_ge"] = self.cfg["qb_freq_ge"] - round(
                (self.fit_params[1] - self.cfg["ramsey_freq"]), 2
            )
            print(
                f"over detune {round((self.fit_params[1] - self.cfg['ramsey_freq']), 5)}MHz"
            )
            return round(self.cfg["qb_freq_ge"], 5)
        else:
            print("Detune < 5kHz")
            return self.cfg["qb_freq_ge"]

    def _save_comment(self, dict_val):
        if self.cfg["ramsey_freq"] != 0:
            return f"T2 Ramsey = {self.fit_params[3]:.2f} us\n{dict_val}"
        else:
            return f"T2 Ramsey = {self.fit_params[2]:.2f} us\n{dict_val}"
