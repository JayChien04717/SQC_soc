"""
Coherence/spin_echo — s007: Spin Echo (Hahn echo, ge).
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import SpinEchoAnalysis
from ...tools.fitting import decaysin, fitdecaysin, expfunc, fitexp
from ...plotter.plot_utils import plot_final


class SpinEchoProgram(BaseProgram):
    """QICK program for Hahn echo: π/2 — wait/2 — π — wait/2 — π/2."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse_pi", gain_key="pi_gain_ge")
        ramsey_phase = cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse2", gain_key="pi2_gain_ge", phase=ramsey_phase)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto((cfg["wait_time"] / 2) + 0.01, tag="wait1")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi", t=0)
        self.delay_auto((cfg["wait_time"] / 2) + 0.01, tag="wait2")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(0.01)
        self.measure(cfg)


class SpinEcho(BaseExperiment):
    """Spin Echo (ge): Hahn echo, extracts T2 Echo."""

    EXPT_NAME = "s007_SpinEcho_ge"
    TAG = "Spin Echo"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit SpinEcho ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = SpinEchoAnalysis

    def _create_program(self):
        return SpinEchoProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = (
            prog.get_time_param("wait1", "t", as_array=True)
            + prog.get_time_param("wait2", "t", as_array=True)
        )
        return self.delay_times

    def _post_fit(self, x_vals):
        if self.cfg["ramsey_freq"] != 0:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Times (us)", fitdecaysin, decaysin
            )
            fig.suptitle(
                f"T2 Echo = {self.fit_params[3]:.2f} us, "
                f"detune = {self.fit_params[1]:.5f}MHz "
                f"± {error[1] * 1e3:.3f}kHz",
                fontsize=15,
            )
        else:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Times (us)", fitexp, expfunc
            )
            fig.suptitle(f"T2 Echo = {self.fit_params[2]:.2f} us", fontsize=15)
        self.fit_errors = error
        fig.tight_layout()
        return self.fit_params, error

    def _save_comment(self, dict_val):
        if self.fit_params is not None:
            if self.cfg["ramsey_freq"] != 0:
                return f"T2 Spin Echo = {self.fit_params[3]:.2f} us\n{dict_val}"
            return f"T2 Spin Echo = {self.fit_params[2]:.2f} us\n{dict_val}"
        return str(dict_val)
