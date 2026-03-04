"""
s002 — Resonator Spectroscopy (ge)
===================================
Sweeps resonator frequency and performs circle fitting.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_lorentzian
from ..tools.module_fitzcu import resonator_circlefit


class ResonatorSpecProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.measure(cfg)


class ResonatorSpec(BaseExperiment):
    EXPT_NAME = "s002_onetone_ge"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        return ResonatorSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _simulate(self, x_pts):
        f0 = self.cfg.get("res_freq_ge", (x_pts[0] + x_pts[-1]) / 2)
        if hasattr(f0, "start"):
            try:
                span = list(f0.spans.values())[0] if hasattr(f0, "spans") and f0.spans else 0
                f0 = f0.start + span / 2
            except Exception:
                f0 = (x_pts[0] + x_pts[-1]) / 2
        return mock_lorentzian(x_pts, f0=f0, gamma=2, amp=1.0, offset=0.5)

    def _post_fit(self, x_vals):
        self.param = resonator_circlefit(x_vals, self.iqdata)
        return self.param

    def _save_comment(self, dict_val):
        return f"f_res = {self.param[0] / 1e6:.4f} MHz, \n{dict_val}"
