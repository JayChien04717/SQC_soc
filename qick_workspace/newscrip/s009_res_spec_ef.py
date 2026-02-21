"""
s009 — Resonator Spectroscopy (ef)
====================================
ge pi pulse first, then sweeps resonator frequency.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_lorentzian
from ..tools.module_fitzcu import resonator_circlefit


class ResSpecEfProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_pi_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class ResonatorSpec_ef(BaseExperiment):
    EXPT_NAME = "s009_onetone_ef"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        return ResSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def run(self, py_avg, solve_type="hm", simulate=False, **kwargs):
        """Override to pass solve_type to circle fit."""
        self._solve_type = solve_type
        return super().run(py_avg, simulate=simulate, **kwargs)

    def _simulate(self, x_pts):
        f0 = self.cfg.get("res_freq_ge", (x_pts[0] + x_pts[-1]) / 2)
        if hasattr(f0, "start"):
            f0 = (f0.start + f0.stop) / 2
        return mock_lorentzian(x_pts, f0=f0, gamma=2, amp=1.0, offset=0.5)

    def _post_fit(self, x_vals):
        self.param = resonator_circlefit(x_vals, self.iqdata, solve_type=self._solve_type)
        return self.param

    def _save_comment(self, dict_val):
        return f"f_res = {self.param[0] / 1e6:.4f} MHz, \n{dict_val}"
