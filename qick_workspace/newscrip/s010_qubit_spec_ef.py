"""
s010 — Qubit Spectroscopy (ef)
================================
Two-tone for ef: ge pi pulse first, then sweep ef drive frequency.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_lorentzian
from ..tools.fitting import fitlor, lorfunc
from ..plotter.plot_utils import plot_final


class QubitSpecEfProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_pulse_ef", gain_key="pi_gain_ef",
                            pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse_ef", t=0)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.measure(cfg)


class QubitSpec_ef(BaseExperiment):
    EXPT_NAME = "s010_qubit_spec_ef"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ef Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        return QubitSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse_ef", "freq", as_array=True)

    def _simulate(self, x_pts):
        f0 = self.cfg.get("qb_freq_ef", (x_pts[0] + x_pts[-1]) / 2)
        if hasattr(f0, "start"):
            f0 = (f0.start + f0.stop) / 2
        return mock_lorentzian(x_pts, f0=f0, gamma=5, amp=0.8, offset=0.5)

    def _post_fit(self, x_vals):
        fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Frequency(MHz)", fitlor, lorfunc
        )
        fig.suptitle(f"Qubit ef Spectrum, Qubit freq = {fit_params[2]:.6f} MHz")
        fig.tight_layout()
        self.fit_params = fit_params
        return round(fit_params[2], 6)
