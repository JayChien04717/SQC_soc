"""
s002 — Resonator Spectroscopy (ge)
===================================
Sweeps resonator frequency and performs circle fitting.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.module_fitzcu import resonator_circlefit


class ResonatorSpecProgram(BaseProgram):
    """QICK program for resonator spectroscopy: sweeps resonator frequency."""

    def _initialize(self, cfg):
        """Set up resonator pulse and frequency sweep loop."""
        self.setup_resonator(cfg, prefix="ge")
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        """Apply optional cooling then measure resonator transmission."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.measure(cfg)


class ResonatorSpec(BaseExperiment):
    """
    Resonator spectroscopy (ge) experiment.

    Sweeps ``res_freq_ge`` and fits a circle to extract resonator parameters.
    """

    EXPT_NAME = "s002_onetone_ge"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        """Instantiate and return the ResonatorSpecProgram."""
        return ResonatorSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the frequency sweep axis in MHz."""
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _post_fit(self, x_vals):
        """
        Perform circle fit on the IQ data.

        Parameters
        ----------
        x_vals : ndarray
            Frequency axis in MHz.

        Returns
        -------
        param : tuple
            Fitted resonator parameters from ``resonator_circlefit``.
        """
        self.param = resonator_circlefit(x_vals, self.iqdata)
        return self.param

    def _save_comment(self, dict_val):
        """Return a comment string including the fitted resonance frequency."""
        return f"f_res = {self.param[0] / 1e6:.4f} MHz, \n{dict_val}"
