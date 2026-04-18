"""
s009 — Resonator Spectroscopy (ef)
====================================
ge pi pulse first, then sweeps resonator frequency.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.module_fitzcu import resonator_circlefit


class ResSpecEfProgram(BaseProgram):
    """QICK program for ef resonator spectroscopy: ge pi pulse then resonator frequency sweep."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, frequency loop, and ge pi pulse."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_pi_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        """Apply optional cooling, ge pi pulse, then measure resonator."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class ResonatorSpec_ef(BaseExperiment):
    """
    Resonator Spectroscopy (ef) experiment.

    Prepares the qubit in |e> via a ge pi pulse, then sweeps the resonator
    drive frequency and applies a circle fit to extract resonator parameters
    in the excited-state dressed frame.
    """

    EXPT_NAME = "s009_onetone_ef"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        """Instantiate and return the ResSpecEfProgram."""
        return ResSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the resonator frequency sweep axis in MHz."""
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def run(self, py_avg, solve_type="hm", **kwargs):
        """
        Run resonator spectroscopy (ef) and pass ``solve_type`` to the circle fit.

        Parameters
        ----------
        py_avg : int
            Hardware averages (rounds) per frequency point.
        solve_type : str, optional
            Circle fit solver type passed to ``resonator_circlefit``
            (e.g. ``"hm"`` for Hanger model).
        **kwargs
            Additional keyword arguments forwarded to ``BaseExperiment.run()``.

        Returns
        -------
        param : array
            Resonator fit parameters from ``resonator_circlefit``.
        """
        self._solve_type = solve_type
        return super().run(py_avg, **kwargs)

    def _post_fit(self, x_vals):
        """
        Apply a circle fit to the resonator IQ data.

        Parameters
        ----------
        x_vals : ndarray
            Frequency sweep axis in MHz.

        Returns
        -------
        param : array
            Resonator parameters returned by ``resonator_circlefit``.
        """
        self.param = resonator_circlefit(x_vals, self.iqdata, solve_type=self._solve_type)
        return self.param

    def _save_comment(self, dict_val):
        """Return a comment string including the fitted resonator frequency."""
        return f"f_res = {self.param[0] / 1e6:.4f} MHz, \n{dict_val}"
