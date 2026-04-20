"""
s010 — Qubit Spectroscopy (ef)
================================
Two-tone for ef: ge pi pulse first, then sweep ef drive frequency.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import fitlor, lorfunc
from ..plotter.plot_utils import plot_final


class QubitSpecEfProgram(BaseProgram):
    """QICK program for ef qubit spectroscopy: ge pi pulse then ef frequency sweep."""

    def _initialize(self, cfg):
        """Set up resonator, ge and ef generators, frequency loop, and probe pulses."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_pulse_ef", pulse_type="flat_top")

    def _body(self, cfg):
        """Apply optional cooling, ge pi, ef probe, optional ge ref, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse_ef", t=0)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.02)
        self.delay_auto(0.02)
        self.measure(cfg)


class QubitSpec_ef(BaseExperiment):
    """
    Qubit Spectroscopy (ef) experiment.

    Prepares the qubit in |e> via a ge pi pulse, sweeps the ef drive
    frequency, and fits a Lorentzian to locate the ef transition frequency.
    """

    EXPT_NAME = "s010_qubit_spec_ef"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ef Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def _create_program(self):
        """Instantiate and return the QubitSpecEfProgram."""
        return QubitSpecEfProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the ef drive frequency sweep axis in MHz."""
        return prog.get_pulse_param("qb_pulse_ef", "freq", as_array=True)

    def _post_fit(self, x_vals):
        """
        Fit a Lorentzian and plot the ef qubit spectrum.

        Parameters
        ----------
        x_vals : ndarray
            Frequency sweep axis in MHz.

        Returns
        -------
        qb_freq_ef : float
            Fitted ef transition frequency in MHz, rounded to 6 decimal places.
        """
        fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Frequency(MHz)", fitlor, lorfunc
        )
        fig.suptitle(f"Qubit ef Spectrum, Qubit freq = {fit_params[2]:.6f} MHz")
        fig.tight_layout()
        self.fit_params = fit_params
        return round(fit_params[2], 6)

    def _save_comment(self, dict_val):
        """Return a comment string including the fitted ef frequency."""
        return f"f_q_ef = {self.fit_params[2]:.4f} MHz, \n{dict_val}"
