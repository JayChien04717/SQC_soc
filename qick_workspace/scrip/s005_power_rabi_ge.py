"""
s005 — Power Rabi (ge)
=======================
Amplitude Rabi: sweeps qubit drive gain at fixed length.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..plotter.plot_utils import plot_final


class PowerRabiProgram(BaseProgram):
    """QICK program for power Rabi: sweeps qubit drive gain."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, and gain-swept qubit pulse."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_pulse")

    def _body(self, cfg):
        """Apply optional cooling, send qubit pulse, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabi(BaseExperiment):
    """
    Power Rabi (ge) experiment.

    Sweeps the qubit drive gain and fits a decaying sinusoid to extract
    the pi and pi/2 pulse gains.
    """

    EXPT_NAME = "s005_power_rabi_ge"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the PowerRabiProgram."""
        return PowerRabiProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the gain sweep axis."""
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _post_fit(self, x_vals):
        """
        Fit a decaying sinusoid and return the pi and pi/2 pulse gains.

        Parameters
        ----------
        x_vals : ndarray
            DAC gain sweep axis.

        Returns
        -------
        pi_gain : float
            Pi pulse gain (rounded to 6 decimal places).
        pi2_gain : float
            Pi/2 pulse gain (rounded to 6 decimal places).
        """
        self.fit_params, error, fig, ax = plot_final(
            x_vals, self.iqdata, "Dac Gain(a.u)", fitdecaysin, decaysin,
            return_ax=True,
        )
        fig.suptitle("Power Rabi ge")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        ax.axvline(pi_gain, color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")
        return round(pi_gain, 6), round(pi2_gain, 6)
