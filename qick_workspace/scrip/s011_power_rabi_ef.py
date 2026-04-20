"""
s011 — Power Rabi (ef)
=======================
Amplitude Rabi on ef: ge pi pulse first, then sweep ef drive gain.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..plotter.plot_utils import plot_final


class PowerRabiEfProgram(BaseProgram):
    """QICK program for ef Power Rabi: ge pi pulse then ef gain sweep."""

    def _initialize(self, cfg):
        """Set up resonator, ge and ef generators, gain loop, and probe pulses."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_pulse_ef")

    def _body(self, cfg):
        """Apply optional cooling, ge pi, swept ef pulse, optional ge ref, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse_ef", t=0)
        self.delay_auto(0.02)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class PowerRabi_ef(BaseExperiment):
    """
    Power Rabi (ef) experiment.

    Prepares the qubit in |e> via a ge pi pulse, sweeps the ef drive gain,
    and fits a decaying sinusoid to extract the pi and pi/2 gains for the
    ef transition.
    """

    EXPT_NAME = "s011_power_rabi_ef"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ef"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the PowerRabiEfProgram."""
        return PowerRabiEfProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the ef gain sweep axis in DAC units."""
        self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)
        return self.gains

    def _post_fit(self, x_vals):
        """
        Fit a decaying sinusoid and mark pi / pi/2 gains on the plot.

        Parameters
        ----------
        x_vals : ndarray
            ef gain sweep axis in DAC units.

        Returns
        -------
        pi_gain : float
            Pi pulse gain rounded to 6 decimal places.
        pi2_gain : float
            Pi/2 pulse gain rounded to 6 decimal places.
        """
        self.fit_params, error, fig, ax = plot_final(
            x_vals,
            self.iqdata,
            "Dac Gain(a.u)",
            fitdecaysin,
            decaysin,
            return_ax=True,
        )
        fig.suptitle("Power Rabi ef")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        ax.axvline(pi_gain, color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")
        return round(pi_gain, 6), round(pi2_gain, 6)
