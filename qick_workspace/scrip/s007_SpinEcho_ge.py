"""
s007 — Spin Echo (ge)
======================
Hahn echo: pi/2 — wait/2 — pi — wait/2 — pi/2 — readout.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, expfunc, fitexp
from ..plotter.plot_utils import plot_final


class SpinEchoProgram(BaseProgram):
    """QICK program for Hahn spin echo: pi/2 — wait/2 — pi — wait/2 — pi/2."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, wait loop, and pi/2 / pi pulses."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])

        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse_pi", gain_key="pi_gain_ge")
        ramsey_phase = (
            cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        )
        self.setup_qb_pulse(
            cfg, "ge", name="qb_pulse2", gain_key="pi2_gain_ge", phase=ramsey_phase
        )

    def _body(self, cfg):
        """Apply optional cooling, Hahn echo sequence, then measure."""
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
    """
    Spin Echo (ge) experiment.

    Hahn echo sequence sweeping the total inter-pulse delay.  Fits either a
    decaying sinusoid (when ``ramsey_freq != 0``) or a pure exponential decay
    (when ``ramsey_freq == 0``) to extract T2 Echo.
    """

    EXPT_NAME = "s007_SpinEcho_ge"
    TAG = "Spin Echo"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit SpinEcho ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the SpinEchoProgram."""
        return SpinEchoProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the total wait-time sweep axis (wait1 + wait2) in microseconds."""
        self.delay_times = prog.get_time_param(
            "wait1", "t", as_array=True
        ) + prog.get_time_param("wait2", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        """
        Fit and plot the Spin Echo decay.

        When ``cfg["ramsey_freq"] != 0`` a decaying sinusoid is fit; otherwise
        a pure exponential decay is fit.

        Parameters
        ----------
        x_vals : ndarray
            Total wait-time sweep axis in microseconds.

        Returns
        -------
        fit_params : array
            Best-fit parameters from the chosen model.
        error : array
            One-sigma parameter uncertainties.
        """
        if self.cfg["ramsey_freq"] != 0:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Times (us)", fitdecaysin, decaysin
            )
            fig.suptitle(
                f"T2 Echo = {self.fit_params[3]:.2f} us, "
                f"detune = {self.fit_params[1]:.5f}MHz "
                f"\u00b1 {error[1] * 1e3:.3f}kHz",
                fontsize=15,
            )
        else:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Times (us)", fitexp, expfunc
            )
            fig.suptitle(f"T2 Echo = {self.fit_params[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params, error

    def _save_comment(self, dict_val):
        """Return a comment string including T2 Spin Echo."""
        if self.cfg["ramsey_freq"] != 0:
            return f"T2 Spin Echo = {self.fit_params[3]:.2f} us\n{dict_val}"
        else:
            return f"T2 Spin Echo = {self.fit_params[2]:.2f} us\n{dict_val}"
