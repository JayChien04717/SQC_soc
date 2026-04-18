"""
s013 — T1 (ef)
================
T1 decay on ef: ge pi -> ef pi -> wait -> readout.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import expfunc, fitexp
from ..plotter.plot_utils import plot_final


class T1EfProgram(BaseProgram):
    """QICK program for ef T1: ge pi then ef pi then swept wait delay."""

    def _initialize(self, cfg):
        """Set up resonator, ge and ef generators, wait loop, and pi pulses."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        """Apply optional cooling, ge pi, ef pi, swept wait, optional ge ref, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.measure(cfg)


class T1_ef(BaseExperiment):
    """
    T1 (ef) experiment.

    Prepares the qubit in |f> via ge pi + ef pi pulses, sweeps the wait
    delay before readout, and fits an exponential decay to extract the
    ef energy relaxation time T1.
    """

    EXPT_NAME = "s013_T1_ef"
    TAG = "T1"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit T1 ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the T1EfProgram."""
        return T1EfProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the wait-time sweep axis in microseconds."""
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _post_fit(self, x_vals):
        """
        Fit an exponential decay and plot the ef T1 result.

        Parameters
        ----------
        x_vals : ndarray
            Wait-time sweep axis in microseconds.

        Returns
        -------
        fit_params : array
            Best-fit parameters ``[A, B, T1]``.
        """
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Times(us)", fitexp, expfunc
        )
        fig.suptitle(f"T1 = {self.fit_params[2]:.2f} +-{error[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params

    def _save_comment(self, dict_val):
        """Return a comment string including ef T1."""
        return f"T1 = {self.fit_params[2]:.2f} us \n{dict_val}"
