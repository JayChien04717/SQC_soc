"""
s008 — T1 (ge)
================
T1 decay: pi pulse followed by variable wait, then readout.
"""

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import expfunc, fitexp
from ..plotter.plot_utils import plot_final


class T1Program(BaseProgram):
    """QICK program for T1 measurement: pi pulse then swept wait delay."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, wait loop, and pi pulse."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        """Apply optional cooling, pi pulse, swept wait, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        self.measure(cfg)


class T1(BaseExperiment):
    """
    T1 (ge) experiment.

    Applies a pi pulse and sweeps the wait delay before readout.  Fits an
    exponential decay to extract the energy relaxation time T1.
    """

    EXPT_NAME = "s008_T1_ge"
    TAG = "T1"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit T1 ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    def _create_program(self):
        """Instantiate and return the T1Program."""
        return T1Program(
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
        Fit an exponential decay and plot the T1 result.

        Parameters
        ----------
        x_vals : ndarray
            Wait-time sweep axis in microseconds.

        Returns
        -------
        fit_params : array
            Best-fit parameters ``[A, B, T1]``.
        error : array
            One-sigma parameter uncertainties.
        """
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Times(us)", fitexp, expfunc
        )
        fig.suptitle(f"T1 = {self.fit_params[2]:.2f} +-{error[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params, error

    def _save_comment(self, dict_val):
        """Return a comment string including T1."""
        return f"T1 = {self.fit_params[2]:.2f} us \n{dict_val}"
