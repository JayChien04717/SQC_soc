"""
s004 — Time Rabi (ge)
======================
Length Rabi: sweeps qubit pulse length at fixed gain (flat_top only).
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_rabi_time
from ..tools.fitting import decaysin, fitdecaysin
from ..plotter.plot_utils import plot_final


class TimeRabiProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.add_loop("lenloop", cfg["steps"])
        self.setup_qb_pulse(
            cfg, 'ge', name="qb_pulse",
            pulse_type="flat_top", gain_key="pi_gain_ge",
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class TimeRabi(BaseExperiment):
    EXPT_NAME = "s004_time_rabi_ge"
    TAG = "Rabi"
    X_LABEL = "Pulse Length (us)"
    TITLE_PREFIX = "Qubit Time Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_flat_top_length_ge"]
    X_SAVE_NAME = "Pulse Length"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return TimeRabiProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "length", as_array=True)

    def _simulate(self, x_pts):
        return mock_rabi_time(x_pts, amp=0.5, freq=5.0, decay=2.0, offset=0.5)

    def _post_fit(self, x_vals):
        self.fit_params, error, fig = plot_final(
            x_vals, self.iqdata, "Pulse Length (us)", fitdecaysin, decaysin
        )
        fig.suptitle("Time Rabi ge")
        fig.tight_layout()
        return self.fit_params
