"""
s011 — Power Rabi (ef)
=======================
Amplitude Rabi on ef: ge pi pulse first, then sweep ef drive gain.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_decaysin
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..plotter.plot_utils import plot_final


class PowerRabiEfProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_pulse_ef")

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


class PowerRabi_ef(BaseExperiment):
    EXPT_NAME = "s011_power_rabi_ef"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ef"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return PowerRabiEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)
        return self.gains

    def _simulate(self, x_pts):
        return mock_decaysin(x_pts, amp=0.5, freq=2.0, decay=3.0, offset=0.5)

    def _post_fit(self, x_vals):
        self.fit_params, error, fig, ax = plot_final(
            x_vals, self.iqdata, "Dac Gain(a.u)", fitdecaysin, decaysin,
            return_ax=True,
        )
        fig.suptitle("Power Rabi ef")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(self.fit_params)
        ax.axvline(pi_gain, color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")
        return round(pi_gain, 6), round(pi2_gain, 6)
