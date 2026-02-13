"""
s012 — Ramsey (ef)
===================
Ramsey on ef: ge pi pulse, then ef pi/2 — wait — pi/2.
"""
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_decaysin, mock_exp_decay
from ..tools.fitting import decaysin, fitdecaysin, expfunc, fitexp
from ..plotter.plot_utils import plot_final


class RamseyEfProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("waitloop", cfg["steps"])

        self.setup_qb_pulse(cfg, 'ge', name="qb_pi_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_pulse1", gain_key="pi2_gain_ef")
        ramsey_phase = cfg.get("qb_phase_ef", 0) + cfg["wait_time"] * 360 * cfg["ramsey_freq"]
        self.setup_qb_pulse(cfg, 'ef', name="qb_pulse2", gain_key="pi2_gain_ef", phase=ramsey_phase)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse2", t=0)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.measure(cfg)


class Ramsey_ef(BaseExperiment):
    EXPT_NAME = "s012_Ramsey_ef"
    TAG = "Ramsey"
    X_LABEL = "Ramsey Times (us)"
    TITLE_PREFIX = "Qubit Ramsey ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        return RamseyEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _simulate(self, x_pts):
        ramsey_freq = self.cfg.get("ramsey_freq", 0.5)
        if ramsey_freq != 0:
            return mock_decaysin(x_pts, amp=0.5, freq=ramsey_freq, decay=10.0, offset=0.5)
        else:
            return mock_exp_decay(x_pts, amp=0.5, tau=10.0, offset=0.0)

    def _post_fit(self, x_vals):
        if self.cfg["ramsey_freq"] != 0:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitdecaysin, decaysin
            )
            fig.suptitle(
                f"T2 Ramsey = {self.fit_params[3]:.2f} us, "
                f"detune = {self.fit_params[1]:.5f}MHz "
                f"\u00b1 {error[1] * 1e3:.3f}kHz", fontsize=15,
            )
        else:
            self.fit_params, error, fig = plot_final(
                x_vals, self.iqdata, "Ramsey Times", fitexp, expfunc
            )
            fig.suptitle(f"T2 Ramsey = {self.fit_params[2]:.2f} us", fontsize=15)
        fig.tight_layout()
        return self.fit_params

    def correct_detune(self):
        """Correct qubit ef frequency based on fitted detuning."""
        if abs(self.fit_params[1] - self.cfg["ramsey_freq"]) > 0.005:
            self.cfg["qb_freq_ef"] -= round(self.fit_params[1] - self.cfg["ramsey_freq"], 2)
            print(f"over detune {round(self.fit_params[1] - self.cfg['ramsey_freq'], 5)}MHz")
            return round(self.cfg["qb_freq_ef"], 5)
        else:
            print("Detune < 5kHz")
            return self.cfg["qb_freq_ef"]

    def _save_comment(self, dict_val):
        return f"T2 Ramsey = {self.fit_params[3]:.2f} us\n{dict_val}"
