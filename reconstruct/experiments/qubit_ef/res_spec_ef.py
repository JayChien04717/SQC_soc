"""
QubitEF/res_spec_ef — s009: Resonator spectroscopy (ef).
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.resonator import ResonatorSpecAnalysis


class ResSpecEfProgram(BaseProgram):
    """QICK program for ef resonator spectroscopy: ge π pulse then resonator frequency sweep."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pi_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class ResonatorSpec_ef(BaseExperiment):
    """
    Resonator spectroscopy (ef).

    Prepares |e⟩ via ge π pulse, then sweeps resonator frequency.
    Circle fit extracts resonator parameters in excited-state frame.
    """

    EXPT_NAME = "s009_onetone_ef"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy (ef)"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = ResonatorSpecAnalysis

    def _create_program(self):
        return ResSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def run(self, py_avg, solve_type="hm", **kwargs):
        self._solve_type = solve_type
        return super().run(py_avg, **kwargs)

    def _post_fit(self, x_vals):
        try:
            try:
                from abcd_rf_fit import analyze
            except ImportError:
                from ...tools.abcd_rf_fit.abcd_rf_fit import analyze

            solve_type = getattr(self, "_solve_type", "hm")
            fit = analyze(x_vals * 1e6, self.iqdata, solve_type, fit_edelay=True)
            p = fit.tolist()
            self.param = {"f0_Hz": p[0], "kappa_Hz": p[1], "kappa_c_Hz": p[2]}
        except Exception as e:
            self.param = None
            print(f"Resonator fit failed: {e}")
        return self.param

    def _save_comment(self, dict_val):
        if self.param is not None and hasattr(self.param, '__len__'):
            return f"f_res = {self.param[0] / 1e6:.4f} MHz, \n{dict_val}"
        return str(dict_val)
