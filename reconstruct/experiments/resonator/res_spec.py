"""
Resonator/res_spec — s002: Resonator spectroscopy (ge).
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.resonator import ResonatorSpecAnalysis


class ResonatorSpecProgram(BaseProgram):
    """QICK program for resonator spectroscopy: sweeps resonator frequency."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.measure(cfg)


class ResonatorSpec(BaseExperiment):
    """
    Resonator spectroscopy (ge).

    Sweeps ``res_freq_ge`` and fits a circle (ABCD / hanger model) or
    Lorentzian to extract resonator parameters.
    """

    EXPT_NAME = "s002_onetone_ge"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = ResonatorSpecAnalysis

    def _create_program(self):
        return ResonatorSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def run(self, py_avg, solve_type="hm", **kwargs):
        """Run resonator spectroscopy.  ``solve_type`` passed to circle fit."""
        self._solve_type = solve_type
        return super().run(py_avg, **kwargs)

    def _post_fit(self, x_vals):
        result_dict = None
        try:
            try:
                from abcd_rf_fit import analyze
            except ImportError:
                from ...tools.abcd_rf_fit.abcd_rf_fit import analyze

            solve_type = getattr(self, "_solve_type", "hm")
            fit = analyze(x_vals * 1e6, self.iqdata, solve_type, fit_edelay=True)
            fit.plot()
            p = fit.tolist()
            f0, kappa, kappa_c = p[0], p[1], p[2]
            result_dict = {
                "Fres(GHz)": round(f0 / 1e9, 4),
                "Qi": round(f0 / (kappa - kappa_c)) if kappa > kappa_c else 0,
                "absQc": round(f0 / kappa_c) if kappa_c > 0 else 0,
                "Ql": round(f0 / kappa) if kappa > 0 else 0,
                "κ(MHz)": round(kappa * 1e-6, 2),
            }
            print(f"abcd_rf_fit result:\n {result_dict}")
            self.param = result_dict
        except Exception as e:
            print(f"Circle fit failed ({e}). No fit_params set.")
            self.param = None

        return self.param

    def _save_comment(self, dict_val):
        if isinstance(self.param, dict):
            f_res_mhz = self.param.get("Fres(GHz)", 0) * 1000
            return f"f_res = {f_res_mhz:.4f} MHz, \n{dict_val}"
        return f"{dict_val}"
