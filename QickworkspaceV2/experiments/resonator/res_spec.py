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
        # QICK mutates cfg in _create_program, replacing QickSweep1D (MHz)
        # with a compiled QickParam (register units). Use the pre-compile snapshot.
        cfg = self._cfg_snap if self._cfg_snap is not None else self.cfg
        sweep = cfg.get("res_freq_ge")
        steps = int(cfg.get("steps", 100))
        if hasattr(sweep, 'start') and hasattr(sweep, 'stop'):
            return np.linspace(float(sweep.start), float(sweep.stop), steps)
        return np.array([float(sweep)])

    def run(self, py_avg, solve_type="hm", **kwargs):
        """Run resonator spectroscopy.  ``solve_type`` passed to circle fit."""
        self.cfg["_solve_type"] = solve_type
        return super().run(py_avg, **kwargs)

    def _save_comment(self, dict_val):
        if self.result is not None:
            f0 = self.result.fit_result.get("f0_GHz", (None,))[0]
            if f0 is not None:
                return f"f_res = {f0 * 1000:.4f} MHz, \n{dict_val}"
        return str(dict_val)
