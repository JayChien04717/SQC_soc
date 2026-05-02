"""
Resonator/res_spec_flux — s002c: Resonator spec vs flux (2D sweep).
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment


class ResonatorSpecFluxProgram(BaseProgram):
    """QICK program for resonator spectroscopy vs flux: 2D sweep."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        if "flux_ch" in cfg:
            self.declare_gen(ch=cfg["flux_ch"], nqz=1)
            self.add_pulse(
                ch=cfg["flux_ch"], name="flux_pulse", style="const",
                length=cfg["flux_length"], freq=0, phase=0, gain=cfg["flux_gain"],
            )
            self.add_loop("fluxloop", cfg["steps_flux"])
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if "flux_ch" in cfg:
            self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
            self.delay(cfg.get("saturate_times", 0.1))
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.measure(cfg)


class ResonatorSpecFlux(BaseExperiment):
    """
    Resonator spectroscopy vs flux (ge).

    Sweeps resonator frequency (inner axis) and flux/Yoko (outer axis).
    """

    EXPT_NAME = "s002c_onetone_flux_ge"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Flux Gain / Yoko (A)"
    TITLE_PREFIX = "Resonator Flux Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge", "flux_gain"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "Flux"
    Y_SAVE_UNIT = "DAC or A"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return ResonatorSpecFluxProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        yoko_val = self.cfg.get("yoko_value")
        if yoko_val is not None:
            return np.asarray(yoko_val)
        if "flux_ch" in self.cfg:
            return prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        return None

    def saveLabber(self, qb_idx, config_all=None, title=None, **kwargs):
        if self._yoko_mode == "voltage":
            self.Y_SAVE_UNIT = "V"
        elif self._yoko_mode == "current":
            self.Y_SAVE_UNIT = "A"
        super().saveLabber(qb_idx, yoko_value=None, config_all=config_all, title=title)
