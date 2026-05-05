"""
QubitGE/qubit_spec — s003: Qubit spectroscopy (ge) + flux variant (s003a).
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.resonator import LorentzianAnalysis


class QubitSpecProgram(BaseProgram):
    """QICK program for two-tone qubit spectroscopy: sweeps qubit frequency."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class QubitSpec(BaseExperiment):
    """
    Qubit spectroscopy (ge).

    Sweeps qubit drive frequency and fits a Lorentzian to extract ge
    transition frequency.
    """

    EXPT_NAME = "s003_qubit_spec_ge"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ge Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = LorentzianAnalysis

    def _create_program(self):
        return QubitSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        cfg = self._cfg_snap if self._cfg_snap is not None else self.cfg
        sweep = cfg.get("qb_freq_ge")
        steps = int(cfg.get("steps", 100))
        if hasattr(sweep, 'start') and hasattr(sweep, 'stop'):
            return np.linspace(float(sweep.start), float(sweep.stop), steps)
        return np.array([float(sweep)])

    def _save_comment(self, dict_val):
        if self.result is not None:
            f0 = self.result.fit_result.get("f0_MHz", (None,))[0]
            if f0 is not None:
                return f"f_q_ge = {f0:.4f} MHz, \n{dict_val}"
        return str(dict_val)


# ── Flux variant ─────────────────────────────────────────────────────────────

class QubitSpecFluxProgram(BaseProgram):
    """QICK program for qubit spec vs flux: adds flux pulse outer loop."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        if "flux_ch" in cfg:
            self.declare_gen(ch=cfg["flux_ch"], nqz=1)
            self.add_pulse(
                ch=cfg["flux_ch"], name="flux_pulse", style="const",
                length=cfg["flux_length"], freq=0, phase=0, gain=cfg["flux_gain"],
            )
            self.add_loop("fluxloop", cfg["steps_flux"])
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if "flux_ch" in cfg:
            self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
            self.delay(cfg.get("saturate_times", 0.1))
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class QubitSpecFlux(BaseExperiment):
    """Qubit spectroscopy vs flux (s003a): 2D sweep."""

    EXPT_NAME = "s003a_qubit_flux_spec_ge"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Flux / Yoko"
    TITLE_PREFIX = "Qubit Flux Spectroscopy"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "Flux"
    Y_SAVE_UNIT = "DAC or A"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return QubitSpecFluxProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        cfg = self._cfg_snap if self._cfg_snap is not None else self.cfg
        sweep = cfg.get("qb_freq_ge")
        steps = int(cfg.get("steps", 100))
        if hasattr(sweep, 'start') and hasattr(sweep, 'stop'):
            return np.linspace(float(sweep.start), float(sweep.stop), steps)
        return np.array([float(sweep)])

    def _extract_sweep_axis_y(self, prog):
        yoko_val = self.cfg.get("yoko_value")
        if yoko_val is not None:
            return np.asarray(yoko_val)
        if "flux_ch" in self.cfg:
            return prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        return None
