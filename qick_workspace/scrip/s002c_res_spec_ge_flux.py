"""
s002c — Resonator Spec vs Flux (ge)
=====================================
Sweeps resonator frequency and flux (hardware gain or Yoko).
"""
import numpy as np
from .base_program import BaseProgram
from .base_experiment import BaseExperiment


class ResonatorSpecFluxProgram(BaseProgram):
    """QICK program for resonator spectroscopy vs flux: 2D sweep."""

    def _initialize(self, cfg):
        """Set up resonator and optional flux-bias generator with sweep loops."""
        self.setup_resonator(cfg, prefix="ge")

        # Flux setup (hardware sweep)
        if "flux_ch" in cfg:
            self.declare_gen(ch=cfg["flux_ch"], nqz=1)
            self.add_pulse(
                ch=cfg["flux_ch"],
                name="flux_pulse",
                style="const",
                length=cfg["flux_length"],
                freq=0, phase=0,
                gain=cfg["flux_gain"]
            )
            self.add_loop("fluxloop", cfg["steps_flux"])

        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        """Apply optional flux pulse and cooling, then measure resonator."""
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
    Resonator spectroscopy vs flux (ge) experiment.

    Sweeps resonator frequency on the inner axis and flux (hardware DAC
    gain or Yokogawa current) on the outer axis.
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
        """Instantiate and return the ResonatorSpecFluxProgram."""
        return ResonatorSpecFluxProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the resonator frequency sweep axis in MHz."""
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        """
        Return the flux sweep axis.

        Prefers ``cfg['yoko_value']`` if provided; otherwise reads the flux
        pulse gain from the program.

        Parameters
        ----------
        prog : ResonatorSpecFluxProgram
            Compiled QICK program instance.

        Returns
        -------
        ndarray or None
            Flux axis values.
        """
        # Prefer yoko_value if provided in config
        yoko_val = self.cfg.get("yoko_value")
        if yoko_val is not None:
            return np.asarray(yoko_val)

        if "flux_ch" in self.cfg:
            return prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        return None
