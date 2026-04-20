"""
s003a — Qubit Flux Spectroscopy (ge)
=====================================
Sweeps qubit frequency and flux (hardware gain or Yoko).
"""
import numpy as np
from .base_program import BaseProgram
from .base_experiment import BaseExperiment


class QubitSpecFluxProgram(BaseProgram):
    """QICK program for qubit flux spectroscopy: 2D sweep over frequency and flux."""

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, and optional flux bias with sweep loops."""
        self.setup_resonator(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_qb_pulse(cfg, prefix="ge")

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
        """Apply optional flux pulse and cooling, send qubit probe, then measure."""
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        if "flux_ch" in cfg:
            self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)

        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(cfg.get("wait_time", 0.05))

        self.measure(cfg)


class QubitSpecFlux(BaseExperiment):
    """
    Qubit flux spectroscopy (ge) experiment.

    Sweeps qubit drive frequency on the inner axis and flux (DAC gain or
    Yokogawa current) on the outer axis to map the qubit flux dispersion.
    """

    EXPT_NAME = "s003a_qubit_flux_spec_ge"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Flux Gain / Yoko (A)"
    TITLE_PREFIX = "Qubit Flux Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ge", "flux_gain"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Y_SAVE_NAME = "Flux"
    Y_SAVE_UNIT = "DAC or A"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        """Instantiate and return the QubitSpecFluxProgram."""
        return QubitSpecFluxProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Return the qubit frequency sweep axis in MHz."""
        return prog.get_pulse_param("qb_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        """
        Return the flux sweep axis.

        Prefers ``cfg['yoko_value']`` if provided; otherwise reads the flux
        pulse gain from the program.

        Parameters
        ----------
        prog : QubitSpecFluxProgram
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
