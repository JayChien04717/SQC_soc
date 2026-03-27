"""
s003a — Qubit Flux Spectroscopy (ge)
=====================================
Sweeps qubit frequency and flux (hardware gain or Yoko).
"""
import numpy as np
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_lorentzian

class QubitSpecFluxProgram(BaseProgram):
    def _initialize(self, cfg):
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
        return QubitSpecFluxProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        # Prefer yoko_value if provided in config
        yoko_val = self.cfg.get("yoko_value")
        if yoko_val is not None:
            return np.asarray(yoko_val)
            
        if "flux_ch" in self.cfg:
            return prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        return None

    def _simulate(self, x_pts, y_pts=None):
        f_center = self.cfg.get("qb_freq_ge", (x_pts[0] + x_pts[-1]) / 2)
        if hasattr(f_center, "start"):
            f_center = (f_center.start + f_center.stop) / 2
            
        if y_pts is not None:
            # 2D Simulation: shift qubit frequency with flux
            data = np.zeros((len(y_pts), len(x_pts)), dtype=complex)
            for i, y in enumerate(y_pts):
                # Mock a parabolic flux shift (qubit is usually more sensitive than resonator)
                shift = 50.0 * (y / 30000)**2 
                data[i] = mock_lorentzian(x_pts, f0=f_center - shift, amp=-0.5, offset=1.0)
            return data
        else:
            return mock_lorentzian(x_pts, f0=f_center, amp=-0.5, offset=1.0)
