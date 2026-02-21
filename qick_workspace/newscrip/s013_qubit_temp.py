"""
s013 — Qubit Temperature Measurement
=====================================
Uses Amplitude Rabi on ef transition to estimate qubit temperature.
P_e = exp(-E_e/kT) / Z
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_decaysin
from ..tools.fitting import decaysin, fitdecaysin
from ..tools.module_fitzcu import post_rotate

class QubitTempProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ef")
        self.setup_qubit_gen(cfg, prefix="ge")
        self.setup_qubit_gen(cfg, prefix="ef")
        
        # Pi pulse on ge
        self.setup_qb_pulse(cfg, prefix="ge", name="qubit_pi_pulse", gain_key="qubit_pi_gain_ge")
        # Sweep pulse on ef
        self.setup_qb_pulse(cfg, prefix="ef", name="qubit_pulse_ef")

        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        
        # If temp_ref is False, we apply ge pi pulse first (check population in e)
        if not cfg.get("temp_ref", False):
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
            
        # ef pulse sweep
        self.pulse(ch=cfg["qubit_ch_ef"], name="qubit_pulse_ef", t=0)
        self.delay_auto(0.01)
        
        # Optional post-rotation ge pi pulse
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
            
        self.measure(cfg)

class QubitTemperature(BaseExperiment):
    EXPT_NAME = "s013_qubit_temp"
    TAG = "Temperature"
    X_LABEL = "Gain (DAC unit)"
    TITLE_PREFIX = "Qubit Temperature (Rabi ef)"
    SWEEP_KEYS_TO_REMOVE = ["qubit_gain_ef"]
    
    def run(self, py_avg, simulate=False):
        # This experiment requires two separate runs: 
        # 1. Measurement run (temp_ref=False)
        # 2. Reference run (temp_ref=True)
        
        print("Running Measurement Run...")
        self.cfg["temp_ref"] = False
        data_meas = super().run(py_avg, simulate=simulate)
        iq_meas = self.iqdata
        
        print("Running Reference Run...")
        self.cfg["temp_ref"] = True
        data_ref = super().run(py_avg, simulate=simulate)
        iq_ref = self.iqdata
        
        self.iqdata_meas = iq_meas
        self.iqdata_ref = iq_ref
        
        return self.analyze_temperature()

    def analyze_temperature(self):
        # Simple analysis extracted from original script
        mag_meas = np.abs(post_rotate(self.iqdata_meas))
        mag_ref = np.abs(post_rotate(self.iqdata_ref))
        
        pOpt, _ = fitdecaysin(self._sweep_vals_x, mag_meas)
        pOpt_ref, _ = fitdecaysin(self._sweep_vals_x, mag_ref)
        
        temp_amplitude = max(decaysin(self._sweep_vals_x, *pOpt)) - min(decaysin(self._sweep_vals_x, *pOpt))
        temp_ref_amplitude = max(decaysin(self._sweep_vals_x, *pOpt_ref)) - min(decaysin(self._sweep_vals_x, *pOpt_ref))
        
        popu_e = temp_amplitude / (temp_amplitude + temp_ref_amplitude)
        
        try:
            T_k = self.solve_temperature(self.cfg['qubit_freq_ge'], self.cfg['qubit_freq_ef'], popu_e)
            print(f"Estimated Temperature: {T_k*1e3:.2f} mK")
            return T_k
        except Exception as e:
            print(f"Temperature calculation failed: {e}")
            return None

    def solve_temperature(self, fge_MHz, fef_MHz, Pe_target):
        h = 6.62607015e-34
        kB = 1.380649e-23
        E_e = h * fge_MHz * 1e6
        E_f = h * (fge_MHz + fef_MHz) * 1e6

        def Pe(T):
            exp_ee = np.exp(-E_e / (kB * T))
            exp_ef = np.exp(-E_f / (kB * T))
            Z = 1 + exp_ee + exp_ef
            return exp_ee / Z

        sol = root_scalar(lambda T: Pe(T) - Pe_target, bracket=[0.001, 1], method="brentq")
        return sol.root if sol.converged else None

    def _simulate(self, x_pts):
        # Mocking two different amplitudes based on a "theoretical" 50mK temperature
        if not self.cfg.get("temp_ref", False):
            return mock_decaysin(x_pts, amp=0.4, freq=2.0)
        else:
            return mock_decaysin(x_pts, amp=0.6, freq=2.0)
