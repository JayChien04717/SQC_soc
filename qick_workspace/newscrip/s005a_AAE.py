"""
s005a — Amplified Amplitude Error (AAE)
========================================
pi/2 followed by N repetitions of pi (or pi/2+pi/2) pulses.
Uses callback-based liveplot (one program per iteration count).
"""
import matplotlib.pyplot as plt
import numpy as np

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fit_probg_Xhalf, fit_probg_X, probg_Xhalf, probg_X
from ..plotter.liveplot import liveplotfun


# ── Program ──

class AAEProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qb_pulse(cfg, 'ge', name="qb_pulse_pi2", gain_key="pi2_gain_ge")
        self.setup_qb_pulse(cfg, 'ge', name="qb_pulse_pi", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi2", t=0)
        self.delay_auto(0.02)
        if cfg["AAE"] == "pi":
            for _ in range(cfg["steps"]):
                self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi", t=0)
                self.delay_auto(0.02)
        elif cfg["AAE"] == "pi2":
            for _ in range(cfg["steps"]):
                self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi2", t=0)
                self.delay_auto(0.02)
                self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi2", t=0)
                self.delay_auto(0.02)
        self.delay_auto(0.01)
        self.measure(cfg)


# ── Experiment ──

class AAE:
    """Amplified Amplitude Error: callback-based liveplot over iteration count."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, iteration_count):
        """Run with liveplot over N=0..iteration_count."""
        self.iter = np.arange(0, iteration_count, 1)

        def create_prog(n):
            self.cfg["steps"] = int(n)
            return AAEProgram(
                self.soccfg, reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"], cfg=self.cfg,
            )

        self.iqdata, interrupted, done_avg = liveplotfun(
            soc=self.soc, py_avg=py_avg,
            scan_x_axis=self.iter, get_prog_callback=create_prog,
            x_label="N", title_prefix="Amplified Amplitude Error",
            show_final_plot=True,
        )

    def analyze_and_plot(self):
        """Fit AAE data and extract angle error."""
        n_pts = self.iter
        z_pts = np.real(self.iqdata)
        aae_mode = self.cfg.get("AAE", "pi2")

        try:
            if aae_mode == "pi2":
                pOpt, pCov = fit_probg_Xhalf(n_pts, z_pts)
                fit_func = probg_Xhalf
            else:
                pOpt, pCov = fit_probg_X(n_pts, z_pts)
                fit_func = probg_X

            a_fit, delta_fit = pOpt
            plt.figure(figsize=(8, 5))
            plt.scatter(n_pts, z_pts, color="blue", label="Raw Data", zorder=3)
            n_fine = np.linspace(min(n_pts), max(n_pts), 1000)
            z_fit = [fit_func(n, *pOpt) for n in n_fine]
            plt.plot(n_fine, z_fit, color="orange",
                     label=f"Fit ($\\delta$={delta_fit:.4f} deg)")
            plt.xlabel("N (Number of repetitions)")
            plt.ylabel("<Z>")
            plt.title(f"Amplified Amplitude Error ({aae_mode})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            print(f"[{aae_mode}] offset(a): {a_fit:.4f}, angle error(delta): {delta_fit:.6f} deg")
            return pOpt
        except Exception as e:
            print(f"Fitting failed: {e}")
            return None

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = f"s005a_AAE_ge_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        dict_val = config_to_yaml(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "N", "unit": "#", "values": self.iter},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=f"\n{dict_val}", tag="AAE",
        )
        print(f"Data save to {file_path}")
