"""
s005a — DRAG Calibration (Alpha Sweep)
=====================================
Sweeps the DRAG parameter alpha for a fixed number of X180, X-180 repetitions
to find the alpha that minimizes phase leakage errors.
Uses callback-based liveplot (one program per alpha iteration).
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..plotter.liveplot import liveplotfun


# ── Program ──

class DragProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.declare_gen_auto(cfg["qb_ch"], cfg["nqz_qb"], "qb_mixer", cfg)

        # Retrieve drag alpha and anharmonicity
        alpha = cfg.get("drag_alpha", 0.5)
        # delta represents the anharmonicity: the difference between f_ge and f_ef
        delta = cfg['qb_freq_ge'] - cfg['qb_freq_ef']

        # We define a custom DRAG envelope using the add_DRAG method provided by qick
        self.add_DRAG(
            ch=cfg["qb_ch"],
            name="drag_env",
            sigma=cfg["sigma_ge"],
            length=cfg.get("length_mult", 5) * cfg["sigma_ge"],
            delta=delta, 
            alpha=alpha,
            even_length=True
        )

        # Add x180 and mx180 pulses using the DRAG envelope
        self.add_pulse(
            ch=cfg["qb_ch"],
            name="x180",
            style="arb",
            envelope="drag_env",
            freq=cfg["qb_freq_ge"],
            phase=0,
            gain=cfg["pi_gain_ge"]
        )
        self.add_pulse(
            ch=cfg["qb_ch"],
            name="mx180",
            style="arb",
            envelope="drag_env",
            freq=cfg["qb_freq_ge"],
            phase=180,  # Negative X pulse
            gain=cfg["pi_gain_ge"]
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # DRAG error amplification sequence: [X180, X-180] * N
        # The number of repetitions 'iter' dictates how much the error is amplified
        iterations = int(cfg.get("iter", 10))
        for _ in range(iterations):
            self.pulse(ch=cfg["qb_ch"], name="x180", t=0)  
            self.delay_auto(0.01)
            self.pulse(ch=cfg["qb_ch"], name="mx180", t=0)
            self.delay_auto(0.01)

        self.measure(cfg)


# ── Experiment ──

class DragCalibration:
    """DRAG Parameter Calibration: callback-based liveplot over alpha."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config
        self.iqdata = None
        self.alphas = None

    def run(self, py_avg, simulate=False):
        """Run with liveplot over alpha (and optionally iter as a 2D sweep)."""
        if "alpha_start" not in self.cfg or "alpha_stop" not in self.cfg or "alpha_steps" not in self.cfg:
            raise ValueError("Config must contain 'alpha_start', 'alpha_stop', and 'alpha_steps'.")

        # Create the array of alphas to sweep
        self.alphas = np.linspace(
            self.cfg["alpha_start"], self.cfg["alpha_stop"], self.cfg["alpha_steps"]
        )

        # Create the array of iterations if iter_start and iter_stop are provided
        if "iter_start" in self.cfg and "iter_stop" in self.cfg:
            self.iters = np.arange(
                self.cfg["iter_start"], self.cfg["iter_stop"] + 1, self.cfg.get("iter_step", 1)
            )
        else:
            # Fallback to 1D sweep using the fixed 'iter' value
            self.iters = None

        if simulate:
            if self.iters is not None:
                a_ideal = 0.5
                self.iqdata = np.zeros((len(self.iters), len(self.alphas)), dtype=complex)
                for i, it in enumerate(self.iters):
                    z_mock = it * 10 * (self.alphas - a_ideal)**2 + 100 + np.random.normal(0, 2, len(self.alphas))
                    self.iqdata[i, :] = z_mock + 1j * np.random.normal(0, 1, len(self.alphas))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.pcolormesh(self.alphas, self.iters, np.abs(self.iqdata), shading='auto', cmap='viridis')
                fig.colorbar(im, label="ADC Units (Abs)")
                ax.set_ylabel("Repetitions (N)")
            else:
                a_ideal = 0.5
                z_mock = 50 * (self.alphas - a_ideal)**2 + 100 + np.random.normal(0, 2, len(self.alphas))
                self.iqdata = z_mock + 1j * np.random.normal(0, 1, len(self.alphas))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(self.alphas, np.abs(self.iqdata), "o-", markersize=4)
                ax.set_ylabel("ADC Units (Abs)")
            
            ax.set_xlabel("DRAG Parameter ($\\alpha$)")
            ax.set_title("DRAG Calibration [SIMULATED]")
            fig.tight_layout()
            plt.show()
            return

        if self.iters is not None:
            def create_prog_2d(a, n):
                self.cfg["drag_alpha"] = a
                self.cfg["iter"] = int(n)
                return DragProgram(
                    self.soccfg, reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"], cfg=self.cfg,
                )

            self.iqdata, interrupted, done_avg = liveplotfun(
                soc=self.soc, py_avg=py_avg,
                scan_x_axis=self.alphas,
                scan_y_axis=self.iters,
                get_prog_callback=create_prog_2d,
                x_label="DRAG Alpha", 
                y_label="Iterations (N)",
                title_prefix="DRAG Calibration",
                show_final_plot=True,
            )
        else:
            def create_prog_1d(a):
                self.cfg["drag_alpha"] = a
                return DragProgram(
                    self.soccfg, reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"], cfg=self.cfg,
                )

            self.iqdata, interrupted, done_avg = liveplotfun(
                soc=self.soc, py_avg=py_avg,
                scan_x_axis=self.alphas, get_prog_callback=create_prog_1d,
                x_label="DRAG Alpha", 
                title_prefix=f"DRAG Calibration (iter={int(self.cfg.get('iter', 10))})",
                show_final_plot=True,
            )

    def analyze_and_plot(self):
        """Fit DRAG data (parabola) and extract optimal alpha."""
        if self.iqdata is None:
            print("No data acquired yet. Call run() first.")
            return None

        x_pts = self.alphas
        
        # Determine whether to analyze 1D or the last row of 2D
        if getattr(self, "iters", None) is not None:
            z_pts = np.abs(self.iqdata[-1, :])
            title_str = f"DRAG Calibration (iter={self.iters[-1]})"
        else:
            z_pts = np.abs(self.iqdata)
            title_str = f"DRAG Calibration (iter={int(self.cfg.get('iter', 10))})"

        def parabola(x, a, b, c):
            return a * (x - b)**2 + c

        try:
            idx_max = np.argmax(z_pts)
            idx_min = np.argmin(z_pts)
            
            b_guess = x_pts[idx_min]
            c_guess = z_pts[idx_min]
            denom = max((x_pts[idx_max] - b_guess)**2, 1e-10)
            a_guess = (z_pts[idx_max] - c_guess) / denom

            pOpt, pCov = curve_fit(parabola, x_pts, z_pts, p0=[a_guess, b_guess, c_guess])
            a_fit, b_fit, c_fit = pOpt
            optimal_alpha = b_fit

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_pts, z_pts, "o", color="blue", label="Raw Data (Max Iter)", markersize=6, alpha=0.7)
            
            x_fine = np.linspace(min(x_pts), max(x_pts), 1000)
            z_fit = parabola(x_fine, *pOpt)
            
            ax.plot(x_fine, z_fit, color="orange", linewidth=2, label=f"Fit (Optimum $\\alpha$={optimal_alpha:.4f})")
            ax.axvline(optimal_alpha, color="red", linestyle="--", alpha=0.8)
            
            ax.set_xlabel("DRAG Parameter ($\\alpha$)")
            ax.set_ylabel("Magnitude (a.u.)")
            ax.set_title(title_str)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"[DRAG] Optimal alpha found: {optimal_alpha:.4f}")
            return optimal_alpha

        except Exception as e:
            print(f"Fitting failed: {e}")
            return None

    def saveLabber(self, qb_idx, yoko_value=None):
        if self.iqdata is None:
            print("No data to save.")
            return

        expt_name = f"s005a_drag_ge_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        dict_val = config_to_yaml(self.cfg)
        
        x_info = {"name": "Alpha", "unit": "a.u.", "values": self.alphas}
        y_info = None
        if getattr(self, "iters", None) is not None:
             y_info = {"name": "Iterations", "unit": "N", "values": self.iters}

        hdf5_generator(
            filepath=file_path,
            x_info=x_info,
            y_info=y_info,
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=f"\n{dict_val}", 
            tag="DRAGCalibration",
        )
        print(f"Data save to {file_path}")
