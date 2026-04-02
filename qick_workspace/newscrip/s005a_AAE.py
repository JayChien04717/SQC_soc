"""
s005 — Power Rabi (ge)
=======================
Amplitude Rabi: sweeps qubit drive gain at fixed length.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import display, clear_output, update_display
from tqdm.auto import tqdm

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..plotter.plot_utils import plot_final


class PowerRabiProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        for _ in range(cfg["iteration"]):
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(t=0.02)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiChevron(BaseExperiment):
    EXPT_NAME = "s005_power_rabi_chevron"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Y_SAVE_NAME = "Iterations"
    Y_SAVE_UNIT = "N"
    Y_SAVE_SCALE = 1.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_scan_axes(self):
        cfg = self.cfg
        prog = self._create_program()
        gains = self._extract_sweep_axis(prog)

        if "iter_start" in cfg and "iter_stop" in cfg:
            iters = np.arange(
                cfg["iter_start"],
                cfg["iter_stop"] + 1,
                cfg.get("iter_step", 1),
                dtype=int,
            )
        else:
            iters = np.array([int(cfg.get("iteration", 21))])

        return gains, iters

    # ── override run() ───────────────────────────────────────────────────────

    def run(self, py_avg, show_final_plot=False, **kwargs):
        """
        Mixed 2D parameter scan:
        - Inner loop: Hardware sweep over gain (handled by PowerRabiProgram).
        - Outer loop: Software sweep over iterations.
        """
        gains, iters = self._build_scan_axes()
        self._sweep_vals_x = gains
        self._sweep_vals_y = iters

        iqdata_full = np.zeros((len(iters), len(gains)), dtype=complex)
        data_to_plot = np.zeros((len(iters), len(gains)))

        fig, ax = plt.subplots(figsize=(6, 4))
        mesh = ax.pcolormesh(gains, iters, data_to_plot, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="ADC Units (Abs)")
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel("Iterations (N)")
        title = ax.set_title(f"{self.TITLE_PREFIX} (Initializing...)")

        plot_display_id = f"live-plot-aae-{np.random.randint(1e9)}"
        display(fig, display_id=plot_display_id)

        interrupted = False
        try:
            for y_idx, iter_val in enumerate(
                tqdm(iters, desc="Outer Sweep: Iterations")
            ):
                self.cfg["iteration"] = int(iter_val)
                prog = self._create_program()

                # Hardware sweep provides the inner loop natively.
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                iq_data_row = iq_list[0][0].dot([1, 1j])

                iqdata_full[y_idx, :] = iq_data_row
                data_to_plot = np.abs(iqdata_full)

                mesh.set_array(data_to_plot.ravel())

                # Update color scale based on measured data only
                measured_data = data_to_plot[: y_idx + 1, :]
                c_min, c_max = np.min(measured_data), np.max(measured_data)
                if c_max > c_min:
                    mesh.set_clim(vmin=c_min, vmax=c_max)
                elif c_max > 0:
                    mesh.set_clim(vmin=0, vmax=c_max)

                title.set_text(f"{self.TITLE_PREFIX} | N={iter_val}")
                update_display(fig, display_id=plot_display_id)

        except KeyboardInterrupt:
            interrupted = True

        clear_output(wait=True)
        plt.close(fig)

        if interrupted:
            print(f"Interrupted at iteration {iters[y_idx]}.")

        self.iqdata = iqdata_full
        return self._post_fit()

    # ── _create_program / _extract_sweep_axis (required) ─────────────────────

    def _create_program(self):
        self.cfg.setdefault("iteration", self.cfg.get("iter_start", 1))
        return PowerRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return self._sweep_vals_y

    # ── analysis ─────────────────────────────────────────────────────────────

    def analyze_and_plot(self):
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        gains = self._sweep_vals_x
        iters = self._sweep_vals_y

        # Sum all iteration traces (abs) to amplify the error
        sum_trace = np.sum(np.abs(self.iqdata), axis=0)  # shape: (n_gains,)

        # Perfect pi gain should result in max excitation amplitude in summed trace
        idx_max = int(np.argmax(sum_trace))
        optimal_gain = gains[idx_max]

        # Parabola sub-pixel refinement around the max peak
        try:

            def parabola(x, a, b, c):
                return a * (x - b) ** 2 + c

            i0 = max(idx_max - 3, 0)
            i1 = min(idx_max + 4, len(gains))
            popt, _ = curve_fit(
                parabola,
                gains[i0:i1],
                sum_trace[i0:i1],
                p0=[-1.0, gains[idx_max], sum_trace[idx_max]],
            )
            if gains.min() <= popt[1] <= gains.max():
                optimal_gain = popt[1]
        except Exception:
            pass

        print(f"\n[AAE Gain] Optimal pi gain = {optimal_gain:.6f}")

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: 2D heatmap
        ax0 = axes[0]
        im = ax0.pcolormesh(
            gains, iters, np.abs(self.iqdata), shading="auto", cmap="viridis"
        )
        fig.colorbar(im, ax=ax0, label="ADC Units (Abs)")
        ax0.axvline(
            optimal_gain,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Gain={optimal_gain:.4f}",
        )
        ax0.set_xlabel(self.X_LABEL)
        ax0.set_ylabel("Iterations (N)")
        ax0.set_title("Power Rabi Chevron")
        ax0.legend(fontsize=9)

        # Right: Summed trace
        ax1 = axes[1]
        ax1.plot(
            gains,
            sum_trace,
            "o-",
            color="steelblue",
            linewidth=2,
            markersize=5,
            label="Σ iterations",
        )
        ax1.axvline(
            optimal_gain,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Gain={optimal_gain:.4f}",
        )
        ax1.set_xlabel(self.X_LABEL)
        ax1.set_ylabel("Summed ADC Units (Abs)")
        ax1.set_title("Summed Trace")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        fig.suptitle(self.TITLE_PREFIX, fontsize=13)
        fig.tight_layout()
        plt.show()

        self.fit_params = {
            "optimal_gain": round(float(optimal_gain), 6),
        }
        return self.fit_params

    def _save_comment(self, dict_val):
        if self.fit_params:
            g = self.fit_params.get("optimal_gain", "N/A")
            return f"AAE Power Rabi\nOptimal gain = {g}\n{dict_val}"
        return f"{dict_val}"
