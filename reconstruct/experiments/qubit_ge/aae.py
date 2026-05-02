"""
QubitGE/aae — s005a_AAE: Amplitude-Amplitude-Envelope (power Rabi chevron).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output, update_display
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...tools.fitting import decaysin, fitdecaysin, fix_phase
from ...plotter.plot_utils import plot_final


class PowerRabiProgram(BaseProgram):
    """QICK program for AAE power Rabi: repeats the pulse ``iteration`` times."""

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


# Aliases for backward compatibility
AAEProgram = PowerRabiProgram


class PowerRabiChevron(BaseExperiment):
    """
    AAE Power Rabi Chevron experiment.

    Performs a 2D scan: inner loop sweeps gain (hardware), outer loop
    sweeps iteration count (software).
    """

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

    def run(self, py_avg, show_final_plot=False, **kwargs):
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

                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                iq_data_row = iq_list[0][0].dot([1, 1j])

                iqdata_full[y_idx, :] = iq_data_row
                data_to_plot = np.abs(iqdata_full)

                mesh.set_array(data_to_plot.ravel())

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

    def analyze_and_plot(self):
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        gains = self._sweep_vals_x
        iters = self._sweep_vals_y
        raw_sum_trace = np.sum(np.abs(self.iqdata), axis=0)
        sum_trace = gaussian_filter1d(raw_sum_trace, sigma=2.0)

        dx = gains[1] - gains[0]
        fft_vals = np.abs(np.fft.rfft(sum_trace - np.mean(sum_trace)))
        fft_freqs = np.fft.rfftfreq(len(gains), d=dx)
        freq_guess = fft_freqs[np.argmax(fft_vals[1:]) + 1]
        width_guess = 0.5 / freq_guess

        amp_guess = (np.max(sum_trace) - np.min(sum_trace)) / 2
        off_guess = np.mean(sum_trace)

        idx_max = int(np.argmax(sum_trace))
        idx_min = int(np.argmin(sum_trace))
        if abs(sum_trace[idx_max] - off_guess) >= abs(sum_trace[idx_min] - off_guess):
            x0_guess = gains[idx_max]
            sign_guess = 1.0
        else:
            x0_guess = gains[idx_min]
            sign_guess = -1.0

        def sinc2_model(x, A, x0, width, offset):
            return A * np.sinc((x - x0) / width) ** 2 + offset

        fit_success = False
        optimal_gain = x0_guess

        try:
            p0 = [sign_guess * amp_guess, x0_guess, width_guess, off_guess]
            popt, _ = curve_fit(
                sinc2_model, gains, sum_trace, p0=p0,
                bounds=([-np.inf, gains.min(), dx, -np.inf],
                        [np.inf, gains.max(), np.inf, np.inf]),
                maxfev=10000,
            )
            A_fit, x0_fit, width_fit, offset_fit = popt
            if gains.min() <= x0_fit <= gains.max():
                optimal_gain = x0_fit
                fit_success = True
            else:
                print("Fit x0 out of range, falling back.")
        except Exception as e:
            print(f"Fit failed: {e}, falling back to smoothed extremum.")

        print(f"\n[PowerRabi] Optimal pi gain = {optimal_gain:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax0 = axes[0]
        im = ax0.pcolormesh(gains, iters, np.abs(self.iqdata), shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax0, label="ADC Units (Abs)")
        ax0.axvline(optimal_gain, color="red", linestyle="--", alpha=0.8,
                    label=f"Fit={optimal_gain:.4f}")
        ax0.set_title("Power Rabi Chevron")
        ax0.legend()

        ax1 = axes[1]
        ax1.scatter(gains, raw_sum_trace, s=20, color="steelblue", alpha=0.5, label="Raw Data")
        ax1.plot(gains, sum_trace, "--", color="gray", alpha=0.7, label="Smoothed")

        if fit_success:
            fine_x = np.linspace(gains.min(), gains.max(), 2000)
            ax1.plot(fine_x, sinc2_model(fine_x, *popt), color="firebrick",
                     lw=2, label="Sinc² Fit")

        ax1.axvline(optimal_gain, color="red", linestyle="--")
        ax1.set_title("Summed Trace & Physical Fit")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        return optimal_gain

    def _save_comment(self, dict_val):
        if self.fit_params:
            g = self.fit_params.get("optimal_gain", "N/A")
            return f"AAE Power Rabi\nOptimal gain = {g}\n{dict_val}"
        return f"{dict_val}"


# Alias
AAE = PowerRabiChevron

__all__ = ["PowerRabiProgram", "PowerRabiChevron", "AAEProgram", "AAE"]
