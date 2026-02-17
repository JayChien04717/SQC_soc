"""
s001 — Time of Flight (TOF)
===============================
Loopback measurement: acquire_decimated with custom liveplot.
Inherits from BaseExperiment for unified saveLabber support,
but overrides run() because the data flow is fundamentally different
(decimated time-domain data, not swept IQ).
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output, update_display

from .base_program import BaseProgram
from .base_experiment import BaseExperiment


# ── Program ──

class LoopbackProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0)


# ── Experiment ──

class TOF(BaseExperiment):
    """Time-of-flight: uses acquire_decimated (not a standard sweep)."""

    EXPT_NAME = "s001_tof"
    TAG = "TOF"
    X_LABEL = r"Time ($\mu$s)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Time of Flight"

    X_SAVE_NAME = "Time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6  # μs → s

    def __init__(self, soc, soccfg, config):
        super().__init__(soc, soccfg, config)
        self.iq_list = None
        self.t = None  # time axis (μs)

    def _create_program(self):
        return LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_time_axis(ro_index=0)

    # ── Override run: acquire_decimated with custom liveplot ──

    def run(self, py_avg=1, simulate=False):
        """Override: uses acquire_decimated instead of standard sweep."""
        if simulate:
            return self._run_simulate(py_avg)
        return self.liveplot(py_avg=py_avg)

    def _run_simulate(self, py_avg=1):
        """Generate mock TOF data without hardware."""
        from .mock_signals import mock_tof
        n_pts = self.cfg.get("readout_length", 200)
        self.t = np.linspace(0, 2.0, n_pts)
        self._sweep_vals_x = self.t
        self.iqdata = mock_tof(self.t, pulse_start=0.3, pulse_end=0.7, amp=1.0, noise=0.02)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.t, np.abs(self.iqdata), "o-", markersize=2, label="Simulated Data")
        mean = np.mean(np.abs(self.iqdata))
        cross_idx = np.argmax(np.abs(self.iqdata) > 1.5 * mean)
        trig_time = self.t[cross_idx]
        ax.axvline(trig_time, c="r", ls="--", label=f"TOF: {trig_time:.2f} μs")
        ax.set_title(f"{self.TITLE_PREFIX} [SIMULATED], trig = {trig_time:.2f} μs")
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel("ADC unit")
        ax.legend()
        plt.show()
        return self.iqdata, True, py_avg

    def liveplot(self, py_avg=1, threshold=1.5):
        prog = self._create_program()
        self.t = self._extract_sweep_axis(prog)
        self._sweep_vals_x = self.t  # for BaseExperiment.saveLabber

        iq_sum = 0
        fig, ax = plt.subplots(figsize=(7, 5))
        nan_data = np.full_like(self.t, np.nan, dtype=float)
        (line,) = ax.plot(self.t, nan_data, alpha=0.8)
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        title = ax.set_title(f"{self.TITLE_PREFIX} | Average: 0 / 0")
        t_min, t_max = np.min(self.t), np.max(self.t)
        ax.set_xlim(t_min, t_max)

        plot_id = f"live-plot-tof-{np.random.randint(1e9)}"
        display(fig, display_id=plot_id)

        interrupted = False
        i = 0
        try:
            for i in tqdm(range(py_avg), desc="Software Average Count"):
                self.iq_list = prog.acquire_decimated(self.soc, rounds=1, progress=False)
                current_iq = self.iq_list[0].dot([1, 1j])
                iq_sum = current_iq if i == 0 else iq_sum + current_iq
                self.iqdata = iq_sum / (i + 1)

                plot_data = np.abs(self.iqdata)
                line.set_ydata(plot_data)
                cmin, cmax = np.min(plot_data), np.max(plot_data)
                span = max(cmax - cmin, 1e-9)
                ax.set_ylim(cmin - 0.1 * span, cmax + 0.1 * span)
                title.set_text(f"{self.TITLE_PREFIX} | Average: {i+1} / {py_avg}")
                update_display(fig, display_id=plot_id)
        except KeyboardInterrupt:
            interrupted = True
            print(f"Interrupted at average count: {i+1}")

        clear_output(wait=True)
        if self.iqdata is not None:
            final_fig, final_ax = plt.subplots(figsize=(7, 5))
            final_ax.plot(self.t, np.abs(self.iqdata), "o-", markersize=2, label="Averaged Data")
            mean = np.mean(np.abs(self.iqdata))
            cross_idx = np.argmax(np.abs(self.iqdata) > threshold * mean)
            trig_time = self.t[cross_idx]
            final_ax.axvline(trig_time, c="r", ls="--", label=f"TOF: {trig_time:.2f} μs")
            title_text = f"{self.TITLE_PREFIX}, trig = {trig_time:.2f} μs"
            if interrupted:
                title_text += " (Interrupted)"
            final_ax.set_title(title_text)
            final_ax.set_xlabel(self.X_LABEL)
            final_ax.set_ylabel("ADC unit")
            final_ax.set_xlim(t_min, t_max)
            final_ax.legend()
            display(final_fig)
            plt.close(final_fig)
        plt.close(fig)
        return self.iqdata, not interrupted, i + 1

    def plot(self, threshold=1.5):
        if self.iq_list is None:
            print("No data to plot. Run the experiment first.")
            return
        plt.plot(self.t, self.iq_list[0].T[0])
        plt.plot(self.t, self.iq_list[0].T[1])
        plt.plot(self.t, np.abs(self.iq_list[0].dot([1, 1j])))
        plt.xlabel("Time (us)")
        plt.ylabel("a.u")
        mean = np.mean(np.abs(self.iq_list[0].dot([1, 1j])))
        plt.axvline(
            self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > threshold * mean)],
            c="r", ls="--",
        )
        plt.title(f"Time of Flight, trig = {round(self.t[np.argmax(np.abs(self.iq_list[0].dot([1,1j])) > 1.5*mean)], 2)} us")


