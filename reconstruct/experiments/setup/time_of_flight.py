"""
Setup/time_of_flight — s001: Time-of-flight (TOF) measurement.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output, update_display

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...core.experiment_data import ExperimentData, QualityFlag


class LoopbackProgram(BaseProgram):
    """QICK loopback program for TOF measurement."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg, prefix="ge")
        self.has_qb_pulse = False
        self.has_qb_pulse_ef = False
        try:
            self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")
            self.has_qb_pulse = True
        except Exception:
            pass
        try:
            self.setup_qb_pulse(cfg, "ef", name="qb_pulse_ef", gain_key="pi_gain_ef")
            self.has_qb_pulse_ef = True
        except Exception:
            pass

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        do_check_e = cfg.get("check_e", cfg.get("cheek_e", False))
        do_check_f = cfg.get("check_f", cfg.get("cheek_f", False))
        if do_check_e and self.has_qb_pulse:
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(0.05, tag="check_delay")
        if do_check_f and self.has_qb_pulse_ef:
            if self.has_qb_pulse:
                self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
                self.delay_auto(0.05)
            ch_ef = cfg.get("qb_ch_ef", cfg.get("qb_ch"))
            self.pulse(ch=ch_ef, name="qb_pulse_ef", t=0)
            self.delay_auto(0.05, tag="check_delay_ef")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0)


class TOF(BaseExperiment):
    """Time-of-flight: acquire_decimated mode with live plot."""

    EXPT_NAME = "s001_tof"
    TAG = "TOF"
    X_LABEL = r"Time ($\mu$s)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Time of Flight"
    X_SAVE_NAME = "Time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    def __init__(self, config, backend=None):
        super().__init__(config, backend)
        self.iq_list = None
        self.t = None

    def _create_program(self):
        return LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_time_axis(ro_index=0)

    def run(self, py_avg=1, **kwargs) -> ExperimentData:
        """Override: uses acquire_decimated instead of standard sweep."""
        return self.liveplot(py_avg=py_avg)

    def liveplot(self, py_avg=1, threshold=1.5) -> ExperimentData:
        prog = self._create_program()
        self.t = self._extract_sweep_axis(prog)
        self._sweep_vals_x = self.t

        iq_sum = 0
        fig, ax = plt.subplots(figsize=(7, 5))
        nan_data = np.full_like(self.t, np.nan, dtype=float)
        (line,) = ax.plot(self.t, nan_data, alpha=0.8)
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        title = ax.set_title(f"{self.TITLE_PREFIX} | Average: 0 / 0")
        t_min, t_max = np.min(self.t), np.max(self.t)
        ax.set_xlim(t_min, t_max)
        plot_id = f"live-plot-tof-{np.random.randint(int(1e9))}"
        display(fig, display_id=plot_id)

        interrupted = False
        i = 0
        trig_time = 0.0
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
                title.set_text(f"{self.TITLE_PREFIX} | Average: {i + 1} / {py_avg}")
                update_display(fig, display_id=plot_id)
        except KeyboardInterrupt:
            interrupted = True

        clear_output(wait=True)
        if self.iqdata is not None:
            final_fig, final_ax = plt.subplots(figsize=(7, 5))
            final_ax.plot(self.t, np.abs(self.iqdata), "o-", markersize=2)
            mean = np.mean(np.abs(self.iqdata))
            cross_idx = np.argmax(np.abs(self.iqdata) > threshold * mean)
            trig_time = float(self.t[cross_idx])
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

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.t,
            scalar_result=trig_time,
            fit_result={"trig_time_us": (trig_time, None)},
            interrupted=interrupted,
            avg_count=i + 1,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
        )
        self.result = result
        return result

    def plot(self, threshold=1.5):
        if self.iq_list is None:
            print("No data. Run the experiment first.")
            return
        plt.plot(self.t, self.iq_list[0].T[0])
        plt.plot(self.t, self.iq_list[0].T[1])
        plt.plot(self.t, np.abs(self.iq_list[0].dot([1, 1j])))
        plt.xlabel("Time (us)")
        mean = np.mean(np.abs(self.iq_list[0].dot([1, 1j])))
        trig = self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > threshold * mean)]
        plt.axvline(trig, c="r", ls="--")
        plt.title(f"Time of Flight, trig = {round(float(trig), 2)} us")
