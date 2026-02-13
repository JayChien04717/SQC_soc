"""
s002b — Resonator Punchout (ge)
================================
2D sweep: resonator gain × frequency.
Uses BaseProgram for generator setup, but adds custom gain/freq loops.
"""
import numpy as np
import matplotlib.pyplot as plt

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..plotter.liveplot import liveplotfun


# ── Program ──

class PunchoutProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.measure(cfg)


# ── Experiment ──

class Punchout:
    """Resonator punchout: 2D sweep over gain and frequency."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=True, simulate=False):
        if simulate:
            return self._simulate()

        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = PunchoutProgram(
                self.soccfg, reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"], cfg=self.cfg,
            )
            self.iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def _simulate(self):
        """Generate mock 2D punchout data without hardware."""
        from .mock_signals import mock_lorentzian_2d
        res_freq = self.cfg.get("res_freq_ge")
        if hasattr(res_freq, "start"):
            self.freqs = np.linspace(res_freq.start, res_freq.stop, self.cfg.get("f_steps", 51))
        else:
            self.freqs = np.linspace(6000, 6100, self.cfg.get("f_steps", 51))

        res_gain = self.cfg.get("res_gain_ge")
        if hasattr(res_gain, "start"):
            self.gains = np.linspace(res_gain.start, res_gain.stop, self.cfg.get("g_steps", 11))
        else:
            self.gains = np.linspace(0, 1, self.cfg.get("g_steps", 11))

        self.iqdata = mock_lorentzian_2d(self.freqs, self.gains,
                                          f0=(self.freqs[0] + self.freqs[-1]) / 2)
        # Plot
        data = np.abs(self.iqdata)
        data_norm = np.array([
            (row - row.min()) / (row.max() - row.min())
            if row.max() != row.min() else row
            for row in data
        ])
        plt.figure(figsize=(8, 5))
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title("Resonator Punch Out [SIMULATED]")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("DAC Gains [a.u.]")
        plt.colorbar(pcm)
        plt.show()

    def liveplot(self, py_avg):
        prog = PunchoutProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog, soc=self.soc, py_avg=py_avg,
            x_axis_vals=self.freqs, y_axis_vals=self.gains,
            x_label="Frequency (MHz)", y_label="DAC Gain",
            title_prefix="Resonator Punchout", show_final_plot=False,
        )
        if self.iqdata is None:
            print("No data acquired.")
        elif interrupted:
            print(f"Interrupted at {avg_count} averages (data is partial).")

    def plot(self):
        data = np.abs(self.iqdata)
        data_norm = np.array([
            (row - row.min()) / (row.max() - row.min())
            if row.max() != row.min() else row
            for row in data
        ])
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title("Resonator Punch Out")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("DAC Gains [a.u.]")
        plt.colorbar(pcm)

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = f"s002b_res_ge_punchout_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        dict_val = config_to_yaml(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            y_info={"name": "DAC Gains", "unit": "a.u.", "values": self.gains},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=f"{dict_val}", tag="OneTone",
        )
        print(f"Data save to {file_path}")
