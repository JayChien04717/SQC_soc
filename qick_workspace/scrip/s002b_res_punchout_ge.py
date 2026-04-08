# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2, QickSpan, QickSweep1D

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.fitting import *
from ..tools.yamltool import yml_comment
from ..plotter.liveplot import liveplotfun

##################
# Define Program #
##################


class SingleToneSpectroscopyPunchoutProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleToneSpectroscopyPunchout:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = SingleToneSpectroscopyPunchoutProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            self.iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def plot(self):
        data = np.abs(self.iqdata)  # shape: (n_gain, n_freq)
        data_norm = np.array(
            [
                (row - np.min(row)) / (np.max(row) - np.min(row))
                if np.max(row) != np.min(row)
                else row
                for row in data
            ]
        )
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title("Resonator Punch Out")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Dac Gains [a.us]")
        plt.colorbar(pcm)

    def liveplot(self, py_avg):
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self.freqs,
            y_axis_vals=self.gains,
            x_label="Frequency (MHz)",
            y_label="DAC Gain",
            title_prefix="Resonator Punchout",
            yoko_inst_addr=None,
            show_final_plot=False,
        )
        if self.iqdata is None:
            print("No data was acquired.")
            return None

        if interrupted:
            print(
                f"Experiment interrupted at {avg_count} averages. "
                "Circle fit is based on partial data."
            )

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "002b_res_ge_punchout" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        try:
            self.cfg.pop("res_freq_ge")
            self.cfg.pop("res_gain_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            y_info={"name": "DAC Gains", "unit": "a.u.", "values": self.gains},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="OneTone",
        )
        print(f"Data save to {file_path}")
