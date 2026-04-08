# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output

# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2, QickSpan, QickSweep1D

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.yamltool import yml_comment
from ..plotter.liveplot import liveplotfun

##################
# Define Program #
##################


class SingleToneSpectroscopyProgram_yoko(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_loop("freqloop", cfg["steps"])
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


class SingleToneSpectroscopyProgram_hardware(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        flux_ch = cfg["flux_ch"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_gen(ch=flux_ch, nqz=1)
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_loop("fluxloop", cfg["steps_flux"])
        self.add_loop("freqloop", cfg["steps"])
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
        self.add_pulse(
            ch=flux_ch,
            name="flux_pulse",
            style="const",
            length=cfg["flux_length"],
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay(cfg["saturate_times"])
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Resonator_onetone_flux:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(
        self,
        py_avg,
        liveplot=False,
        yoko_inst: str = None,
        yoko_value=None,
        mode: str = "current",
    ):
        if yoko_inst is not None:
            if yoko_value is None:
                raise ValueError("Please provide yoko sweep values for liveplot_yoko.")
            self.liveplot_yoko(
                py_avg, yoko_value=yoko_value, yoko_inst=yoko_inst, mode=mode
            )
        elif yoko_inst is None and liveplot is True:
            self.liveplot_hardware(py_avg)

        else:
            prog = SingleToneSpectroscopyProgram_yoko(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        pass

    def liveplot_yoko(
        self,
        py_avg,
        yoko_value: np.ndarray,
        yoko_inst: str = None,
        mode: str = "current",
    ):
        prog = SingleToneSpectroscopyProgram_yoko(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

        iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self.freqs,
            y_axis_vals=yoko_value,
            x_label="Frequency (MHz)",
            y_label=mode.capitalize(),
            title_prefix="Resonator Onetone Flux",
            yoko_inst_addr=yoko_inst,
            yoko_mode=mode,
        )
        self.iqdata = iqdata
        self.yoko_currnet = yoko_value

        if interrupted:
            print(f"Yoko sweep interrupted at step {avg_count}. Data stored.")

    def liveplot_hardwre(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyProgram_hardware(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            ax.cla()
            ax.pcolormesh(self.freqs, self.gains, np.abs(self.iqdata))
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Flux Gains")

            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        ax.pcolormesh(self.freqs, self.gains, np.abs(self.iqdata))

    def saveLabber(self, qb_idx, yoko_value=None, mode: str = "current"):
        expt_name = "s002_onetone_flux" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        try:
            self.cfg.pop("res_freq_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        if yoko_value is not None:
            if mode == "current":
                hdf5_generator(
                    filepath=file_path,
                    x_info={
                        "name": "Frequency",
                        "unit": "Hz",
                        "values": self.freqs * 1e6,
                    },
                    y_info={"name": "Yoko", "unit": "A", "values": yoko_value},
                    z_info={
                        "name": "Signal",
                        "unit": "ADC unit",
                        "values": self.iqdata,
                    },
                    comment=(f"{dict_val}"),
                    tag="OneTone",
                )
            elif mode == "voltage":
                hdf5_generator(
                    filepath=file_path,
                    x_info={
                        "name": "Frequency",
                        "unit": "Hz",
                        "values": self.freqs * 1e6,
                    },
                    y_info={"name": "Yoko", "unit": "V", "values": yoko_value},
                    z_info={
                        "name": "Signal",
                        "unit": "ADC unit",
                        "values": self.iqdata,
                    },
                    comment=(f"{dict_val}"),
                    tag="OneTone",
                )
        print(f"Data save to {file_path}")
