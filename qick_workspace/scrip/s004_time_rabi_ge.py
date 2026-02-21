# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# ===================================================================
# 2. QICK Libraries
# ===================================================================
from qick import *
from qick.asm_v2 import AveragerProgramV2

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..tools.module_fitzcu import lengthrabi_analyze
from ..plotter.liveplot import liveplotfun
from ..plotter.plot_utils import plot_final

##################
# Define Program #
##################


class LengthRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
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
        self.add_gauss(
            ch=qb_ch,
            name="qb",
            sigma=cfg["sigma_ge"],
            length=5 * cfg["sigma_ge"],
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_pulse",
            style="flat_top",
            envelope="qb",
            length=cfg["qb_length_ge"],
            freq=cfg["qb_freq_ge"],
            phase=0,
            gain=cfg["qb_gain_ge"],
        )

    def apply_cool(self, cfg):
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]
        if self.soccfg["gens"][cool_ch1]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch1, nqz=cfg["nqz_cool_ch1"], mixer_freq=cfg["cool_mixer1"]
            )
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg["nqz_cool_ch1"])

        if self.soccfg["gens"][cool_ch2]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch2, nqz=cfg["nqz_cool_ch2"], mixer_freq=cfg["cool_mixer2"]
            )
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg["nqz_cool_ch2"])

        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg["cooling"] is True:
            self.apply_cool(cfg)
            self.pulse(ch=self.cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=self.cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(0.5, tag="Ring down")
        else:
            pass
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0.05, tag="waiting")

        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Time_Rabi:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False, time_axis: np.ndarray = None):
        if liveplot:
            if time_axis is None:
                raise ValueError("time_axis must be provided for live plotting.")
            else:
                return self.liveplot(py_avg, time_axis)
        else:
            prog = LengthRabiProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.gains = prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def plot(self):
        lengthrabi_analyze(self.time_step, self.iqdata)

    def liveplot(self, py_avg, time_axis):
        def create_rabi_prog(length_val):
            self.cfg["qb_length_ge"] = length_val
            return LengthRabiProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

        self.time_step = time_axis

        self.iqdata, interrupted, done_avg = liveplotfun(
            soc=self.soc,
            py_avg=py_avg,
            scan_x_axis=self.time_step,
            get_prog_callback=create_rabi_prog,
            x_label="Time (us)",
            title_prefix="Length Rabi",
            show_final_plot=False,
        )

        ### Final plot ###
        self.fit_params, error, fig, ax = plot_final(
            self.time_step,
            self.iqdata,
            "Times(us)",
            fitdecaysin,
            decaysin,
            return_ax=True,
        )
        fig.suptitle(f"Time Rabi ge, Rabi frequency = {self.fit_params[1]:.3f} MHz")
        fig.tight_layout()
        pi_len, pi2_len = fix_phase(self.fit_params)
        ax.axvline(pi_len, color="red", linestyle="--", label=r"$\pi$ length")
        ax.axvline(pi2_len, color="red", linestyle="--", label=r"$\pi/2$ length")
        ax.legend()

        return round(pi_len, 3), round(pi2_len, 3)

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "004_time_rabi_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        dict_val = config_to_yaml(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.time_step * 1e-6},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"Rabi frequency = {self.fit_params[1]:.3f} MHz\n{dict_val}"),
            tag="Rabi",
        )

        print(f"Data save to {file_path}")
