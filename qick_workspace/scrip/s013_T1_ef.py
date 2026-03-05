# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

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
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.fitting import expfunc, fitexp
from ..tools.module_fitzcu import T1_analyze
from ..tools.yamltool import yml_comment
from ..plotter.liveplot import liveplotfun
from ..plotter.plot_utils import plot_final


##################
# Define Program #
##################
class T1Program(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]
        qb_ch_ef = cfg["qb_ch_ef"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])
        if self.soccfg["gens"][qb_ch_ef]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qb_ch_ef, nqz=cfg["nqz_qb_ef"], mixer_freq=cfg["qb_mixer_ef"]
            )
        else:
            self.declare_gen(ch=qb_ch_ef, nqz=cfg["nqz_qb_ef"])

        self.add_loop("waitloop", cfg["steps"])
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
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pi_pulse",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
            )
        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pi_pulse",
                ro_ch=ro_ch,
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
            )
        self.add_gauss(
            ch=qb_ch_ef,
            name="ramp_ef",
            sigma=cfg["sigma_ef"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch_ef,
                name="qb_pulse",
                style="arb",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["pi_gain_ef"],
            )

        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch_ef,
                name="qb_pulse",
                style="flat_top",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["pi_gain_ef"],
                length=cfg["qb_flat_top_length_ef"],
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

        self.add_gauss(
            ch=cool_ch1,
            name="cooling1",
            sigma=cfg["res_sigma"],
            length=cfg["res_sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            envelope="cooling1",
            style="flat_top",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=cfg["res_sigma"],
            length=cfg["res_sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            envelope="cooling2",
            style="flat_top",
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

        self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=self.cfg["qb_ch_ef"], name="qb_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")

        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class T1_ef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)

        else:
            prog = T1Program(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iq_list[0][0].dot([1, 1j])
            self.delay_times = prog.get_time_param("wait", "t", as_array=True)

    def plot(self):
        T1_analyze(self.delay_times, self.iq_list[0][0].dot([1, 1j]))

    def liveplot(self, py_avg):
        prog = T1Program(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self.delay_times,
            y_axis_vals=None,
            x_label="Times (us)",
            y_label="ADC Units",
            title_prefix="Qubit T1 ef",
            yoko_inst_addr=None,
            show_final_plot=False,
        )

        ### Final plot ###
        self.fit_params, error, fig = plot_final(
            self.delay_times, self.iqdata, "Times(us)", fitexp, expfunc
        )

        fig.suptitle(f"T1 = {self.fit_params[2]:.2f} +-{error[2]:.2f} us", fontsize=15)
        fig.tight_layout()

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s013_T1_ef" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        try:
            self.cfg.pop("wait_time")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Times", "unit": "us", "values": self.delay_times},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"T1 = {self.fit_params[2]:.2f}$\mu s$ {dict_val}"),
            tag="T1",
        )

        print(f"Data save to {file_path}")
