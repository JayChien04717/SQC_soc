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

from qick_workspace.scrip.s005_power_rabi_ge import AmplitudeRabiProgram

# ===================================================================
# 3. User/Local Libraries
# ===================================================================
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml
from ..tools.fitting import decaysin, fitdecaysin, fix_phase
from ..tools.module_fitzcu import amprabi_analyze
from ..plotter.liveplot import liveplotfun
from ..plotter.plot_utils import plot_final


##################
# Define Program #
##################


class AmplitudeRabiProgram(AveragerProgramV2):
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

        self.add_loop("gainloop", cfg["steps"])
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
            sigma=cfg["sigma_ge"],
            length=cfg["sigma_ge"] * 5,
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
                name="qb_pulse_ef",
                style="arb",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["qb_gain_ef"],
            )
        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch_ef,
                name="qb_pulse_ef",
                style="flat_top",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["qb_gain_ef"],
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
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_pulse_ef", t=0)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Amp_Rabi_ef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            prog = AmplitudeRabiProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)

    def plot(self):
        pi_gain, pi2_gain = amprabi_analyze(self.gains, self.iqdata)
        return pi_gain, pi2_gain

    def liveplot(self, py_avg):
        prog = AmplitudeRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.gains = prog.get_pulse_param("qb_pulse_ef", "gain", as_array=True)

        iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self.gains,
            y_axis_vals=None,
            x_label="Dac Gain (a.u)",
            y_label="ADC Units",
            title_prefix="Qubit Power Rabi ef",
            yoko_inst_addr=None,
            show_final_plot=False,
        )

        self.iqdata = iqdata

        if self.iqdata is None:
            print("No data was acquired.")
            return None
        if interrupted:
            print(
                f"Experiment interrupted at {avg_count} averages. "
                "Fit is based on partial data."
            )
        ### Final plot ###
        fit_params, error, fig, ax = plot_final(
            self.gains,
            self.iqdata,
            "Dac Gain(a.u)",
            fitdecaysin,
            decaysin,
            return_ax=True,
        )
        fig.suptitle("Power Rabi ef")
        fig.tight_layout()
        pi_gain, pi2_gain = fix_phase(fit_params)
        ax.axvline(pi_gain, color="red", linestyle="--", label=r"$\pi$ Gain")
        ax.axvline(pi2_gain, color="red", linestyle="--", label=r"$\pi/2$ Gain")

        return round(pi_gain, 6), round(pi2_gain, 6)

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s011_power_rabi_ef" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        dict_val = config_to_yaml(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="Rabi",
        )

        print(f"Data save to {file_path}")
