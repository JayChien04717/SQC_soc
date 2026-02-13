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
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml

### AllXY Sequence ###
sequence = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]


class AllXYprogram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        # self.declare_gen(ch=qb_ch, nqz=cfg['nqz_qb'])
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
            name="ramp",
            sigma=cfg["sigma_ge"],
            length=cfg["sigma_ge"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            for name, phase, gain in [
                ("x180", 0, cfg["pi_gain_ge"]),
                ("y180", 9, cfg["pi_gain_ge"]),
                ("x90", 0, cfg["pi2_gain_ge"]),
                ("y90", 90, cfg["pi2_gain_ge"]),
            ]:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="arb",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                )

        elif cfg["pulse_type"] == "flat_top":
            for name, phase, gain in [
                ("x180", 0, cfg["pi_gain_ge"]),
                ("y180", 9, cfg["pi_gain_ge"]),
                ("x90", 0, cfg["pi2_gain_ge"]),
                ("y90", 90, cfg["pi2_gain_ge"]),
            ]:
                self.add_pulse(
                    ch=qb_ch,
                    name=name,
                    style="flat_top",
                    envelope="ramp",
                    freq=cfg["qb_freq_ge"],
                    phase=phase,
                    gain=gain,
                    length=cfg["flat_top_len"],
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
            length=5 * cfg["res_sigma"],
            even_length=True,
        )

        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="flat_top",
            envelope="cooling1",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="flat_top",
            envelope="cooling2",
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

        ## allxy gates ##
        if cfg["allxy_gates"][0] == "I":
            pass
        else:
            self.pulse(ch=self.cfg["qb_ch"], name=f"{self.cfg['allxy_gates'][0]}", t=0)
        self.delay_auto(0.01)
        if cfg["allxy_gates"][1] == "I":
            pass
        else:
            self.pulse(ch=self.cfg["qb_ch"], name=f"{self.cfg['allxy_gates'][1]}", t=0)

        ## readout pulse ##
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class AllXY:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            print("This program is not supported in liveplot mode")
        else:
            allxy_lst = []
            for gate in tqdm(sequence):
                self.cfg["allxy_gates"] = gate
                allxy = AllXYprogram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = allxy.acquire(self.soc, rounds=py_avg, progress=False)

                allxy_lst.append(iq_list[0][0].dot([1, 1j]))
            self.allxy_lst = np.array(allxy_lst)

    def plot(self):
        amp = np.abs(self.allxy_lst)

        if amp[0] < amp[-1]:
            ref = (
                [np.min(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.max(amp)] * 4
            )
        else:
            ref = (
                [np.max(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.min(amp)] * 4
            )

        if len(ref) != len(amp):
            if len(ref) > len(amp):
                ref = ref[: len(amp)]
            else:
                ref = ref + [ref[-1]] * (len(amp) - len(ref))

        plt.figure(figsize=(10, 5))
        plt.plot(amp, "bo", label="Data")
        plt.plot(ref, "r-", label="Reference Line")
        plt.xticks(np.arange(len(sequence)), sequence, rotation=45)
        plt.ylabel(r"$F_{\left|1\right\rangle}$")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s014_AllXY_ge" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        dict_val = config_to_yaml(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "Sequence",
                "unit": "None",
                "values": np.arange(len(sequence)),
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.allxy_lst},
            comment=(f"{dict_val}"),
            tag="ALLXY",
        )
        print(f"Data save to {file_path}")
