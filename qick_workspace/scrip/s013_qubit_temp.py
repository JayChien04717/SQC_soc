# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import root_scalar

# ----- User Library ----- #
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.module_fitzcu import amprabi_analyze, post_rotate, pipulse_analyze
from ..tools.fitting import decaysin, fitdecaysin
from ..system_tool.yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class AmplitudeRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]
        qubit_ch_ef = cfg["qubit_ch_ef"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])

        if self.soccfg["gens"][qubit_ch_ef]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch_ef,
                nqz=cfg["nqz_qubit_ef"],
                mixer_freq=cfg["qmixer_freq_ef"],
            )
        else:
            self.declare_gen(ch=qubit_ch_ef, nqz=cfg["nqz_qubit_ef"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ef"], gen_ch=res_ch
        )

        self.add_loop("gainloop", cfg["steps"])

        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="const",
            length=cfg["res_length"],
            freq=cfg["res_freq_ef"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ef"],
        )

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pi_pulse",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi_gain_ge"],
        )

        self.add_gauss(
            ch=qubit_ch_ef,
            name="ramp2",
            sigma=cfg["sigma_ef"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch_ef,
            name="qubit_pulse_ef",
            style="arb",
            envelope="ramp2",
            freq=cfg["qubit_freq_ef"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_gain_ef"],
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
        # use Rabi to chek temperature
        if cfg["temp_ref"] is False:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
        elif cfg["temp_ref"] is True:
            pass

        self.pulse(ch=self.cfg["qubit_ch_ef"], name="qubit_pulse_ef", t=0)
        self.delay_auto(0.01)
        if cfg["ge_ref"] is True:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pi_pulse", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.01)

        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Qubit_temperature:
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
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.gains = prog.get_pulse_param("qubit_pulse_ef", "gain", as_array=True)

    def plot(self):
        pi_gain, pi2_gain = amprabi_analyze(self.gains, self.iqdata)
        return pi_gain, pi2_gain

    def liveplot(self, py_avg):
        iq = 0
        iq_ref = 0
        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))

        self.cfg["temp_ref"] = False
        prog = AmplitudeRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.cfg["temp_ref"] = True
        prog_ref = AmplitudeRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

        self.gains = prog.get_pulse_param("qubit_pulse_ef", "gain", as_array=True)

        for avg in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            self.iq_list_ref = prog_ref.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq_data_ref = self.iq_list_ref[0][0].dot([1, 1j])
            iq = iq_data if avg == 0 else iq + iq_data
            iq_ref = iq_data_ref if avg == 0 else iq_ref + iq_data_ref
            self.iqdata = iq / (avg + 1)
            self.iqdata_ref = iq_ref / (avg + 1)

            ax.cla()
            ax.plot(self.gains, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.plot(self.gains, np.abs(post_rotate(self.iqdata_ref)), **marker_style)
            ax.set_title(f"average: {avg + 1} / {py_avg}")
            ax.set_xlabel("Gain (Dac unit)")
            ax.set_ylabel("Signal (ADC unit)")

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)
        ax.set_title("Power Rabi ef")
        ax.plot(self.gains, np.abs(post_rotate(self.iqdata)), **marker_style)
        ax.plot(self.gains, np.abs(post_rotate(self.iqdata_ref)), **marker_style)
        pOpt, _ = fitdecaysin(self.gains, np.abs(post_rotate(self.iqdata)))
        pOpt_ref, _ = fitdecaysin(self.gains, np.abs(post_rotate(self.iqdata_ref)))
        ax.plot(self.gains, decaysin(self.gains, *pOpt), label="witho ref")
        ax.plot(self.gains, decaysin(self.gains, *pOpt_ref), label="with ref")
        ax.legend()
        temp = max(decaysin(self.gains, *pOpt)) - min(decaysin(self.gains, *pOpt))
        temp_ref = max(decaysin(self.gains, *pOpt_ref)) - min(
            decaysin(self.gains, *pOpt_ref)
        )
        popu = temp / (temp + temp_ref)
        # temperture = self.solve_temperature(self.cfg['qubit_freq_ge'], self.cfg['qubit_freq_ef'], popu)
        # print(f'T = {temperture*1e3}mK')
        return temp, temp_ref

    def solve_temperature(self, fge_Hz, fef_Hz, Pe_target):
        """
        根據 fge (Hz), fef (Hz), 和目標 Pe，反推出溫度 T (K)
        """
        # 能量
        h = 6.62607015e-34  # Planck 常數 (J·s)
        kB = 1.380649e-23  # Boltzmann 常數 (J/K)

        E_e = h * fge_Hz * 1e6
        E_f = h * (fge_Hz + fef_Hz) * 1e6

        # 定義 P_e 函數
        def Pe(T):
            exp_ee = np.exp(-E_e / (kB * T))
            exp_ef = np.exp(-E_f / (kB * T))
            Z = 1 + exp_ee + exp_ef
            return exp_ee / Z

        # 定義 root function
        def f_root(T):
            return Pe(T) - Pe_target

        # 解方程，設定初始搜索範圍 1 mK 到 1 K
        sol = root_scalar(f_root, bracket=[0.001, 1], method="brentq")
        if sol.converged:
            return sol.root
        else:
            raise ValueError("無法求解，請檢查輸入值是否合理")

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "s011_power_rabi_ef" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("qubit_gain_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        if save_sim:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
                y_info={"name": "simulate", "unit": "None", "values": np.array([0, 1])},
                z_info={
                    "name": "Signal",
                    "unit": "ADC unit",
                    "values": np.array([self.iqdata, self.sim]),
                },
                comment=(f"{dict_val}"),
                tag="Rabi",
            )
        else:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
                z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
                comment=(f"{dict_val}"),
                tag="Rabi",
            )

        print(f"Data save to {file_path}")


if __name__ == "__main__":
    pass
