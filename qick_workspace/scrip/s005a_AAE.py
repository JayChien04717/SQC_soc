# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from scipy.optimize import curve_fit

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
from ..tools.fitting import fit_probg_Xhalf, fit_probg_X, probg_Xhalf, probg_X
from ..tools.module_fitzcu import T2fring_analyze
from ..tools.yamltool import yml_comment
from ..plotter.liveplot import liveplotfun
from ..plotter.plot_utils import plot_final


##################
# Define Program #
##################


class AmplifiedAmplitudeError(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
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
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse_pi2",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi2_gain_ge"],
            )

            # pi pulse
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse_pi",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
            )
        elif cfg["pulse_type"] == "flat_top":
            if cfg["qb_flat_top_length_ge"] is None:
                raise ValueError("Please set qb_flat_top_length_ge in config")
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse_pi2",
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi2_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
            )

            # pi pulse
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse_pi",
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
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

        self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse_pi2", t=0)
        self.delay_auto(0.02)
        if cfg["AAE"] == "pi":
            for i in range(cfg["steps"]):
                self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse_pi", t=0)
                self.delay_auto(0.02)
        elif cfg["AAE"] == "pi2":
            for i in range(cfg["steps"]):
                self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse_pi2", t=0)
                self.delay_auto(0.02)
                self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse_pi2", t=0)
                self.delay_auto(0.02)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class AAE:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False, iteration=None):
        if liveplot:
            if iteration is None:
                raise ValueError("Please provide iteration values for live plot")
            else:
                self.liveplot(py_avg, iteration)
        else:
            prog = AmplifiedAmplitudeError(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.delay_times = prog.get_time_param(
                "wait1", "t", as_array=True
            ) + prog.get_time_param("wait2", "t", as_array=True)

    def plot(self):
        pass

    def liveplot(self, py_avg, iteration_count):
        iter_axis = np.arange(0, iteration_count, 1)

        def create_aae_prog(n):
            self.cfg["steps"] = int(n)
            return AmplifiedAmplitudeError(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

        self.iter = iter_axis

        self.iqdata, interrupted, done_avg = liveplotfun(
            soc=self.soc,
            py_avg=py_avg,
            scan_x_axis=self.iter,
            get_prog_callback=create_aae_prog,
            x_label="N",
            title_prefix="Amplified Amplitude Error",
            show_final_plot=True,
        )

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s007_SpinEcho_ge" + f"_{qb_idx}"
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
            comment=(f"\n{dict_val}"),
            tag="Spin Echo",
        )

        print(f"Data save to {file_path}")

    def analyze_and_plot(self):
        # 取得數據
        n_pts = self.iter
        z_pts = np.real(self.iqdata)

        # 根據實驗類型選擇擬合函數 (X/2 閘或 X 閘)
        # 假設 cfg["AAE"] 可以是 'pi2' 或 'pi'
        aae_mode = self.cfg.get("AAE", "pi2")

        try:
            if aae_mode == "pi2":
                # 使用你提供的 X/2 擬合邏輯
                pOpt, pCov = fit_probg_Xhalf(n_pts, z_pts)
                a_fit, delta_fit = pOpt
                fit_func = probg_Xhalf
            else:
                # 使用你提供的 X 擬合邏輯
                pOpt, pCov = fit_probg_X(n_pts, z_pts)
                a_fit, delta_fit = pOpt
                fit_func = probg_X

            # 繪圖
            plt.figure(figsize=(8, 5))
            plt.scatter(n_pts, z_pts, color="blue", label="Raw Data", zorder=3)

            # 繪製擬合曲線
            n_fine = np.linspace(min(n_pts), max(n_pts), 1000)
            # 注意：probg_Xhalf 內部有 (-1)**n，繪圖時建議只畫整數點或清楚標示
            z_fit = [fit_func(n, *pOpt) for n in n_fine]
            plt.plot(
                n_fine,
                z_fit,
                color="orange",
                label=f"Fit ($\delta$={delta_fit:.4f} deg)",
            )

            plt.xlabel("N (Number of repetitions)")
            plt.ylabel("<Z>")
            plt.title(f"Amplified Amplitude Error Fit ({aae_mode})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            print(
                f"[{aae_mode}] 擬合成功！偏移(a): {a_fit:.4f}, 角度誤差(delta): {delta_fit:.6f} deg"
            )
            return pOpt

        except Exception as e:
            print(f"擬合過程出錯: {e}")
            return None
