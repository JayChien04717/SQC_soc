# ===================================================================
# 1. Standard & Third-Party Scientific Libraries
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from IPython.display import display, clear_output, update_display

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
from ..tools.system_tool import hdf5_generator, get_next_filename_labber


##################
# Define Program #
##################


class LoopbackProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        if self.soccfg["gens"][res_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg["res_freq_ge"])
        else:
            self.declare_gen(ch=res_ch, nqz=2)

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
            name="loopback_pulse",
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
        self.pulse(ch=cfg["res_ch"], name="loopback_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0)


class TOF:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config
        self.iqdata = None
        self.iq_list = None
        self.t = None

    def run(self, py_avg=1, liveplot=True):
        if liveplot:
            return self.liveplot(py_avg=py_avg)
        else:
            prog = LoopbackProgram(
                self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
            )
            self.iq_list = prog.acquire_decimated(self.soc, rounds=py_avg)
            self.t = prog.get_time_axis(ro_index=0)
            self.iqdata = self.iq_list[0].dot([1, 1j])
            return self.iqdata

    def plot(self, thressold=1.5):
        if self.iq_list is not None:
            plt.plot(self.t, self.iq_list[0].T[0])
            plt.plot(self.t, self.iq_list[0].T[1])
            plt.plot(self.t, np.abs((self.iq_list[0]).dot([1, 1j])))
            plt.xlabel("Time (us)")
            plt.ylabel("a.u")

            mean = np.mean(np.abs(self.iq_list[0].dot([1, 1j])))
            plt.axvline(
                self.t[
                    np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > thressold * mean)
                ],
                c="r",
                ls="--",
            )
            plt.title(
                f"Time of Flight Experiment, trig = {round(self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > 1.5 * mean)], 2)} us"
            )

        else:
            print("No data to plot. Run the experiment first.")

    def liveplot(self, py_avg=1, thressold=1.5):
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        self.t = prog.get_time_axis(ro_index=0)

        iq_sum = 0

        fig, ax = plt.subplots(figsize=(7, 5))

        nan_data = np.full_like(self.t, np.nan, dtype=float)
        (line,) = ax.plot(self.t, nan_data, alpha=0.8)

        ax.set_xlabel("Time ($\mu$s)")
        ax.set_ylabel("ADC Units (Abs)")
        title = ax.set_title("Time of Flight (TOF) | Average: 0 / 0")

        plot_display_id = f"live-plot-tof-{np.random.randint(1e9)}"
        display(fig, display_id=plot_display_id)

        interrupted = False
        i = 0

        # --- 設定 X 軸範圍 (只需要設定一次) ---
        t_min = np.min(self.t)
        t_max = np.max(self.t)
        ax.set_xlim(t_min, t_max)
        # ------------------------------------

        try:
            for i in tqdm(range(py_avg), desc="Software Average Count"):
                self.iq_list = prog.acquire_decimated(
                    self.soc, rounds=1, progress=False
                )

                current_iq_data = self.iq_list[0].dot([1, 1j])

                if i == 0:
                    iq_sum = current_iq_data
                else:
                    iq_sum += current_iq_data

                self.iqdata = iq_sum / (i + 1)

                plot_data_abs = np.abs(self.iqdata)
                line.set_ydata(plot_data_abs)

                current_min, current_max = np.min(plot_data_abs), np.max(plot_data_abs)
                range_span = current_max - current_min
                if range_span == 0:
                    range_span = 1e-9
                ax.set_ylim(
                    current_min - 0.1 * range_span, current_max + 0.1 * range_span
                )
                # ----------------------------------------------------

                title.set_text(f"Time of Flight (TOF) | Average: {i + 1} / {py_avg}")

                update_display(fig, display_id=plot_display_id)

        except KeyboardInterrupt:
            interrupted = True
            print(f"Interrupted by user at average count: {i + 1}")

        clear_output(wait=True)

        if self.iqdata is not None:
            final_fig, final_ax = plt.subplots(figsize=(7, 5))
            final_ax.plot(
                self.t, np.abs(self.iqdata), "o-", markersize=2, label="Averaged Data"
            )

            mean = np.mean(np.abs(self.iqdata))
            cross_index = np.argmax(np.abs(self.iqdata) > thressold * mean)
            trig_time = self.t[cross_index]

            final_ax.axvline(
                trig_time, c="r", ls="--", label=f"TOF: {trig_time:.2f} $\mu$s"
            )

            title_text = f"Time of Flight Experiment, trig = {trig_time:.2f} $\mu$s"
            if interrupted:
                title_text += " (Interrupted)"

            final_ax.set_title(title_text)
            final_ax.set_xlabel("Time ($\mu$s)")
            final_ax.set_ylabel("ADC unit")
            final_ax.set_xlim(t_min, t_max)
            final_ax.legend()
            display(final_fig)
            plt.close(final_fig)

        plt.close(fig)

        return self.iqdata, not interrupted, i + 1

    def saveLabber(self, qb_idx):
        expt_name = "s001_tof" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.t * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "ADC unit",
                "values": self.iqdata,
            },
            comment=(),
            tag="TOF",
        )
        print(f"Data save to {file_path}")
