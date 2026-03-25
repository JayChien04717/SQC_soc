from .base_program import BaseProgram
from .base_experiment import BaseExperiment
import numpy as np
import matplotlib.pyplot as plt


class CryoscopeProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")

        # Ramsey pulses
        self.setup_qb_pulse(cfg, "ge", name="pi2_1", gain_key="pi2_gain_ge")
        self.setup_qb_pulse(cfg, "ge", name="pi2_2", gain_key="pi2_gain_ge", phase=0)
        self.setup_qb_pulse(cfg, "ge", name="pi2_3", gain_key="pi2_gain_ge", phase=90)

        # Flux pulse
        self.declare_gen(ch=cfg["flux_ch"], nqz=1, mixer_freq=0)
        self.add_pulse(
            ch=cfg["flux_ch"],
            name="flux_pulse",
            style="const",
            length=cfg["flux_length"],
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )

        self.add_loop("tloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # π/2
        self.pulse(ch=cfg["qb_ch"], name="pi2_1", t=0)
        self.delay_auto(0.01)
        # flux pulse
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay_auto(0.01)
        # second π/2
        self.pulse(ch=cfg["qb_ch"], name="pi2_2", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)

        self.delay_auto(30)

        self.pulse(ch=cfg["qb_ch"], name="pi2_1", t=0)
        self.delay_auto(0.01)
        # flux pulse
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay_auto(0.01)
        # second π/2
        self.pulse(ch=cfg["qb_ch"], name="pi2_3", t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


class Cryoscope(BaseExperiment):
    EXPT_NAME = "sXXX_cryoscope"
    TAG = "Cryoscope"
    X_LABEL = "Flux Pulse Length (us)"
    TITLE_PREFIX = "Cryoscope"
    SWEEP_KEYS_TO_REMOVE = ["flux_length"]

    X_SAVE_NAME = "Time"
    X_SAVE_UNIT = "us"

    def _create_program(self):
        return CryoscopeProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("flux_pulse", "length", as_array=True)

    # ---- projection hook（之後可放 readout rotation）----
    def _project_iq(self, iq):
        return np.real(iq)

    def _post_fit(self, x_vals):
        iq = self.iqdata

        signal = self._project_iq(iq)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals, signal, "o-")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title(f"Cryoscope raw ({'X' if self.cfg['x'] else 'Y'})")
        plt.tight_layout()
        plt.show()

        return signal

    @classmethod
    def run_xy(cls, soc, soccfg, cfg, py_avg):
        # 1. 執行實驗 (只跑一次，因為 body 裡有兩個 measure)
        expt = cls(soc, soccfg, cfg)
        # 假設 expt.run 回傳的是你提到的 result[0] 結構 (2, 81, 2)
        result = expt.acquire(soc, rounds=py_avg)

        x_vals = expt._sweep_vals_x  # 時間軸 (81,)

        X = result[0]
        Y = result[1, :, 0]

        S = X + 1j * Y
        S = S - np.mean(S)  # 移除 DC 偏移
        phase = np.unwrap(np.angle(S))
        phase = phase - phase[0]  # 令起點為 0

        # 4. 數值微分還原 Step Response (重要！)
        dt = x_vals[1] - x_vals[0]
        # diff 之後長度變為 80
        freq = np.diff(phase) / (2 * np.pi * dt)

        # 5. 繪圖 (注意 x_vals[1:] 匹配 80 個點)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0].plot(x_vals, X, label="X (0°)")
        axs[0].plot(x_vals, Y, label="Y (90°)")
        axs[0].set_ylabel("Voltage (a.u.)")
        axs[0].legend()

        axs[1].plot(x_vals, phase)
        axs[1].set_ylabel("Accumulated Phase (rad)")

        axs[2].plot(x_vals[1:], freq, color="red")
        axs[2].set_ylabel("Instantaneous Δf (MHz)")
        axs[2].set_xlabel("Time (us)")

        plt.tight_layout()
        plt.show()

        return {"freq": freq, "phase": phase, "time": x_vals[1:]}
