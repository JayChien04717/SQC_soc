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

        if cfg.get("ps_resolution", False):
            t = cfg.get("ps_time", 0.0)
            ch = cfg["flux_ch"]
            gencfg = self.soccfg["gens"][ch]
            samps_per_clk = gencfg["samps_per_clk"]
            fs = gencfg["f_fabric"] * samps_per_clk
            min_samps = 3 * samps_per_clk

            n_samples = t * fs
            n_int = int(np.ceil(n_samples))

            # The length of an 'arb' envelope MUST be an exact multiple of samps_per_clk.
            # We first enforce the 3-cycle hardware minimum.
            min_req_samples = max(n_int, min_samps)
            # Then we pad it to the next multiple of samps_per_clk.
            padded_len = int(np.ceil(min_req_samples / samps_per_clk)) * samps_per_clk

            idata = np.zeros(padded_len)
            n_floor = int(np.floor(n_samples))  # number of fully-on samples
            frac = n_samples - n_floor  # fractional part [0, 1)
            if n_floor > 0:
                idata[:n_floor] = 1  # full-amplitude samples
            if frac > 0 and n_floor < padded_len:
                idata[n_floor] = frac  # last partial sample

            maxv = self.get_maxv(cfg["flux_ch"])
            idata_int = (idata * maxv).astype(np.int16)
            self.add_envelope(ch, "flux_step", idata=idata_int)

            self.add_pulse(
                ch=cfg["flux_ch"],
                name="flux_pulse",
                style="arb",
                envelope="flux_step",
                freq=0,
                phase=0,
                gain=cfg["flux_gain"],
            )

        else:
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
        prog = expt._create_program()
        iq_list = prog.acquire(soc, rounds=py_avg, progress=True)

        # QICK V2 AveragerProgramV2 returns iq_list with structure where
        # iq_list[0][0] has shape (steps, triggers_per_step, 2) since there are 2 measures per step.
        # Transpose to (2, steps, 2) to get trigger -> step -> I/Q
        result = np.transpose(iq_list[0][0], (1, 0, 2))

        x_vals = expt._extract_sweep_axis(prog)  # 時間軸 (steps,)

        X = result[0, :, 0]
        Y = result[1, :, 0]

        # Normalize each quadrature independently (zero-mean, unit-range)
        X = (X - X.mean()) / (X.max() - X.min() + 1e-12)
        Y = (Y - Y.mean()) / (Y.max() - Y.min() + 1e-12)

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

    @classmethod
    def run_ps(cls, soc, soccfg, cfg, py_avg, times_us, smooth_window=9):
        """
        Run Cryoscope with picosecond (sub-cycle) resolution.
        This uses a Python loop to recompile the program for each time point,
        ensuring the arbitrary envelope and delay are updated accurately.

        smooth_window : int (odd)
            Savitzky-Golay window length applied to phase before differentiation.
            Larger = smoother Δf but lower time resolution. Set to 0 to disable.
        """
        cfg["ps_resolution"] = True

        results_X = []
        results_Y = []

        from tqdm import tqdm

        for t in tqdm(times_us, desc="Cryoscope PS Sweep"):
            # Update time config and force 1 step per hardware execution
            cfg["ps_time"] = t
            cfg["steps"] = 1

            # Recompile the experiment for the new envelope length
            expt = cls(soc, soccfg, cfg)
            prog = expt._create_program()

            # Run acquisition directly on the program object
            # Disable the inner progress bar to prevent screen clutter
            res = prog.acquire(soc, rounds=py_avg, progress=False)

            # res[0] shape: (steps, reads_per_shot, 2) = (1, 2, 2)
            # axis 0 = steps loop, axis 1 = trigger index (0=X, 1=Y), axis 2 = I/Q
            results_X.append(res[0][0, 0, 0])  # step=0, trigger=0 (0° pulse), I
            results_Y.append(res[0][1, 0, 0])  # step=0, trigger=1 (90° pulse), I

        # 3. Process results
        X = np.array(results_X)
        Y = np.array(results_Y)
        x_vals = np.array(times_us)

        # Normalize each quadrature independently (zero-mean, unit-range)
        X = (X - X.mean()) / (X.max() - X.min() + 1e-12)
        Y = (Y - Y.mean()) / (Y.max() - Y.min() + 1e-12)

        # Calculation logic similar to run_xy
        S = X + 1j * Y
        S = S - np.mean(S)
        phase = np.unwrap(np.angle(S))
        phase = phase - phase[0]
        dt = x_vals[1] - x_vals[0]

        # Smooth phase before differentiation to suppress noise amplified by small dt
        from scipy.signal import savgol_filter

        n_pts = len(phase)
        win = min(
            smooth_window if smooth_window % 2 == 1 else smooth_window + 1,
            n_pts if n_pts % 2 == 1 else n_pts - 1,
        )
        phase_smooth = (
            savgol_filter(phase, window_length=win, polyorder=3) if win >= 5 else phase
        )

        freq = np.diff(phase_smooth) / (2 * np.pi * dt)

        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0].plot(x_vals, X, label="X (0°)", marker="o")
        axs[0].plot(x_vals, Y, label="Y (90°)", marker="s")
        axs[0].set_ylabel("DAC (a.u.)")
        axs[0].legend()

        axs[1].plot(x_vals, phase)
        axs[1].set_ylabel("Accumulated Phase (rad)")

        axs[2].plot(x_vals[1:], freq, color="red")
        axs[2].set_ylabel("Instantaneous Δf (MHz)")
        axs[2].set_xlabel("Time (us)")

        plt.tight_layout()
        plt.show()
        # Return results
        return {"freq": freq, "phase": phase, "time": x_vals, "X": X, "Y": Y}

    def analyze(X, Y, x_vals, smooth_window=9):
        S = X + 1j * Y
        S = S - np.mean(S)
        phase = np.unwrap(np.angle(S))
        phase = phase - phase[0]
        dt = x_vals[1] - x_vals[0]

        # Smooth phase before differentiation to suppress noise amplified by small dt
        from scipy.signal import savgol_filter

        n_pts = len(phase)
        win = min(
            smooth_window if smooth_window % 2 == 1 else smooth_window + 1,
            n_pts if n_pts % 2 == 1 else n_pts - 1,
        )
        phase_smooth = (
            savgol_filter(phase, window_length=win, polyorder=s) if win >= 5 else phase
        )

        freq = np.diff(phase_smooth) / (2 * np.pi * dt)

        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0].plot(x_vals, X, label="X (0°)", marker="o")
        axs[0].plot(x_vals, Y, label="Y (90°)", marker="s")
        axs[0].set_ylabel("DAC (a.u.)")
        axs[0].legend()

        axs[1].plot(x_vals, phase)
        axs[1].set_ylabel("Accumulated Phase (rad)")

        axs[2].plot(x_vals[1:], freq, color="red")
        axs[2].set_ylabel("Instantaneous Δf (MHz)")
        axs[2].set_xlabel("Time (us)")

        plt.tight_layout()
        plt.show()
        # Return results
        return {"freq": freq, "phase": phase, "time": x_vals[1:], "X": X, "Y": Y}
