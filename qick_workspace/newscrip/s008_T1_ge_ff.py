"""
s008 — T1 Fast Flux (ge)
==========================
T1 decay: pi pulse followed by variable wait with flux, then readout.
2D sweep: wait time (x-axis) vs flux gain (y-axis).
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_exp_decay


class T1FFProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")

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

        self.add_loop("fluxloop", cfg["steps_flux"])
        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay_auto(0.05, tag="wait")
        self.measure(cfg)


class T1FastFlux(BaseExperiment):
    EXPT_NAME = "s008_T1_ff_ge"
    TAG = "T1"
    X_LABEL = "Wait Time (us)"
    Y_LABEL = "Flux Gain (DAC)"
    TITLE_PREFIX = "Qubit T1 Fast Flux ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time", "flux_gain"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0
    Y_SAVE_NAME = "Flux"
    Y_SAVE_UNIT = "DAC"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return T1FFProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_time_param("wait", "t", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return prog.get_pulse_param("flux_pulse", "gain", as_array=True)

    def _simulate(self, x_pts, y_pts=None):
        if y_pts is not None:
            data = np.zeros((len(y_pts), len(x_pts)), dtype=complex)
            for i, flux in enumerate(y_pts):
                flux_norm = flux / (np.max(np.abs(y_pts)) + 1e-9)
                tau = 20.0 / (1.0 + 3.0 * flux_norm**2)
                data[i] = mock_exp_decay(x_pts, amp=0.5, tau=tau, offset=0.0)
            return data
        else:
            return mock_exp_decay(x_pts, amp=0.5, tau=20.0, offset=0.0)

    # def _post_fit(self, x_vals, y_vals=None):
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     im = ax.pcolormesh(
    #         x_vals,
    #         y_vals,
    #         np.abs(self.iqdata),
    #         cmap="RdBu_r",
    #         shading="auto",
    #     )
    #     plt.colorbar(im, ax=ax, label="|IQ|")
    #     ax.set_xlabel(self.X_LABEL)
    #     ax.set_ylabel(self.Y_LABEL)
    #     ax.set_title(self.TITLE_PREFIX)
    #     fig.tight_layout()

    #     return None

    def _post_fit(self, x_vals, y_vals=None):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- 左：pcolormesh ---
        ax = axes[0]
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            np.abs(self.iqdata),
            cmap="RdBu_r",
            shading="auto",
        )
        plt.colorbar(im, ax=ax, label="|IQ|")
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        ax.set_title(self.TITLE_PREFIX)

        # --- 右：selected row cuts ---
        ax2 = axes[1]
        mid = len(self.iqdata) // 2
        for idx, label in [
            (0, f"flux={y_vals[0]:.0f}"),
            (mid, f"flux={y_vals[mid]:.0f}"),
            (-1, f"flux={y_vals[-1]:.0f}"),
        ]:
            ax2.plot(x_vals, np.abs(self.iqdata[idx]), label=label)
        ax2.set_xlabel(self.X_LABEL)
        ax2.set_ylabel("|IQ|")
        ax2.set_title("Row Cuts")
        ax2.legend()

        fig.tight_layout()
        return None
