"""
s005 — Power Rabi Chevron (ge)
================================
2D sweep: qubit drive gain (x-axis) vs flux gain (y-axis).
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from .mock_signals import mock_decaysin


class PowerRabiChevronProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

        self.add_loop("freqloop", cfg["steps_freq"])
        self.add_loop("gainloop", cfg["steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiChevron(BaseExperiment):
    EXPT_NAME = "s005_power_rabi_chevron_ge"
    TAG = "Rabi"
    X_LABEL = "DAC Gain (a.u.)"
    Y_LABEL = "Probe Frequency (MHz)"
    TITLE_PREFIX = "Power Rabi Chevron ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge", "qb_freq_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0
    Y_SAVE_NAME = "Frequency"
    Y_SAVE_UNIT = "MHz"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return PowerRabiChevronProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return prog.get_pulse_param("qb_pulse", "freq", as_array=True)

    def _simulate(self, x_pts, y_pts=None):
        if y_pts is not None:
            data = np.zeros((len(y_pts), len(x_pts)), dtype=complex)
            for i, flux in enumerate(y_pts):
                flux_norm = flux / (np.max(np.abs(y_pts)) + 1e-9)
                freq_shift = 2.0 + 1.5 * flux_norm**2
                decay = 3.0 + abs(flux_norm) * 2.0
                data[i] = mock_decaysin(
                    x_pts, amp=0.5, freq=freq_shift, decay=decay, offset=0.5
                )
            return data
        else:
            return mock_decaysin(x_pts, amp=0.5, freq=2.0, decay=3.0, offset=0.5)

    def _post_fit(self, x_vals, y_vals=None):
        fig, ax = plt.subplots(figsize=(7, 5))
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
        fig.tight_layout()
        return None
