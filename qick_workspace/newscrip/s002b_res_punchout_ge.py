from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..plotter.liveplot import liveplotfun
import numpy as np
import matplotlib.pyplot as plt


# ── Program ──


class PunchoutProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.measure(cfg)


# ── Experiment ──


class Punchout(BaseExperiment):
    """Resonator punchout: 2D sweep over gain and frequency."""

    EXPT_NAME = "s002b_res_ge_punchout"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "DAC Gains"
    TITLE_PREFIX = "Resonator Punchout"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "DAC Gains"
    Y_SAVE_UNIT = "a.u."
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        return PunchoutProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def _mock_sweep_axis(self, **kwargs):
        # Determine frequency (x) axis
        res_freq = self.cfg.get("res_freq_ge")
        if hasattr(res_freq, "start"):
            return np.linspace(
                res_freq.start, res_freq.stop, self.cfg.get("f_steps", 51)
            )
        # Default fallback
        return np.linspace(6000, 6100, self.cfg.get("f_steps", 51))

    def _mock_sweep_axis_y(self, **kwargs):
        # Determine gain (y) axis
        res_gain = self.cfg.get("res_gain_ge")
        if hasattr(res_gain, "start"):
            return np.linspace(
                res_gain.start, res_gain.stop, self.cfg.get("g_steps", 11)
            )
        # Default fallback
        return np.linspace(0, 1, self.cfg.get("g_steps", 11))

    def _simulate(self, x_pts, y_pts):
        from .mock_signals import mock_lorentzian_2d

        return mock_lorentzian_2d(x_pts, y_pts, f0=(x_pts[0] + x_pts[-1]) / 2)
