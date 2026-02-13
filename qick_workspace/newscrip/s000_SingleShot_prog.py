"""
s000 — Single Shot Readout (g/e/f)
=====================================
Multi-readout per body: ground → pi → excited → (optional ef pi → f).
Does not use BaseExperiment because of unique multi-trigger data flow.
Histogram utilities (plot_hist, general_hist, hist) are preserved as-is.
"""
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy.integrate import quad

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fit_doublegauss, double_gaussian, fit_gauss, gaussian


# ── Histogram Utilities ──

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle = ["o", "*", "s", "^"]


def plot_hist(data, bins, ax=None, xlims=None, color=None, linestyle=None,
              label=None, alpha=None, normalize=True):
    if color is None:
        color = next(cycle(default_colors))
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        hist_sum = hist_data.sum()
        if hist_sum > 0:
            hist_data = hist_data / hist_sum
    for i in range(len(hist_data)):
        ax.plot([bin_edges[i], bin_edges[i+1]], [hist_data[i], hist_data[i]],
                color=color, linestyle=linestyle, label=label if i == 0 else None,
                alpha=alpha, linewidth=0.9)
        if i < len(hist_data) - 1:
            ax.plot([bin_edges[i+1], bin_edges[i+1]], [hist_data[i], hist_data[i+1]],
                    color=color, linestyle=linestyle, alpha=alpha, linewidth=0.9)
    ax.relim()
    ax.set_ylim((0, None))
    return hist_data, bin_edges


# Re-export general_hist and hist from original module to keep compatibility
from ..scrip.s000_SingleShot_prog import general_hist, hist


# ── Program ──

class SingleShotProgram_gef(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_ge_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        # Ground state readout
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        # Excited state readout
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        # Optional f state readout
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


# ── Experiment ──

class SingleShot_gef:
    """Single-shot readout for g/e/f state discrimination."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False, simulate=False):
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f

        if simulate:
            return self._simulate(SHOTS, shot_f)

        prog = SingleShotProgram_gef(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        iq_list = prog.acquire(self.soc, rounds=1, progress=True)

        Ig = iq_list[0][0, :, 0]
        Qg = iq_list[0][0, :, 1]
        Ie = iq_list[0][1, :, 0]
        Qe = iq_list[0][1, :, 1]

        if shot_f:
            If = iq_list[0][2, :, 0]
            Qf = iq_list[0][2, :, 1]
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe, "If": If, "Qf": Qf}
        else:
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe}
        return self.data

    def _simulate(self, n_shots, shot_f=False):
        """Generate mock single-shot Gaussian IQ blobs."""
        from .mock_signals import mock_singleshot
        result = mock_singleshot(n_shots, separation=3.0, sigma=1.0, include_f=shot_f)
        self.data = {"Ig": result["ig"], "Qg": result["qg"],
                     "Ie": result["ie"], "Qe": result["qe"]}
        if shot_f:
            self.data["If"] = result["if"]
            self.data["Qf"] = result["qf"]
        return self.data

    def plot(self, fid_avg=False, fit=False, normalize=False, verbose=True, overlap=False):
        return hist(
            self.data, amplitude_mode=False, ps_threshold=None, theta=None,
            plot=True, verbose=verbose, gauss_overlap=overlap, fid_avg=fid_avg,
            fit=fit, fitparams=[None, None, 5, None, None, 5], normalize=normalize,
        )

    def saveLabber(self, qb_idx, yoko_value=None):
        has_f = "If" in self.data
        expt_name = ("s000_singleshot_gef" if has_f else "s000_singleshot_ge") + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        print("Current data file: " + file_path)

        dict_val = config_to_yaml(self.cfg)
        keys = ["Ig", "Qg", "Ie", "Qe"] + (["If", "Qf"] if has_f else [])
        shotdata = np.array([self.data[k] for k in keys])
        states = [0, 1, 2] if has_f else [0, 1]

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "# shot", "unit": "#", "values": np.arange(self.cfg["shots"])},
            y_info={"name": "State", "unit": "", "values": states},
            z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
            comment=f"{dict_val}", tag="SingleShot",
        )
