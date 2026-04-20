"""
s000 — Single Shot Readout (g/e/f)
=====================================
Multi-readout per body: ground → pi → excited → (optional ef pi → f).
Does not use BaseExperiment because of unique multi-trigger data flow.
"""
import numpy as np

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from .singleshot_utils import plot_hist, general_hist, hist


# ── Program ──

class SingleShotProgram_gef(BaseProgram):
    """QICK program for g/e/f single-shot readout with multi-trigger body."""

    def _initialize(self, cfg):
        """Set up resonator, ge/ef generators, and pi-pulse definitions."""
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, 'ge')
        self.setup_qubit_gen(cfg, 'ef')
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, 'ge', name="qb_ge_pulse", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, 'ef', name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        """Execute one shot cycle: ground, excited, and optional f-state readout."""
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

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Experiment configuration.  Requires
            ``BaseExperiment.setup(soc, soccfg, data_path)`` to have been
            called before instantiation.
        """
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False):
        """
        Acquire single-shot IQ data for ground, excited, and optionally f state.

        Parameters
        ----------
        SHOTS : int
            Number of single shots per state.
        shot_f : bool, optional
            If True, also acquire f-state shots (requires ef pi-pulse).

        Returns
        -------
        data : dict
            Keys ``'Ig'``, ``'Qg'``, ``'Ie'``, ``'Qe'``, and optionally
            ``'If'``, ``'Qf'``.
        """
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f

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

    def plot(self, fid_avg=False, verbose=True):
        """
        Plot the IQ histogram and return fidelity metrics.

        Parameters
        ----------
        fid_avg : bool, optional
            API-compatibility flag passed to ``hist()``.
        verbose : bool, optional
            Print numerical results.

        Returns
        -------
        list
            ``[fids, thresholds, angle_deg, conf_matrix_pct]``.
        """
        return hist(
            self.data, plot=True, verbose=verbose, fid_avg=fid_avg,
        )

    def saveLabber(self, qb_idx, yoko_value=None):
        """
        Save IQ shot data to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the filename.
        yoko_value : float or None, optional
            Yokogawa current value appended to the filename.
        """
        has_f = "If" in self.data
        expt_name = ("s000_singleshot_gef" if has_f else "s000_singleshot_ge") + f"_{qb_idx}"
        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        print("Current data file: " + file_path)

        dict_val = config_to_yaml(self.cfg)
        shotdata = np.array([
            self.data["Ig"] + 1j * self.data["Qg"],
            self.data["Ie"] + 1j * self.data["Qe"],
        ] + ([self.data["If"] + 1j * self.data["Qf"]] if has_f else []))
        states = [0, 1, 2] if has_f else [0, 1]

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "# shot", "unit": "#", "values": np.arange(self.cfg["shots"])},
            y_info={"name": "State", "unit": "", "values": states},
            z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
            comment=f"{dict_val}", tag="SingleShot",
        )
