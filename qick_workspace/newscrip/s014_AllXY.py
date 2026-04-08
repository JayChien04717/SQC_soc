"""
s014 — AllXY
=============
Gate error diagnostic: iterates over 21 gate-pair sequences.
Uses setup_standard_gates() for automatic pulse declaration.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .base_program import BaseProgram, resolve_gate
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml


# AllXY gate sequence (21 pairs)
ALLXY_SEQUENCE = [
    ("I", "I"),
    ("X", "X"),
    ("Y", "Y"),
    ("X", "Y"),
    ("Y", "X"),
    ("X/2", "I"),
    ("Y/2", "I"),
    ("X/2", "Y/2"),
    ("Y/2", "X/2"),
    ("X/2", "Y"),
    ("Y/2", "X"),
    ("X", "Y/2"),
    ("Y", "X/2"),
    ("X/2", "X"),
    ("X", "X/2"),
    ("Y/2", "Y"),
    ("Y", "Y/2"),
    ("X", "I"),
    ("Y", "I"),
    ("X/2", "X/2"),
    ("Y/2", "Y/2"),
]


# ── Program ──


class AllXYProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_standard_gates(cfg, prefix="ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # Apply gate pair from cfg (resolve shorthand → actual pulse name)
        gate1, gate2 = cfg["allxy_gates"]
        g1 = resolve_gate(gate1)
        g2 = resolve_gate(gate2)
        if g1 != "I":
            self.pulse(ch=cfg["qb_ch"], name=g1, t=0)
        self.delay_auto(0.01)
        if g2 != "I":
            self.pulse(ch=cfg["qb_ch"], name=g2, t=0)

        self.delay_auto(0.05)
        self.measure(cfg)


# ── Experiment ──


class AllXY:
    """AllXY gate error diagnostic (21 gate-pair sequences)."""

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, iq_process="abs"):
        """
        Parameters
        ----------
        iq_process : "abs" | "real"
            How to convert complex IQ → scalar for display/analysis.
            Use "real" after readout optimization.
        """
        self._iq_process = iq_process
        allxy_lst = []
        for gate in tqdm(ALLXY_SEQUENCE, desc="AllXY"):
            self.cfg["allxy_gates"] = gate
            prog = AllXYProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
            allxy_lst.append(iq_list[0][0].dot([1, 1j]))
        self.allxy_lst = np.array(allxy_lst)

    def plot(self):
        _proc = np.real if getattr(self, "_iq_process", "abs") == "real" else np.abs
        amp = _proc(self.allxy_lst)
        if amp[0] < amp[-1]:
            ref = (
                [np.min(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.max(amp)] * 4
            )
        else:
            ref = (
                [np.max(amp)] * 5
                + [(np.max(amp) + np.min(amp)) / 2] * 12
                + [np.min(amp)] * 4
            )
        if len(ref) != len(amp):
            ref = (
                ref[: len(amp)]
                if len(ref) > len(amp)
                else ref + [ref[-1]] * (len(amp) - len(ref))
            )

        plt.figure(figsize=(10, 5))
        plt.plot(amp, "bo", label="Data")
        plt.plot(ref, "r-", label="Reference Line")
        plt.xticks(np.arange(len(ALLXY_SEQUENCE)), ALLXY_SEQUENCE, rotation=45)
        plt.ylabel(r"$F_{\left|1\right\rangle}$")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = f"s014_AllXY_ge_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        dict_val = config_to_yaml(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={
                "name": "Sequence",
                "unit": "None",
                "values": np.arange(len(ALLXY_SEQUENCE)),
            },
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.allxy_lst},
            comment=f"{dict_val}",
            tag="ALLXY",
        )
        print(f"Data save to {file_path}")
