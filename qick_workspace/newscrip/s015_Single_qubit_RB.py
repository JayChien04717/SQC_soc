"""
s015 — Single Qubit Randomized Benchmarking (RB)
==================================================
Clifford group generation + random sequence + inverse gate.

Gate-ordering convention
------------------------
A composite gate label like "Y/2,X" means "apply Y/2 first, then X",
i.e. the physical unitary is M = M_X @ M_Y2  (right-to-left matrix product).
Equivalently, when tracking a Pauli-frame column vector |ψ⟩ in the
symplectic representation, we apply gates LEFT-TO-RIGHT:

    |ψ'⟩ = M_Y2 @ M_X @ |ψ⟩

which is just iterating through the sub-gate names in forward order.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .base_program import BaseProgram
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err
from .RB_generator import generate_rb_sequence, generate_irb_sequence, GATE_MATRIX


# ── Program ───────────────────────────────────────────────────────────────────


class RBProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")

        # Primitive gates present in the Clifford group.
        # X and Y are self-inverse (no "-X" or "-Y" needed).
        rb_pulses = [
            ("I", 0, 0),  # zero-gain placeholder
            ("X", 0, cfg["pi_gain_ge"]),
            ("X/2", 0, cfg["pi2_gain_ge"]),
            ("-X/2", 0, -cfg["pi2_gain_ge"]),
            ("Y", 90, cfg["pi_gain_ge"]),
            ("Y/2", 90, cfg["pi2_gain_ge"]),
            ("-Y/2", 90, -cfg["pi2_gain_ge"]),
        ]
        for pulse_name, phase, gain in rb_pulses:
            self.setup_qb_pulse(
                cfg,
                "ge",
                name=pulse_name,
                phase=phase,
                gain_override=gain,
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        for gate in cfg["gate_seq"]:
            if gate == "I":
                self.delay_auto(cfg["sigma_ge"] * 5)
            else:
                self.pulse(ch=cfg["qb_ch"], name=gate, t=0)
                self.delay_auto(0.01)

        self.delay_auto(0.05)
        self.measure(cfg)


# ── Experiment ────────────────────────────────────────────────────────────────

# Valid gates for interleaved RB (must match GATE_MATRIX keys in RB_generator)
_VALID_IRB_GATES = set(GATE_MATRIX.keys()) - {"I"}  # I is not useful to interleave

# Gate name → filename suffix mapping for interleaved RB
_INTERLEAVED_FILE_SUFFIX = {
    "X/2": "halfX",
    "-X/2": "halfXm",
    "X": "X",
    "Y": "Y",
    "Y/2": "halfY",
    "-Y/2": "halfYm",
}


class RandomizedBenchmarking:
    """
    Standard or Interleaved Single-Qubit Randomized Benchmarking.

    Saving convention (saveLabber):
        x-axis : circuit depth values  (len = number of depth points)
        y-axis : sample index          (len = number_sample)
        data   : complex IQ, shape (number_sample, n_depths)

        Filenames:
            Standard RB        →  s015_RB_{qidx}
            Interleaved X/2    →  s015_RB_halfX_{qidx}
            (other gates use _INTERLEAVED_FILE_SUFFIX table)

    Usage::

        rb = RandomizedBenchmarking(soc, soccfg, cfg)
        rb.run(py_avg=10, max_circuit_depth=200, delta_clifford=20,
               number_sample=30, interleaved_gate=None)
        rb.plot("Standard RB")
        rb.saveLabber(qb_idx="Q3", config_all=config_all)
    """

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

        # Results populated by run()
        self.x = None  # 1-D array of depth values
        self.rb_result = None  # list[list[complex]]  shape: [n_depths][number_sample]
        self._number_sample = None
        self._interleaved_gate = None

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
        seed=None,
    ):
        """
        Parameters
        ----------
        py_avg            : software averages per circuit instance
        max_circuit_depth : maximum number of Cliffords (exclusive upper bound)
        delta_clifford    : step size between depth points
        number_sample     : number of random circuit instances per depth
        interleaved_gate  : if set, run interleaved RB with this gate name
                            (must be a valid Clifford name, e.g. "X/2")
        seed              : random seed for reproducibility
        """
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved_gate = interleaved_gate

        # Parent RNG: child seeds derived from this ensure each sample is unique
        # but the full experiment is still reproducible when seed is given.
        parent_rng = np.random.default_rng(seed)

        if interleaved_gate is not None:
            assert interleaved_gate in _VALID_IRB_GATES, (
                f"'{interleaved_gate}' is not a valid IRB gate. "
                f"Must be one of: {sorted(_VALID_IRB_GATES)}"
            )
            desc = f"Interleaved RB ({interleaved_gate}) depth"
        else:
            desc = "Standard RB depth"

        rb_result = []
        for depth in tqdm(self.x, desc=desc):
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                # Draw a unique child seed for each sample so circuits differ.
                child_seed = int(parent_rng.integers(0, 2**31))
                if interleaved_gate is None:
                    c_idx, r_idx, pulses, U_acc, U_rec = generate_rb_sequence(
                        depth, seed=child_seed)
                else:
                    c_idx, r_idx, pulses, U_acc, U_rec = generate_irb_sequence(
                        depth, interleave_gate=interleaved_gate, seed=child_seed)
                self.cfg["gate_seq"] = pulses

                prog = RBProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                # Store raw complex IQ
                rblist.append(iq_list[0][0].dot([1, 1j]))
            rb_result.append(rblist)
        self.rb_result = rb_result

    # ── Save ──────────────────────────────────────────────────────────────────

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """
        Save RB data to HDF5 (Labber) format.

        Layout
        ------
        x-axis : Circuit Depth   (values = self.x, each depth point)
        y-axis : Sample Number   (values = 0 … number_sample-1)
        data   : complex IQ,     shape (number_sample, n_depths)

        Filenames
        ---------
        Standard RB        →  s015_RB_{qb_idx}
        Interleaved X/2    →  s015_RB_halfX_{qb_idx}
        Interleaved -X/2   →  s015_RB_halfXm_{qb_idx}
        (see _INTERLEAVED_FILE_SUFFIX for full table)

        Parameters
        ----------
        qb_idx     : qubit index / name (used in filename and YAML comment)
        config_all : optional ExperimentConfig — extracts nested YAML for comment
        yoko_value : optional dict with 'value'/'unit' keys for filename
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before saveLabber().")

        # ── Build filename ─────────────────────────────────────────────────────
        gate = self._interleaved_gate
        if gate is None:
            expt_name = f"s015_RB_{qb_idx}_ref"
        else:
            suffix = _INTERLEAVED_FILE_SUFFIX.get(
                gate, gate.replace("/", "").replace("-", "m")
            )
            expt_name = f"s015_RB_{suffix}_{qb_idx}"

        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        # ── Config comment ─────────────────────────────────────────────────────
        if config_all is not None:
            dict_val = config_all.to_yaml(q_id=qb_idx)
        else:
            dict_val = config_to_yaml(self.cfg)
        comment = str(dict_val)

        # ── Axes ───────────────────────────────────────────────────────────────
        n_depths = len(self.x)
        n_samples = self._number_sample

        # x-axis: circuit depth (number of Cliffords)
        x_info = {
            "name": "Circuit Depth",
            "unit": "",
            "values": self.x.astype(float),  # length = n_depths
        }

        # y-axis: sample number
        y_info = {
            "name": "Sample Number",
            "unit": "",
            "values": np.arange(n_samples, dtype=float),  # length = n_samples
        }

        # data: shape (n_samples, n_depths) — one row per sample
        raw = np.array(self.rb_result)  # shape (n_depths, n_samples)
        iq_data = raw.T  # shape (n_samples, n_depths)

        hdf5_generator(
            filepath=file_path,
            x_info=x_info,
            y_info=y_info,
            z_info={"name": "Signal", "unit": "ADC unit", "values": iq_data},
            comment=comment,
            tag="RB",
        )
        print(f"RB data saved to {file_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self, label, color=None, ax=None, marker="o", show_individual=False):
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before plot().")

        raw = np.array(self.rb_result)
        amp_per_sample = np.abs(raw)
        std_r_avg = amp_per_sample.mean(axis=1)
        std_r_std = amp_per_sample.std(axis=1)
        std_r_sem = std_r_std / np.sqrt(amp_per_sample.shape[1])

        pOpt, pCov = fitrb(self.x, std_r_avg)
        p_fit = pOpt[0]
        p_fit_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = (
            float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        )

        print(f"\n--- {label} ---")
        print(f"  p   = {p_fit * 100:.6f} ± {p_fit_err * 100:.6f} %")
        print(f"  EPC = {epc * 100:.6f} ± {epc_err * 100:.6f} %")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        base_color = color if color is not None else "steelblue"

        # Individual sample points (small, gray)
        if show_individual:
            for s in range(amp_per_sample.shape[1]):
                ax.scatter(
                    self.x,
                    amp_per_sample[:, s],
                    s=6,
                    color="gray",
                    alpha=0.25,
                    linewidths=0,
                    zorder=1,
                )

        # Fit line
        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        yfit = rb_func(xfit, *pOpt)
        ax.plot(
            xfit,
            yfit,
            color=base_color,
            linewidth=2.0,
            zorder=3,
            solid_capstyle="round",
        )

        # Error bars + mean data points
        ax.errorbar(
            self.x,
            std_r_avg,
            yerr=std_r_sem,
            fmt="none",
            ecolor=base_color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
            alpha=0.8,
        )
        ax.scatter(
            self.x,
            std_r_avg,
            s=60,
            color=base_color,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=label,
        )

        return epc, epc_err, p_fit, p_fit_err, pCov
