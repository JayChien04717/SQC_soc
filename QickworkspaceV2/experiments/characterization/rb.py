"""
Characterization/rb — s015: Single Qubit RB, Interleaved RB, AutoRB.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...core.experiment_data import ExperimentData, QualityFlag
from ...analysis.rb import RBAnalysis
from ...tools.fitting import fitrb, rb_func, rb_error, error_fit_err


_GEN_TO_QICK = {
    "I":    None,
    "X":    "x180_{pfx}",
    "Y":    "y180_{pfx}",
    "X/2":  "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2":  "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
}

_INTERLEAVED_FILE_SUFFIX = {
    "X":    "X",
    "Y":    "Y",
    "X/2":  "halfX",
    "-X/2": "halfXm",
    "Y/2":  "halfY",
    "-Y/2": "halfYm",
}


class RBProgram(BaseProgram):
    """QICK program that unrolls a Clifford gate sequence at compile time."""

    def _initialize(self, cfg):
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)
        pfx = cfg.get("prefix", "ge")
        for gate in cfg["gate_seq"]:
            if gate == "I":
                self.delay_auto(cfg[f"sigma_{pfx}"] * 5)
            else:
                template = _GEN_TO_QICK.get(gate)
                if template is None:
                    raise ValueError(f"Unknown gate '{gate}' in gate_seq")
                self.pulse(ch=cfg["qb_ch"], name=template.format(pfx=pfx), t=0)
                self.delay_auto(0.01)
        self.delay_auto(0.05)
        self.measure(cfg)


class RandomizedBenchmarking(BaseExperiment):
    """Single-qubit RB (s015): standard and interleaved."""

    EXPT_NAME = "s015_RB"
    Analysis = RBAnalysis

    def __init__(self, config):
        super().__init__(config)
        self.x = None
        self.rb_result = None
        self._number_sample = None
        self._interleaved = None
        self._iq_process = "abs"

    def run(
        self,
        py_avg: int,
        max_circuit_depth: int,
        delta_clifford: int,
        number_sample: int,
        interleaved_gate: str | None = None,
        seed: int | None = None,
        prefix: str = "ge",
        iq_process: str = "abs",
        randomize_depth_order: bool = False,
    ) -> ExperimentData:
        from ...tools.rb_generator import single_qb_rb, INTERLEAVE_GATES

        self._iq_process = iq_process
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved = interleaved_gate
        self._prefix = prefix

        if interleaved_gate is not None and interleaved_gate not in INTERLEAVE_GATES:
            raise ValueError(
                f"interleaved_gate '{interleaved_gate}' not in {list(INTERLEAVE_GATES.keys())}"
            )

        is_irb = interleaved_gate is not None
        desc = f"IRB ({interleaved_gate})" if is_irb else "Standard RB"
        rng = np.random.default_rng(seed)
        n_depths = len(self.x)
        seeds_matrix = [
            [int(rng.integers(0, 2**31)) for _ in range(number_sample)]
            for _ in range(n_depths)
        ]
        depth_indices = np.arange(n_depths)
        if randomize_depth_order:
            rng.shuffle(depth_indices)

        rb_accum = [[None] * number_sample for _ in range(n_depths)]
        for avg_i in tqdm(range(py_avg), desc="Software Average"):
            for idx in tqdm(depth_indices, desc=desc, leave=False):
                depth = self.x[idx]
                for s_i in tqdm(range(number_sample), desc="Samples", leave=False):
                    seqs = single_qb_rb(
                        n_clifford=depth, n_sample=1,
                        interleave=interleaved_gate, seed=seeds_matrix[idx][s_i],
                    )
                    self.cfg["gate_seq"] = seqs[0]
                    self.cfg["prefix"] = prefix
                    prog = RBProgram(
                        self.soccfg, reps=self.cfg["reps"],
                        final_delay=self.cfg["relax_delay"], cfg=self.cfg,
                    )
                    iq_data = prog.acquire(self.soc, rounds=1, progress=False)[0][0].dot([1, 1j])
                    if avg_i == 0:
                        rb_accum[idx][s_i] = iq_data
                    else:
                        rb_accum[idx][s_i] = rb_accum[idx][s_i] + iq_data

        self.rb_result = [
            [rb_accum[idx][s_i] / py_avg for s_i in range(number_sample)]
            for idx in range(n_depths)
        ]

        _proc = np.real if iq_process == "real" else np.abs
        avg = _proc(np.array(self.rb_result)).mean(axis=1)
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            x_axis=self.x.astype(float),
            y_axis=avg,
            quality=QualityFlag.NO_INFORMATION,
        )
        if self.Analysis is not None:
            result = self.Analysis().run(result)
        self.result = result
        return result

    def plot(self, label: str, color=None, ax=None, marker="o", show_individual=False):
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")
        _proc = np.real if self._iq_process == "real" else np.abs
        raw = np.array(self.rb_result)
        amp = _proc(raw)
        avg = amp.mean(axis=1)
        pOpt, pCov = fitrb(self.x, avg)
        p_fit = pOpt[0]
        p_fit_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc = rb_error(p_fit, d=2)
        epc_err = float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        print(f"\n--- {label} ---")
        print(f"  p   = {p_fit*100:.6f} ± {p_fit_err*100:.6f} %")
        print(f"  EPC = {epc*100:.6f} ± {epc_err*100:.6f} %")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        c = color or "steelblue"
        if show_individual:
            for s in range(amp.shape[1]):
                ax.scatter(self.x, amp[:, s], s=6, color="gray", alpha=0.25, linewidths=0, zorder=1)
        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=c, linewidth=2.0, zorder=3)
        ax.errorbar(self.x, avg, yerr=amp.std(axis=1)/np.sqrt(amp.shape[1]),
                    fmt="none", ecolor=c, capsize=3, zorder=4)
        ax.scatter(self.x, avg, s=60, color=c, marker=marker,
                   edgecolors="black", label=label, zorder=5)
        return epc, epc_err, p_fit, p_fit_err, pCov

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None, title=None):
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")
        from ...tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
        if title is not None:
            expt_name = f"s015_RB_{title}_{qb_idx}"
        elif self._interleaved is not None:
            suffix = _INTERLEAVED_FILE_SUFFIX.get(self._interleaved, self._interleaved)
            expt_name = f"s015_RB_{suffix}_{qb_idx}"
        else:
            expt_name = f"s015_RB_{qb_idx}_ref"
        save_dir = BaseExperiment._data_path
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = config_all.to_yaml(q_id=qb_idx) if config_all is not None else config_to_yaml(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Circuit Depth", "unit": "", "values": self.x.astype(float)},
            y_info={"name": "Sample Number", "unit": "", "values": np.arange(self._number_sample, dtype=float)},
            z_info={"name": "Signal", "unit": "ADC unit", "values": np.array(self.rb_result).T},
            comment=str(dict_val), tag="RB",
        )
        print(f"RB data saved to {file_path}")


def _gate_fidelity(p_ref, p_irb, d=2):
    epc = (d - 1) / d * (1 - p_irb / p_ref)
    return 1 - epc, epc


def _gate_fidelity_err(p_ref, p_irb, var_p_ref, var_p_irb, d=2):
    c = (d - 1) / d
    depc_dpref = c * p_irb / p_ref**2
    depc_dpirb = -c / p_ref
    return float(np.sqrt(depc_dpref**2 * var_p_ref + depc_dpirb**2 * var_p_irb))


class AutoRB:
    """Automated Standard + Interleaved RB in one call (s015)."""

    def __init__(self, config):
        self.cfg = config
        self._rb_kwargs: dict = {}
        self.results: dict = {}
        self._rb_objects: dict = {}

    def run(
        self,
        py_avg: int,
        max_circuit_depth: int,
        delta_clifford: int,
        number_sample: int,
        interleaved_gates: list[str] | None = None,
        seed: int | None = None,
        prefix: str = "ge",
        iq_process: str = "abs",
    ):
        self._rb_kwargs = dict(
            max_circuit_depth=max_circuit_depth,
            delta_clifford=delta_clifford,
            number_sample=number_sample,
            seed=seed, prefix=prefix, iq_process=iq_process,
        )
        gates_to_run = [None] + (interleaved_gates or [])
        for gate in tqdm(gates_to_run, desc="AutoRB"):
            label = "ref" if gate is None else gate
            rb = RandomizedBenchmarking(self.cfg)
            rb.run(py_avg, interleaved_gate=gate, **self._rb_kwargs)
            self._rb_objects[label] = rb

    def plot(self, show_individual=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ref_rb = self._rb_objects.get("ref")
        if ref_rb is None:
            print("No reference RB — call run() first.")
            return

        ref_epc, ref_epc_err, p_ref, p_ref_err, ref_cov = ref_rb.plot(
            "Reference RB", color=colors[0], ax=ax, show_individual=show_individual,
        )
        self.results["ref"] = dict(epc=ref_epc, epc_err=ref_epc_err, p=p_ref, p_err=p_ref_err)

        for i, (label, rb) in enumerate(self._rb_objects.items()):
            if label == "ref":
                continue
            epc, epc_err, p_irb, p_irb_err, irb_cov = rb.plot(
                f"IRB ({label})", color=colors[(i + 1) % len(colors)],
                ax=ax, show_individual=show_individual,
            )
            var_ref = ref_cov[0, 0] if ref_cov is not None else 0
            var_irb = irb_cov[0, 0] if irb_cov is not None else 0
            f_gate, epc_gate = _gate_fidelity(p_ref, p_irb)
            epc_gate_err = _gate_fidelity_err(p_ref, p_irb, var_ref, var_irb)
            self.results[label] = dict(
                fidelity=f_gate, epc=epc_gate, epc_err=epc_gate_err,
                p=p_irb, p_err=p_irb_err,
            )
            print(f"  Gate '{label}': F = {f_gate*100:.4f}%, EPC = {epc_gate*100:.4f} ± {epc_gate_err*100:.4f} %")

        ax.set_xlabel("Circuit Depth (# Cliffords)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def summary(self) -> str:
        lines = ["AutoRB Summary", "=" * 50]
        for key, val in self.results.items():
            lines.append(f"  {key:<10s}  F={val['fidelity']*100:.4f}%  EPC={val['epc']*100:.5f}%")
        return "\n".join(lines)

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        for label, rb in self._rb_objects.items():
            rb.saveLabber(qb_idx, config_all=config_all, yoko_value=yoko_value, title=label)
