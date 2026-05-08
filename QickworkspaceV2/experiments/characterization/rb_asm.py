"""
characterization/rb_asm — Single Qubit RB with ASMv2 DMEM dispatch.

Drop-in replacement for rb.py that solves the pmem/dmem overflow at
large Clifford depths by moving the gate-sequence loop from Python
compile-time unrolling into a tProc v2 hardware loop.

Architecture
------------
``rb.py`` (compile-time unroll)::

    for gate in cfg["gate_seq"]:          # Python: O(N) pmem instructions
        self.pulse(name=..., t=0)

``rb_asm.py`` (ASMv2 DMEM dispatch, this file)::

    compile_datamem()  →  pack 8 gate codes per int32 dmem word (4 bits each)
    _body():
        write_reg / init word_reg, shift_reg, word_addr
        open_loop(N, "gate_idx")
            gate_code = (word_reg ASR shift_reg) AND 0xF    ← 3 REG_WR
            cond_jump GATE_I if code == 0
            cond_jump GATE_X if code == 1  ...              ← O(1) dispatch table
        POST_GATE:
            shift_reg += 4 ; reload word every 8th gate
        close_loop()

Memory comparison
-----------------
+------------------+-----------------+------------------+
| metric           | rb.py (unroll)  | rb_asm.py (DMEM) |
+==================+=================+==================+
| pmem             | O(N_gates)      | ~120 words fixed |
| dmem             | —               | ceil(N/8) words  |
| max gates (pmem) | ~480            | unlimited*       |
| max gates (dmem) | —               | 32 768           |
| max depth (RB)   | ~160 Clifford   | ~17 476 Clifford |
| max depth (IRB)  | ~105 Clifford   | ~11 398 Clifford |
+------------------+-----------------+------------------+

\\* pmem is fixed; dmem ceiling is 4096 words × 8 codes = 32 768 gates.

Notes
-----
``delay()`` is used inside branch blocks instead of ``delay_auto()``
because ``delay_auto`` computes the inc_ref immediate at compile time
from the accumulated timeline; in runtime branches (where only one
branch executes per iteration) this produces incorrect timing.
``delay()`` encodes a literal tick count that is branch-independent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from qick.asm_v2 import AsmInst, Macro

from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from ...analysis.rb import RBAnalysis
from ...tools.fitting import fitrb, rb_func, rb_error, error_fit_err


# ── Internal: ALU macro ──────────────────────────────────────────────────────

class _RegOp(Macro):
    """Store ALU result in-place: ``dst = dst {op} src``.

    The tProc v2 ``REG_WR -op(...)`` instruction supports the full
    aluList (AND, ASR, SL, OR, …) but the standard Python API only
    exposes ADD via ``inc_reg``.  This macro exposes AND and ASR for
    the bit-unpack decode step.

    Parameters
    ----------
    dst : str
        Destination (and first operand) register name.
    src : int or str
        Immediate value or source register name.
    op : str
        ALU operation string recognised by tprocv2_assembler
        ``aluList``, e.g. ``'AND'``, ``'ASR'``.
    """

    def expand(self, prog):
        dst = prog._get_reg(self.dst)
        src = "#%d" % self.src if isinstance(self.src, int) else prog._get_reg(self.src)
        return [
            AsmInst(
                inst={"CMD": "REG_WR", "DST": dst, "SRC": "op", "OP": f"{dst} {self.op} {src}"},
                addr_inc=1,
            )
        ]


# ── Gate encoding ────────────────────────────────────────────────────────────

_GATE_CODES: dict[str, int] = {
    "I":    0,
    "X":    1,
    "Y":    2,
    "X/2":  3,
    "-X/2": 4,
    "Y/2":  5,
    "-Y/2": 6,
}

_INTERLEAVED_FILE_SUFFIX: dict[str, str] = {
    "X":    "X",
    "Y":    "Y",
    "X/2":  "halfX",
    "-X/2": "halfXm",
    "Y/2":  "halfY",
    "-Y/2": "halfYm",
}


# ── QICK Program ─────────────────────────────────────────────────────────────

class RBAsmProgram(BaseProgram):
    """QICK program: ASMv2 DMEM dispatch loop for RB.

    Gate codes are packed 8-per-word in dmem (4 bits each); each
    iteration decodes one code via ``ASR`` + ``AND 0xF`` and dispatches
    with ``cond_jump``.  pmem footprint is ~120 words regardless of depth.
    """

    def _initialize(self, cfg):
        """Set up resonator, qubit, standard gates, and unpack registers.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dictionary.
        """
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

        # gate_idx is allocated automatically by open_loop()
        self.add_reg("gate_code")  # unpacked 4-bit gate code from current word
        self.add_reg("word_reg")   # cached packed dmem word (8 codes × 4 bits)
        self.add_reg("shift_reg")  # bit offset within word_reg: 0/4/8/.../28
        self.add_reg("word_addr")  # current dmem word index

    def compile_datamem(self):
        """Pack the gate sequence into dmem (8 codes per int32 word).

        Bit layout of each 32-bit word::

            bits  3: 0  → gate code for position 8k+0
            bits  7: 4  → gate code for position 8k+1
            bits 11: 8  → gate code for position 8k+2
            ...
            bits 31:28  → gate code for position 8k+7

        Returns
        -------
        packed : numpy.ndarray of int32
            Length ``ceil(len(gate_seq) / 8)``.  Fits in 4096-word
            dmem for sequences up to 32 768 gate codes.
        """
        codes = [_GATE_CODES[g] for g in self.cfg["gate_seq"]]
        packed = []
        for i in range(0, len(codes), 8):
            word = 0
            for j in range(min(8, len(codes) - i)):
                word |= (codes[i + j] & 0xF) << (j * 4)
            packed.append(word)
        return np.array(packed, dtype=np.int32)

    def _body(self, cfg):
        """Hardware gate-dispatch loop followed by measurement.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dictionary.  Required keys:
            ``ro_ch``, ``qb_ch``, ``sigma_{prefix}``, ``relax_delay``.
        """
        pfx = cfg.get("prefix", "ge")
        # gate slot duration — must be a compile-time constant (see module notes)
        slot = float(cfg[f"sigma_{pfx}"]) * 5 + 0.01   # µs
        ch   = cfg["qb_ch"]

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        # ── initialise unpack registers ──────────────────────────────────
        self.write_reg("shift_reg", 0)
        self.write_reg("word_addr", 0)
        self.read_dmem("word_reg", "word_addr")   # preload word 0

        # ── hardware loop ────────────────────────────────────────────────
        self.open_loop(len(cfg["gate_seq"]), name="gate_idx")

        # decode: gate_code = (word_reg >> shift_reg) & 0xF
        self.write_reg("gate_code", "word_reg")
        self.append_macro(_RegOp(dst="gate_code", src="shift_reg", op="ASR"))
        self.append_macro(_RegOp(dst="gate_code", src=0xF,         op="AND"))

        # dispatch
        self.cond_jump("GATE_I",   "gate_code", "Z")
        self.cond_jump("GATE_X",   "gate_code", "Z", op="-", arg2=1)
        self.cond_jump("GATE_Y",   "gate_code", "Z", op="-", arg2=2)
        self.cond_jump("GATE_X2",  "gate_code", "Z", op="-", arg2=3)
        self.cond_jump("GATE_mX2", "gate_code", "Z", op="-", arg2=4)
        self.cond_jump("GATE_Y2",  "gate_code", "Z", op="-", arg2=5)
        self.cond_jump("GATE_mY2", "gate_code", "Z", op="-", arg2=6)
        self.jump("POST_GATE")

        # gate blocks
        self.label("GATE_I")
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_X")
        self.pulse(ch=ch, name=f"x180_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_Y")
        self.pulse(ch=ch, name=f"y180_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_X2")
        self.pulse(ch=ch, name=f"x90_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_mX2")
        self.pulse(ch=ch, name=f"x90m_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_Y2")
        self.pulse(ch=ch, name=f"y90_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_mY2")
        self.pulse(ch=ch, name=f"y90m_{pfx}", t=0)
        self.delay(slot)

        # ── advance unpack counters ──────────────────────────────────────
        self.label("POST_GATE")
        self.inc_reg("shift_reg", 4)
        self.cond_jump("NO_WORD_ADVANCE", "shift_reg", "NZ", op="-", arg2=32)
        self.write_reg("shift_reg", 0)
        self.inc_reg("word_addr", 1)
        self.read_dmem("word_reg", "word_addr")
        self.label("NO_WORD_ADVANCE")

        self.close_loop()

        self.delay_auto(0.05)
        self.measure(cfg)


# ── Experiment ───────────────────────────────────────────────────────────────

class RandomizedBenchmarkingAsm(BaseExperiment):
    """Single-qubit RB using ASMv2 DMEM dispatch (constant pmem).

    Functionally identical to ``RandomizedBenchmarking`` but uses
    ``RBAsmProgram`` so program memory does not grow with circuit
    depth.  Hardware ``rounds`` replace the Python averaging loop,
    reducing per-circuit overhead.

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    Raises
    ------
    RuntimeError
        If ``BaseExperiment.connect_pyro4()`` has not been called.

    Examples
    --------
    >>> rb = RandomizedBenchmarkingAsm(cfg)
    >>> result = rb.run(py_avg=100, max_circuit_depth=500,
    ...                 delta_clifford=20, number_sample=5)
    >>> rb.plot("Standard RB")
    """

    EXPT_NAME = "s015_RB_asm"
    Analysis   = RBAnalysis

    def __init__(self, config):
        super().__init__(config)
        self.cfg          = config
        self.x            = None
        self.rb_result    = None
        self._number_sample = None
        self._interleaved = None
        self._iq_process  = "abs"

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
        """Acquire standard RB or IRB data.

        Parameters
        ----------
        py_avg : int
            Hardware rounds (averages) per circuit.  Passed directly
            as ``rounds`` to ``prog.acquire()``; no Python loop.
        max_circuit_depth : int
            Maximum Clifford depth (exclusive upper bound).
        delta_clifford : int
            Step size between sampled depths.
        number_sample : int
            Number of random circuits per depth point.
        interleaved_gate : str or None, optional
            Gate name for IRB (must be in ``INTERLEAVE_GATES``), or
            ``None`` for standard RB.
        seed : int or None, optional
            Random seed for reproducibility.
        prefix : str, optional
            Transition prefix: ``"ge"`` (default) or ``"ef"``.
        iq_process : str, optional
            ``"abs"`` (default) or ``"real"`` for I-axis readout
            after readout optimisation.
        randomize_depth_order : bool, optional
            Shuffle depth order to average out slow time drift.

        Returns
        -------
        result : ExperimentData
            Fitted decay result including EPC and fidelity.

        Raises
        ------
        ValueError
            If ``interleaved_gate`` is not in the supported gate set.
        """
        from ...tools.rb_generator import single_qb_rb, INTERLEAVE_GATES

        self._iq_process    = iq_process
        self.x              = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved   = interleaved_gate
        self._prefix        = prefix

        if interleaved_gate is not None and interleaved_gate not in INTERLEAVE_GATES:
            raise ValueError(
                f"interleaved_gate '{interleaved_gate}' not in "
                f"{list(INTERLEAVE_GATES.keys())}"
            )

        is_irb = interleaved_gate is not None
        desc   = f"IRB ({interleaved_gate})" if is_irb else "Standard RB (ASM)"
        rng    = np.random.default_rng(seed)
        n_depths = len(self.x)

        depth_indices = np.arange(n_depths)
        if randomize_depth_order:
            rng.shuffle(depth_indices)
            print(f"  Depth order: {self.x[depth_indices].tolist()}")

        rb_result: list = [None] * n_depths
        for idx in tqdm(depth_indices, desc=desc):
            depth  = self.x[idx]
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                child_seed = int(rng.integers(0, 2**31))
                seqs = single_qb_rb(
                    n_clifford=depth, n_sample=1,
                    interleave=interleaved_gate, seed=child_seed,
                )
                self.cfg["gate_seq"] = seqs[0]
                self.cfg["prefix"]   = prefix
                prog = RBAsmProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_data = (
                    prog.acquire(self.soc, rounds=py_avg, progress=False)[0][0]
                    .dot([1, 1j])
                )
                rblist.append(iq_data)
            rb_result[idx] = rblist

        self.rb_result = rb_result

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

    def plot(
        self,
        label: str,
        color=None,
        ax=None,
        marker: str = "o",
        show_individual: bool = False,
    ):
        """Fit and plot the RB decay curve.

        Parameters
        ----------
        label : str
            Legend label for this curve.
        color : str or None, optional
            Line and marker colour.  Defaults to ``"steelblue"``.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot into.  A new figure is created when ``None``.
        marker : str, optional
            Marker style.  Default ``"o"``.
        show_individual : bool, optional
            Scatter individual circuit samples as grey points.

        Returns
        -------
        epc : float
            Error per Clifford.
        epc_err : float
            One-sigma uncertainty on EPC.
        p_fit : float
            Fitted decay parameter.
        p_fit_err : float
            One-sigma uncertainty on the decay parameter.
        pCov : numpy.ndarray
            Covariance matrix from the fit.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")

        _proc = np.real if self._iq_process == "real" else np.abs
        raw   = np.array(self.rb_result)
        amp   = _proc(raw)
        avg   = amp.mean(axis=1)

        pOpt, pCov    = fitrb(self.x, avg)
        p_fit         = pOpt[0]
        p_fit_err     = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc           = rb_error(p_fit, d=2)
        epc_err       = (
            float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        )

        print(f"\n--- {label} (ASM) ---")
        print(f"  p   = {p_fit*100:.6f} ± {p_fit_err*100:.6f} %")
        print(f"  EPC = {epc*100:.6f} ± {epc_err*100:.6f} %")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        c = color or "steelblue"
        if show_individual:
            for s in range(amp.shape[1]):
                ax.scatter(self.x, amp[:, s], s=6, color="gray",
                           alpha=0.25, linewidths=0, zorder=1)
        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=c, linewidth=2.0, zorder=3)
        ax.errorbar(self.x, avg, yerr=amp.std(axis=1) / np.sqrt(amp.shape[1]),
                    fmt="none", ecolor=c, capsize=3, zorder=4)
        ax.scatter(self.x, avg, s=60, color=c, marker=marker,
                   edgecolors="black", label=label, zorder=5)
        return epc, epc_err, p_fit, p_fit_err, pCov

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None, title=None):
        """Save RB data to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the experiment name.
        config_all : object or None, optional
            Full config object with a ``to_yaml(q_id)`` method.
        yoko_value : float or None, optional
            Yokogawa flux bias embedded in the filename.
        title : str or None, optional
            Custom title string for the saved file name.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Call run() first.")
        from ...tools.system_tool import (
            config_to_yaml, get_next_filename_labber, hdf5_generator,
        )
        if title is not None:
            expt_name = f"s015_RB_asm_{title}_{qb_idx}"
        elif self._interleaved is not None:
            suffix = _INTERLEAVED_FILE_SUFFIX.get(self._interleaved, self._interleaved)
            expt_name = f"s015_RB_asm_{suffix}_{qb_idx}"
        else:
            expt_name = f"s015_RB_asm_{qb_idx}_ref"

        save_dir  = BaseExperiment._data_path
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val  = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Circuit Depth", "unit": "", "values": self.x.astype(float)},
            y_info={"name": "Sample Number",  "unit": "",
                    "values": np.arange(self._number_sample, dtype=float)},
            z_info={"name": "Signal", "unit": "ADC unit",
                    "values": np.array(self.rb_result).T},
            comment=str(dict_val),
            tag="RB",
        )
        print(f"RB data saved to {file_path}")


# ── AutoRBAsm ─────────────────────────────────────────────────────────────────

def _gate_fidelity(p_ref: float, p_irb: float, d: int = 2):
    epc = (d - 1) / d * (1 - p_irb / p_ref)
    return 1 - epc, epc


def _gate_fidelity_err(
    p_ref: float, p_irb: float,
    var_p_ref: float, var_p_irb: float,
    d: int = 2,
) -> float:
    c = (d - 1) / d
    return float(
        np.sqrt((c * p_irb / p_ref**2) ** 2 * var_p_ref
                + (c / p_ref) ** 2 * var_p_irb)
    )


class AutoRBAsm:
    """Automated Standard + Interleaved RB using ASMv2 dispatch.

    Runs one standard RB reference followed by one IRB pass per
    requested gate, then reports gate fidelity from the combined fit.

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    Examples
    --------
    >>> auto = AutoRBAsm(cfg)
    >>> auto.run(py_avg=100, max_circuit_depth=500,
    ...          delta_clifford=20, number_sample=5,
    ...          interleaved_gates=["X", "X/2"])
    >>> auto.plot()
    >>> print(auto.summary())
    """

    def __init__(self, config):
        self.cfg           = config
        self._rb_kwargs: dict = {}
        self.results: dict    = {}
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
        """Run reference RB and optionally one IRB per gate.

        Parameters
        ----------
        py_avg : int
            Hardware rounds per circuit.
        max_circuit_depth : int
            Maximum Clifford depth (exclusive).
        delta_clifford : int
            Step between sampled depths.
        number_sample : int
            Random circuits per depth.
        interleaved_gates : list of str or None, optional
            Gates to characterise via IRB (e.g. ``["X", "X/2"]``).
        seed : int or None, optional
            Global random seed.
        prefix : str, optional
            Transition prefix.  Default ``"ge"``.
        iq_process : str, optional
            ``"abs"`` or ``"real"``.
        """
        self._rb_kwargs = dict(
            max_circuit_depth=max_circuit_depth,
            delta_clifford=delta_clifford,
            number_sample=number_sample,
            seed=seed,
            prefix=prefix,
            iq_process=iq_process,
        )
        gates_to_run = [None] + (interleaved_gates or [])
        for gate in tqdm(gates_to_run, desc="AutoRBAsm"):
            label = "ref" if gate is None else gate
            rb = RandomizedBenchmarkingAsm(self.cfg)
            rb.run(py_avg, interleaved_gate=gate, **self._rb_kwargs)
            self._rb_objects[label] = rb

    def plot(self, show_individual: bool = False):
        """Plot all RB/IRB decay curves and print gate fidelities.

        Parameters
        ----------
        show_individual : bool, optional
            Scatter individual circuit samples as grey points.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ref_rb  = self._rb_objects.get("ref")
        if ref_rb is None:
            print("No reference RB — call run() first.")
            return

        ref_epc, ref_epc_err, p_ref, p_ref_err, ref_cov = ref_rb.plot(
            "Reference RB", color=colors[0], ax=ax,
            show_individual=show_individual,
        )
        self.results["ref"] = dict(
            epc=ref_epc, epc_err=ref_epc_err, p=p_ref, p_err=p_ref_err,
        )

        for i, (label, rb) in enumerate(self._rb_objects.items()):
            if label == "ref":
                continue
            epc, epc_err, p_irb, p_irb_err, irb_cov = rb.plot(
                f"IRB ({label})", color=colors[(i + 1) % len(colors)],
                ax=ax, show_individual=show_individual,
            )
            var_ref = ref_cov[0, 0] if ref_cov is not None else 0.0
            var_irb = irb_cov[0, 0] if irb_cov is not None else 0.0
            f_gate, epc_gate     = _gate_fidelity(p_ref, p_irb)
            epc_gate_err         = _gate_fidelity_err(p_ref, p_irb, var_ref, var_irb)
            self.results[label]  = dict(
                fidelity=f_gate, epc=epc_gate, epc_err=epc_gate_err,
                p=p_irb, p_err=p_irb_err,
            )
            print(
                f"  Gate '{label}': F = {f_gate*100:.4f}%  "
                f"EPC = {epc_gate*100:.4f} ± {epc_gate_err*100:.4f} %"
            )

        ax.set_xlabel("Circuit Depth (# Cliffords)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def summary(self) -> str:
        """Return a formatted text summary of all fidelity results.

        Returns
        -------
        summary : str
            Multi-line string with EPC and fidelity for each gate.
        """
        lines = ["AutoRBAsm Summary", "=" * 50]
        for key, val in self.results.items():
            if "fidelity" in val:
                lines.append(
                    f"  {key:<10s}  F={val['fidelity']*100:.4f}%  "
                    f"EPC={val['epc']*100:.5f}%"
                )
            else:
                lines.append(
                    f"  {key:<10s}  EPC={val['epc']*100:.5f}%"
                )
        return "\n".join(lines)

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None):
        """Save all RB/IRB datasets to HDF5/Labber files.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to each experiment name.
        config_all : object or None, optional
            Full config with a ``to_yaml(q_id)`` method.
        yoko_value : float or None, optional
            Yokogawa flux bias embedded in filenames.
        """
        for label, rb in self._rb_objects.items():
            rb.saveLabber(qb_idx, config_all=config_all,
                          yoko_value=yoko_value, title=label)
