"""
s015_RB_asm — Randomized Benchmarking with ASMv2 gate-dispatch loop
====================================================================
與 s015_Single_qubit_RB.py 功能完全相同，但把 Python for 迴圈改成
ASMv2 open_loop + register-based gate dispatch。

架構差異
--------
原版（Python for-loop）：
    for gate in cfg["gate_seq"]:         ← Python compile-time unroll
        self.pulse(name=gate_pulse, t=0)

本版（ASMv2 dispatch）：
    _initialize()  →  add_reg("gate_code")   ← gate_idx 由 open_loop 自動分配
    compile_datamem() →  gate_seq 打包：8 個 gate code 存入一個 int32 word
                         (每個 code 佔 4 bit，位置 0/4/8/12/16/20/24/28)
    _body() 中：
        ; 初始化 word_reg / shift_reg / word_addr
        open_loop(N, "gate_idx")
            ; 解碼：gate_code = (word_reg >> shift_reg) & 0xF
            write_reg gate_code  word_reg
            REG_WR    gate_code -op(gate_code ASR shift_reg)
            REG_WR    gate_code -op(gate_code AND #15)
            cond_jump GATE_I    if gate_code == 0    ← 線性分派
            cond_jump GATE_X    if gate_code == 1
            ...
        LABEL POST_GATE:
            ; 更新 shift_reg，每 8 gate 換下一個 word
            inc_reg   shift_reg 4
            cond_jump NO_ADVANCE  shift_reg NZ - 32
            write_reg shift_reg 0
            inc_reg   word_addr 1
            read_dmem word_reg  word_addr
        NO_ADVANCE:
        close_loop()
        measure

dmem 容量比較
-------------
              原版 (1 code/word)   本版 (8 codes/word)
dmem 容量            4096 gates         32768 gates
等效最大深度 (RB)    ~2184 Clifford     ~17476 Clifford
等效最大深度 (IRB)   ~1425 Clifford     ~11398 Clifford

為何用 delay() 而非 delay_auto()
---------------------------------
delay_auto() 在編譯時計算「compile-time timeline 累積值 + t」作為 inc_ref 的立即數。
在 runtime 分支（每次只走其中一條路）中，各分支的累積值是 compile-time 疊加的，
會產生錯誤的 inc_ref 值。
delay(t) 只編碼 Python 端計算好的固定 t_ticks，不受 compile-time 累積影響，
適合在 runtime jump table 的各分支中使用。

pmem / dmem 大小（不受序列長度影響）
--------------------------------------
pmem ≈ 120 words (固定)
dmem = ceil(N_gates / 8) words ← O(N/8)，實際 dmem 用量降為原來 1/8

限制
----
每個 circuit 仍需 recompile（compile_datamem() 是 binprog 的一部分，
每次 prog.acquire() 都會重新編譯並上傳），無法消除 acquire() call 的數量。
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from qick.asm_v2 import Macro, AsmInst

from .base_program import BaseProgram
from .RB_generator import single_qb_rb, INTERLEAVE_GATES
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err


# ── Helper macro ────────────────────────────────────────────────────────────
class _RegOp(Macro):
    """Emit: REG_WR dst -op(dst {op} src)  — stores the ALU result in dst.

    The tProc v2 REG_WR instruction supports the full aluList (AND, ASR, SL,
    OR, …) but the standard Python API (inc_reg / write_reg) only exposes ADD.
    This helper fills the gap for AND and ASR needed by the bit-unpacking loop.

    Parameters
    ----------
    dst : str   destination register name
    src : int or str   immediate value or source register name
    op  : str   ALU op string, e.g. ``'AND'``, ``'ASR'``
    """
    def expand(self, prog):
        dst = prog._get_reg(self.dst)
        src = '#%d' % self.src if isinstance(self.src, int) else prog._get_reg(self.src)
        return [AsmInst(inst={'CMD': 'REG_WR', 'DST': dst, 'SRC': 'op',
                              'OP': f'{dst} {self.op} {src}'}, addr_inc=1)]

# ── Gate code encoding ──────────────────────────────────────────────────────
# I=0, X=1, Y=2, X/2=3, -X/2=4, Y/2=5, -Y/2=6

_GATE_CODES = {
    "I":    0,
    "X":    1,
    "Y":    2,
    "X/2":  3,
    "-X/2": 4,
    "Y/2":  5,
    "-Y/2": 6,
}

_INTERLEAVED_FILE_SUFFIX = {
    "X":    "X",
    "Y":    "Y",
    "X/2":  "halfX",
    "-X/2": "halfXm",
    "Y/2":  "halfY",
    "-Y/2": "halfYm",
}


# ====================================================== #
# QICK Program
# ====================================================== #
class RBAsmProgram(BaseProgram):
    """
    RB program using ASMv2 gate dispatch.

    ``compile_datamem()`` encodes the gate sequence into data memory (dmem).
    ``_body()`` implements a hardware dispatch loop:
    read gate_code from dmem then cond_jump to the corresponding gate block.
    """

    def _initialize(self, cfg):
        """Set up resonator, qubit generator, standard gates, and dispatch registers."""
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

        # gate_idx allocated automatically by open_loop()
        self.add_reg("gate_code")  # unpacked gate code (0-6) from current dmem word
        self.add_reg("word_reg")   # cached packed dmem word (8 codes × 4 bits)
        self.add_reg("shift_reg")  # current bit offset within word_reg (0/4/8/.../28)
        self.add_reg("word_addr")  # current dmem word index

    def compile_datamem(self):
        """
        Override ``QickProgramV2.compile_datamem()`` to statically initialise dmem.

        Packs 8 gate codes per 32-bit dmem word (4 bits each at bit positions
        0, 4, 8, …, 28).  dmem capacity = 4096 words × 8 = 32 768 gate codes,
        equivalent to ~17 476 Clifford depth for standard RB.

        Returns
        -------
        packed : ndarray of int32
            dmem initialisation array, length = ceil(len(gate_seq) / 8).
        """
        gate_seq = self.cfg["gate_seq"]
        codes = [_GATE_CODES[g] for g in gate_seq]

        packed = []
        for i in range(0, len(codes), 8):
            word = 0
            for j in range(min(8, len(codes) - i)):
                word |= (codes[i + j] & 0xF) << (j * 4)
            packed.append(word)
        return np.array(packed, dtype=np.int32)

    def _body(self, cfg):
        """Implement the ASMv2 gate-dispatch loop and final measurement."""
        pfx = cfg.get("prefix", "ge")

        # 使用 delay(slot) 而非 delay_auto()，原因見模組 docstring。
        gate_len = float(cfg[f"sigma_{pfx}"]) * 5   # µs；float() 防止 QickParam 傳入
        gap      = 0.01                              # µs
        slot     = gate_len + gap
        ch       = cfg["qb_ch"]

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        # ── 初始化 bit-unpack 用的 registers（每次 acquire 執行一次）────
        self.write_reg("shift_reg", 0)
        self.write_reg("word_addr", 0)
        self.read_dmem("word_reg", "word_addr")   # 預載第 0 個 word

        # ── 硬體迴圈：open_loop 自動分配 gate_idx register ──────────────
        # gate_idx 從 0 遞增到 N-1，close_loop() 負責遞增與跳回。
        self.open_loop(len(cfg["gate_seq"]), name="gate_idx")

        # ── 解碼：gate_code = (word_reg >> shift_reg) & 0xF ─────────────
        # 三條指令；loop 內只能用 delay() 不能用 delay_auto()，但算術指令無此限制。
        self.write_reg("gate_code", "word_reg")
        self.append_macro(_RegOp(dst="gate_code", src="shift_reg", op="ASR"))
        self.append_macro(_RegOp(dst="gate_code", src=0xF,         op="AND"))

        # ── 線性分派鏈（switch-case 等效）────────────────────────────────
        self.cond_jump("GATE_I",   "gate_code", "Z")                   # code 0
        self.cond_jump("GATE_X",   "gate_code", "Z", op="-", arg2=1)  # code 1
        self.cond_jump("GATE_Y",   "gate_code", "Z", op="-", arg2=2)  # code 2
        self.cond_jump("GATE_X2",  "gate_code", "Z", op="-", arg2=3)  # code 3
        self.cond_jump("GATE_mX2", "gate_code", "Z", op="-", arg2=4)  # code 4
        self.cond_jump("GATE_Y2",  "gate_code", "Z", op="-", arg2=5)  # code 5
        self.cond_jump("GATE_mY2", "gate_code", "Z", op="-", arg2=6)  # code 6
        self.jump("POST_GATE")  # safety fallthrough（正常不會走到）

        # ── Gate blocks ──────────────────────────────────────────────────
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

        # ── POST_GATE：更新 bit-unpack counters，然後 close_loop ─────────
        self.label("POST_GATE")
        # shift_reg 每 gate 加 4；滿 32（即處理完 8 個 gate）時換下一個 word
        self.inc_reg("shift_reg", 4)
        self.cond_jump("NO_WORD_ADVANCE", "shift_reg", "NZ", op="-", arg2=32)
        self.write_reg("shift_reg", 0)
        self.inc_reg("word_addr", 1)
        self.read_dmem("word_reg", "word_addr")
        self.label("NO_WORD_ADVANCE")

        self.close_loop()   # 遞增 gate_idx；若未完成則跳回 open_loop 標籤

        self.delay_auto(0.05)
        self.measure(cfg)


# ====================================================== #
# Experiment class
# ====================================================== #
class RandomizedBenchmarkingAsm:
    """
    RB experiment using ASMv2 gate dispatch (constant pmem regardless of depth).

    Same execution logic as ``RandomizedBenchmarking`` but uses
    ``RBAsmProgram`` so program memory size does not grow with circuit depth.
    Each circuit still requires a ``prog.acquire()`` call for recompilation
    because ``compile_datamem()`` is part of the binary program.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Experiment configuration dictionary.

        Raises
        ------
        RuntimeError
            If ``BaseExperiment.setup()`` has not been called first.
        """
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.soc     = BaseExperiment._soc
        self.soccfg  = BaseExperiment._soccfg
        self.cfg     = config
        self.x       = None
        self.rb_result = None
        self._number_sample = None
        self._interleaved   = None

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        interleaved_gate=None,
        seed=None,
        prefix="ge",
        iq_process="abs",
        randomize_depth_order=False,
    ):
        """
        Acquire RB or IRB data using the ASMv2 dispatch program.

        Parameters
        ----------
        py_avg : int
            Hardware averages (rounds) per circuit.
        max_circuit_depth : int
            Maximum Clifford depth (exclusive upper bound).
        delta_clifford : int
            Step size between circuit depths.
        number_sample : int
            Number of random circuit samples per depth point.
        interleaved_gate : str or None, optional
            Gate name for IRB, or ``None`` for standard RB.
        seed : int or None, optional
            Random seed for reproducibility.
        prefix : str, optional
            Gate prefix (``"ge"`` or ``"ef"``).
        iq_process : str, optional
            IQ processing mode: ``"abs"`` or ``"real"``.
        randomize_depth_order : bool, optional
            Measure circuit depths in random order to average out time drift.

        Raises
        ------
        ValueError
            If ``interleaved_gate`` is not in the supported gate set.
        """
        self._iq_process = iq_process
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        self._number_sample = number_sample
        self._interleaved   = interleaved_gate
        self._prefix        = prefix

        if interleaved_gate is not None and interleaved_gate not in INTERLEAVE_GATES:
            raise ValueError(
                f"interleaved_gate '{interleaved_gate}' not supported. "
                f"Choose from: {list(INTERLEAVE_GATES.keys())}"
            )

        is_irb = interleaved_gate is not None
        desc   = f"IRB ({interleaved_gate})" if is_irb else "Standard RB (ASMv2)"
        rng    = np.random.default_rng(seed)

        depth_indices = np.arange(len(self.x))
        if randomize_depth_order:
            rng.shuffle(depth_indices)
            print(f"  Depth order: {self.x[depth_indices].tolist()}")

        rb_result = [None] * len(self.x)
        for idx in tqdm(depth_indices, desc=desc):
            depth = self.x[idx]
            rblist = []
            for _ in tqdm(range(number_sample), desc="Samples", leave=False):
                child_seed = int(rng.integers(0, 2**31))

                seqs = single_qb_rb(
                    n_clifford=depth,
                    n_sample=1,
                    interleave=interleaved_gate,
                    seed=child_seed,
                )
                gate_seq = seqs[0]

                self.cfg["gate_seq"] = gate_seq
                self.cfg["prefix"]   = prefix

                # ── 重新 compile：compile_datamem() 把 gate_seq 寫入 dmem ──
                prog = RBAsmProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                rblist.append(iq_list[0][0].dot([1, 1j]))

            rb_result[idx] = rblist

        self.rb_result = rb_result

    def saveLabber(self, qb_idx, config_all=None, yoko_value=None, title=None):
        """
        Save RB data to an HDF5/Labber file.

        Parameters
        ----------
        qb_idx : int
            Qubit index appended to the experiment name.
        config_all : object or None, optional
            Full config object with a ``to_yaml(q_id)`` method.
        yoko_value : float or None, optional
            Yokogawa flux bias value embedded in the filename.
        title : str or None, optional
            Custom title string for the saved file name.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before saveLabber().")

        if title is not None:
            expt_name = f"s015_RB_asm_{title}_{qb_idx}"
        elif self._interleaved is not None:
            suffix = _INTERLEAVED_FILE_SUFFIX.get(self._interleaved, self._interleaved)
            expt_name = f"s015_RB_asm_{suffix}_{qb_idx}"
        else:
            expt_name = f"s015_RB_asm_{qb_idx}_ref"

        from .base_experiment import BaseExperiment
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )
        raw = np.array(self.rb_result)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Circuit Depth", "unit": "", "values": self.x.astype(float)},
            y_info={"name": "Sample Number", "unit": "", "values": np.arange(self._number_sample, dtype=float)},
            z_info={"name": "Signal", "unit": "ADC unit", "values": raw.T},
            comment=str(dict_val),
            tag="RB",
        )
        print(f"RB data saved to {file_path}")

    def plot(self, label, color=None, ax=None, marker="o", show_individual=False):
        """
        Fit and plot the RB decay curve.

        Parameters
        ----------
        label : str
            Legend label for this curve.
        color : str or None, optional
            Line and marker colour.  Defaults to ``"steelblue"``.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot into.  A new figure is created when ``None``.
        marker : str, optional
            Marker style for the average data points.
        show_individual : bool, optional
            Whether to show individual circuit samples as scatter points.

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
        pCov : ndarray
            Covariance matrix from the fit.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called first.
        """
        if self.x is None or self.rb_result is None:
            raise RuntimeError("Must call run() before plot().")

        _proc = np.real if getattr(self, "_iq_process", "abs") == "real" else np.abs
        raw = np.array(self.rb_result)
        amp = _proc(raw)
        avg = amp.mean(axis=1)

        pOpt, pCov = fitrb(self.x, avg)
        p_fit     = pOpt[0]
        p_fit_err = float(np.sqrt(np.diag(pCov))[0]) if pCov is not None else 0.0
        epc       = rb_error(p_fit, d=2)
        epc_err   = (
            float(np.sqrt(error_fit_err(pCov[0, 0], d=2))) if pCov is not None else 0.0
        )

        print(f"\n--- {label} (ASMv2 dispatch) ---")
        print(f"  p   = {p_fit * 100:.6f} ± {p_fit_err * 100:.6f} %")
        print(f"  EPC = {epc * 100:.6f} ± {epc_err * 100:.6f} %")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        c = color or "steelblue"

        if show_individual:
            for s in range(amp.shape[1]):
                ax.scatter(self.x, amp[:, s], s=6, color="gray",
                           alpha=0.25, linewidths=0, zorder=1)

        xfit = np.linspace(self.x.min(), self.x.max(), 400)
        ax.plot(xfit, rb_func(xfit, *pOpt), color=c, linewidth=2.0, zorder=3)
        ax.errorbar(self.x, avg,
                    yerr=amp.std(axis=1) / np.sqrt(amp.shape[1]),
                    fmt="none", ecolor=c, capsize=3, zorder=4)
        ax.scatter(self.x, avg, s=60, color=c, marker=marker,
                   edgecolors="black", label=label, zorder=5)

        return epc, epc_err, p_fit, p_fit_err, pCov
