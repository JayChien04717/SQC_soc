"""
s015_RB_asm — Randomized Benchmarking with ASMv2 gate-dispatch loop
====================================================================
與 s015_Single_qubit_RB.py 功能完全相同，但把 Python for 迴圈改成
ASMv2 register-based gate dispatch。

架構差異
--------
原版（Python for-loop）：
    for gate in cfg["gate_seq"]:         ← Python compile-time unroll
        self.pulse(name=gate_pulse, t=0)

本版（ASMv2 dispatch）：
    _initialize()  →  add_reg("gate_idx"), add_reg("gate_code")
    compile_datamem() →  gate_seq 編碼為整數寫入 dmem（不消耗 pmem）
    _body() 中：
        write_reg gate_idx = 0
        LABEL dispatch_loop:
            read_dmem gate_code ← gate_idx          ← 間接定址
            cond_jump seq_done  if gate_code == END  ← sentinel 偵測
            cond_jump GATE_I    if gate_code == 0    ← 線性分派
            cond_jump GATE_X    if gate_code == 1
            ...
        LABEL GATE_X:
            pulse x180
            delay(slot)                              ← fixed delay，非 delay_auto
            jump POST_GATE
        ...
        LABEL POST_GATE:
            inc_reg gate_idx
            jump dispatch_loop
        LABEL seq_done:
            measure

為何用 delay() 而非 delay_auto()
---------------------------------
delay_auto() 在編譯時計算「compile-time timeline 累積值 + t」作為 inc_ref 的立即數。
在 runtime 分支（每次只走其中一條路）中，各分支的累積值是 compile-time 疊加的，
會產生錯誤的 inc_ref 值。
delay(t) 只編碼 Python 端計算好的固定 t_ticks，不受 compile-time 累積影響，
適合在 runtime jump table 的各分支中使用。

pmem 大小比較（不同電路深度）
------------------------------
          深度 10       深度 50      深度 200
原版     ~30+10*3     ~30+50*3    ~30+200*3  ← O(N·gate_per_clifford)
本版     ~50 (固定)    ~50 (固定)  ~50 (固定)  ← O(1)

限制
----
每個 circuit 仍需 recompile（compile_datamem() 是 binprog 的一部分，
每次 prog.acquire() 都會重新編譯並上傳），無法消除 acquire() call 的數量。
pmem 恆定是本方案的唯一硬體優勢。
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .base_program import BaseProgram
from .RB_generator import single_qb_rb, INTERLEAVE_GATES
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..tools.fitting import fitrb, rb_func, rb_error, error_fit_err

# ── Gate code encoding ──────────────────────────────────────────────────────
# I=0, X=1, Y=2, X/2=3, -X/2=4, Y/2=5, -Y/2=6, END=7

_GATE_CODES = {
    "I":    0,
    "X":    1,
    "Y":    2,
    "X/2":  3,
    "-X/2": 4,
    "Y/2":  5,
    "-Y/2": 6,
}
_END_CODE = 7   # sentinel written at the end of the gate sequence in dmem

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

    compile_datamem() encodes the gate sequence into data memory (dmem).
    _body() implements a hardware dispatch loop:
        read gate_code from dmem → cond_jump to the corresponding gate block.
    """

    def _initialize(self, cfg):
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)

        # ── 宣告 dispatch loop 用的暫存器 ──────────────────────────────
        # 必須在 _initialize() 中明確呼叫 add_reg()，
        # 否則 write_reg / read_dmem / cond_jump 中的 _get_reg() 會找不到名稱。
        self.add_reg("gate_idx")   # 當前 gate 的 dmem 位址 index
        self.add_reg("gate_code")  # 從 dmem 讀出的 gate 代碼

    def compile_datamem(self):
        """
        Override QickProgramV2.compile_datamem() 以靜態初始化 dmem。

        將 gate_seq 編碼為整數陣列寫入 dmem，末尾附上 END sentinel (=7)。
        這樣不消耗 pmem — dmem 是獨立的 binary 區段，由 QICK 框架在
        prog.acquire() 時上傳到 tProc 資料記憶體。

        Returns
        -------
        np.ndarray of int64  (dmem 初始值)
        """
        gate_seq = self.cfg["gate_seq"]
        codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]
        return np.array(codes, dtype=np.int64)

    def _body(self, cfg):
        pfx = cfg.get("prefix", "ge")

        # ── 計算每個 gate 的固定時間槽 ─────────────────────────────────
        # 所有非 identity gate 的 pulse length = sigma * 5（arb / Gaussian）。
        # 使用 delay(slot) 而非 delay_auto()，原因見模組 docstring。
        gate_len = cfg[f"sigma_{pfx}"] * 5   # µs，Gaussian pulse 長度
        gap      = 0.01                       # µs，inter-gate gap
        slot     = gate_len + gap             # µs，每個 gate 的固定時間槽

        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        # ── 初始化 index 暫存器 ──────────────────────────────────────────
        # write_reg(dst, src): tProc REG_WR 指令，將立即值寫入具名暫存器
        self.write_reg("gate_idx", 0)

        # ── Dispatch loop ────────────────────────────────────────────────
        # label(name): 為下一條指令貼上標籤，供 jump / cond_jump 使用
        self.label("dispatch_loop")

        # read_dmem(dst, addr): 從 dmem[gate_idx] 讀值到 gate_code 暫存器
        # addr 為暫存器名稱時使用間接定址 → dmem[r_gate_idx]
        self.read_dmem("gate_code", "gate_idx")

        # ── END sentinel 偵測 ──────────────────────────────────────────
        # cond_jump(label, arg1, test, op, arg2):
        #   TEST 指令：計算 arg1 op arg2，測試結果
        #   test="Z"：結果為零時跳躍（即 gate_code - 7 == 0 → gate_code == 7）
        #   TEST 是 non-destructive，gate_code 暫存器不被修改
        self.cond_jump("seq_done", "gate_code", "Z", op="-", arg2=_END_CODE)

        # ── 線性分派鏈（switch-case 等效）────────────────────────────────
        # QICK ASMv2 沒有 computed jump（無法用暫存器值當跳躍位址），
        # 因此用一系列 cond_jump 模擬 jump table。
        # 每次比較都是 TEST + JUMP，共 7 條指令（含 END check 共 8 條）。
        self.cond_jump("GATE_I",   "gate_code", "Z")                   # code 0
        self.cond_jump("GATE_X",   "gate_code", "Z", op="-", arg2=1)  # code 1
        self.cond_jump("GATE_Y",   "gate_code", "Z", op="-", arg2=2)  # code 2
        self.cond_jump("GATE_X2",  "gate_code", "Z", op="-", arg2=3)  # code 3
        self.cond_jump("GATE_mX2", "gate_code", "Z", op="-", arg2=4)  # code 4
        self.cond_jump("GATE_Y2",  "gate_code", "Z", op="-", arg2=5)  # code 5
        self.cond_jump("GATE_mY2", "gate_code", "Z", op="-", arg2=6)  # code 6
        # 正常情況下不會到這裡（gate_code 只有 0–7）
        self.jump("POST_GATE")

        # ── Gate blocks ──────────────────────────────────────────────────
        # 每個 block 結構：
        #   (optional) pulse(ch, name, t=0)  →  fire at current ref_time
        #   delay(slot)                      →  inc_ref by slot ticks
        #   jump("POST_GATE")                →  回到公共後處理
        #
        # identity gate：不打 pulse，只用 delay 佔據相同時間槽，
        # 確保 identity 與非 identity gate 的總時序一致。

        self.label("GATE_I")
        # delay(t)：TIME inc_ref #slot_ticks，不讀 compile-time 累積時間軸
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_X")
        self.pulse(ch=cfg["qb_ch"], name=f"x180_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_Y")
        self.pulse(ch=cfg["qb_ch"], name=f"y180_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_X2")
        self.pulse(ch=cfg["qb_ch"], name=f"x90_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_mX2")
        self.pulse(ch=cfg["qb_ch"], name=f"x90m_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_Y2")
        self.pulse(ch=cfg["qb_ch"], name=f"y90_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        self.label("GATE_mY2")
        self.pulse(ch=cfg["qb_ch"], name=f"y90m_{pfx}", t=0)
        self.delay(slot)
        self.jump("POST_GATE")

        # ── 每個 gate 共用的後處理 ────────────────────────────────────────
        # inc_reg(dst, src)：dst = dst + src，tProc REG_WR + OP
        self.label("POST_GATE")
        self.inc_reg("gate_idx", 1)
        # jump(label)：無條件跳躍，tProc JUMP 指令
        self.jump("dispatch_loop")

        # ── 序列結束，執行量測 ────────────────────────────────────────────
        self.label("seq_done")
        # 使用 delay(0.05) 而非 delay_auto(0.05)，
        # 避免 compile-time 累積時間軸污染 waiting gap 的數值。
        # ref_time 此時已正確指向最後一個 gate 結束後 gap µs 的位置。
        self.delay(0.05)
        self.measure(cfg)


# ====================================================== #
# Experiment class
# ====================================================== #
class RandomizedBenchmarkingAsm:
    """
    與 RandomizedBenchmarking 相同的執行邏輯，
    但使用 RBAsmProgram（ASMv2 dispatch）代替原版 RBProgram。

    pmem 大小不隨電路深度增長；適合需要長電路（depth > 50）的實驗。
    注意：每個 circuit 仍需呼叫 prog.acquire() 重新編譯（compile_datamem()
    是 binprog 的一部分），不可跳過。
    """

    def __init__(self, config):
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
