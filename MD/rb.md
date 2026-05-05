# Randomized Benchmarking (RB)

## 兩種實作

| 類別 | 檔案 | pmem | dmem 容量 | 最大深度 (RB) |
|---|---|---|---|---|
| `RandomizedBenchmarking` | `experiments/characterization/rb.py` | O(N_gates) | — | ~160 Clifford |
| `RandomizedBenchmarkingAsm` | `experiments/characterization/rb_asm.py` | **~120 words 固定** | ceil(N/4) words | **~8 738 Clifford** |

`RandomizedBenchmarkingAsm` / `AutoRBAsm` 是 drop-in 替換，介面完全相同。  
深度 > ~150 Clifford 或跑 IRB 時請使用 ASM 版本。

---

## 快速使用

```python
from QickworkspaceV2.experiments import RandomizedBenchmarkingAsm, AutoRBAsm

# Standard RB
rb = RandomizedBenchmarkingAsm(cfg)
result = rb.run(
    py_avg=100,
    max_circuit_depth=500,
    delta_clifford=20,
    number_sample=5,
)
rb.plot("Standard RB")

# Standard + IRB in one call
auto = AutoRBAsm(cfg)
auto.run(
    py_avg=100,
    max_circuit_depth=500,
    delta_clifford=20,
    number_sample=5,
    interleaved_gates=["X", "X/2"],
)
auto.plot()
print(auto.summary())
auto.saveLabber(qb_idx=0)
```

---

## ASM 版本架構

### 問題根源（rb.py）

```python
# compile-time unroll → pmem 隨序列長度線性增長
for gate in cfg["gate_seq"]:
    self.pulse(name=gate_pulse, t=0)   # 每個 gate ≈ 8 條 pmem 指令
```

深度 > ~160 Clifford 時超過 tProc v2 的 4096 word pmem 限制。

### 解法架構（rb_asm.py）

```
Python 端 (compile time)             硬體端 (run time)
─────────────────────────            ────────────────────────────
compile_datamem()                    dmem[0..M-1] = packed words
  pack 4 codes per int32 word

_body():
  write_reg  shift_reg ← 0
  write_reg  word_addr ← 0
  read_dmem  word_reg  ← dmem[0]     preload word 0

  open_loop(N, "gate_idx")           loop counter = 0 .. N-1
    ; decode
    write_reg gate_code ← word_reg
    REG_WR    gate_code  ASR shift_reg   gate_code >>= shift_reg
    REG_WR    gate_code  AND #255        gate_code &= 0xFF
    ; dispatch
    cond_jump GATE_I   if code == 0
    cond_jump GATE_X   if code == 1
    ...
  POST_GATE:
    inc_reg  shift_reg + 8
    if shift_reg == 32:              every 4th gate: reload word
        shift_reg = 0
        word_addr += 1
        word_reg = dmem[word_addr]
  close_loop()
```

### dmem 打包格式

```
int32 word = (code[4k] & 0xFF)
           | (code[4k+1] & 0xFF) << 8
           | (code[4k+2] & 0xFF) << 16
           | (code[4k+3] & 0xFF) << 24
```

gate code 對照：`I=0, X=1, Y=2, X/2=3, -X/2=4, Y/2=5, -Y/2=6`

### 為何用 `delay()` 不用 `delay_auto()`

`delay_auto()` 在 compile time 計算 `inc_ref` 立即數（= 目前累積時間 + t）。
在 runtime branch（每次只走一條路）中，compile-time 累積值是所有 branch 疊加的，
會產生錯誤的 timing。  
`delay()` 只編碼固定 tick 數，不受 compile-time timeline 影響，適合用在 dispatch 各分支內。

`close_loop()` 之後回到 compile-time territory，可以安全使用 `delay_auto(0.05)`。

### `_RegOp` 自訂 Macro

tProc v2 `REG_WR -op(...)` 支援完整 `aluList`（AND、ASR、SL、OR…），
但標準 Python API（`write_reg` / `inc_reg`）只暴露 ADD。  
`_RegOp` 透過繼承 `qick.asm_v2.Macro` 補足 AND 與 ASR：

```python
class _RegOp(Macro):
    def expand(self, prog):
        dst = prog._get_reg(self.dst)
        src = "#%d" % self.src if isinstance(self.src, int) else prog._get_reg(self.src)
        return [AsmInst(inst={"CMD": "REG_WR", "DST": dst, "SRC": "op",
                              "OP": f"{dst} {self.op} {src}"}, addr_inc=1)]

# 用法
self.append_macro(_RegOp(dst="gate_code", src="shift_reg", op="ASR"))
self.append_macro(_RegOp(dst="gate_code", src=0xFF,        op="AND"))
```

---

## 記憶體限制對照

| | pmem (4096 word) | dmem (4096 word) |
|---|---|---|
| `rb.py` | 超過 > ~160 Clifford | 未使用 |
| `rb_asm.py` | 固定 ~120 words | ceil(N_gates / 4)，上限 16 384 gates |

dmem 上限 16 384 gates ÷ 平均 1.875 gates/Clifford ≈ **8 738 Clifford（RB）**  
dmem 上限 16 384 gates ÷ 平均 2.875 gates/Clifford ≈ **5 700 Clifford（IRB）**

---

## py_avg 語意差異

| | `RandomizedBenchmarking` | `RandomizedBenchmarkingAsm` |
|---|---|---|
| `py_avg` | 軟體迴圈次數（Python for loop） | 硬體 rounds（`prog.acquire(rounds=py_avg)`） |
| 實際行為 | 同一電路測 py_avg 次，Python 累加 | 同一電路跑 py_avg 硬體 rounds，QICK 平均 |
| 速度 | 較慢（Python 開銷） | 較快（無 Python 開銷） |
| 統計等效 | 是 | 是 |
