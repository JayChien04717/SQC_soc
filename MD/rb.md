# Randomized Benchmarking (RB)

## 兩種實作

| 類別 | 檔案 | pmem | dmem 容量 | 最大深度 (RB) |
|---|---|---|---|---|
| `RandomizedBenchmarking` | `experiments/characterization/rb.py` | O(N_gates) | — | ~160 Clifford |
| `RandomizedBenchmarkingAsm` | `experiments/characterization/rb_asm.py` | **~120 words 固定** | ceil(N/8) words | **~17 476 Clifford** |

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
  pack 8 codes per int32 word
  (4 bits each, positions 0/4/8/.../28)

_body():
  write_reg  shift_reg ← 0
  write_reg  word_addr ← 0
  read_dmem  word_reg  ← dmem[0]     preload word 0

  open_loop(N, "gate_idx")           loop counter = 0 .. N-1
    ; decode
    write_reg gate_code ← word_reg
    REG_WR    gate_code  ASR shift_reg   gate_code >>= shift_reg
    REG_WR    gate_code  AND #15         gate_code &= 0xF
    ; dispatch
    cond_jump GATE_I   if code == 0
    cond_jump GATE_X   if code == 1
    ...
  POST_GATE:
    inc_reg  shift_reg + 4
    if shift_reg == 32:              every 8th gate: reload word
        shift_reg = 0
        word_addr += 1
        word_reg = dmem[word_addr]
  close_loop()
```

### dmem 打包格式

```
int32 word = (code[8k]   & 0xF)
           | (code[8k+1] & 0xF) << 4
           | (code[8k+2] & 0xF) << 8
           | (code[8k+3] & 0xF) << 12
           | (code[8k+4] & 0xF) << 16
           | (code[8k+5] & 0xF) << 20
           | (code[8k+6] & 0xF) << 24
           | (code[8k+7] & 0xF) << 28
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
| `rb_asm.py` | 固定 ~120 words | ceil(N_gates / 8)，上限 32 768 gates |

dmem 上限 32 768 gates ÷ 平均 1.875 gates/Clifford ≈ **~17 476 Clifford（RB）**  
dmem 上限 32 768 gates ÷ 平均 2.875 gates/Clifford ≈ **~11 398 Clifford（IRB）**

---

## py_avg 語意差異

| | `RandomizedBenchmarking` | `RandomizedBenchmarkingAsm` |
|---|---|---|
| `py_avg` | 軟體迴圈次數（Python for loop） | 硬體 rounds（`prog.acquire(rounds=py_avg)`） |
| 實際行為 | 同一電路測 py_avg 次，Python 累加 | 同一電路跑 py_avg 硬體 rounds，QICK 平均 |
| 速度 | 較慢（Python 開銷） | 較快（無 Python 開銷） |
| 統計等效 | 是 | 是 |

---

## Appendix：tProc v2 Register 操作完整教學

### 1. Register 是什麼

tProc v2 是一顆 32-bit RISC-like 軟核處理器，內建一個 **register file**（暫存器堆），
每個 register 存一個有符號 32-bit 整數。
QICK Python API 以字串名稱（`"shift_reg"`、`"word_addr"` …）管理 register，
底層由 `_get_reg(name)` 把名稱對映到實際硬體 register 編號。

Register 沒有型別之分，只有「目前存的 int32 值」。

---

### 2. `add_reg` — 宣告一個 named register

```python
self.add_reg("my_counter")
```

在 `_initialize()` 裡呼叫，告訴 QICK 分配一個 register slot 並命名。
`open_loop()` 會自動分配一個 loop-counter register，不需手動宣告。

> **規則**：同一個程式內名稱必須唯一；重複 `add_reg` 同一名稱會報錯。

---

### 3. `write_reg` — 寫入立即數或複製另一個 register

```python
self.write_reg("shift_reg", 0)          # shift_reg ← 0      (立即數)
self.write_reg("gate_code", "word_reg") # gate_code ← word_reg（register copy）
```

| 第二個參數 | 行為 |
|---|---|
| `int` 或 `float` | 把立即數寫入 dst |
| `str` (register name) | 把 src register 的值複製到 dst |

底層組語：`REG_WR dst, #imm` 或 `REG_WR dst, src`。

---

### 4. `inc_reg` — 加法（唯一標準 API 暴露的 ALU 運算）

```python
self.inc_reg("shift_reg", 4)   # shift_reg += 4
self.inc_reg("word_addr", 1)   # word_addr += 1
```

底層組語：`REG_WR dst -op(dst + #imm)`。  
只接受 **立即數**；不能用另一個 register 當加數。  
要做其他 ALU 運算（AND、ASR 等）必須用 `_RegOp` macro（見第 6 節）。

---

### 5. `read_dmem` — 從 data memory 讀值

```python
self.read_dmem("word_reg", "word_addr")  # word_reg ← dmem[word_addr]
```

第二個參數是 **存著位址的 register 名稱**（間接定址）。  
直接用整數立即數也行（固定位址）：

```python
self.read_dmem("word_reg", 0)  # word_reg ← dmem[0]
```

dmem 是 4096 個 int32 word 的陣列，由 `compile_datamem()` 在程式上傳前初始化。

---

### 6. `_RegOp` Macro — 補足 AND / ASR 等 ALU 運算

#### 背景：aluList vs aluList_s

tProc v2 assembler 有兩張 ALU op 表：

| 表 | 支援的 op | 用途 |
|---|---|---|
| `aluList` | `+`, `-`, `*`, `AND`, `OR`, `XOR`, `SL`, `SR`, `ASR` | `REG_WR -op(...)` 算術指令 |
| `aluList_s` | `+`, `-`, `AND`, `ASR` | `JUMP -cond(...)` 條件跳轉的 ALU |

標準 Python API `inc_reg` 只暴露 `+`（ADD）。  
`write_reg` 只做純搬移，不計算。  
AND、ASR 沒有對應的標準 API → 用 Macro 自造。

#### `_RegOp` 的實作

```python
from qick.asm_v2 import Macro, AsmInst

class _RegOp(Macro):
    def expand(self, prog):
        dst = prog._get_reg(self.dst)          # 名稱 → 硬體 register 編號字串
        src = ("#%d" % self.src                # 立即數：加 # 前綴
               if isinstance(self.src, int)
               else prog._get_reg(self.src))   # register 名稱 → 編號
        return [AsmInst(
            inst={"CMD": "REG_WR",
                  "DST": dst,
                  "SRC": "op",                 # "op" 表示走 aluList 路徑
                  "OP":  f"{dst} {self.op} {src}"},
            addr_inc=1,
        )]
```

`Macro` 的建構子自動把 `**kwargs` 存成屬性，所以 `self.dst`、`self.src`、`self.op`
是直接傳進去的關鍵字引數。

#### 用法

```python
# dst = dst >> src_reg  (算術右移，保留符號位)
self.append_macro(_RegOp(dst="gate_code", src="shift_reg", op="ASR"))

# dst = dst & 0xF  (取低 4 bit)
self.append_macro(_RegOp(dst="gate_code", src=0xF, op="AND"))

# dst = dst | mask
self.append_macro(_RegOp(dst="flags", src=0b1010, op="OR"))

# dst = dst << 2
self.append_macro(_RegOp(dst="val", src=2, op="SL"))
```

**注意**：所有操作都是 **in-place**（結果寫回 dst）。  
若需要保留原始值，先 `write_reg("tmp", "src")` 複製再操作。

---

### 7. `cond_jump` — 條件跳轉

#### 基本形式：jump if reg == 0

```python
self.cond_jump("LABEL", "reg_name", "Z")
```

`"Z"` = Zero flag：若 `reg_name` 目前值 == 0 則跳。  
`"NZ"` = Not Zero：若 != 0 則跳。

#### 帶 ALU 的形式：jump if (reg OP arg2) 的結果符合條件

```python
self.cond_jump("LABEL", "reg_name", "Z", op="-", arg2=n)
# 語義：if (reg_name - n) == 0  →  if reg_name == n
```

底層組語：`JUMP -cond(reg OP arg2) Z LABEL`

| 參數 | 意義 | 可用值 |
|---|---|---|
| `op` | ALU op（aluList_s） | `"+"`, `"-"`, `"AND"`, `"ASR"` |
| `arg2` | 第二運算元（立即數） | 任意 int32 |
| condition | `"Z"` / `"NZ"` | Zero / Not-Zero |

常用模式：

```python
# if reg == n: jump
self.cond_jump("TARGET", "reg", "Z", op="-", arg2=n)

# if reg != n: jump
self.cond_jump("TARGET", "reg", "NZ", op="-", arg2=n)

# if reg == 0: jump（最簡形式）
self.cond_jump("TARGET", "reg", "Z")

# if (reg & mask) == 0: jump（測試特定 bit）
self.cond_jump("TARGET", "reg", "Z", op="AND", arg2=mask)
```

> **限制**：`arg2` 只能是立即數，不能是 register。  
> 若要比較兩個 register，先用 `_RegOp(op="-")` 計算差，再對結果用 `cond_jump(..., "Z")`。

---

### 8. `open_loop` / `close_loop` — 硬體計數迴圈

```python
self.open_loop(N, name="gate_idx")
# ... 迴圈本體 ...
self.close_loop()
```

- `open_loop(N, name)` 自動分配一個 loop-counter register（名稱為 `name`），
  從 0 遞增到 N-1，共執行 N 次。
- `close_loop()` 發出 `LOOP` 指令：遞增 counter，若 < N 則跳回 `open_loop` 的下一條。
- counter register 在迴圈體內**可讀不可寫**（寫入結果未定義）。
- 迴圈可以巢狀（每層各自分配一個 register）。

```python
# 讀取 loop counter 目前值（例如用來計算位址偏移）
self.write_reg("tmp", "gate_idx")
```

---

### 9. `jump` — 無條件跳轉

```python
self.label("MY_LABEL")
# ...
self.jump("MY_LABEL")
```

搭配 `label()` 使用。常用於 dispatch chain 的 fallthrough 跳過其他 gate block：

```python
self.label("GATE_X")
self.pulse(...)
self.delay(slot)
self.jump("POST_GATE")   # 執行完 X gate 後直接跳到計數更新，不落入下一個 block

self.label("GATE_Y")
# ...
```

---

### 10. 完整範例：bit-unpack 迴圈（rb_asm.py 的核心）

下面把上述所有操作串在一起，對照逐行說明：

```python
# ── _initialize ─────────────────────────────────────────────────────
self.add_reg("gate_code")   # 解碼後的 gate code（0–6）
self.add_reg("word_reg")    # 從 dmem 讀出的 packed word（8 codes × 4 bit）
self.add_reg("shift_reg")   # 目前要讀的 bit 偏移量（0, 4, 8, …, 28）
self.add_reg("word_addr")   # 目前 dmem word 的索引

# ── _body ────────────────────────────────────────────────────────────

# 1. 初始化所有 register
self.write_reg("shift_reg", 0)
self.write_reg("word_addr", 0)
self.read_dmem("word_reg", "word_addr")       # word_reg ← dmem[0]

# 2. 硬體迴圈：自動分配 gate_idx register，從 0 到 N-1
self.open_loop(len(cfg["gate_seq"]), name="gate_idx")

#   3a. 複製 word_reg 到 gate_code（保留原始 word 供下次迭代使用）
self.write_reg("gate_code", "word_reg")

#   3b. gate_code >>= shift_reg  (取出正確的 4-bit slot)
self.append_macro(_RegOp(dst="gate_code", src="shift_reg", op="ASR"))

#   3c. gate_code &= 0xF  (遮掉高位，只留低 4 bit)
self.append_macro(_RegOp(dst="gate_code", src=0xF, op="AND"))

#   4. 線性 dispatch：依 gate_code 的值跳到對應 gate block
self.cond_jump("GATE_I",   "gate_code", "Z")                  # code == 0
self.cond_jump("GATE_X",   "gate_code", "Z", op="-", arg2=1)  # code == 1
self.cond_jump("GATE_Y",   "gate_code", "Z", op="-", arg2=2)  # code == 2
# ... (略)
self.jump("POST_GATE")

#   5. Gate blocks（每個 block 結束都 jump 到 POST_GATE）
self.label("GATE_I");  self.delay(slot);  self.jump("POST_GATE")
self.label("GATE_X");  self.pulse(...);   self.delay(slot); self.jump("POST_GATE")
# ...

# 6. POST_GATE：更新 bit-unpack counters
self.label("POST_GATE")
self.inc_reg("shift_reg", 4)                                   # shift_reg += 4
self.cond_jump("NO_ADVANCE", "shift_reg", "NZ", op="-", arg2=32)  # if shift != 32: skip
self.write_reg("shift_reg", 0)                                 # shift_reg = 0
self.inc_reg("word_addr", 1)                                   # word_addr += 1
self.read_dmem("word_reg", "word_addr")                        # 載入下一個 word
self.label("NO_ADVANCE")

# 7. close_loop：gate_idx += 1；若 < N 則跳回 open_loop 後的第一條指令
self.close_loop()
```

#### 執行流程圖

```text
open_loop(N)
│
├─► [gate_idx = 0..N-1]
│     write_reg gate_code ← word_reg
│     _RegOp ASR shift_reg
│     _RegOp AND 0xF
│     cond_jump → GATE_I / GATE_X / ...
│               ↓ 執行對應 pulse
│     POST_GATE:
│       inc_reg shift_reg +4
│       if shift_reg == 32:
│           write_reg shift_reg 0
│           inc_reg word_addr +1
│           read_dmem word_reg word_addr
│
close_loop() ──► gate_idx < N? → 跳回 open_loop
                              → 繼續往下
```

---

### 11. 常見錯誤與注意事項

| 錯誤 | 原因 | 修正 |
|---|---|---|
| `cond_jump` 的 `arg2` 用 register 名稱 | aluList_s 不支援 register-register 比較 | 先 `_RegOp(op="-")` 計算差，再 `cond_jump(..., "Z")` |
| `delay_auto()` 在 dispatch branch 裡 | compile-time timeline 累積值不對 | 改用 `delay(t_µs)` |
| `add_reg` 在 `_body` 裡呼叫 | register 需在 `_initialize` 階段分配 | 移到 `_initialize` |
| 迴圈內寫入 `open_loop` 自動分配的 counter | 行為未定義 | 只讀 counter，不寫 |
| `_RegOp` 的 `op` 字串大小寫錯誤 | assembler 區分大小寫 | 用 `"AND"`、`"ASR"`（全大寫） |
