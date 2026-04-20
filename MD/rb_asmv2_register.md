# RB 實驗中 asmv2 Register 的使用方式

## 核心問題

Randomized Benchmarking (RB) 每次 shot 需要依序執行一串 Clifford gate。
每個 gate 對應一個 pulse（X90、Y180…等），選哪個 pulse 必須在硬體執行時動態決定，
而不是在 Python 端逐一編譯成不同程式。

解法：將 gate 序列存入 **data memory (DMEM)**，每次迴圈從 DMEM 讀出 gate 編號，
再用 **register 比較 + 條件跳轉** 選擇對應的 pulse branch。

---

## 整體架構

```
Python 端 (compile time)          硬體端 (run time)
─────────────────────────          ─────────────────────────────────────
gate_seq = [2, 0, 5, 1, ...]  →   DMEM[0..N-1] = {2, 0, 5, 1, ...}

ScanWith  ──┐                      loop counter = 0
            ├─ Repeat(N)     →     while counter < N:
            └─ LoadValue     →         gate_idx = DMEM[counter]
                                        │
Branch ──────────────────────────  compare gate_idx:
    case 0 → Pulse(Id)                  0 → execute Id pulse
    case 1 → Pulse(X90)                 1 → execute X90 pulse
    case 2 → Pulse(X180)                2 → execute X180 pulse
    ...                                 ...
                                        counter++
```

---

## 三個關鍵模組

### 1. `ScanWith` — 組裝器

[dmem.py:196](../lib/zcu_tools/program/v2/modules/dmem.py#L196)

`ScanWith` 本身不產生任何 asmv2 指令，只是把 `Repeat` 和 `LoadValue` 組合起來。

```python
ScanWith("gate_idx", gate_seq, val_reg="gate_idx")
```

- 內部建立 `Repeat("gate_idx_count", len(gate_seq))` 提供迴圈
- 內部建立 `LoadValue(idx_reg="gate_idx_count", val_reg="gate_idx", values=gate_seq)`
- 提供 `.add_content(...)` 讓 `Branch` 插入迴圈體內

---

### 2. `LoadValue` — DMEM 讀取器

[dmem.py:21](../lib/zcu_tools/program/v2/modules/dmem.py#L21)

負責「把序列存進 DMEM，並在每次迴圈從 DMEM 讀值到 register」。

**init 階段（compile time）：**

```python
self.offset = prog.add_dmem(self._packed_values)  # 寫入 DMEM
prog.add_reg(self.val_reg)                          # 分配 gate_idx register
```

**run 階段（每次迴圈，run time）：**

```
addr_reg = loop_counter + offset       # 計算 DMEM 位址
gate_idx = DMEM[addr_reg]              # 讀出當前 gate 編號
```

對應的 asmv2 指令（非壓縮模式）：

```
REG_WR  addr_reg ← idx_reg + offset
MEM_RD  gate_idx ← DMEM[addr_reg]
```

**壓縮模式（auto_compress，序列長度 ≥ 30）：**

若所有 gate 編號只需少量 bit 表示（如 7 種 gate 只需 3 bit），
多個值會打包進同一個 32-bit DMEM word，用位元移位取出：

```
addr_reg  = idx ASR word_shift          # 找到所在 word
word_reg  = DMEM[addr_reg]              # 讀出整個 word
shift_reg = idx AND slot_mask           # 計算 slot 位移
shift_reg = shift_reg SL bits_shift
gate_idx  = word_reg ASR shift_reg      # 移出目標值
gate_idx  = gate_idx AND value_mask     # 遮罩取低位
```

---

### 3. `Branch` — 條件跳轉選擇器

[control.py:119](../lib/zcu_tools/program/v2/modules/control.py#L119)

根據 `gate_idx` register 的值，跳轉到對應的 pulse branch 執行。

```python
Branch("basic_gate",
    Pulse("gate_Id",   id_pulse),
    Pulse("gate_X90",  x90_pulse),
    Pulse("gate_X180", x180_pulse),
    ...
    compare_by="gate_idx",
)
```

展開成的 asmv2 指令（以 4 個 branch 為例）：

```
cond_jump  branch_skip_0  gate_idx NZ - 0   # gate_idx != 0 → 跳過 branch 0
  [branch 0 pulses]
jump       branch_end

branch_skip_0:
cond_jump  branch_skip_1  gate_idx NZ - 1
  [branch 1 pulses]
jump       branch_end

branch_skip_1:
cond_jump  branch_skip_2  gate_idx NZ - 2
  [branch 2 pulses]
jump       branch_end

branch_skip_2:
  [branch 3 pulses]        # 最後一個不需要 cond_jump

branch_end:
```

每個 branch 前後均有 `delay` / `delay_auto` 刷新時序，
確保不同長度的 pulse 不互相干擾。

---

## rb.py 中的完整呼叫鏈

[rb.py:248](../lib/zcu_tools/experiment/v2/twotone/rb.py#L248)

```python
gate_seq = reduce_gate_seq(clifford_seq)   # Python 端：Clifford → BasicGate 序列

ModularProgramV2(soccfg, cfg, modules=[
    Reset("reset", ...),
    ScanWith("gate_idx", gate_seq, val_reg="gate_idx").add_content(
        Branch(
            "basic_gate",
            Pulse("gate_Id",   id_pulse),
            Pulse("gate_X90",  x90_pulse),
            Pulse("gate_X180", x180_pulse),
            Pulse("gate_MX90", mx90_pulse),
            Pulse("gate_Y90",  y90_pulse),
            Pulse("gate_Y180", y180_pulse),
            Pulse("gate_MY90", my90_pulse),
            compare_by="gate_idx",
        )
    ),
    Readout("readout", ...),
])
```

`reduce_gate_seq` 會把 Clifford 分解中的虛擬 Z gate 轉換成相位偏移，
只保留真實需要發射的 pulse（X90/X180/Y90 等），輸出為 `BasicGate` enum 整數序列。

---

## Register 分配總覽

| Register 名稱 | 用途 | 在哪分配 |
|---|---|---|
| `gate_idx_count` | `Repeat` 迴圈計數器 | `OpenLoopReg.preprocess()` via `prog.add_reg` |
| `gate_idx` | 當前 gate 編號（DMEM 讀出值） | `LoadValue.init()` via `prog.add_reg` |
| `addr_reg` (temp) | DMEM 位址 / 位元移位暫存 | `LoadValue.init()` via `prog.acquire_temp_reg` |
| `word_reg` (temp) | 壓縮模式：packed DMEM word | `LoadValue.init()` via `prog.acquire_temp_reg`（僅壓縮模式）|

---

## 設計重點

1. **單一程式多序列**：`prog_cache[(seed, depth)]` 快取每個 `(seed, depth)` 組合的程式，
   避免重複編譯相同序列。

2. **硬體端執行序列**：gate 序列在 DMEM 內，迴圈與跳轉在 asmv2 組語層執行，
   Python 不介入每個 gate 的選擇，減少 latency。

3. **虛擬 Z gate**：Z 旋轉不發射 pulse，只在 `reduce_gate_seq` 時累積相位偏移，
   將後續 X/Y pulse 的 phase 欄位調整，硬體端完全看不到 Z gate。

4. **壓縮 DMEM**：當序列夠長（≥ 30 個值），`LoadValue` 自動壓縮打包，
   減少 DMEM 用量，但需要額外的位元運算解包。
