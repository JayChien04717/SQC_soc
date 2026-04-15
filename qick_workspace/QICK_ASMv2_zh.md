# QICK ASMv2 指令參考（中文）

> 適用於 `qick.asm_v2.AveragerProgramV2` 及其父類 `QickProgramV2`。
> 所有方法均在 `_initialize()` 或 `_body()` 中呼叫。

---

## 目錄

1. [迴圈控制](#1-迴圈控制)
2. [標籤與跳躍](#2-標籤與跳躍)
3. [條件跳躍與回饋](#3-條件跳躍與回饋)
4. [暫存器操作](#4-暫存器操作)
5. [記憶體操作](#5-記憶體操作)
6. [時序控制](#6-時序控制)
7. [波形與脈衝](#7-波形與脈衝)
8. [子程序（Subroutine）](#8-子程序subroutine)
9. [其他工具指令](#9-其他工具指令)
10. [使用場景速查表](#10-使用場景速查表)
11. 補充：`delay(t) + delay_auto(t=0)` 模式分析
12. 實戰移植：Active Reset（v1 RAveragerProgram → v2 AveragerProgramV2）
13. v1 ↔ v2 時序 API 完整對照

---

## 1. 迴圈控制

### `add_loop(name, count)` — 僅限 `_initialize()`

```python
self.add_loop("gainloop", cfg["steps"])
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 建立 **sweep loop**，讓 tProc 在硬體層面自動掃描參數 |
| `name` | 迴圈名稱，需與 `QickSweep1D` / `QickSpan` 中使用的名稱一致 |
| `count` | 迴圈總次數（必須為 `int`） |
| 位置 | **只能在 `_initialize()` 中呼叫**，不可放在 `_body()` |
| 底層行為 | 框架自動在每次迭代遞增所有綁定到這個 loop 的 `QickParam` |

> **使用情境**：掃描 qubit gain、頻率、時間等量，產生 2D / 多維 sweep 資料。

---

### `open_loop(n, name=None)` / `close_loop()` — 用於 `_body()`

```python
self.open_loop(cfg["iteration"], name="iter_loop")
self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
self.delay_auto(t=0.02)
self.close_loop()
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 在 `_body()` 內部建立 **硬體計數器迴圈**，不掃描任何參數 |
| `n` | 重複次數（必須為 plain `int`，不可為 `QickParam`） |
| `name` | 具名暫存器，省略時自動命名；同一 `_body` 有多個迴圈時需指定不同名稱 |
| 底層行為 | **open_loop**：分配暫存器 → 初始化為 0 → 寫入 label<br>**close_loop**：暫存器 +1 → 若 < n 則 JUMP 回 label |

編譯後的 ASM 示意：

```
iter_loop_reg = 0
LABEL: iter_loop
  PULSE qb_ch qb_pulse
  DELAY 0.02
  iter_loop_reg = iter_loop_reg + 1
  JUMP IF NZ (iter_loop_reg - n) → iter_loop
```

> **vs Python `for _ in range(n)`**：
> - Python for 迴圈在編譯時展開，程式記憶體大小與 n 成正比。
> - `open_loop`/`close_loop` 使用硬體計數器，程式大小固定，不受 n 影響。
> - 適合 n 較大（> 20）或 n 在執行時才決定的情境。

---

## 2. 標籤與跳躍

### `label(label)`

```python
self.label("my_start")
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 為**下一條指令**貼上標籤，作為跳躍目標 |
| `label` | 字串，同一程式中不可重複 |
| 注意 | 若下一條指令是 macro，標籤貼在 macro 展開的**第一條** ASM 指令上 |

---

### `jump(label)`

```python
self.jump("my_start")
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 無條件跳到 `label` 標記的指令 |
| 底層指令 | `JUMP LABEL` |

---

## 3. 條件跳躍與回饋

### `cond_jump(label, arg1, test, op=None, arg2=None)`

```python
# 若 my_reg > 0，跳到 "loop_start"
self.cond_jump("loop_start", arg1="my_reg", test="NZ", op="-", arg2=1)
```

| 參數 | 說明 |
| --- | --- |
| `label` | 條件成立時跳到的標籤 |
| `arg1` | 暫存器名稱（運算的第一個運算元） |
| `test` | 測試條件：`"Z"` (==0) / `"NZ"` (≠0) / `"S"` (<0) / `"NS"` (≥0) / `"1"` (永遠跳) / `"0"` (永遠不跳) / `"F"` (外部旗標) |
| `op` | 運算：`"+"` / `"-"` / `"AND"` (位元 AND) / `"ASR"` (右移) |
| `arg2` | 24-bit 有號整數，或另一個暫存器名稱 |

底層行為：計算 `arg1 op arg2`，對結果進行 `test`，通過則 JUMP。

> **使用情境**：自訂計數器迴圈、條件性跳過某段波形。

---

### `read_input(ro_ch)`

```python
self.read_input(ro_ch=cfg["ro_ch"])
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 從 tProc 輸入埠讀取**已積分**的 I/Q 值 |
| 讀取結果存放位置 | I → 特殊暫存器 `s_port_l`；Q → `s_port_h` |
| 注意 | 必須確保 readout window 已結束，否則讀到的是舊值 |

---

### `read_and_jump(ro_ch, component, threshold, test, label)`

```python
# 若 I >= threshold，跳到 "ground_state"
self.read_and_jump(
    ro_ch=cfg["ro_ch"],
    component="I",
    threshold=cfg["threshold"],
    test=">=",
    label="ground_state",
)
```

| 參數 | 說明 |
| --- | --- |
| `component` | `"I"` 或 `"Q"` |
| `threshold` | 24-bit 有號整數或暫存器名稱 |
| `test` | `">="` 或 `"<"` |

等同於 `read_input()` + `cond_jump()`，是實現**主動回饋（active reset）**的核心指令。

> **使用情境**：主動量子態重置（active reset）：量測到 |e⟩ 就發 π pulse，量測到 |g⟩ 就跳過。

---

## 4. 暫存器操作

tProc v2 有一組通用暫存器（user-accessible registers），可用名稱存取。

### `write_reg(dst, src)`

```python
self.write_reg("my_counter", 0)       # 寫入立即值
self.write_reg("my_counter", "other") # 從另一暫存器複製
```

| 參數 | 說明 |
| --- | --- |
| `dst` | 目的暫存器名稱（字串） |
| `src` | 立即整數值，或來源暫存器名稱 |

---

### `inc_reg(dst, src)`

```python
self.inc_reg("my_counter", 1)         # my_counter += 1
self.inc_reg("my_counter", "step_reg")
```

| 參數 | 說明 |
| --- | --- |
| `dst` | 目的暫存器名稱 |
| `src` | 要加上的立即值，或另一暫存器 |

---

### `set_ext_counter(addr, val)` / `inc_ext_counter(addr, val)`

```python
self.set_ext_counter(addr=1, val=0)   # 初始化外部可讀計數器
self.inc_ext_counter(addr=1, val=1)   # 遞增外部可讀計數器
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 寫入可從 Python 端讀取的計數器暫存器（`s_core_w1` / `s_core_w2`） |
| `addr` | 1 或 2 |
| 使用情境 | 從 Python 端監控程式執行進度（shot counter） |

---

## 5. 記憶體操作

### `read_dmem(dst, addr)` / `write_dmem(addr, src)`

```python
self.write_dmem(addr=0, src=42)       # 把 42 寫進 data memory[0]
self.read_dmem(dst="my_reg", addr=0)  # 把 data memory[0] 讀進暫存器
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 存取 tProc 的 data memory（可用作暫存陣列） |
| `addr` | 記憶體位址（int 或暫存器名稱） |
| `src` / `dst` | 立即值或暫存器名稱 |

---

### `read_wmem(name)` / `write_wmem(name)`

```python
self.read_wmem("env_ge_gauss")        # 從波形記憶體讀到暫存器
# （修改波形參數）
self.write_wmem("env_ge_gauss")       # 寫回波形記憶體
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 在執行期動態修改波形參數（振幅、頻率等） |
| `name` | 波形名稱，由 `list_pulse_waveforms()` 取得 |
| 使用情境 | 執行時才決定 pulse 參數的進階實驗（如 closed-loop calibration） |

---

## 6. 時序控制

### `delay_auto(t=0, gens=True, ros=True, tag=None)`

```python
self.delay_auto(t=0.02)           # 等到最後一個 pulse/readout 結束再 +0.02 µs
self.delay_auto(t=0.05, tag="waiting")
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 將時間基準推進到「所有已排程事件結束的時刻 + t」 |
| `t` | 額外延遲（µs），預設 0 |
| `gens` | 是否考慮 generator pulse 的結束時間 |
| `ros` | 是否考慮 readout window 的結束時間 |
| `tag` | 賦予這個時間點一個名稱，之後可用 `get_time_param()` 取得 |

> **最常用的時序指令**，大多數程式只需要這一個。

---

### `delay(t, tag=None)`

```python
self.delay(t=1.0, tag="qubit_reset")
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 無條件將時間基準**遞增** t µs |
| 與 `delay_auto` 的差異 | `delay_auto` 先找最晚結束的事件再加 t；`delay` 直接在目前時間基準上加 t |

---

### `wait(t, tag=None)`

```python
self.wait(t=2.0)
```

| 項目 | 說明 |
| --- | --- |
| 用途 | **暫停 tProc 執行**，直到時間基準 + t 到達（不會推進時間基準本身） |
| 使用情境 | 確保回饋測量的 readout window 已結束後再讀取結果 |

---

## 7. 波形與脈衝

這些方法主要在 `_initialize()` 中呼叫，用來宣告脈衝波形。

### `add_gauss(ch, name, sigma, length, even_length)`

```python
self.add_gauss(ch=cfg["qb_ch"], name="env_ge_gauss",
               sigma=cfg["sigma_ge"], length=5*cfg["sigma_ge"],
               even_length=True)
```

建立 Gaussian envelope，存入波形記憶體。

---

### `add_cosine(ch, name, length, even_length)`

```python
self.add_cosine(ch=cfg["qb_ch"], name="env_ge_cos",
                length=cfg["sigma_ge"], even_length=True)
```

建立 raised-cosine envelope。

---

### `add_DRAG(ch, name, sigma, length, delta, alpha, even_length)`

```python
self.add_DRAG(ch=cfg["qb_ch"], name="env_ge_drag",
              sigma=cfg["sigma_ge"], length=5*cfg["sigma_ge"],
              delta=cfg["qb_freq_ge"]-cfg["qb_freq_ef"],
              alpha=cfg["drag_alpha"], even_length=True)
```

建立 DRAG envelope，用來抑制 leakage 到 |f⟩ 態。

---

### `add_pulse(ch, name, style, ...)`

```python
self.add_pulse(ch=cfg["qb_ch"], name="qb_pulse",
               style="arb", envelope="env_ge_gauss",
               freq=cfg["qb_freq_ge"], phase=0, gain=cfg["qb_gain_ge"])
```

| `style` | 說明 |
| --- | --- |
| `"const"` | 方波，需指定 `length` |
| `"arb"` | 任意波形，需指定 `envelope` |
| `"flat_top"` | 平頂波（中間方波 + 兩端 envelope），需指定 `envelope` 與 `length` |

---

### `pulse(ch, name, t)`

```python
self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
```

在 `_body()` 中排程播放一個脈衝。`t=0` 表示「在目前時間基準立即播放」。

---

## 8. 子程序（Subroutine）

### `call(label)` / `ret()`

```python
# 定義子程序（在 _body 末段）
self.label("my_subroutine")
self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
self.delay_auto(t=0.02)
self.ret()

# 呼叫子程序
self.call("my_subroutine")
```

| 項目 | 說明 |
| --- | --- |
| 用途 | 避免重複程式碼；`call` 儲存程式計數器並跳到子程序，`ret` 跳回 |
| 注意 | `call`/`ret` 是 tProc v2 的硬體指令，不是 Python 函式呼叫 |

---

## 9. 其他工具指令

### `nop()`

```python
self.nop()
```

空操作，浪費一個 tProc 時鐘週期。偶爾用於精確時序對齊。

---

### `end()`

```python
self.end()
```

結束程式執行（實作為無限迴圈，因為 tProc v2 沒有真正的 END 狀態）。框架通常自動處理，不需要手動呼叫。

---

## 10. 使用場景速查表

| 需求 | 推薦方法 |
| --- | --- |
| 掃描 gain / 頻率 / 時間（sweep 軸） | `add_loop()` + `QickSweep1D()` |
| 在 `_body` 中重複某段 pulse N 次 | `open_loop(N)` ... `close_loop()` |
| 主動重置（active reset） | `read_and_jump()` |
| 根據測量結果條件跳躍 | `read_input()` + `cond_jump()` |
| 程式內通用計數器 | `write_reg()` + `inc_reg()` + `cond_jump()` |
| 確保 readout 完成再讀值 | `wait_auto(ros=True)` |
| 確保 readout 完成後重新同步 tProc 時間軸 | `wait_auto(ros=True)` + `resync()` |
| 一般 pulse 間隔 | `delay_auto()` |
| Runtime 分支（dispatch loop）中的 pulse 間隔 | `delay(固定值)` ← 不用 `delay_auto` |
| 動態修改 pulse 波形 | `read_wmem()` → 修改 → `write_wmem()` |
| 重複使用的 pulse 序列 | `label()` + `call()` + `ret()` |

---

## 附錄：open_loop / close_loop 完整編譯示意

```
# open_loop(3, name="iter_loop") 展開：
iter_loop = 0
LABEL: iter_loop

# --- loop body ---
PULSE qb_ch qb_pulse @ t=0
DELAY_AUTO 0.02 µs

# close_loop() 展開：
JUMP IF NZ (iter_loop + 1 → iter_loop) < 3  →  LABEL: iter_loop
# 即：iter_loop++，若仍 < 3 就跳回
```

相比之下，Python `for _ in range(3)` 展開為：

```
PULSE qb_ch qb_pulse @ t=0
DELAY_AUTO 0.02 µs
PULSE qb_ch qb_pulse @ t=0
DELAY_AUTO 0.02 µs
PULSE qb_ch qb_pulse @ t=0
DELAY_AUTO 0.02 µs
```

程式大小為 O(N)，N 越大程式越長；`open_loop` 則固定為 O(1)。

---

## 11. 補充：`delay(t) + delay_auto(t=0)` 模式分析

這個組合**等同於只寫 `delay(t)`**，`delay_auto(t=0)` 是 no-op。

原因：`delay(t)` 執行後，compile-time gen_timestamp 被減回 ≈0。
`delay_auto(t=0)` 計算 `max(0) + 0 = 0` → 編譯成 `inc_ref #0`（什麼都不做）。

| 組合 | 等效結果 |
| --- | --- |
| `delay(pulse_len) + delay_auto(t=0)` | `delay(pulse_len)`（少了 gap，通常錯誤） |
| `delay(pulse_len) + delay_auto(t=gap)` | `delay(pulse_len) + delay(gap)` = `delay(slot)` |
| `delay(slot)` | 最簡潔，直接用即可 |

**結論**：在 runtime dispatch 分支中，直接用 `delay(pulse_len + gap)` 是最清晰正確的寫法。

---

## 12. 實戰移植：Active Reset（v1 RAveragerProgram → v2 AveragerProgramV2）

### 原版 v1 程式流程

```
initialize():
  regwi page r_gain2 ← start          # 掃描起始 gain
  regwi page r_thresh ← threshold × readout_length

body():
  mathi r_gain ← r_gain2 + 0          # 把 sweep gain 複製到 pulse 暫存器
  pulse(qubit_ch)                      # 探測 pulse（sweeping gain）
  sync_all(0.05 µs)
  measure(res_ch, adcs)                # 第一次量測（為 active reset 用，不 wait）
  wait_all(200 clocks)                 # 等 ADC 積分完成
  read(ch=0, "lower", reg=2)           # 把 ADC I 值讀入暫存器 2
  read(ch=0, "upper", reg=3)
  condj(reg2 < r_thresh, 'after_reset')# 若 I < threshold → |g⟩ → 跳過 pi
  regwi page r_gain ← pi_gain         # 設 pi pulse gain
  pulse(qubit_ch)                      # pi pulse 把 |e⟩ 打回 |g⟩
  label('after_reset')
  sync_all(1 µs)
  measure(res_ch, adcs, wait=True, syncdelay=relax_delay)  # 第二次量測（正式資料）

update():
  mathi r_gain2 ← r_gain2 + step      # 遞增 gain → next sweep point
```

---

### v1 → v2 指令對照表

| v1 | v2 | 說明 |
| --- | --- | --- |
| `RAveragerProgram` | `AveragerProgramV2` | 基礎框架 |
| `initialize()` | `_initialize(cfg)` | |
| `body()` | `_body(cfg)` | |
| `update()` + `mathi` gain sweep | `add_loop("gainloop", steps)` + `QickSweep1D` | v2 的 sweep 由框架管理 |
| `regwi(page, reg, val)` | `add_reg("name")` + `write_reg("name", val)` | 具名暫存器，不用 page 概念 |
| `mathi(page, dst, src, "+", 0)` | pulse gain 由 QickSweep1D 自動管理，無需手動複製 | v2 直接把 swept 值綁定到 pulse |
| `set_pulse_registers(style="arb", gain=start)` | `add_pulse(gain=QickSweep1D(...))` | gain 成為 swept 參數 |
| `sync_all(cycles)` | `delay_auto(t=µs)` | 對齊所有 ch 並等待 |
| `measure(pulse_ch, adcs, ...)` | `pulse(res_ch) + trigger(ros, t=trig_time)` | v2 拆開 pulse 和 trigger |
| `wait_all(N cycles)` | `wait_auto(t=0, ros=True)` | **tProc 硬體停等** ADC 結束 |
| `read(ch, idx, "lower", reg)` + `condj(...)` | `read_and_jump(ro_ch, "I", threshold, "<", label)` | **v2 最大亮點：一行搞定** |
| `regwi(page, r_gain, pi_gain)` + `pulse(...)` | 預先宣告 `"pi_pulse"`，直接 `pulse(name="pi_pulse")` | 不需要 runtime 修改 gain |
| `sync_all(1 µs)` | `resync(t=0.05)` 或 `delay_auto(t=1.0)` | |
| `measure(..., wait=True, syncdelay=relax_delay)` | `pulse + trigger` + 外層 `final_delay=relax_delay` | v2 用 program 參數管理 relax |

---

### v2 實作概念（程式骨架）

```python
from qick.asm_v2 import QickSweep1D
from .base_program import BaseProgram

class ActiveResetProgram(BaseProgram):

    def _initialize(self, cfg):
        pfx = cfg.get("prefix", "ge")

        # ── 宣告 gen / readout ──────────────────────────────────────────
        self.setup_resonator(cfg, prefix=pfx)
        self.setup_qubit_gen(cfg, prefix=pfx)

        # ── Gain sweep：probe pulse 的 gain 是 swept 參數 ───────────────
        # add_loop 建立掃描軸，QickSweep1D 把 gain 綁定到這個 loop
        self.add_loop("gainloop", cfg["steps"])
        probe_gain = QickSweep1D("gainloop", cfg["start"], cfg["stop"])

        self.add_gauss(
            ch=cfg["qb_ch"], name="env_probe",
            sigma=cfg["sigma_ge"], length=cfg["sigma_ge"] * 4,
            even_length=True,
        )
        self.add_pulse(
            ch=cfg["qb_ch"], name="probe_pulse",
            style="arb", envelope="env_probe",
            freq=cfg["qb_freq_ge"], phase=0,
            gain=probe_gain,          # ← swept
        )

        # ── Pi pulse：固定 gain，用於 active reset ─────────────────────
        self.add_pulse(
            ch=cfg["qb_ch"], name="pi_pulse",
            style="arb", envelope="env_probe",
            freq=cfg["qb_freq_ge"], phase=0,
            gain=cfg["pi_gain_ge"],   # ← 固定
        )

    def _body(self, cfg):

        # ── 送 readout config ───────────────────────────────────────────
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # ── 探測 pulse（gain 由 gainloop 控制）──────────────────────────
        self.pulse(ch=cfg["qb_ch"], name="probe_pulse", t=0)
        self.delay_auto(t=0.05)

        # ── 第一次量測：用於 active reset 判斷 ──────────────────────────
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        # wait_auto(ros=True)：tProc WAIT 指令，硬體停等直到 ADC 積分視窗結束
        # 這是真正的等待（WAIT 指令），不只是推進 ref_time
        self.wait_auto(t=0, ros=True)

        # ── Active reset：讀取 I 值並條件跳躍 ───────────────────────────
        # read_and_jump = read_input(ro_ch) + cond_jump
        # 讀出的 I 值存入特殊暫存器 s_port_l
        # 若 I < threshold → 量子位元在 |g⟩ → 不需要 reset → 跳到 after_reset
        # 若 I ≥ threshold → 量子位元在 |e⟩ → 繼續執行 pi_pulse
        self.read_and_jump(
            ro_ch=cfg["ro_ch"],
            component="I",
            threshold=int(cfg["threshold"]),
            test="<",
            label="after_reset",
        )

        # ── Pi pulse（只有 |e⟩ 時才執行）──────────────────────────────
        self.pulse(ch=cfg["qb_ch"], name="pi_pulse", t=0)

        # ── after_reset：對齊時間軸 ────────────────────────────────────
        self.label("after_reset")
        # delay_auto(0.05)：讓兩條 branch（有/無 pi_pulse）共用同一個
        # compile-time max_timestamp，只等 pi_pulse 長度 + 50 ns。
        # 不使用 resync()：resync 可能引入過長等待，使 qubit 自然 relax
        # 到 |g⟩，讓 active reset 的意義消失。
        self.delay_auto(t=0.05)

        # ── 第二次量測：reset 結果驗證（非正式 Rabi 資料） ─────────────
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
```

**Experiment class（caller 端）**：

```python
prog = ActiveResetProgram(
    soccfg,
    reps=cfg["reps"],
    final_delay=cfg["relax_delay"],   # relax delay 由框架的 outer loop 管理
    cfg=cfg,
)
iq_list = prog.acquire(soc, rounds=py_avg)

# 兩次 trigger 的資料結構：
#   iq_list[0][0]  →  trigger 1：探測 pulse 後的量測（實際 Rabi 資料）
#   iq_list[0][1]  →  trigger 2：active reset 後的驗證（可監測 reset 效率）
#
# 注意：liveplotfun 硬寫 iq_list[0][0]，剛好對應實際資料，
#       但若要同時存取兩次 trigger，需使用 standalone class 自行呼叫 acquire()。
iq_rabi  = iq_list[0][0].dot([1, 1j])   # Rabi 資料
iq_reset = iq_list[0][1].dot([1, 1j])   # reset 後驗證
```

---

### 關鍵設計說明

#### 1. `wait_auto(ros=True)` vs `wait(t)`

```
wait_auto(t=0, ros=True)
  → WAIT 指令，等到硬體 ADC 計時器超過 ro_end_time
  → ro_end_time 在 compile time 由 readout 長度推算
  → 是真正的硬體停等（tProc clock-cycles 停止推進）

wait(t=2.0)
  → WAIT 指令，等到 ref_time + 2.0 µs
  → 需要手動確認等待時間 ≥ readout 長度
```

#### 2. `delay_auto` vs `resync` 在條件分支後

在 `label("after_reset")` 之後選擇 `delay_auto(t=0.05)` 而不是 `resync(t=0.05)`：

| 比較項目 | `delay_auto(t=0.05)` | `resync(t=0.05)` |
| --- | --- | --- |
| 等待時間 | ≈ pi_pulse 長度 + 50 ns（幾十 ns 量級） | 可能等到所有 channel 完全結束，引入額外長等待 |
| Active reset 效果 | qubit 沒有時間自然 relax，reset 有意義 | 若等待時間遠大於 T1，qubit 自然回到基態，reset 失效 |
| compile-time 行為 | max_timestamp 包含 pi_pulse，無 pi 那條 branch 多等幾十 ns，可接受 | 可能呼叫 sync_all 類指令 |

**結論：active reset 後用 `delay_auto(t=0.05)`，不用 `resync()`。**

#### 3. 兩次 trigger 的資料結構與設計

| trigger | 對應 | 說明 |
| --- | --- | --- |
| `iq_list[0][0]` | 第一次量測 | 探測 pulse 後的 Rabi 資料（正式量測） |
| `iq_list[0][1]` | 第二次量測 | Active reset 後的驗證資料（可監測 reset 效率） |

**兩次 trigger 的設計原則**：

- v1 中第一次 measure 是 actual data；第二次 measure（`wait=True, syncdelay=relax_delay`）負責 timing 控制
- v2 中 `liveplotfun` 硬寫 `iq_list[0][0]`，剛好對應 Rabi 資料，可直接用 `BaseExperiment`
- 若需要同時讀兩個 trigger（例如監測 reset 效率），改用 standalone class 自行呼叫 `prog.acquire()` 並分別存取 `iq_list[0][0]` 和 `iq_list[0][1]`
- v2 不需要第二次量測來控制 relax timing：`AveragerProgramV2(final_delay=relax_delay)` 已在每個 body 末端加入等待

#### 4. threshold 校準

v1 的 `regwi(0, r_thresh, cfg["threshold"] * cfg["readout_length"])` 乘上 readout_length
是因為 v1 `read()` 讀到的是整個 window 的 **accumulated sum**。

v2 的 `read_input()` 編譯為 `DPORT_RD`，同樣讀取 avg_buffer 推送到 tProc port 的
**accumulated sum**（`int64`）。因此 threshold 的慣例與 v1 相同：應使用 accumulated 後的整數值。

但 v1 與 v2 的 threshold 數值**不可直接沿用**，原因：
- v1 `readout_length` 單位是 clock cycles；v2 `ro_length` 單位是 µs，換算係數不同
- v2 的 accumulated buffer 是 64-bit（v1 是 32-bit），相同輸入可能對應不同數值
- tProc port 推送的時機和格式可能有差異

**正確做法**：在 v2 環境下跑 SingleShot 實驗，
直接量測 `s_port_l`（I channel）的 histogram，由此決定 threshold 值。

---

### 完整流程 ASM 示意

```
# _body() 展開後的執行順序（runtime）：

send_readoutconfig myro
PULSE qb_ch probe_pulse @ t=0       ← 探測（gain 由 gainloop 掃描）
delay_auto 0.05 µs

PULSE res_ch res_pulse @ t=0        ← 第一次量測 pulse
TRIGGER ro_ch @ trig_time           ← 第一次 ADC 觸發

WAIT until ro_end_time              ← wait_auto(ros=True)：硬體停等 ADC 完成

READ_INPUT ro_ch                    ← 把 I/Q 值存到 s_port_l / s_port_h
TEST s_port_l - threshold
JUMP IF S (< 0) → after_reset       ← I < threshold → |g⟩ → 跳過

PULSE qb_ch pi_pulse @ t=0          ← |e⟩ → pi pulse reset

LABEL after_reset:
delay_auto 0.05 µs                  ← 推進 ref_time（非硬體停等），兩條路徑對齊後繼續

PULSE res_ch res_pulse @ t=0        ← 第二次量測 pulse
TRIGGER ro_ch @ trig_time           ← 第二次 ADC 觸發（正式資料）
```

---

## 13. v1 ↔ v2 時序 API 完整對照

| 功能 | v1 (`RAveragerProgram`) | v2 (`AveragerProgramV2`) | 說明 |
| ---- | ---------------------- | ------------------------ | ---- |
| 推進 ref\_time（非停等） | `sync_all(cycles)` | `delay_auto(t=µs)` | tProc 繼續跑，只推 ref\_time |
| 硬體停等 ADC 完成 | `wait_all(N_cycles)` | `wait_auto(t=0, ros=True)` | tProc 真正暫停到 ADC window 結束 |
| wait 後是否推進 ref\_time | 不推進（需再呼叫 sync\_all） | 自動推進到 readout 結束 | v2 省去一次 sync\_all |
| 條件跳躍（Active Reset） | `condj(page, reg, '<', thresh, label)` | `read_and_jump(ro_ch, "I", thresh, "<", label)` | v2 合一指令 |
| 條件分支後重新對齊 | `sync_all(small_t)` | `delay_auto(t=0.05)` | 短暫 settle，不用 resync（避免過長等待） |
| Relax delay / inter-shot delay | `syncdelay=us2cycles(relax_delay)` 在 `measure()` 內 | `AveragerProgramV2(final_delay=relax_delay)` | v2 由框架管理，body 內不需要 |
| 量測（readout pulse + trigger） | `measure(pulse_ch, adcch, ...)` | `pulse(res_ch) + trigger(ros, pins, t)` | v2 拆開，更靈活 |
| 多次 trigger 資料存取 | `iq_list[ro_ch][trigger_idx]` | `iq_list[ro_ch][trigger_idx]` | 結構相同 |

### 關鍵差異摘要

- **`wait_all` vs `wait_auto`**：兩者都是硬體停等（tProc 真正暫停）。差別在於 `wait_all` 結束後 **不推進** ref\_time，必須再呼叫 `sync_all`；`wait_auto` 結束後 **自動推進** ref\_time 到 readout 結束點。

- **`sync_all` vs `delay_auto`**：兩者都只推進 ref\_time（tProc 繼續跑），不是硬體停等。`sync_all` 是 v1 用語，`delay_auto` 是 v2 對應語法。

- **`delay_auto` vs `resync` 在條件分支後**：`delay_auto(t=0.05)` 只等 ~50 ns（編譯期 max\_timestamp + 50 ns），是正確選擇。`resync` 可能引入更長的等待，導致 qubit 自然 relax 到 ground state，使 active reset 失去意義。

- **Active Reset 完整時序**（v2）：

  ```text
  probe pulse → delay_auto(0.05 µs)
  → res_pulse + trigger
  → wait_auto(ros=True)          # 硬體停等 ADC
  → read_and_jump(threshold, "<", label)
  → [pi_pulse if |e⟩]
  → label("after_reset")
  → delay_auto(0.05 µs)         # 兩路徑對齊
  → res_pulse + trigger          # 第二次量測（驗證 reset）
  # AveragerProgramV2 final_delay = relax_delay（框架管理）
  ```
