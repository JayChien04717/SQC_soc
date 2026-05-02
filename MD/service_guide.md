# Service Usage Guide & Multi-User Coordination

## Quick Start

### 1. Start the server on the lab machine (where QICK is connected)

```python
# server_start.py  — run once on the lab PC
import uvicorn
from # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.service.api import create_app
from # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.tools.system_tool import ExperimentConfig
from # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.config.system_cfg import config_list, DATA_PATH
from # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.calibration.store import CalibrationStore
from # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.backend.qick_backend import QICKBackend

backend = QICKBackend.from_pyro4("192.168.10.82", 8888)
backend.activate()

config_all = ExperimentConfig(config_list)
store      = CalibrationStore("cal_Q1.json")

app = create_app(cal_store=store, config_all=config_all, backend=backend)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# or from shell (bare app — calibration endpoints need the factory above)
uvicorn # MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
.service.api:app --host 0.0.0.0 --port 8000
```

### 2. Connect from any client machine on the same network

```python
import requests, time

SERVER = "http://192.168.10.100:8000"   # lab PC IP
```

---

## Endpoints Reference

| Method | URL | Body / Params | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{ "status": "ok" }` |
| `POST` | `/experiments/run` | `RunRequest` JSON | `{ "job_id": "...", "status": "pending" }` |
| `GET` | `/experiments/{id}/status` | — | `{ "status": "pending\|running\|done\|error" }` |
| `GET` | `/experiments/{id}/result` | — | `ExperimentData` JSON |
| `GET` | `/experiments` | — | list of all jobs |
| `GET` | `/calibrations/{qubit}/params` | — | all stored params |
| `POST` | `/calibrations/{qubit}/set` | `{ "key": ..., "value": ... }` | echo |
| `GET` | `/calibrations/{qubit}/stale` | — | list of stale keys |
| `POST` | `/calibrate/{qubit}/run` | `{ "skip": [...] }` | `{ "job_id": ... }` |

### `RunRequest` schema

```json
{
  "experiment_type": "qubit_spec",
  "config": { "res_freq_ge": 6717, "qb_freq_ge": 2872, ... },
  "py_avg": 10,
  "kwargs": {}
}
```

Valid `experiment_type` strings:
`res_spec`, `qubit_spec`, `time_rabi`, `power_rabi`, `ramsey`, `spin_echo`, `t1`,
`res_spec_ef`, `qubit_spec_ef`, `power_rabi_ef`, `ramsey_ef`, `t1_ef`,
`allxy`, `rb`, `tomography`, `tof`

---

## Basic Client Workflow

```python
import requests, time

SERVER = "http://192.168.10.100:8000"

# 1. Submit
resp = requests.post(f"{SERVER}/experiments/run", json={
    "experiment_type": "qubit_spec",
    "config": run_cfg,       # flat dict from config_all.get_qubit("Q1")
    "py_avg": 10,
})
job_id = resp.json()["job_id"]
print("submitted:", job_id)

# 2. Poll
while True:
    status = requests.get(f"{SERVER}/experiments/{job_id}/status").json()["status"]
    if status == "done":
        break
    if status == "error":
        raise RuntimeError(requests.get(f"{SERVER}/experiments/{job_id}/status").json()["error"])
    time.sleep(3)

# 3. Fetch result
data = requests.get(f"{SERVER}/experiments/{job_id}/result").json()
print("fit_params:", data["fit_params"])
```

### Helper class for notebooks

```python
class RemoteExperiment:
    def __init__(self, server, exp_type, config, py_avg=10, poll_interval=3):
        self.server = server
        self.payload = {"experiment_type": exp_type, "config": config, "py_avg": py_avg}
        self.poll_interval = poll_interval

    def run(self):
        r = requests.post(f"{self.server}/experiments/run", json=self.payload)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        while True:
            s = requests.get(f"{self.server}/experiments/{job_id}/status").json()
            if s["status"] == "done":
                return requests.get(f"{self.server}/experiments/{job_id}/result").json()
            if s["status"] == "error":
                raise RuntimeError(s["error"])
            time.sleep(self.poll_interval)

# Usage
result = RemoteExperiment(SERVER, "t1", run_cfg, py_avg=20).run()
```

---

## Multi-User Coordination

The current service runs jobs **one-at-a-time in the background**. If two users submit simultaneously, their QICK programs will overlap on the hardware and both results will be corrupted.

### The problem

```
User A  POST /experiments/run  → job a1f3  (running on QICK)
User B  POST /experiments/run  → job 9c2e  (also starts on QICK immediately)
                                            ← hardware conflict, both results bad
```

### Solution 1 — Hardware lock (recommended, minimal change)

Add a `threading.Lock` around the QICK run on the server side. Edit `api.py`'s `_run_job`:

```python
# in server_start.py, before create_app()
import threading
_HW_LOCK = threading.Lock()

# monkey-patch into the app after creation, or pass as dependency
# simplest: subclass create_app and wrap _run_job
```

Or edit `# MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA_tuning_guide.md](TWPA_tuning_guide.md) | AI-TWPA-C 完整校正操作步驟（對應官方 Fig. 1.1 流程圖） |
| [TWPA_gain_analyze_explained.md](TWPA_gain_analyze_explained.md) | `TWPAGain.analyze()` 計算流程逐行說明 |
| [TWPA_official_notebook_flow.md](TWPA_official_notebook_flow.md) | 官方 AI-TWPA-C scoring notebook 流程說明（xr.Dataset 格式） |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK Qubit Measurement GUI 產品設計規劃（量測工程師 + PM 雙重視角） |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/` 套件架構說明（給 Claude Code 使用） |
| [../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md](../# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
/CHECKPOINT.md) | `# s015_RB_asm — open_loop 重構備忘錄

## 改動動機

原版用手動 label + inc_reg + jump 實作迴圈，還需要在 dmem 末尾放 sentinel（END code = 7）偵測終止。
改用 QICK 內建的 `open_loop` / `close_loop` 後，迴圈計數由硬體管理，更簡潔且符合 QICK v2 慣用法。

---

## 改動對照

### 1. `_initialize`

```python
# 舊版
self.add_reg("gate_idx")   # 手動分配迴圈計數器
self.add_reg("gate_code")

# 新版
self.add_reg("gate_code")  # gate_idx 由 open_loop 自動分配，不需手動宣告
```

### 2. `compile_datamem`

```python
# 舊版：末尾加 sentinel
codes = [_GATE_CODES[g] for g in gate_seq] + [_END_CODE]  # _END_CODE = 7

# 新版：純 gate code，不需 sentinel
codes = [_GATE_CODES[g] for g in gate_seq]
```

### 3. `_body` 迴圈結構

```
# 舊版
write_reg  gate_idx = 0
LABEL dispatch_loop:
    read_dmem gate_code ← gate_idx
    cond_jump seq_done  if gate_code == 7    ← sentinel 偵測
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
    inc_reg gate_idx + 1
    jump dispatch_loop
LABEL seq_done:
    measure
```

```
# 新版
open_loop(N, "gate_idx")                     ← gate_idx 自動從 0 遞增到 N-1
    read_dmem gate_code ← gate_idx
    cond_jump GATE_I    if gate_code == 0
    ...
LABEL POST_GATE:
close_loop()                                 ← inc gate_idx，未完成則跳回
measure
```

### 4. 移除的常數

```python
# 舊版有，新版移除
_END_CODE = 7
```

---

## 改動總結

| 項目 | 舊版 | 新版 |
|---|---|---|
| gate_idx register | 手動 `add_reg("gate_idx")` | `open_loop` 自動分配 |
| dmem sentinel | 末尾附加 `_END_CODE = 7` | 不需要 |
| 迴圈初始化 | `write_reg("gate_idx", 0)` | `open_loop(N, "gate_idx")` |
| 迴圈跳回 | `inc_reg` + `jump("dispatch_loop")` | `close_loop()` |
| END 偵測 | `cond_jump("seq_done", ..., arg2=7)` | 完全移除 |
| pmem 指令數 | 多 3 條（write_reg + END check + label） | 較少 |

---

## open_loop / close_loop 行為說明

`open_loop(n, name)` 展開成：
```
REG_WR  name ← 0          # 初始化計數器
LABEL   name:              # 迴圈標籤
```

`close_loop()` 展開成：
```
TEST    name - (n-1)       # 比較計數器與上限
JUMP    name  if NZ  WR name ← name + 1   # 未完成：遞增並跳回
```

計數器從 0 數到 n-1，共執行 n 次，退出時不再跳回。

---

## 注意事項

- `delay()` 而非 `delay_auto()`：dispatch branch 內仍需用 `delay(slot)`，原因不變——`delay_auto` 在 compile-time 累積時間軸，在 runtime branch 中會產生錯誤的 inc_ref 值。
- `open_loop` 內部用的 register 名稱與 `name` 參數相同（本例為 `"gate_idx"`），`read_dmem("gate_code", "gate_idx")` 可直接用此名稱做間接定址。
- 每次 acquire 仍需 recompile（`compile_datamem` 隨 gate_seq 變動），pmem 恆定的優勢不變。
` 套件建置進度（CP1–CP11，已完成） |
/service/api.py` directly — add one lock around the experiment `.run()` call:

```python
_HW_LOCK = threading.Lock()          # module-level, next to _JOBS

def _run_job(job_id, exp_type, cfg, py_avg, kwargs):
    _set_job(job_id, status="queued")
    with _HW_LOCK:                   # blocks until hardware is free
        _set_job(job_id, status="running", started_at=datetime.now().isoformat())
        try:
            cls = _resolve_experiment(exp_type)
            expt = cls(cfg, backend=backend)
            result = expt.run(py_avg, **kwargs)
            ...
```

Now jobs queue automatically. Clients see `"queued"` while waiting, `"running"` when on hardware.

### Solution 2 — Per-qubit locks

If users work on different qubits and the hardware is truly independent per-channel:

```python
_QB_LOCKS: dict[str, threading.Lock] = {}

def _get_qb_lock(cfg: dict) -> threading.Lock:
    qb = cfg.get("name", "default")
    if qb not in _QB_LOCKS:
        _QB_LOCKS[qb] = threading.Lock()
    return _QB_LOCKS[qb]
```

This allows Q1 and Q2 experiments to run concurrently while preventing two Q1 jobs from overlapping.

### Solution 3 — Priority booking via CalibrationStore

For labs where scheduled runs are preferred over first-come-first-served, add a simple reservation field to the store:

```python
# Client A reserves a slot
requests.post(f"{SERVER}/calibrations/Q1/set", json={
    "key": "__reserved_by__", "value": "Alice"
})

# Client B checks before submitting
params = requests.get(f"{SERVER}/calibrations/Q1/params").json()
if "__reserved_by__" in params:
    print("Q1 is reserved by", params["__reserved_by__"])
else:
    # safe to submit
    ...
```

This is a soft convention, not enforced — useful for team communication, not race-condition safety.

---

## Practical Lab Protocol (team of 2–4 people)

1. **One person starts the server** at the beginning of the day and keeps it running.
2. **Check `/health`** before submitting to confirm the server is up.
3. **Check `/experiments`** to see if a job is currently `running` — don't submit while someone else is active.
4. **Use descriptive qubit names** in your config (`"name": "Q1"`) so the job list shows who is using which qubit.
5. **Fetch your result immediately** after `done` — the in-memory job registry is lost on server restart.
6. After a run, **write key results back** to `CalibrationStore` via `POST /calibrations/{qubit}/set` so others see the latest params.

```python
# After fitting qubit spec
requests.post(f"{SERVER}/calibrations/Q1/set", json={
    "key": "qb_freq_ge", "value": 4502.3
})
```

---

## Checking Stale Parameters Before a Run

Before starting a long calibration, check what needs re-running:

```python
stale = requests.get(f"{SERVER}/calibrations/Q1/stale").json()["stale_keys"]
print("needs recal:", stale)
# → ['qb_freq_ge', 'pi_gain_ge']
```

Then run only the stale steps via AutoCalibrate:

```python
requests.post(f"{SERVER}/calibrate/Q1/run", json={
    "skip": ["spin_echo", "ss_opt"]   # skip steps that are still fresh
})
```
