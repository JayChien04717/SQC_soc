# s015_RB_asm — open_loop 重構備忘錄

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
