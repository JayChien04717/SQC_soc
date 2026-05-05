# RB pmem 溢出修復指南

## 錯誤訊息

```
RuntimeError: compiled program uses 4161 words of pmem,
              but the size of that tProc memory is only 4096
```

---

## 根本原因

`RBProgram._body()` 目前在 **compile time** 展開整個 gate 序列：

```python
# s015_Single_qubit_RB.py — 現在的寫法
for gate in cfg["gate_seq"]:          # compile-time 迴圈
    self.pulse(...)                    # 每個 gate → 約 7-10 條 pmem 指令
    self.delay_auto(...)
```

pmem 用量 ≈ `len(gate_seq) × ~8 words + 固定開銷（~100 words）`。

| max_circuit_depth | 平均 gate 數 (×3 basis) | 估計 pmem |
|---|---|---|
| 100 | 300 | ~2 500 words (安全) |
| 140 | 420 | ~3 460 words (安全) |
| **160** | **480** | **~3 940 words (接近上限)** |
| **170** | **510** | **~4 180 words (溢出)** |

---

## 解法一：短期 — 降低 max_circuit_depth

在 `Auto_RB.run()` 呼叫時，確保 `max_circuit_depth` 不超過安全值：

```python
# 安全上限約 150，視 delta_clifford 而定
rb_ref.run(
    py_avg=...,
    max_circuit_depth=150,   # ← 從原來的值調低
    delta_clifford=...,
    number_sample=...,
)
```

**缺點**：無法跑到高 Clifford depth，限制了 EPC 擬合準確度。

---

## 解法二：根本解 — DMEM + open_loop 動態分發

將 gate 序列存入 **data memory (DMEM)**，pmem 只存固定大小的分發邏輯，
不隨序列長度增長。架構如 [rb_asmv2_register.md](rb_asmv2_register.md) 所述。

### pmem 用量比較

| 方法 | pmem 指令數 |
|---|---|
| 現在（compile-time unroll） | ~8 × N_gates |
| DMEM dispatch（本方案） | **固定 ~80 words**（與 depth 無關）|

### gate code 對照

```python
_GATE_CODES = {
    "I":    0,
    "X":    1,
    "Y":    2,
    "X/2":  3,
    "-X/2": 4,
    "Y/2":  5,
    "-Y/2": 6,
}
```

### 重構後的 RBProgram

```python
# s015_Single_qubit_RB.py — 修改後版本

_GATE_CODES = {
    "I":    0,
    "X":    1,
    "Y":    2,
    "X/2":  3,
    "-X/2": 4,
    "Y/2":  5,
    "-Y/2": 6,
}


class RBProgram(BaseProgram):
    """QICK program — DMEM dispatch, pmem 與序列長度無關。"""

    def _initialize(self, cfg):
        prefix = cfg.get("prefix", "ge")
        self.setup_resonator(cfg, prefix=prefix)
        self.setup_qubit_gen(cfg, prefix=prefix)
        self.setup_standard_gates(cfg, prefix=prefix)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
        self.add_reg("gate_code")   # 用於存放從 DMEM 讀出的 gate 編號

    def compile_datamem(self):
        """將 gate_seq 轉成 gate code 整數陣列寫入 DMEM。"""
        gate_seq = self.cfg["gate_seq"]
        return [_GATE_CODES[g] for g in gate_seq]

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        pfx = cfg.get("prefix", "ge")
        n = len(cfg["gate_seq"])
        ch = cfg["qb_ch"]

        # 每個 gate slot 的持續時間：pulse 長度 (sigma×5) + inter-gate buffer
        # 必須是 compile-time 常數；loop 內不能用 delay_auto
        sigma = float(cfg[f"sigma_{pfx}"])
        gate_slot = sigma * 5 + 0.01   # µs

        # open_loop 自動分配 "gate_idx" register，從 0 數到 n-1
        self.open_loop(n, "gate_idx")

        # 從 DMEM[gate_idx] 讀出當前 gate 的 code
        self.read_dmem("gate_code", "gate_idx")

        # --- 分發邏輯：依序比對 gate_code ---
        # code 0: Identity
        self.cond_jump("GATE_X", "gate_code", "NZ")
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 1: X180
        self.label("GATE_X")
        self.cond_jump("GATE_Y", "gate_code", "NZ", op="-", arg2=1)
        self.pulse(ch=ch, name=f"x180_{pfx}", t=0)
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 2: Y180
        self.label("GATE_Y")
        self.cond_jump("GATE_X90", "gate_code", "NZ", op="-", arg2=2)
        self.pulse(ch=ch, name=f"y180_{pfx}", t=0)
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 3: X90
        self.label("GATE_X90")
        self.cond_jump("GATE_MX90", "gate_code", "NZ", op="-", arg2=3)
        self.pulse(ch=ch, name=f"x90_{pfx}", t=0)
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 4: -X90
        self.label("GATE_MX90")
        self.cond_jump("GATE_Y90", "gate_code", "NZ", op="-", arg2=4)
        self.pulse(ch=ch, name=f"x90m_{pfx}", t=0)
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 5: Y90
        self.label("GATE_Y90")
        self.cond_jump("GATE_MY90", "gate_code", "NZ", op="-", arg2=5)
        self.pulse(ch=ch, name=f"y90_{pfx}", t=0)
        self.delay(gate_slot)
        self.jump("POST_GATE")

        # code 6: -Y90（最後一個 branch，不需要 cond_jump）
        self.label("GATE_MY90")
        self.pulse(ch=ch, name=f"y90m_{pfx}", t=0)
        self.delay(gate_slot)

        self.label("POST_GATE")
        self.close_loop()           # 遞增 gate_idx，若未完成則跳回迴圈頂

        self.delay_auto(0.05)
        self.measure(cfg)
```

---

## 注意事項

### 1. `delay` vs `delay_auto`

loop 內的 branch 中 **必須用 `delay(t)`**，禁止 `delay_auto`。  
原因：`delay_auto` 在 compile time 計算 "最後一個 pulse 的結束時間"，
在 runtime branch 中無法正確追蹤時序，會產生錯誤的 `inc_ref` 值。

### 2. `sigma` 必須是純 float

`gate_slot = float(cfg[f"sigma_{pfx}"]) * 5 + 0.01` 必須是 compile-time 常數。
若 `sigma` 是 `QickParam`（掃描參數），此方案不適用。

### 3. compile_datamem 每次仍重新編譯

每次建立 `RBProgram(...)` 時，DMEM 內容（gate 序列）會隨 `gate_seq` 更新，
但 pmem（程式本身）保持固定大小，不會再溢出。

### 4. DMEM 大小限制

`compile_datamem` 回傳的陣列長度不能超過 `tproccfg["dmem_size"]`（通常 ≥ 8192）。
對應的 gate 序列上限約為 8192 個 gate（等效 ~2730 Clifford），
遠超實驗所需。

---

## 修改範圍總結

| 項目 | 原版 | 修改後 |
|---|---|---|
| `_initialize` | 無 gate_code register | `add_reg("gate_code")` |
| `compile_datamem` | 不覆寫（回傳 None） | 回傳 gate code 整數 list |
| `_body` 迴圈 | compile-time `for gate in gate_seq` | runtime `open_loop` + `read_dmem` + `cond_jump` |
| pmem 用量 | O(N_gates) → 溢出 | O(1) 固定 ~80 words |

詳細 open_loop / close_loop 行為見 [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md)。
