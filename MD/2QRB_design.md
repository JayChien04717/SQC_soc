# 2-Qubit Randomized Benchmarking — Design Document

> 本文對齊 `s015` / `RB_generator.py` 的設計語言，並以 **symplectic formalism** 從頭建構 2Q Clifford group。

---

## 1. 檔案結構

```
scrip/
├── RB_generator.py           ← 已有 (1Q)
├── RB_generator_2q.py        ← 新增：2Q Clifford 生成器
├── s015_Single_qubit_RB.py   ← 已有
└── s016_Two_qubit_RB.py      ← 新增：2Q RB 實驗 + QICK program
```

---

## 2. Gate Layer 格式規範

與 comfit 一致，sequence 是 `List[List[Dict]]`：

```python
# 型別定義
GateDict  = dict              # {'Name': str | list[str], 'gate': str}
Layer     = list[GateDict]    # 同時執行的 gates（一個 moment）
Sequence  = list[Layer]       # 完整 gate sequence

# 1Q gate moment (兩個 qubit 同時執行不同 gate)
[{'Name': 'Q1', 'gate': 'X/2'}, {'Name': 'Q2', 'gate': 'Y/2'}]

# 2Q gate moment (CZ 涉及兩個 qubit，Name 用 list)
[{'Name': ['Q1', 'Q2'], 'gate': 'CZ'}]

# Identity moment (只等待，不送 pulse)
[{'Name': 'Q1', 'gate': 'I'}, {'Name': 'Q2', 'gate': 'I'}]
```

一個完整 2Q RB sequence 範例（length=2 Cliffords + recovery）：

```python
sequence = [
    # ─── Clifford 0 ───────────────────────────────────────
    [{'Name': 'Q1', 'gate': 'X/2'}, {'Name': 'Q2', 'gate': 'Y/2'}],
    [{'Name': ['Q1', 'Q2'], 'gate': 'CZ'}],
    [{'Name': 'Q1', 'gate': '-Y/2'}, {'Name': 'Q2', 'gate': 'X/2'}],
    # ─── Clifford 1 ───────────────────────────────────────
    [{'Name': 'Q1', 'gate': 'I'}, {'Name': 'Q2', 'gate': 'X/2'}],
    # ─── Recovery Clifford ────────────────────────────────
    [{'Name': 'Q1', 'gate': 'X/2'}, {'Name': 'Q2', 'gate': 'X/2'}],
    [{'Name': ['Q1', 'Q2'], 'gate': 'CZ'}],
    [{'Name': 'Q1', 'gate': 'Y/2'}, {'Name': 'Q2', 'gate': '-X/2'}],
]
```

---

## 3. Symplectic Formalism

### 3.1 基本編碼

n=2 qubits 的 Pauli 算子以 4-bit 二進位向量表示：

```
v = (x₁, x₂ | z₁, z₂)  ∈ GF(2)⁴

X₁ → (1,0, 0,0)
X₂ → (0,1, 0,0)
Z₁ → (0,0, 1,0)
Z₂ → (0,0, 0,1)
Y₁ → X₁Z₁ → (1,0, 1,0)   (phase 由 p vector 另行記錄)
```

### 3.2 Clifford 的 Symplectic 表示

一個 2Q Clifford C 的作用：

```
C Pᵢ C† = (-1)^(pᵢ) Pⱼ    (up to phase)
```

被兩個物件完全描述：
- **F** ∈ GF(2)⁴ˣ⁴：symplectic matrix，描述 Pauli 怎麼被映射  
- **p** ∈ GF(2)⁴：phase vector，記錄 ±1 phase

Symplectic 條件（保持 commutativity structure）：

```
F^T Ω F = Ω  (mod 2)

    ⎡ 0  0  1  0 ⎤
Ω = ⎢ 0  0  0  1 ⎥   (symplectic form)
    ⎢ 1  0  0  0 ⎥
    ⎣ 0  1  0  0 ⎦
```

### 3.3 Generator Gates 的 Symplectic Matrix

以下是 native gates 在 symplectic 表示下的 F 矩陣：

```
行排列: x₁, x₂, z₁, z₂

H₁ (Hadamard Q1):  X₁↔Z₁
    F = [[0,0,1,0],
         [0,1,0,0],
         [1,0,0,0],
         [0,0,0,1]]

S₁ (Phase Q1):  X₁→Y₁, Z₁→Z₁
    F = [[1,0,0,0],
         [0,1,0,0],
         [1,0,1,0],
         [0,0,0,1]]

CZ (on Q1,Q2):  X₁→X₁Z₂, Z₁→Z₁, X₂→Z₁X₂, Z₂→Z₂
    F = [[1,0,0,0],
         [0,1,0,0],
         [0,1,1,0],
         [1,0,0,1]]

H₂, S₂: 類比 H₁, S₁，作用在第二個 qubit 的 index 上
```

### 3.4 Clifford 群的乘法

兩個 Clifford (F₁, p₁) 和 (F₂, p₂) 的複合：

```python
# C₂ followed by C₁ (C₁ C₂)
F_composed = F₁ @ F₂  (mod 2)
p_composed = p₁ + F₁ @ p₂  (mod 2)   # phase 的傳播需要額外的 correction term
```

完整的 phase update 需要考慮 Y = iXZ 的 phase：

```python
def compose_cliffords(F1, p1, F2, p2):
    """Compose Clifford (F2 applied first, then F1)."""
    F  = (F1 @ F2) % 2
    # Phase: p1 acts on F2's output, plus correction from Y-type Paulis
    p  = (p1 + F1 @ p2 + phase_correction(F1, F2)) % 2
    return F, p
```

---

## 4. 2Q Clifford 群的結構（11520 個元素）

依照 CZ gate 數量最少化分成 4 個 class：

| Class | 描述 | CZ 數 | 元素數 | 備註 |
|-------|------|--------|--------|------|
| 1 | SQ ⊗ SQ | 0 | 576 | 24 × 24 個 1Q Clifford 積 |
| 2 | CNOT-like | 1 | 5040 | 一個 CZ 糾纏 |
| 3 | iSWAP-like | 2 | 4320 | 兩個 CZ |
| 4 | SWAP-like | 3 | 1584 | 三個 CZ |
| **總計** | | | **11520** | |

每個 class 的 gate 模板：

```
Class 1:  [SQ₁ ⊗ SQ₂]

Class 2:  [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂]

Class 3:  [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂]

Class 4:  [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂] – CZ – [SQ₁ ⊗ SQ₂]
```

其中每個 `SQ₁ ⊗ SQ₂` 是兩個獨立的 1Q Clifford（各 24 種）。

---

## 5. `RB_generator_2q.py` 設計

### 5.1 Native Gate Set

對齊 1Q 的命名慣例：

```python
# 單 qubit gates（與 1Q RB 完全相同的字串）
SQ_GATE_NAMES = ['I', 'X/2', '-X/2', 'Y/2', '-Y/2', 'X', 'Y',
                 'X/2-Y/2', ...]   # 24 個 1Q Clifford 展開後的標籤

# 兩 qubit gate
TQ_GATE = 'CZ'
```

### 5.2 Clifford Table 預計算（建議離線做一次）

```python
# ── 預計算步驟（初始化時執行一次）─────────────────────────────────────

def _build_2q_clifford_table():
    """
    Enumerate all 11520 2Q Clifford elements.
    Returns:
        F_table  : ndarray (11520, 4, 4) uint8, symplectic matrices
        p_table  : ndarray (11520, 4)    uint8, phase vectors
        seq_table: list[list[Layer]]     pre-built gate layer sequences
    """
    # Step 1: 枚舉 Class 1 (SQ ⊗ SQ)
    #   -> 24 × 24 = 576 個 (F, p)
    #   -> 每個 decompose 成 1 個 layer [1Q ⊗ 1Q]

    # Step 2: 枚舉 Class 2
    #   -> 對每個 (SQ₁ ⊗ SQ₂) × CZ × (SQ₁ ⊗ SQ₂) 組合
    #   -> 共 24² × 24² = 331776 候選，但只有 5040 個不等價（需過濾重複）

    # Step 3, 4: 類似

    # 最終驗證：確認全部 11520 個 F 都是合法 symplectic 矩陣且互不重複
    ...
    return F_table, p_table, seq_table


def _build_inverse_table(F_table, p_table):
    """
    inverse_table[i] = j  使得  C_j ∘ C_i = Identity
    Clifford inverse: (F, p)^(-1) = (F^(-1 mod 2), F^(-T) p mod 2)
    """
    ...
```

> **Tip**：把 `F_table`, `p_table`, `seq_table` 存成 `.npz` 文件，
> 之後直接 `np.load()` 就好，不需每次重算。

### 5.3 核心函數

```python
def two_qb_rb(
    n_clifford : int,
    n_sample   : int,
    q1         : str = 'Q1',
    q2         : str = 'Q2',
    interleave : str | None = None,   # 2Q IRB 擴展點
    seed       : int | None = None,
    debug      : bool = False,
) -> list[Sequence]:
    """
    Generate n_sample 2Q RB sequences of Clifford length n_clifford.

    Returns
    -------
    list of Sequence
        每個 Sequence = List[Layer] = List[List[Dict]]
        最後一個 Clifford 是 recovery gate.
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_sample):
        layers_all: Sequence = []

        # ── Accumulate Clifford product ──────────────────────────────
        F_acc = np.eye(4, dtype=np.uint8)   # identity symplectic
        p_acc = np.zeros(4, dtype=np.uint8) # identity phase

        for _ in range(n_clifford):
            idx = rng.integers(0, 11520)

            # 累積 Clifford：C_acc ← C_idx ∘ C_acc
            F_acc, p_acc = compose_cliffords(
                F_table[idx], p_table[idx], F_acc, p_acc
            )

            # 加入這個 Clifford 的 gate layers
            layers_all.extend(seq_table[idx])

        # ── Recovery Clifford ─────────────────────────────────────────
        acc_idx       = find_clifford_index(F_acc, p_acc)   # 在 table 中查 C_acc
        inv_idx       = inverse_table[acc_idx]               # 取 C_acc^(-1)
        layers_all.extend(seq_table[inv_idx])

        if debug:
            print(f"  Clifford layers: {len(layers_all)} | "
                  f"recovery idx: {inv_idx}")

        results.append(layers_all)

    return results


def verify_2q_sequence(sequence: Sequence) -> bool:
    """
    Execute the sequence's symplectic matrices and verify product = Identity.
    """
    F = np.eye(4, dtype=np.uint8)
    p = np.zeros(4, dtype=np.uint8)
    for layer in sequence:
        for gd in layer:
            F, p = compose_cliffords(GATE_SYMPLECTIC[gd['gate']], ..., F, p)
    return np.all(F == np.eye(4)) and np.all(p == 0)
```

### 5.4 Layer 建構輔助函數

```python
def _sq_layer(gate_q1: str, gate_q2: str, q1: str, q2: str) -> Layer:
    """Build one simultaneous 1Q gate layer."""
    return [
        {'Name': q1, 'gate': gate_q1},
        {'Name': q2, 'gate': gate_q2},
    ]

def _cz_layer(q1: str, q2: str) -> Layer:
    """Build a CZ gate layer."""
    return [{'Name': [q1, q2], 'gate': 'CZ'}]

def _clifford_to_layers(sq_gates_list: list[tuple[str, str]],
                         include_cz: list[bool],
                         q1: str, q2: str) -> Sequence:
    """
    Convert a Clifford class decomposition to a list of layers.

    Parameters
    ----------
    sq_gates_list : list of (gate_q1, gate_q2) for each 1Q moment
    include_cz    : list of bool, whether to insert CZ after each 1Q moment
    """
    layers = []
    for (g1, g2), add_cz in zip(sq_gates_list, include_cz):
        layers.append(_sq_layer(g1, g2, q1, q2))
        if add_cz:
            layers.append(_cz_layer(q1, q2))
    return layers

# 範例 Class 2 Clifford（1 CZ）：
# sq_gates_list = [('X/2', 'Y/2'), ('-Y/2', 'X/2')]
# include_cz    = [True, False]
# → layers:
#   [{'Name':'Q1','gate':'X/2'}, {'Name':'Q2','gate':'Y/2'}]
#   [{'Name':['Q1','Q2'], 'gate':'CZ'}]
#   [{'Name':'Q1','gate':'-Y/2'}, {'Name':'Q2','gate':'X/2'}]
```

---

## 6. `s016_Two_qubit_RB.py` 設計

### 6.1 Gate Map（對齊 s015 的 `_GEN_TO_QICK`）

```python
_GEN_TO_QICK_2Q = {
    # ── Single qubit gates ───────────────────────────────────────
    'I':    None,
    'X/2':  'x90_{pfx}',
    '-X/2': 'x90m_{pfx}',
    'Y/2':  'y90_{pfx}',
    '-Y/2': 'y90m_{pfx}',
    'X':    'x180_{pfx}',
    'Y':    'y180_{pfx}',
    # ── Two qubit gate ───────────────────────────────────────────
    'CZ':   'cz_{pfx1}_{pfx2}',   # 依照你的 BaseProgram 命名
}
```

### 6.2 RBProgram（QICK Program）

```python
class RBProgram2Q(BaseProgram):

    def _initialize(self, cfg):
        pfx1 = cfg.get('prefix_q1', 'ge')
        pfx2 = cfg.get('prefix_q2', 'ge')
        self.setup_resonator(cfg, prefix=pfx1)   # 或雙 resonator
        self.setup_qubit_gen(cfg, prefix=pfx1)
        self.setup_qubit_gen(cfg, prefix=pfx2)
        self.setup_standard_gates(cfg, prefix=pfx1)
        self.setup_standard_gates(cfg, prefix=pfx2)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name='myro', t=0)

        pfx1 = cfg.get('prefix_q1', 'ge')
        pfx2 = cfg.get('prefix_q2', 'ge')

        for layer in cfg['gate_seq']:     # gate_seq = Sequence = List[Layer]
            for gd in layer:
                name_field = gd['Name']
                gate       = gd['gate']

                if isinstance(name_field, list):
                    # ── 2Q gate ──────────────────────────────────
                    if gate == 'CZ':
                        self.pulse(ch=cfg['cz_ch'],
                                   name=f'cz_{pfx1}_{pfx2}', t=0)
                        self.delay_auto(0.01)
                else:
                    # ── 1Q gate ───────────────────────────────────
                    pfx = pfx1 if name_field == cfg['q1_name'] else pfx2
                    if gate == 'I':
                        self.delay_auto(cfg[f'sigma_{pfx}'] * 5)
                    else:
                        tmpl = _GEN_TO_QICK_2Q[gate]
                        self.pulse(ch=cfg['qb_ch'], 
                                   name=tmpl.format(pfx=pfx), t=0)
                        self.delay_auto(0.01)

            # layer 之間加小 delay（確保不同 qubit 的 gate 在 timing 上分開）
            self.delay_auto(cfg.get('layer_gap', 0.01))

        self.delay_auto(0.05)
        self.measure(cfg)
```

### 6.3 Experiment Class

```python
class TwoQubitRB:
    """
    兩 qubit Randomized Benchmarking（Standard 和 Interleaved）。

    完全對齊 s015 RandomizedBenchmarking 的 API：
        rb = TwoQubitRB(config)
        rb.run(py_avg=10, max_circuit_depth=50, delta_clifford=5, number_sample=20)
        rb.plot('2Q RB')
        rb.saveLabber(qb_idx=1)
    """

    def __init__(self, config):
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup() first.")
        self.soc     = BaseExperiment._soc
        self.soccfg  = BaseExperiment._soccfg
        self.cfg     = config
        self.x       = None
        self.rb_result = None

    def run(
        self,
        py_avg,
        max_circuit_depth,
        delta_clifford,
        number_sample,
        q1: str = 'Q1',
        q2: str = 'Q2',
        interleaved_gate=None,   # 預留 2Q IRB
        seed=None,
        iq_process='abs',
        randomize_depth_order=False,
    ):
        self._iq_process = iq_process
        self.x = np.arange(1, max_circuit_depth, delta_clifford)
        n_depths = len(self.x)

        rng = np.random.default_rng(seed)
        seeds_matrix = [
            [int(rng.integers(0, 2**31)) for _ in range(number_sample)]
            for _ in range(n_depths)
        ]

        depth_indices = np.arange(n_depths)
        if randomize_depth_order:
            rng.shuffle(depth_indices)

        rb_accum = [[None] * number_sample for _ in range(n_depths)]

        for avg_i in tqdm(range(py_avg), desc='Software Average'):
            for idx in tqdm(depth_indices, desc='2Q RB', leave=False):
                depth = self.x[idx]
                for s_i in tqdm(range(number_sample), desc='Samples', leave=False):

                    seqs = two_qb_rb(
                        n_clifford=depth,
                        n_sample=1,
                        q1=q1, q2=q2,
                        interleave=interleaved_gate,
                        seed=seeds_matrix[idx][s_i],
                    )
                    gate_seq = seqs[0]   # List[Layer] = List[List[Dict]]

                    self.cfg['gate_seq'] = gate_seq
                    self.cfg['q1_name']  = q1
                    self.cfg['q2_name']  = q2

                    prog = RBProgram2Q(
                        self.soccfg,
                        reps=self.cfg['reps'],
                        final_delay=self.cfg['relax_delay'],
                        cfg=self.cfg,
                    )
                    iq_list = prog.acquire(self.soc, rounds=1, progress=False)
                    # 2Q readout：需要決定你量的是 joint state 還是 individual qubits
                    iq_data = iq_list[0][0].dot([1, 1j])

                    if avg_i == 0:
                        rb_accum[idx][s_i] = iq_data
                    else:
                        rb_accum[idx][s_i] += iq_data

        self.rb_result = [
            [rb_accum[idx][s_i] / py_avg for s_i in range(number_sample)]
            for idx in range(n_depths)
        ]

    def plot(self, label, color=None, ax=None):
        """Fit and plot: same interface as s015 plot()."""
        # RB decay for 2Q: d = 4 (instead of 2 for 1Q)
        # epc = (1 - p) * (d-1)/d  →  d=4
        ...

    def saveLabber(self, qb_idx, config_all=None, title=None):
        """Save to HDF5/Labber, same interface as s015."""
        ...
```

---

## 7. 2Q IRB 擴展預留

```python
INTERLEAVE_GATES_2Q = {
    'CZ':    (F_CZ, p_CZ, [{'Name': ['Q1', 'Q2'], 'gate': 'CZ'}]),
    'iSWAP': (...),
    # ── 1Q gates 也可以 interleave（對齊 s015）──────────────────
    'X_Q1': (F_X1, p_X1, [{'Name': 'Q1', 'gate': 'X'}]),
}
```

IRB 在 `two_qb_rb` 中只需在每個隨機 Clifford 後插入：

```python
if interleave is not None:
    F_acc, p_acc = compose_cliffords(F_ilv, p_ilv, F_acc, p_acc)
    layers_all.extend(interleave_layers)
```

---

## 8. 實作優先順序

```
Phase 1 — Symplectic engine
  [x] 實作 compose_cliffords(F1, p1, F2, p2)
  [x] 實作 invert_clifford(F, p)
  [x] 驗證 6 個 generator gates 的 F 矩陣正確
  [x] 用 BFS/枚舉確認能生成 11520 個不同元素

Phase 2 — Clifford table
  [x] 枚舉 4 個 class
  [x] 為每個 index 建立 seq_table（gate layers）
  [x] 建立 inverse_table
  [x] verify_2q_sequence() 通過全部 11520 個

Phase 3 — RB_generator_2q.py
  [x] two_qb_rb() 回傳正確格式
  [x] verify_2q_sequence() 通過所有 sample

Phase 4 — s016 integration
  [x] RBProgram2Q._body() 正確 dispatch 1Q/2Q gates
  [x] TwoQubitRB.run() 資料正確累積
  [x] plot() 用 d=4 的 EPC 公式
```

---

## 9. 關鍵公式備查

```
2Q EPC (Error Per Clifford):
    EPC = (d-1)/d × (1 - p_fit)    ,  d = 2^n = 4 for 2Q

RB decay function (same as 1Q):
    S(m) = A × p^m + B

Average gate fidelity (for CZ interleaved):
    r_gate = EPC_irb - EPC_ref × (d-1)/d / ((d-1)/d + 1)
```

---

## 10. 設計決策總結

| 決策 | 選擇 | 理由 |
|------|------|------|
| Gate sequence 格式 | `List[List[Dict]]` | 與 comfit 一致，`Name` 用 list 表示 2Q gate |
| Clifford tracking | Symplectic matrix (GF2) | 精確、不累積數值誤差（純 binary 運算）|
| Recovery lookup | Pre-built inverse table | 和 1Q 的 `inverse_table` 完全相同策略 |
| Clifford table | 離線枚舉 + `.npz` 儲存 | 初始化快，runtime 只查表 |
| EPC formula | `d=4` | 2Q space: d = 2^2 |
| IRB 擴展 | `interleave` 參數 | 對齊 `single_qb_rb` 介面 |
