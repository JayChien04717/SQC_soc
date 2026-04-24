# TWPAGain.analyze() 計算流程說明

對應檔案：[s002e_TWPA_gain.py](../qick_workspace/scrip/s002e_TWPA_gain.py)（第 287–420 行）

---

## 概覽

`analyze()` 接收一個「無泵浦（pump OFF）」的參考量測，對整個 3D gain 資料集打分數，找出最佳操作點，並自動畫圖。

整體流程分為五個階段：

```
原始 S21 資料
    │
    ▼
① 建立 3D xarray（pump_freq × ifbl × frequency）
    │
    ▼
② 除以參考，得到 normalized gain（dB）
    │
    ▼
③ 遮蔽要排除的頻段（freq_exclude）
    │
    ▼
④ 打分數（scoring）
    │
    ▼
⑤ 找最佳操作點 + 畫圖
```

---

## 各步驟詳細說明

### ① 建立 3D Gain xarray

```python
gain = self._build_gain_xarray()  # shape: (pump_freq × ifbl × frequency)
```

`run()` 在每個 pump 頻率跑一次 `TWPAFlux`，每次結果是一個 2D slice `(ifbl × frequency)`，
`_build_gain_xarray()` 用 `xr.concat` 把所有 slice 沿 `pump_freq` 軸疊成 3D。

---

### ② Normalize（歸一化）

```python
ref_s21 = reference._build_s21_xarray()          # shape: (ifbl × frequency)
smallest_flux = float(np.abs(ref_s21["ifbl"]).min())
ref_row = ref_s21.sel(ifbl=smallest_flux).drop_vars("ifbl")  # shape: (frequency,)
gain_normalized = gain / ref_row                 # broadcast 到全部 pump_freq & ifbl
```

**目的**：消除系統背景（cable loss、amplifier response 等）的影響，只留下 TWPA 帶來的真實 gain。

- 參考用的是 `ifbl` 最接近 0 的那一行（零磁通量、無泵浦），代表 TWPA 完全不工作的基準線。
- `gain / ref_row` 是複數振幅相除，之後 scoring 函式會用 `20·log₁₀|·|` 轉成 dB。

**維度調換**：

```python
gain_normalized = gain_normalized.transpose("frequency", "pump_freq", "ifbl")
```

scoring 函式要求 frequency 在第一維。

---

### ③ 頻段遮蔽（freq_exclude）

```python
if freq_exclude:
    mask = xr.ones_like(f, dtype=bool)
    for f_lo, f_hi in freq_exclude:
        mask = mask & ~((f >= f_lo) & (f <= f_hi))
    gain_scored = gain_normalized.where(mask)
```

若某些頻段有已知的干擾（例如腔體共振、qubit 頻率），可以在計算分數前先把那段 mask 成 `NaN`，避免影響評分。

範例：`freq_exclude=[(4.55e9, 4.65e9), (5.75e9, 5.85e9)]`

---

### ④ 打分數（Scoring）

```python
total_score = score_ai_twpa_c_gain_data(
    gain_data=gain_scored,
    gain_min=gain_min,       # 最低 gain 門檻，預設 12 dB
    gain_median=gain_median, # 目標中位數 gain，預設 15 dB
    ripple_max=ripple_max,   # 允許的最大 peak-to-peak ripple，預設 5 dB
    f_min=f_min,             # 計分頻率下限，預設 4 GHz
    f_max=f_max,             # 計分頻率上限，預設 8 GHz
)
```

`score_ai_twpa_c_gain_data()` 內部把三個子分數合成為一個總分：

```
total = (min_gain_score)^gain_min_exp × [ 20 × median_gain_score + 80 × ripple_score ]
```

| 子分數 | 計算方式 | 意義 |
|--------|----------|------|
| **min_gain_score** | `[f_min, f_max]` 內超過 `gain_min` 的頻率點比例（0~1） | 整體 gain 夠不夠高 |
| **median_gain_score** | `median(gain) / gain_median` | 中位數 gain 有沒有達到目標 |
| **ripple_score** | 在滑動窗口（預設 150 MHz）內 pk-to-pk ripple < `ripple_max` 的頻率點比例（0~1） | gain 夠不夠平坦 |

- `min_gain_score` 是**乘積項**，若有很多點低於門檻，總分會被壓低（懲罰機制）。
- `median_gain_score` 和 `ripple_score` 是**加權求和**，權重分別為 20 和 80（ripple 優先）。

結果 `total_score` 是一個 2D xarray，shape 為 `(pump_freq × ifbl)`，每格代表那個操作點的綜合得分。

---

### ⑤ 找最佳操作點

```python
best_points = []
for _ in range(n_best):
    pt = find_best_operation_point(
        total_score,
        excluded_points=best_points,
        exclusion_radius=exclusion_radius,  # 預設 20
    )
    if pt is None:
        break
    best_points.append(pt)
```

`find_best_operation_point()` 每次找 score 最高的點，但會在已找到的點周圍設一個「排除球」，避免連續找到相鄰的點。

排除距離由自定義的距離函式計算（考量 `pump_freq` 和 `ifbl` 的物理尺度）：

```
d = sqrt( (Δpump_freq / 1e6)² + (Δifbl / 2e-6)² )
```

若 `d ≤ exclusion_radius`，該區域被遮蔽，下一次迴圈在剩餘空間再找最大值。

---

### 輸出圖

**圖一：Score Heatmap**
- 橫軸：`ifbl`（µA）
- 縱軸：`pump_freq`（GHz）
- 顏色：總分 `total_score`
- 紅點：前 N 個最佳操作點

**圖二：各操作點的 Gain 曲線**
- 每個子圖顯示對應 `pump_freq`、`ifbl` 下的 gain vs frequency 曲線（dB）
- y 軸範圍固定 −5 ~ 25 dB

---

### 回傳值

| 名稱 | 型別 | 說明 |
|------|------|------|
| `best_points` | `list[xr.DataArray]` | 最佳操作點，每個帶有 `pump_freq`、`ifbl`、`score` 座標 |
| `total_score` | `xr.DataArray` | 2D 得分地圖（pump_freq × ifbl） |

---

## 使用範例

```python
ref = TWPAFlux(run_cfg_unpumped)
ref.run(10, yoko_inst=yoko, yoko_value=yoko_range)

gain = TWPAGain(run_cfg, pump, pump_freqs=np.linspace(10.5e9, 11.0e9, 21), pump_power=-10)
gain.run(10, yoko_inst=yoko, yoko_value=yoko_range)

best_pts, score_map = gain.analyze(
    reference=ref,
    gain_min=12,
    gain_median=15,
    ripple_max=5,
    f_min=4e9,
    f_max=8e9,
    n_best=5,
    exclusion_radius=20,
    freq_exclude=[(4.55e9, 4.65e9)],  # 遮蔽已知干擾頻段
)
```
