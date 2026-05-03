# TWPA 完整參考手冊

本文件整合 AI-TWPA-C 校正流程、`TWPAGain.analyze()` 計算說明，以及官方 Scoring Notebook 格式對照。

---

## 目錄

1. [校正流程（操作步驟）](#1-校正流程操作步驟)
2. [TWPAGain.analyze() 計算流程](#2-twpagainanalyze-計算流程)
3. [官方 Scoring Notebook 格式對照](#3-官方-scoring-notebook-格式對照)

---

## 1. 校正流程（操作步驟）

依照官方手冊 Fig. 1.1 的流程圖，整理成對應本系統（QICK + Yokogawa + MG3692）的操作步驟。

### 流程總覽

```
新冷卻循環
    │
    ▼
【Sanity Check】量測 unpumped zero-flux S21
    │
    ▼
量測 Electrical Length  ──→  取得 Rb（flux line 電阻）
    │
    ▼
確認 unpumped spectrum 正常
    │
    ├─ 是否有 Split dip？→ 見 Troubleshooting
    │
    ▼
【Gain Scan】掃描 pump_freq × flux bias
    │
    ▼
計算 Score，畫 heatmap
    │
    ├─ Gain 不合理？→ 見 Troubleshooting
    │
    ▼
選出前 3～10 個最佳工作點
    │
    ▼
TWPA 準備完畢
```

---

### Step 1：Sanity Check — Unpumped Zero-flux S21

**目的：** 確認 TWPA 工作正常，取得後續 normalize 用的 reference。

**操作：**

```python
from qick_workspace.scrip.s002d_TWPA_flux import TWPAFlux

# pump 關閉，yoko 從 -1 mA 掃到 +1 mA
ref = TWPAFlux(run_cfg)
ref.run(10, yoko_inst=yoko_connect, yoko_value=yoko_range, yoko_mode="current")
```

**檢查項目：**
- S21 heatmap 是否隨 flux 有週期性調變
- IQ 圖是否呈現圓形「甜甜圈」（thin donut）

---

### Step 2：量測 Electrical Length

**目的：** 確認 electrical length 調變量 ≥ 1 ns，並取得 flux line 電阻 Rb。

**操作：**

```python
ref.plot()  # 自動畫 S21 heatmap 和 electrical length vs flux
```

**解讀結果：**

| 狀況 | 意義 |
|------|------|
| 調變量 < 1 ns | Flux 耦合不足，見 Troubleshooting |
| 曲線不對稱 / 不平滑 | 安裝問題 |
| 有 Split dip | 外加磁場干擾 |

**Electrical Length 圖範例（你的量測結果）：**
- 最大調變約 **1.1 ns**（peak-to-peak）
- 工作 flux 範圍：約 **±800 µA** 附近有最大調變

**Rb 估算**（若需要將電壓轉電流）：
```python
Rb = flux_bias_voltage / ifbl  # 單位 Ω
```

---

### Step 3：Gain Scan

**目的：** 掃描 pump_freq × flux bias，量測每個工作點的 gain。

**參數設定建議（根據 datasheet / 前次量測）：**

| 參數 | 建議值 |
|------|--------|
| pump_freq 範圍 | 依 TWPA 型號，約 10–15 GHz |
| pump_freq 步數 | 21–51 點（1 MHz 步距） |
| flux 範圍 | electrical length 最大調變區域 |
| flux 步數 | 21–51 點 |
| pump_power | -10 dBm（初始值，視情況調整） |

**操作：**

```python
from qick_workspace.scrip.s002e_TWPA_gain import TWPAGain
from qick_workspace.tools.mg3692 import AnritsuMG3692

pump = AnritsuMG3692("192.168.10.182")

PUMP_FREQS = np.linspace(10.5e9, 11.0e9, 21)  # 根據 TWPA 調整

gain_scan = TWPAGain(
    run_cfg,
    pump_source=pump,
    pump_freqs=PUMP_FREQS,
    pump_power=-10,
)
gain_scan.run(
    10,
    yoko_inst=yoko_connect,
    yoko_value=yoko_range,
    yoko_mode="current",
)
```

---

### Step 4：Scoring 與選取最佳工作點

**目的：** 對每個 (pump_freq, flux) 組合計算 score，找出最佳工作點。

**評分標準：**

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `gain_min` | 最低 gain 門檻（dB） | 12 |
| `gain_median` | 目標中位數 gain（dB） | 15 |
| `ripple_max` | 最大允許 ripple（dB pk-to-pk） | 5 |
| `f_min / f_max` | 評分的信號頻率範圍（Hz） | 4e9 / 8e9 |

**操作：**

```python
best_points, total_score = gain_scan.analyze(
    reference=ref,        # Step 1 的 unpumped 量測
    gain_min=12,
    gain_median=15,
    ripple_max=5,
    f_min=4e9,
    f_max=8e9,
    n_best=5,             # 找前 5 個工作點
    exclusion_radius=20,  # 點之間的最小間距（grid units）
)
```

**輸出：**
1. Score heatmap（pump_freq × flux），紅點標出最佳工作點
2. 每個最佳點的 gain vs frequency 曲線
3. Console 輸出最佳 pump_freq 和 ifbl 數值

---

### Step 5：確認 Gain 正常

| 狀況 | 處理 |
|------|------|
| Gain 平坦，> 12 dB，ripple < 5 dB | ✅ 正常，進入下一步 |
| Gain 過高且有大 ripple | 降低 pump_power（-1 dBm），重掃 |
| Gain 太低 | 提高 pump_power（+1 dBm），重掃 |
| 有 excess ripple | 見 Troubleshooting |

---

### Step 6：存檔

```python
# 存 netCDF（已 normalize）
gain_scan.saveNetCDF(reference=ref)

# 或存 Labber 格式
# gain_scan.saveLabber(qubit, config_all=config_all)
```

---

### 存檔後分析（從檔案讀取）

```python
import xarray as xr
from qick_workspace.tools.scoring import (
    score_ai_twpa_c_gain_data,
    find_best_operation_point,
    plot_gain_at_operation_point,
)

s21_data = xr.open_dataset("TWPA_gain_xxx.nc")
ref_data = xr.open_dataset("TWPA_gain_xxx_reference.nc")
```

---

### Troubleshooting 快速對照

| 現象 | 可能原因 | 處理方式 |
|------|----------|----------|
| Electrical length 調變 < 1 ns | Flux 耦合不足 | 確認 Yoko 電流 / 配線 |
| Electrical length 不對稱 | 安裝不對稱 | 調整 TWPA 安裝方向 |
| S21 有 split dip | 外加磁場干擾 | 移除雜散磁場來源 |
| Gain 有 excess ripple | 阻抗匹配問題 | 調整工作點或加 isolator |
| Output spectrum 異常 | 非線性過強 | 降低 pump_power |

---

### 本系統對應關係

| AI 手冊術語 | 本系統實作 |
|-------------|------------|
| Unpumped S21 vs flux | `TWPAFlux.run()` |
| Electrical length plot | `TWPAFlux.plot()` |
| Gain scan | `TWPAGain.run()` |
| Score + 找最佳工作點 | `TWPAGain.analyze()` |
| Reference data | 傳入 `reference=ref` |
| MG3692 pump source | `AnritsuMG3692` |
| Yokogawa flux bias | `YOKOGS200` |

---

## 2. TWPAGain.analyze() 計算流程

對應檔案：`s002e_TWPA_gain.py`

### 概覽

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

### 完整使用範例

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

---

## 3. 官方 Scoring Notebook 格式對照

### 資料格式（xr.Dataset）

官方要求兩個 netCDF 檔案：

| 檔案 | 內容 |
|------|------|
| `gain-data_0.nc` | **Pumped** S21，掃 pump_freq × ifbl × frequency |
| `reference-data_0.nc` | **Unpumped zero-flux** S21，只有一個 ifbl 值（≈ 0 A） |

每個 Dataset 必須有：
- **座標**：`frequency`（Hz）、`ifbl`（A）、`pump_freq`（Hz）、`pump_power`（dBm，scalar）、`pump_state`（0/1，scalar）
- **Data variables**：`magnitude`（linear |S21|）、`phase`（arg(S21)，radians）

---

### 官方 normalize 方法

```python
def compute_gain(s21: xr.Dataset, ref_s21: xr.Dataset):
    s21     = s21[MAGNITUDE_LINEAR]    * np.exp(1j * s21[PHASE])
    ref_s21 = ref_s21[MAGNITUDE_LINEAR] * np.exp(1j * ref_s21[PHASE])
    return s21 / ref_s21
```

**重點：**
- 先從 magnitude + phase 重建複數 S21
- 再做複數除法（同時 normalize 振幅 *和* 相位）
- reference 必須是 pump OFF、flux ≈ 0 的單一 row

官方在做之前有三個 assert：
```python
assert bool(reference_s21_data[PUMP_STATE].max()) == False   # pump 必須是 off
assert IFBL not in reference_s21_data.coords or len(np.unique(reference_s21_data[IFBL])) == 1  # reference 只能有一個 flux bias
assert np.abs(s21_data[FREQUENCY] - reference_s21_data[FREQUENCY]).max() < 1e3  # 同一 frequency grid
assert np.abs(flux_bias_in_ref) < 10e-6  # reference flux bias 必須 < 10 µA
```

---

### 本系統對應方式

**`TWPAGain.saveNetCDF(reference=ref)`** 存兩個檔案，格式與官方一致：

| 我們的檔案 | 對應官方 |
|------------|----------|
| `TWPA_gain_<ts>.nc` | `gain-data_0.nc` — pumped，有 pump_freq 維度 |
| `TWPA_gain_<ts>_reference.nc` | `reference-data_0.nc` — unpumped，zero-flux single row |

**注意**：存的是 **raw（未 normalize）** 的 magnitude/phase，跟官方格式相同。官方 notebook 自己做 normalize（`compute_gain()`）。

**`TWPAGain.analyze(reference=ref)`** 在記憶體中直接做 normalize，等效於官方 `compute_gain()`，但用的是 raw complex IQ 相除：

```python
ref_row = ref_s21.sel(ifbl=smallest_flux)  # zero-flux row
gain_normalized = gain / ref_row            # 複數除法，等效官方
```

---

### 從 netCDF 讀回來重新分析（對應官方 notebook）

```python
import xarray as xr
import numpy as np

s21_ds  = xr.open_dataset("TWPA_gain_20250101_120000.nc")
ref_ds  = xr.open_dataset("TWPA_gain_20250101_120000_reference.nc")

# 重建複數 gain（對應官方 compute_gain）
s21_c   = s21_ds["magnitude"] * np.exp(1j * s21_ds["phase"])
ref_c   = ref_ds["magnitude"] * np.exp(1j * ref_ds["phase"])
gain    = s21_c / ref_c   # shape: (frequency × pump_freq × ifbl)

# 接著就可以直接用官方 scoring 函數
from qick_workspace.tools.scoring import score_ai_twpa_c_gain_data, find_best_operation_point

total_score = score_ai_twpa_c_gain_data(
    gain_data=gain,
    gain_min=12, gain_median=15, ripple_max=5,
    f_min=4e9, f_max=8e9,
)
```

---

### 關鍵差異總結

| 項目 | 官方 notebook | 本系統 |
|------|--------------|--------|
| reference 來源 | 獨立檔案（pump off, zero flux） | `TWPAFlux` 的 zero-flux row |
| 存檔格式 | magnitude + phase（raw） | 同左 |
| normalize 時機 | 讀檔後手動 `compute_gain()` | `analyze()` 自動，或讀檔後手動 |
| normalize 方法 | `mag * exp(j*phase)` 複數除法 | 同左（raw complex 相除等效） |
| pump_state coord | 需要（0/1） | 已加入 saveNetCDF |
