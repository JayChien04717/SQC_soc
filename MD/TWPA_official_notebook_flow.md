# 官方 AI-TWPA-C Scoring Notebook 流程說明

## 資料格式（xr.Dataset）

官方要求兩個 netCDF 檔案：

| 檔案 | 內容 |
|------|------|
| `gain-data_0.nc` | **Pumped** S21，掃 pump_freq × ifbl × frequency |
| `reference-data_0.nc` | **Unpumped zero-flux** S21，只有一個 ifbl 值（≈ 0 A） |

每個 Dataset 必須有：
- **座標**：`frequency`（Hz）、`ifbl`（A）、`pump_freq`（Hz）、`pump_power`（dBm，scalar）、`pump_state`（0/1，scalar）
- **Data variables**：`magnitude`（linear |S21|）、`phase`（arg(S21)，radians）

---

## 官方 normalize 方法

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
# pump 必須是 off
assert bool(reference_s21_data[PUMP_STATE].max()) == False

# reference 只能有一個 flux bias
assert IFBL not in reference_s21_data.coords or len(np.unique(reference_s21_data[IFBL])) == 1

# 兩個資料集必須在同一個 frequency grid
assert np.abs(s21_data[FREQUENCY] - reference_s21_data[FREQUENCY]).max() < 1e3

# reference flux bias 必須 < 10 µA（幾乎零 flux）
assert np.abs(flux_bias_in_ref) < 10e-6
```

---

## 本系統對應方式

### `TWPAGain.saveNetCDF(reference=ref)`

存兩個檔案，格式與官方一致：

| 我們的檔案 | 對應官方 |
|------------|----------|
| `TWPA_gain_<ts>.nc` | `gain-data_0.nc` — pumped，有 pump_freq 維度 |
| `TWPA_gain_<ts>_reference.nc` | `reference-data_0.nc` — unpumped，zero-flux single row |

**注意**：我們存的是 **raw（未 normalize）** 的 magnitude/phase，跟官方格式相同。官方 notebook 自己做 normalize（`compute_gain()`）。

### `TWPAGain.analyze(reference=ref)`

在記憶體中直接做 normalize，等效於官方 `compute_gain()`，但用的是 raw complex IQ 相除：

```python
ref_row = ref_s21.sel(ifbl=smallest_flux)  # zero-flux row
gain_normalized = gain / ref_row            # 複數除法，等效官方
```

---

## 從 netCDF 讀回來重新分析（對應官方 notebook）

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

## 關鍵差異總結

| 項目 | 官方 notebook | 本系統 |
|------|--------------|--------|
| reference 來源 | 獨立檔案（pump off, zero flux） | `TWPAFlux` 的 zero-flux row |
| 存檔格式 | magnitude + phase（raw） | 同左 |
| normalize 時機 | 讀檔後手動 `compute_gain()` | `analyze()` 自動，或讀檔後手動 |
| normalize 方法 | `mag * exp(j*phase)` 複數除法 | 同左（raw complex 相除等效） |
| pump_state coord | 需要（0/1） | 已加入 saveNetCDF |
