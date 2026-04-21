# AI-TWPA-C 校正流程指南

依照官方手冊 Fig. 1.1 的流程圖，整理成對應本系統（QICK + Yokogawa + MG3692）的操作步驟。

---

## 流程總覽

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

## Step 1：Sanity Check — Unpumped Zero-flux S21

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

## Step 2：量測 Electrical Length

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

## Step 3：Gain Scan

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

## Step 4：Scoring 與選取最佳工作點

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

## Step 5：確認 Gain 正常

**判斷標準：**

| 狀況 | 處理 |
|------|------|
| Gain 平坦，> 12 dB，ripple < 5 dB | ✅ 正常，進入下一步 |
| Gain 過高且有大 ripple | 降低 pump_power（-1 dBm），重掃 |
| Gain 太低 | 提高 pump_power（+1 dBm），重掃 |
| 有 excess ripple | 見 Troubleshooting |

---

## Step 6：存檔

```python
# 存 Labber HDF5（已 normalize）
gain_scan.saveNetCDF(reference=ref)

# 或存 Labber 格式
# gain_scan.saveLabber(qubit, config_all=config_all)
```

---

## 存檔後分析（從檔案讀取）

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

## Troubleshooting 快速對照

| 現象 | 可能原因 | 處理方式 |
|------|----------|----------|
| Electrical length 調變 < 1 ns | Flux 耦合不足 | 確認 Yoko 電流 / 配線 |
| Electrical length 不對稱 | 安裝不對稱 | 調整 TWPA 安裝方向 |
| S21 有 split dip | 外加磁場干擾 | 移除雜散磁場來源 |
| Gain 有 excess ripple | 阻抗匹配問題 | 調整工作點或加 isolator |
| Output spectrum 異常 | 非線性過強 | 降低 pump_power |

---

## 本系統對應關係

| AI 手冊術語 | 本系統實作 |
|-------------|------------|
| Unpumped S21 vs flux | `TWPAFlux.run()` |
| Electrical length plot | `TWPAFlux.plot()` |
| Gain scan | `TWPAGain.run()` |
| Score + 找最佳工作點 | `TWPAGain.analyze()` |
| Reference data | 傳入 `reference=ref` |
| MG3692 pump source | `AnritsuMG3692` |
| Yokogawa flux bias | `YOKOGS200` |
