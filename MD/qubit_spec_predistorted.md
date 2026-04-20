# qubit_spec_predistorted.py — 備忘錄

來源：`slab_qick_calib-live-plot-ipython/experiments/flux/qubit_spec_predistorted.py`

---

## 功能概述

`QubitSpec` 的替代版本，flux pulse 改用 IIR 逆濾波器生成的 predistorted arb envelope，
補償 flux line 的指數失真（反射、bias-tee droop）。

---

## 脈衝時序

```
t=0           flux_lead_time         +length    +flux_trail_time
│             │                      │          │
▼             ▼                      ▼          ▼
[── flux predist arb pulse ─────────────────────]
              [── qubit probe ──]
                                     [── readout ──]
```

1. flux arb pulse 從 t=0 開始（整段 `lead + length + trail` 長度）
2. 等待 `flux_lead_time` 讓 flux 穩定
3. 打 qubit probe pulse（頻率掃描）
4. 等待 `flux_readout_wait` 後量測
5. （選用）打 `flux_pulse_neg`（負增益）做反向 reset

---

## 使用方法

```python
from slab_qick_calib.experiments.flux.qubit_spec_predistorted import QubitSpecPredist

expt = QubitSpecPredist(cfg_dict, qi=2, style="medium", params={
    "flux_alphas": [0.0484, -0.131, -1.999],
    "flux_taus":   [0.018, 9.077, 12534.6],   # µs
    "flux_alpha0": 1.0969,
    "flux_gain":   -0.35,
    "flux_slow_only": True,   # 只用 τ > 0.5 µs 的慢成分
})
```

`flux_alphas` / `flux_taus` 未填時，自動從 config 讀取：
- `hw.soc.dacs.flux.step_alphas[qi]`
- `hw.soc.dacs.flux.step_taus[qi]`
- `hw.soc.dacs.flux.step_alpha0[qi]`

---

## 主要參數

### 掃描設定（style 預設值）

| style | gain 倍率 | span (MHz) | expts | 適用情境 |
|---|---|---|---|---|
| `huge` | 80× | 1500 | 1000 | 初次找峰 |
| `coarse` | 20× | 500 | 500 | 粗掃 |
| `medium` | 5× | 50 | 200 | 一般用途 |
| `fine` | 1× | 5 | 100 | 精細量測 |

### Flux 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `flux` | `True` | 開啟 flux pulse |
| `flux_gain` | `sweet_spot_ac[qi]` | flux DAC 增益 |
| `flux_lead_time` | 0.035 µs | flux 穩定等待時間 |
| `flux_trail_time` | 0.025 µs | probe 後 flux 繼續維持時間 |
| `flux_readout_wait` | 0.1 µs | probe 結束到量測的等待 |
| `flux_slow_only` | `False` | True → 只用 τ > threshold 的慢成分 |
| `flux_tau_threshold` | 0.5 µs | 慢/快分界 |
| `flux_negative_reset` | `False` | True → 每次 shot 後打負 flux 歸零 |

---

## Gain 計算

```python
scaled_gain = 2 * flux_gain * env["gain_scale"]
```

- `× env["gain_scale"]`：envelope 歸一化補償（predistortion 會放大峰值）
- `× 2`：QICK arb pulse 振幅是 const pulse 的一半，需補回

---

## 輔助方法

### `plot_envelope(ax=None)`
視覺化 predistorted 波形，包含：
- 上圖：qubit 端實際接收到的 flux（predistorted vs uncorrected 比較）
- 下圖：DAC 輸出波形

```python
expt = QubitSpecPredist(cfg_dict, qi=2, go=False, params={...})
expt.plot_envelope()
```

### `analyze()`
用 Lorentzian 擬合，結果存入 `data["new_freq"]`。

---

## 與 QubitSpec 的差異

| | QubitSpec | QubitSpecPredist |
|---|---|---|
| flux pulse 形狀 | const | predistorted arb |
| 需要 step-response 參數 | 否 | 是（`flux_alphas`, `flux_taus`） |
| gain 計算 | 直接用 `flux_gain` | `2 × flux_gain × gain_scale` |
| 額外 reset | 無 | 可選 `flux_negative_reset` |
