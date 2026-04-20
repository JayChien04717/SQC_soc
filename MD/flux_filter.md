# flux_filter.py — Flux Pulse 失真補償備忘錄

來源：`slab_qick_calib-live-plot-ipython/analysis/flux_filter.py`
參考論文：arXiv:2503.04610 Sections 1-3

---

## 問題背景

Flux line 存在指數衰減失真（反射、bias-tee droop），步階響應可表示為：

```
s(t) = α₀ + Σ αᵢ · exp(-t/τᵢ)
```

補償方法：計算逆濾波器 H_inv，對 flux 波形做 predistortion，使通過實體線路後恢復為乾淨步階。

---

## 函式總覽

### 核心計算

#### `step_response_to_tf(alphas, taus, alpha0=1.0)`
將步階響應參數轉換為連續時間傳遞函數 H(s) = N(s)/D(s)。

- `D(s) = Π(s + 1/τᵢ)`
- `N(s) = α₀·D(s) + Σ αᵢ·s·Π_{j≠i}(s + 1/τⱼ)`
- 回傳 `(num, den)` 多項式係數陣列（scipy 格式）

#### `compute_inverse_iir(alphas, taus, fs, alpha0=1.0)`
計算數位逆 IIR 濾波器。

- 對調 H(s) 的分子分母得 H_inv(s) = D(s)/N(s)
- 用 bilinear transform 離散化
- 轉成 second-order sections (SOS) 確保數值穩定
- 回傳 dict：`sos`, `b`, `a`, `num_s`, `den_s`, `fs`, ...

#### `apply_predistortion(waveform, sos)`
對 flux 波形套用逆濾波器。

```python
corrected = apply_predistortion(waveform, result["sos"])
```

#### `validate_filter(alphas, taus, fs, alpha0=1.0)`
驗證濾波器：step → H_inv → H → 應還原為 step。
回傳 `max_error`（穩態後的最大殘差），用來確認補償是否正確。

---

### QICK Envelope 生成

三個函式都回傳相同格式的 dict：

| Key | 說明 |
|---|---|
| `idata` | int16 陣列，直接用於 `prog.add_envelope(ch, name, idata=...)` |
| `waveform` | float 波形（未量化） |
| `gain_scale` | 波形峰值，nominal gain 需乘以此值 |
| `time_us` | 時間軸（µs） |
| `n_samples` | 樣本數 |

#### `make_slow_ramp(alphas, taus, alpha0, duration_us, fs_mhz, tau_threshold_us=0.5)`
只保留 **τ > threshold** 的慢指數（multi-µs droop、bias-tee 衰減）。
適合長 arb pulse，形狀平滑，適合先測試用。

#### `make_fast_correction(alphas, taus, alpha0, fs_mhz, duration_ns=200)`
只保留 **τ ≤ threshold** 的快指數（~ns 級暫態）。
輸出為差值波形（full - slow），從非零衰減到 ~0。
用法：在 slow ramp 開始時疊加播放。

#### `make_full_predistorted_step(alphas, taus, alpha0, duration_us, fs_mhz)`
所有指數項一次全補，最簡單的單 envelope 方案。

---

## 典型使用流程

### 1. 取得參數
從步階響應量測中 fit 出 `alpha0`, `alphas`, `taus`（單位：µs）。

### 2. 生成 idata

```python
from slab_qick_calib.analysis.flux_filter import make_slow_ramp, make_full_predistorted_step

fs_mhz = soccfg['gens'][flux_ch]['f_fabric']   # e.g. 430.08 MHz
maxv = soccfg.get_maxv(flux_ch)

# 方案 A：全補（簡單）
env = make_full_predistorted_step(alphas, taus, alpha0,
                                   duration_us=50, fs_mhz=fs_mhz, maxv=maxv)

# 方案 B：只補慢成分
env = make_slow_ramp(alphas, taus, alpha0,
                     duration_us=50, fs_mhz=fs_mhz, maxv=maxv)
```

### 3. 加入 QICK 程式

```python
self.add_envelope(flux_ch, name="flux_predist", idata=env["idata"])
self.add_pulse(ch=flux_ch, name="flux_pulse",
               style="arb", envelope="flux_predist",
               freq=0, phase=0,
               gain=nominal_gain * env["gain_scale"],
               outsel="product")
```

> **注意**：`gain = nominal_gain × gain_scale`，因為 predistortion 會放大峰值。

### 4. 驗證

```python
from slab_qick_calib.analysis.flux_filter import validate_filter

v = validate_filter(alphas, taus, fs=fs_mhz, alpha0=alpha0)
print(f"max_error = {v['max_error']:.4f}")   # 應接近 0
```

---

## 硬體限制（ZCU216）

| 參數 | 值 |
|---|---|
| Int gen `f_fabric` | 430.08 MHz |
| `maxlen` | 8192 samples → **最長約 19 µs** |
| `maxv_scale` | 0.9（避免插值 overshoot） |
| `samps_per_clk` | 1（int gen）|

envelope 長度需滿足：`duration_us × f_fabric < maxlen`

---

## 慢/快分離策略

```
所有 τ
  ├─ τ > 0.5 µs  → make_slow_ramp    → 長 arb pulse（主補償）
  └─ τ ≤ 0.5 µs  → make_fast_correction → 短 200 ns burst（暫態補償）
```

兩個 envelope 同時播放（疊加增益），可個別調整增益微調。
