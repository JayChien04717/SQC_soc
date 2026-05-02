# reconstruct

自動化超導量子比特校正框架，基於 QICK (Quantum Instruction Control Kit)。
完全獨立套件，不依賴 `qick_workspace`，所有工具均為內部獨立複本。

---

## 快速開始

```python
from reconstruct.tools.system_tool import ExperimentConfig
from reconstruct.config.system_cfg import config_list, DATA_PATH
from reconstruct.core.base_experiment import BaseExperiment

# 連接硬體
soc, soccfg = make_proxy(ns_host="192.168.10.82", ns_port=8888, proxy_name="myqick")
BaseExperiment.setup(soc, soccfg, DATA_PATH)

# 載入設定
qubit = "Q1"
config_all = ExperimentConfig(config_list)
run_cfg = config_all.get_qubit(qubit)
```

---

## 目錄結構

```
reconstruct/
├── core/               ← 實驗執行核心抽象層
├── backend/            ← 硬體介面抽象層
├── config/             ← 硬體通道與頻率設定
├── experiments/        ← 所有實驗類別
│   ├── setup/          ← 初始設定實驗
│   ├── resonator/      ← 共振腔量測
│   ├── qubit_ge/       ← 量子比特 ge 態校正
│   ├── coherence/      ← 同調時間量測
│   ├── qubit_ef/       ← ef 態校正
│   └── characterization/ ← 閘操作品質驗證
├── analysis/           ← 量測資料後處理
├── calibration/        ← 自動校正管線
├── data/               ← 資料儲存與載入
├── plotter/            ← 即時繪圖工具
├── tools/              ← 數值工具與儀器工具
├── instruments/        ← 儀器驅動程式
└── service/            ← FastAPI 遠端控制服務
```

---

## 各模組說明

### `core/` — 核心抽象層

所有實驗共用的基底類別，定義統一的執行、儲存、分析介面。

| 檔案 | 說明 |
| --- | --- |
| `base_experiment.py` | 所有實驗的基底類別。實作 `run()` → `_post_fit()` → `Analysis` → 回傳 `ExperimentData` 的完整流程 |
| `base_program.py` | QICK 程式基底類別，封裝 `_initialize()` / `_body()` 及常用 helper（pulse、gate、measure、cooling） |
| `base_analysis.py` | 分析類別基底，子類別掛載在實驗的 `Analysis` 屬性，於 `_post_fit` 後自動執行 |
| `experiment_data.py` | 統一結果容器 `ExperimentData`，支援舊式 tuple 解包與 scalar 轉型，內含品質旗標 `QualityFlag` |
| `composite.py` | `BatchExperiment`（循序）與 `ParallelExperiment`（平行）組合實驗 |

**子類別只需實作：**
```python
class MyExperiment(BaseExperiment):
    EXPT_NAME = "..."
    def _create_program(self): ...
    def _extract_sweep_axis(self, prog): ...
    def _post_fit(self, x_vals): ...   # 選填
```

---

### `backend/` — 硬體介面抽象層

將實驗程式碼與實際 QICK 硬體解耦，同一份實驗可在真實硬體或模擬環境執行。

| 檔案 | 說明 |
| --- | --- |
| `base_backend.py` | 抽象介面，定義 `activate()`、`run_program()` |
| `qick_backend.py` | 真實硬體後端，透過 Pyro4 連接 QICK 板 |
| `simulated_backend.py` | 軟體模擬後端，不需硬體即可產生合成 IQ 資料（Lorentzian、指數衰減、Rabi 振盪等） |

```python
# 真實硬體
backend = QICKBackend.from_pyro4("192.168.10.82", 8888)

# 離線開發 / 測試
backend = SimulatedBackend(noise_level=0.02)

backend.activate()   # 設定全域 soc / soccfg
```

---

### `config/` — 硬體設定檔

| 檔案 | 說明 |
| --- | --- |
| `system_cfg.py` | **唯一需要手動編輯的檔案**。包含 `DATA_PATH`（資料儲存路徑）與 `config_list`（每顆 qubit 的通道、頻率、脈衝參數） |

`config_list` 使用巢狀結構（`ch` / `res` / `qb` / `cooling`），`ExperimentConfig` 負責展平與存取：

```python
config_all = ExperimentConfig(config_list)
run_cfg = config_all.get_qubit("Q1")           # 取得展平後的副本

config_all.update("res.res_gain_ge", 0.4, q_index="Q1")   # 點記法更新巢狀鍵
config_all.update("qb_freq_ge", 4998.7, q_index="Q1")     # 直接更新展平鍵
```

---

### `experiments/` — 實驗類別

#### `setup/` — 初始設定

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `TOF` | s001 | Time-of-flight：確認 ADC trigger 時間點 |
| `SingleShot_gef` | s000 | 單發讀取（g/e/f 三態），GMM 分析保真度 |
| `SingleShot_ge_opt` | s000_opt | 掃描 (freq, gain, length) 找最佳讀取參數 |

#### `resonator/` — 共振腔量測

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `ResonatorSpec` | s002 | 單音頻譜，ABCD circle fit 萃取 f₀、κ、κ_c |
| `Punchout` | s002b | 2D 掃描 (freq × gain)，找臨界光子數 |
| `ResonatorSpecFlux` | s002c | flux bias 掃描下的共振腔頻率 |
| `TWPAFlux` | s002d | TWPA flux bias 掃描 |
| `TWPAGain` | s002e | TWPA 增益 vs 泵浦頻率 |
| `TWPAGainPower` | s002f | TWPA 增益 vs 泵浦功率 |
| `TWPAPowerScan` | s002g | TWPA 信號功率掃描 |

#### `qubit_ge/` — 量子比特 ge 態校正

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `QubitSpec` | s003 | 雙音頻譜，Lorentzian fit 找 f_ge |
| `QubitSpecFlux` | s003a | Fast flux 下的量子比特頻譜（2D） |
| `TimeRabi` | s004 | 時間 Rabi，掃描脈衝長度 |
| `PowerRabi` | s005 | 功率 Rabi，萃取 π 脈衝增益 |
| `AAE` | s005a | Amplified Amplitude Error，精確校正 π 脈衝 |
| `DragCalibration` | s005a_drag | DRAG 係數 alpha 掃描校正 |

#### `coherence/` — 同調時間量測

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `Ramsey` | s006 | Ramsey 干涉，萃取 T2\*，修正頻率偏移 |
| `SpinEcho` | s007 | Spin Echo，萃取 T2e |
| `T1` | s008 | 能量弛豫時間 T1 |
| `ACStark` | — | AC Stark 位移量測 |

#### `qubit_ef/` — ef 態校正

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `ResonatorSpec_ef` | s009 | ef 態下的共振腔頻移 |
| `QubitSpecEf` | s010 | ef 躍遷頻率 |
| `PowerRabiEf` | s011 | ef 態 π 脈衝校正 |
| `RamseyEf` | s012 | ef 態 T2\* 與頻率修正 |
| `T1Ef` | s013 | ef 態 T1 |
| `QubitTemp` | s013 | 量子比特溫度（ef 功率 Rabi 熱佔據比） |

#### `characterization/` — 閘操作品質驗證

| 類別 | 對應原始腳本 | 功能 |
| --- | --- | --- |
| `AllXY` | s014 | AllXY 序列，診斷旋轉軸與振幅誤差 |
| `RandomizedBenchmarking` | s015 | 標準 RB 與 Interleaved RB，萃取單閘保真度 |
| `AutoRB` | s015_Auto | 自動執行多組 IRB（批次） |
| `Tomography` | s016 | 單量子比特量子態層析 |

---

### `analysis/` — 資料後處理

量測資料的擬合與物理量萃取，掛載在實驗的 `Analysis` 屬性自動執行。

| 檔案 | 說明 |
| --- | --- |
| `resonator.py` | `ResonatorSpecAnalysis`：ABCD hanger model circle fit，萃取 f₀、Qi、Qc、Ql |
| `qubit.py` | `QubitAnalysis`：Lorentzian fit（QubitSpec）、decaysin fit（Ramsey、SpinEcho）、expfit（T1） |
| `rb.py` | `RBAnalysis`：指數衰減 fit，萃取平均閘保真度 r 與 EPC |

---

### `calibration/` — 自動校正管線

| 檔案 | 說明 |
| --- | --- |
| `store.py` | `CalibrationStore`：以時間戳 JSON 持久化校正參數，支援過期偵測（`is_stale()`） |
| `graph.py` | `CalibrationGraph`：以 DAG 表示實驗間的依賴關係，決定執行順序 |
| `monitor.py` | `CalibrationMonitor`：監控各參數的新鮮度，自動觸發再校正 |
| `pipeline.py` | `AutoCalibrate`：7 步驟自動校正流程（res_spec → qubit_spec → power_rabi → ramsey → spin_echo → t1 → ss_opt） |

```python
store = CalibrationStore("cal_Q1.json")
auto  = AutoCalibrate(config_all, "Q1", cal_store=store)
auto.run(skip=("spin_echo", "ss_opt"))   # 跳過選填步驟
```

每一步驟完成後自動更新 `ExperimentConfig` 與 `CalibrationStore`。

---

### `data/` — 資料儲存與載入

| 檔案 | 說明 |
| --- | --- |
| `manager.py` | `save_data()` / `load_data()` / `list_data_files()`：HDF5 儲存與目錄掃描 |
| `serializer.py` | Numpy-safe JSON 序列化（`ExperimentData`、config dict、datetime） |

```python
result.save("path/file.h5")                  # 儲存
data = ExperimentData.load("path/file.h5")   # 載入
files = list_data_files("D:/data/")          # 列出所有 .h5 檔
```

---

### `plotter/` — 繪圖工具

| 檔案 | 說明 |
| --- | --- |
| `liveplot.py` | `liveplotfun`：量測進行中的即時 IQ 圖（producer-consumer 架構，LIFO queue） |
| `plot_utils.py` | `plot_final`：量測結束後的四格 IQ 擬合圖 |

---

### `tools/` — 數值工具

| 檔案 | 說明 |
| --- | --- |
| `system_tool.py` | `ExperimentConfig`（完整版，含點記法、auto-search、addict.Dict、muxconfig）、HDF5 helper、YAML 序列化 |
| `fitting.py` | 常用擬合函數：`fitlor`、`fitdecaysin`、`fitexp`、`fitrb`、`fitgauss` 等 |
| `scoring.py` | TWPA 操作點評分：`score_ai_twpa_c_gain_data`、`find_best_operation_point` |
| `rb_generator.py` | RB Clifford 序列產生器，支援 Standard RB 與 Interleaved RB |
| `electrical_length.py` | TWPA flux-dependent 電氣長度估算與繪圖 |
| `abcd_rf_fit/` | ABCD/hanger model 共振腔 circle fit 套件（完整獨立複本） |
| `qubit.py` | 量子比特物理量計算工具 |

---

### `instruments/` — 儀器驅動程式

| 檔案 | 儀器 | 功能 |
| --- | --- | --- |
| `YOKOGS200.py` | Yokogawa GS200 | 電流 / 電壓源，flux bias 控制 |
| `sgs100a.py` | R&S SGS100A | 微波訊號源，TWPA 泵浦 |
| `mg3692.py` | Anritsu MG3692 | 微波訊號源 |
| `yoko.py` | Yokogawa（通用） | 通用 Yoko 控制介面 |

---

### `service/` — 遠端控制服務

| 檔案 | 說明 |
| --- | --- |
| `api.py` | FastAPI REST 服務，支援非同步實驗提交與 `CalibrationStore` 遠端存取 |

```bash
uvicorn reconstruct.service.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | 功能 |
| --- | --- |
| `POST /experiments/run` | 提交實驗（回傳 job_id） |
| `GET /experiments/{id}/result` | 取得 `ExperimentData` JSON |
| `POST /calibrate/{qubit}/run` | 觸發 `AutoCalibrate` |
| `GET /calibrations/{qubit}/params` | 取得所有校正參數 |

---

## 設計原則

1. **完全獨立** — 不依賴 `qick_workspace`，所有工具為內部複本，可單獨安裝使用
2. **向下相容** — `result = expt.run(py_avg)` 支援舊式 `fit_params, err = result` tuple 解包與 `float(result)` scalar 轉型
3. **統一回傳型別** — 所有實驗回傳 `ExperimentData`，包含原始 IQ、擬合結果、品質旗標、config 快照
4. **硬體抽象** — 同一份實驗程式碼可在 `QICKBackend`（真實硬體）與 `SimulatedBackend`（離線）執行
5. **持久化校正** — `CalibrationStore` 以時間戳 JSON 記錄每次校正結果，支援過期偵測與跨 session 重載
