# GUI Plan for qick_workspace

## Tech Stack

**Framework: PySide6 (Qt6)**

- Native desktop app, no browser needed
- Dockable panels — user can rearrange layout
- Matplotlib 嵌入：`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`
- Threading：`QThread` 跑 acquire，不凍結 UI

**Alternative: Streamlit**（如果想要 browser-based，快速 prototype 用）

- 優點：幾乎不用寫 UI 程式
- 缺點：難做 live plot 更新、docking 彈性差

**建議：PySide6**，適合長期使用的 lab instrument control GUI。

---

## 視窗佈局

主視窗：

```text
┌─────────────────────────────────────────────────────────────────┐
│  Menu Bar: File | Connection | Tools | Data Browser | Help      │
├──────────────┬──────────────────────────┬───────────────────────┤
│              │                          │                       │
│  [1] Setup   │  [2] Experiment Panel    │  [3] Live Plot        │
│  Panel       │                          │                       │
│  ──────────  │  ─────────────────────── │  ─────────────────── │
│  Connection  │  Experiment Selector     │  matplotlib canvas    │
│  Config      │  Parameter Form          │  (auto-updates)       │
│  Qubit Select│  Run Controls            │                       │
│              │                          │                       │
├──────────────┴──────────────────────────┴───────────────────────┤
│  [4] Result Panel                                               │
│  Fit results  |  IQ scatter  |  Save button  |  Log            │
├─────────────────────────────────────────────────────────────────┤
│  [5] Status Bar: Connected | Q2 | Last run: T1=45.3 µs         │
└─────────────────────────────────────────────────────────────────┘
```

Data Browser — 獨立 Dock Window（可浮動）：

```text
┌─────────────────────────────────────────────────────────────────┐
│  [6] Data Browser                                               │
├──────────────────────┬──────────────────────────────────────────┤
│  File List           │  Plot Viewer                             │
│  ──────────────────  │  ──────────────────────────────────────  │
│  Filter: [____] [Q▼] │  View: (●mag) (○phase) (○avgi) (○avgq)  │
│                      │                                          │
│  ▼ 2024-01-15        │  [matplotlib canvas]                     │
│    s003_qubit_q2     │                                          │
│    s008_T1_q2        │                                          │
│  ▼ 2024-01-14        │                                          │
│    s002_res_q2       │  ──────────────────────────────────────  │
│    s006_ramsey_q2    │  Metadata                                │
│                      │  Experiment: s008_T1_ge                  │
│  [Open Folder]       │  Qubit: 2                                │
│  [Refresh]           │  Time: 2024-01-15 10:32                  │
│                      │  T1 = 45.3 µs                            │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Panel 細節

### [1] Setup Panel

```text
[ Connection ]
  SOC IP:     [_____________] [Connect]
  Status:     ● Connected / ✗ Disconnected

[ Config ]
  Config file: [__path__] [Browse] [Load] [Save]
  Data path:   [__path__] [Browse]

[ Qubit Select ]
  Active qubit: ( Q0 ) ( Q1 ) ( Q2 ) ( Q3 )
  Yoko value:   [____] mA
```

對應程式：

- `BaseExperiment.setup(soc, soccfg, data_path)`
- `ExperimentConfig` load/save YAML

---

### [2] Experiment Panel

```text
[ Experiment ]
  Category: [ GE ▼ ]   Experiment: [ Qubit Spec ▼ ]

  ┌─ Parameters ──────────────────────────┐
  │  py_avg       [  10  ]                │
  │  reps         [ 1000 ]                │
  │  relax_delay  [  50  ] µs             │
  │  span         [  50  ] MHz            │
  │  steps        [ 100  ]                │
  │  ...（根據實驗動態生成欄位）            │
  └───────────────────────────────────────┘

  [ ▶ Run ]  [ ■ Stop ]  [ ↺ Repeat ]
  [ Save ]   [ Load params ]
```

**Experiment 分類（Category 下拉）：**

| Category | 實驗 |
| --- | --- |
| Setup | Time of Flight |
| Resonator | Res Spec GE, Punchout, Flux Spec |
| Qubit GE | Qubit Spec, Flux Spec, Time Rabi, Power Rabi, DRAG, AAE |
| Coherence | Ramsey, Spin Echo, T1 |
| Qubit EF | Res Spec EF, Qubit Spec EF, Power Rabi EF, Ramsey EF, T1 EF |
| Advanced | AllXY, SingleShot Opt, Qubit Temp, AC Stark |
| RB | Single Qubit RB, IRB, Auto RB, ASM RB |
| Tomography | State Tomography |
| Auto | AutoCalibrate |

參數欄位從各 experiment class 的 `__init__` 預設值動態生成。

---

### [3] Live Plot Panel

```text
┌─────────────────────────────────────┐
│  [matplotlib figure]                │
│                                     │
│  Qubit Spec Q2                      │
│  ●●●●●●●●● (live updating)         │
│                                     │
│  Controls: [IQ: abs▼] [Autoscale☑] │
└─────────────────────────────────────┘
```

- 呼叫現有的 `liveplot.py` 更新函式
- IQ 選擇：abs / real / imag / IQ scatter
- 每次 acquire 結束後觸發 `_post_fit` 並更新 fit curve

---

### [4] Result Panel

```text
┌─ Fit Results ──────────┬─ IQ Scatter ─┬─ Log ──────────────────┐
│  f_ge:  5234.12 MHz    │              │ [10:32] Qubit Spec done │
│  T1:    45.3 µs        │  [scatter]   │ [10:31] Run started     │
│  T2:    88.1 µs        │              │ [10:30] Config loaded   │
│  EPC:   0.012%         │              │                         │
│                        │              │                         │
│  [→ Update Config]     │              │  [Clear]                │
└────────────────────────┴──────────────┴─────────────────────────┘
```

**「Update Config」按鈕**：把 fit 結果直接寫回 `ExperimentConfig`（f_ge、pi_gain 等）。

---

### [5] Auto Calibrate Panel（獨立 Dialog）

```text
[ Auto Calibrate ]

  Steps to run:
  ☑ Res Spec        ☑ Qubit Spec
  ☑ Power Rabi      ☑ Ramsey
  ☑ Spin Echo       ☑ T1
  ☑ SingleShot Opt

  Skip steps: [____________]

  [ ▶ Start AutoCal ]  [ ■ Stop ]

  Progress:
  ████████░░  Step 4/7: Ramsey
  Status: GP predicting zero crossing...
```

對應 `AutoCalibrate.run(skip=(...))` + `AutoCalibrate.summary()`。

---

### [6] Data Browser（獨立 Dock）

由 `tools/data_manager.py` 驅動，完全不依賴 Labber。

#### File List（左欄）

- `list_data_files(directory)` 掃描資料夾，依日期分組顯示
- Filter bar：關鍵字搜尋 + 按 Qubit 篩選
- 雙擊載入：呼叫 `load_data(path)` 讀取 `.h5`
- `[Open Folder]` 切換資料夾，`[Refresh]` 重新掃描

#### Plot Viewer（右欄）

Radio button 切換顯示頻道：

| 選項 | 資料來源 | 說明 |
| --- | --- | --- |
| **mag** | `data["mag"]` | `sqrt(I² + Q²)`，預設 |
| **phase** | `data["phase"]` | `arctan2(Q, I)`，單位 deg |
| **avgi** | `data["avgi"]` | I 分量 |
| **avgq** | `data["avgq"]` | Q 分量 |

- 1D 實驗：折線圖，x 軸從 `data["x"]`
- 2D 實驗：colormap，x/y 軸從 `data["x"]` / `data["y"]`
- 切換頻道時只換 y 資料，不重新載入檔案

#### Metadata 區（右下）

顯示 HDF5 root attrs：experiment、qubit、timestamp、fit 結果（若有存入 config）。

#### Save 整合

Result Panel 的 **Save** 按鈕呼叫：

```python
from qick_workspace.tools.data_manager import save_data

save_data(expt, qb_idx=2, config_all=cfg)
# 存成 DATA_PATH/s003_qubit_spec_q2_001.h5
```

HDF5 格式（純 h5py，無 Labber 依賴）：

```text
/
├── attrs: experiment, qubit, timestamp, tag, config (JSON)
├── x/    values, attrs: name, unit
├── y/    values, attrs: name, unit   (2D only)
└── data/
    ├── avgi   float32
    ├── avgq   float32
    ├── mag    float32
    └── phase  float32 (degrees)
```

---

## 資料流

```text
UI 操作
  │
  ▼
QThread（背景執行）
  │  呼叫 ExperimentClass.run() / prog.acquire()
  │
  ├─► 每 rep 發 signal → Live Plot 更新
  │
  └─► 完成後發 signal → Result Panel 更新 + Fit
                      → save_data() → .h5 檔案
                      → Data Browser 自動 Refresh
```

`QThread` 確保 acquire 不凍結 UI，Stop 按鈕透過 event flag 中斷。

---

## 檔案結構（建議）

```text
gui/
├── main.py                  # 入口，建立 QApplication + MainWindow
├── main_window.py           # 主視窗，管理 dock panels
├── panels/
│   ├── setup_panel.py       # [1] Connection + Config
│   ├── experiment_panel.py  # [2] Experiment selector + params form
│   ├── plot_panel.py        # [3] Matplotlib canvas
│   ├── result_panel.py      # [4] Fit results + log
│   └── data_browser.py      # [6] Data browser dock
├── dialogs/
│   └── autocal_dialog.py    # Auto Calibrate dialog
├── workers/
│   └── acquire_worker.py    # QThread wrapper for acquire
└── utils/
    ├── param_form.py        # 動態生成參數欄位
    └── config_bridge.py     # ExperimentConfig ↔ GUI 雙向同步
```

---

## 實作優先順序

| Priority | 功能 |
| --- | --- |
| P0 | Connection + Config 載入 |
| P0 | Experiment 選擇 + 靜態參數表單 |
| P0 | Run → acquire → 顯示結果圖 |
| P1 | Live plot 更新（QThread signal） |
| P1 | Fit 結果顯示 + Update Config 按鈕 |
| P1 | save_data() + Data Browser（File List + Plot Viewer） |
| P2 | Data Browser mag/phase/avgi/avgq 切換 |
| P2 | Auto Calibrate dialog |
| P2 | Yoko 控制 |
| P3 | RB / Tomography 頁面 |
| P3 | 完整 Log |

---

## 依賴套件

```text
pip install PySide6 matplotlib numpy h5py
```

qick_workspace 本身的 import 路徑不變，GUI 只是在外層包一個 Qt 殼。
