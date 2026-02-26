# QICK Experiment Job Server

多 PC 共用的 QICK 實驗排程系統。PC1、PC2 透過 HTTP 提交實驗，Worker 按順序執行。

## 架構

```
PC1 (Jupyter) ──┐
                ├── HTTP ──► FastAPI Server (port 8585)
PC2 (Jupyter) ──┘                    │
                                     │ polls DB
                                     ▼
                              Worker (single)
                                     │ Pyro4
                                     ▼
                              QICK RFSoC (qickpyro.py)
```

- **Server**：FastAPI HTTP 伺服器，管理 SQLite job queue
- **Worker**：單一 process，透過 Pyro4 連 QICK，依序執行實驗
- **Client**：Python library，在 Jupyter 中提交 / 查詢 / 取消 job

## 安裝

```bash
pip install fastapi uvicorn sqlalchemy pydantic requests
```

## 啟動

在可以連到 QICK 的 PC 上（例如 PC1）：

```bash
cd /Users/jay/Desktop/test/SQC_soc

# Terminal 1: API Server
python -m uvicorn qick_job_server.server:app --host 0.0.0.0 --port 8585

# Terminal 2: Worker
python -m qick_job_server.worker \
    --ns-host 192.168.10.179 \
    --ns-port 8887 \
    --proxy-name myqick
```

或用 startup script（兩者一起啟動）：

```bash
bash qick_job_server/start.sh
```

啟動後，API 文件在：http://localhost:8585/docs

## 使用方式（Jupyter Notebook）

### 基本：提交實驗並等待

透過新加入的 `.subjob()` 方法，你可以直接把實驗丟到背景執行，等待完成後，**它會自動把 `iqdata` 寫回物件中**給你的分析腳本用！

```python
from qick_workspace.newscrip.s002b_res_punchout_ge import Punchout

# 1. 在 notebook 初始化你的實驗
punchout = Punchout(soc, soccfg, run_cfg)

# 2. 提交給 Job Server，這會自動上傳、排入 queue 並等待結果
# (預設會連線到 http://127.0.0.1:8585)
punchout.subjob(py_avg=10, qubit="Q1", priority=0)

# 3. 跑完後，iqdata 已經在這個 punchout 物件內了，可以直接畫圖或分析！
punchout.analyze()
```

---

如果你想「手動控制」送出與抓取流程，可以呼叫底層的 `JobClient`：

```python
from qick_job_server.client import JobClient
client = JobClient("http://127.0.0.1:8585")

# 提交實驗
job_id = client.submit(...)

# 等待完成
client.wait_for_completion(job_id)

# 取得實驗結果資料 (包含 iqdata)
expt_data = client.get_result(job_id)
# expt_data.iqdata ...
```

### QickSweep1D 的寫法

在 notebook 中本來這樣寫：

```python
run_cfg.update([("res_freq_ge", QickSweep1D("freqloop", 5330, 5370))])
```

提交到 job server 時改用 `client.sweep()` helper：

```python
run_cfg={
    "res_freq_ge": client.sweep("freqloop", 5330, 5370),
    # 等同 {"__sweep__": True, "loop": "freqloop", "start": 5330, "stop": 5370}
}
```

Worker 會自動還原成 `QickSweep1D` 物件。

### 提交多個實驗

```python
jobs = []

# Resonator Spec
jobs.append(client.submit(
    experiment_class="ResonatorSpec",
    experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
    run_cfg={
        "steps": 101,
        "res_freq_ge": client.sweep("freqloop", 5330, 5370),
    },
    qubit="Q1", py_avg=10, user="jay",
))

# T1
jobs.append(client.submit(
    experiment_class="T1",
    experiment_module="qick_workspace.newscrip.s008_T1_ge",
    run_cfg={
        "steps": 100,
        "wait_time": client.sweep("waitloop", 0, 50),
        "relax_delay": 50,
    },
    qubit="Q1", py_avg=50, user="jay",
))

# 等待全部完成
for job_id in jobs:
    client.wait_for_completion(job_id)
```

### 優先排程

```python
# 低優先 (background)
client.submit(..., priority=0)

# 高優先（插隊！）
client.submit(..., priority=10)
```

Priority 越高越先執行。相同 priority 按提交順序 (FIFO)。

### 查看隊列

```python
client.print_queue()
```

輸出：
```
=== Job Queue ===

▶ Running: JOB-20260225-00001
  User: jay  |  ResonatorSpec  |  Q1

Pending: 2 jobs
  1. JOB-20260225-00003 — T1 (user: colleague, qubit: Q1, priority: 5)
  2. JOB-20260225-00002 — PowerRabi (user: jay, qubit: Q1, priority: 0)
```

### 取消 Job

```python
client.cancel_job("JOB-20260225-00002")  # 只能取消 pending 的
```

### 查看歷史紀錄

```python
history = client.get_history(limit=20)
history = client.get_history(user="jay")
history = client.get_history(status="failed")
```

## API Endpoints

| Endpoint | Method | 說明 |
|----------|--------|------|
| `/health` | GET | 健康檢查 + 隊列統計 |
| `/jobs/submit` | POST | 提交新實驗 |
| `/jobs/queue` | GET | 查看 pending/running jobs |
| `/jobs/{job_id}` | GET | 查看特定 job 狀態 |
| `/jobs/{job_id}/result`| GET | 下載實驗結果檔案 (iqdata) |
| `/jobs/{job_id}` | DELETE | 取消 pending job |
| `/jobs/history` | GET | 歷史紀錄 (可用 `?user=` `?status=` 篩選) |

## Worker CLI 選項

```
python -m qick_job_server.worker [OPTIONS]

Options:
  --ns-host TEXT       Pyro4 nameserver host (default: 192.168.10.179)
  --ns-port INT        Pyro4 nameserver port (default: 8887)
  --proxy-name TEXT    QICK proxy name (default: myqick)
  --poll-interval SEC  DB 輪詢間隔 (default: 2.0s)
```

## 實驗對照表

你的 notebook 中用到的實驗 class 和 module：

| 實驗 | experiment_class | experiment_module |
|------|-----------------|-------------------|
| TOF | `TOF` | `qick_workspace.newscrip.s001_time_of_flight` |
| Resonator Spec GE | `ResonatorSpec` | `qick_workspace.newscrip.s002_res_spec_ge` |
| Punchout GE | `Punchout` | `qick_workspace.newscrip.s002b_res_punchout_ge` |
| Qubit Spec GE | `QubitSpec` | `qick_workspace.newscrip.s003_qubit_spec_ge` |
| Time Rabi GE | `TimeRabi` | `qick_workspace.newscrip.s004_time_rabi_ge` |
| Power Rabi GE | `PowerRabi` | `qick_workspace.newscrip.s005_power_rabi_ge` |
| AAE | `AAE` | `qick_workspace.newscrip.s005a_AAE` |
| Ramsey GE | `Ramsey` | `qick_workspace.newscrip.s006_Ramsey_ge` |
| SpinEcho GE | `SpinEcho` | `qick_workspace.newscrip.s007_SpinEcho_ge` |
| T1 GE | `T1` | `qick_workspace.newscrip.s008_T1_ge` |
| SingleShot | `SingleShot_gef` | `qick_workspace.newscrip.s000_SingleShot_prog` |
| Resonator Spec EF | `ResonatorSpec_ef` | `qick_workspace.newscrip.s009_res_spec_ef` |
| Qubit Spec EF | `QubitSpec_ef` | `qick_workspace.newscrip.s010_qubit_spec_ef` |
| Power Rabi EF | `PowerRabi_ef` | `qick_workspace.newscrip.s011_power_rabi_ef` |
| Ramsey EF | `Ramsey_ef` | `qick_workspace.newscrip.s012_Ramsey_ef` |
| T1 EF | `T1_ef` | `qick_workspace.newscrip.s013_T1_ef` |

## Troubleshooting

### Server 啟動失敗 (port 被佔用)
```bash
lsof -ti:8585 | xargs kill -9
```

### Worker 連不上 QICK
- 確認 `qickpyro.py` 已在 QICK 儀器上啟動
- 確認 IP / port / proxy_name 正確
- 確認 PC 和 QICK 在同一個網段

### Job 一直卡在 pending
- 確認 Worker 有在跑
- 看 Worker 的 terminal 輸出有沒有 error

### 重複的 Worker
- Worker 有 PID lock 機制，同時只能跑一個
- 如果誤報 lock，手動刪除 `qick_job_server/worker.lock`

## 檔案結構

```
SQC_soc/qick_job_server/
├── __init__.py        # Package init, exports JobClient
├── models.py          # SQLAlchemy models (Job, IDCounter)
├── database.py        # SQLite connection manager
├── id_generator.py    # JOB-YYYYMMDD-NNNNN ID generator
├── server.py          # FastAPI HTTP server
├── worker.py          # Worker daemon (Pyro4 → QICK)
├── client.py          # Python client library
├── start.sh           # Startup script
├── jobs.db            # SQLite database (auto-created)
├── results/           # 實驗執行完畢生成的 pickle IQ 資料夾
└── README.md          # This file
```
