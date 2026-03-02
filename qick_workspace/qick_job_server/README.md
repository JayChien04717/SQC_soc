# QICK Job Server

多使用者 QICK 實驗排程系統。讓多人可以同時提交實驗到佇列，由單一 Worker 依序執行，確保硬體的獨佔存取。

## 架構

```
┌─────────────┐    HTTP     ┌──────────────┐    SQLite    ┌──────────────┐
│  Notebook   │ ──────────► │  FastAPI     │ ◄──────────► │   Worker     │
│  (Client)   │  port 8585  │  Server      │   jobs.db    │   Daemon     │
└─────────────┘             └──────────────┘              └──────┬───────┘
                                                                 │
                                                           ┌─────▼─────┐
                                                           │   QICK    │
                                                           │   FPGA    │
                                                           └───────────┘
```

- **Server** — FastAPI HTTP 伺服器，管理 job queue（提交、查詢、取消）
- **Worker** — 單一執行緒 daemon，輪詢 DB 取得 pending job 並在 QICK 硬體上執行
- **Client** — Python 函式庫，從 Notebook 提交 job 並追蹤狀態

## 快速開始

### 1. 啟動 Server

```powershell
# 在 qick2env 環境
conda activate qick2env
cd C:\Users\cluster\Desktop\SQC_soc-jobserver
python -m uvicorn qick_workspace.qick_job_server.server:app --host 0.0.0.0 --port 8585
```

啟動後可在瀏覽器開啟 Swagger API 文件：http://127.0.0.1:8585/docs

### 2. 啟動 Worker

```powershell
# Mock 模式（無須 QICK 硬體，用 simulate=True 執行）
conda activate qick2env
cd C:\Users\cluster\Desktop\SQC_soc-jobserver
python -m qick_workspace.qick_job_server.worker --mock

# 真實硬體模式（需要 QICK FPGA 連線）
python -m qick_workspace.qick_job_server.worker
```

Worker 參數：
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--mock` | Mock 模式（simulate=True） | 否 |
| `--poll-interval` | 輪詢間隔（秒） | 2.0 |
| `--data-dir` | 結果儲存目錄 | `qick_workspace/data/job_results` |

### 3. 從 Notebook 提交實驗

```python
from qick_job_server.client import JobClient

client = JobClient("http://127.0.0.1:8585")

# 檢查伺服器狀態
client.health_check()

# 提交一個 job
job_id = client.submit(
    experiment_class="ResonatorSpec",
    experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
    run_cfg=config,        # addict.Dict 或 dict 格式的實驗設定
    qubit="Q1",
    py_avg=10,
    user="jay",
    priority=0,            # 數字越大越優先
)

# 等待完成（會即時串流輸出）
result = client.wait_for_completion(job_id)

# 載入實驗結果
result = client.get_result(job_id)
expt = result.load_expt()  # 取得完整的 BaseExperiment 物件
print(expt.iqdata)          # IQ data
print(expt.fit_params)      # 擬合參數
```

### 4. 直接從 BaseExperiment 提交（subjob）

```python
from qick_workspace.newscrip.s002_res_spec_ge import ResonatorSpec

expt = ResonatorSpec(soc=soc, soccfg=soccfg, config=config)

# 同步模式：等待完成後自動載入結果到 expt.iqdata
expt.subjob(py_avg=10, qubit="Q1", user="jay")
print(expt.iqdata)

# 非同步模式：立即返回 handle
handle = expt.subjob(py_avg=10, qubit="Q1", wait=False)
# ... 做其他事 ...
handle.wait()  # 等待完成
print(expt.iqdata)
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| POST | `/jobs/submit` | 提交新 job |
| GET | `/jobs/{job_id}` | 查詢 job 狀態 |
| GET | `/jobs/queue` | 查看佇列 |
| DELETE | `/jobs/{job_id}` | 取消 pending job |
| GET | `/jobs/{job_id}/output` | 串流 job 輸出 |
| GET | `/jobs/history` | 歷史記錄 |

## 其他 Client 功能

```python
client = JobClient()

# 查看佇列
client.print_queue()

# 取消 job
client.cancel(job_id)

# 查看歷史
client.get_history(limit=10, user="jay")

# 取得 job 輸出
client.get_output(job_id, offset=0)
```

## 檔案結構

```
qick_workspace/qick_job_server/
├── __init__.py          # 匯出 JobClient
├── models.py            # SQLAlchemy ORM (Job, JobOutput, IDCounter)
├── database.py          # SQLite 資料庫管理
├── id_generator.py      # JOB-YYYYMMDD-NNNNN ID 產生器
├── output_capture.py    # stdout/stderr 擷取器
├── server.py            # FastAPI 伺服器 (port 8585)
├── worker.py            # Worker daemon (執行實驗)
└── client.py            # Client 函式庫 (JobClient)
```

## 環境需求

此套件已在 Anaconda `qick2env` (Python 3.9) 環境測試通過，需要以下套件：

- `fastapi`
- `uvicorn`
- `sqlalchemy`
- `requests`
- `addict`
- `pydantic`

## Job 生命週期

```
PENDING ─────► RUNNING ─────► COMPLETED
    │              │
    ▼              ▼
CANCELLED      FAILED
```

- **PENDING** — 在佇列中等待執行
- **RUNNING** — Worker 正在執行
- **COMPLETED** — 執行成功，可透過 `data_path` 取得結果
- **FAILED** — 執行失敗，可透過 `error_message` 查看原因
- **CANCELLED** — 使用者取消
