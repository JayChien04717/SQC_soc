# Service Usage Guide & Multi-User Coordination

## Quick Start

### 1. Start the server on the lab machine (where QICK is connected)

```python
# server_start.py  — run once on the lab PC
import uvicorn
from reconstruct.service.api import create_app
from reconstruct.tools.system_tool import ExperimentConfig
from reconstruct.config.system_cfg import config_list, DATA_PATH
from reconstruct.calibration.store import CalibrationStore
from reconstruct.backend.qick_backend import QICKBackend

backend = QICKBackend.from_pyro4("192.168.10.82", 8888)
backend.activate()

config_all = ExperimentConfig(config_list)
store      = CalibrationStore("cal_Q1.json")

app = create_app(cal_store=store, config_all=config_all, backend=backend)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# or from shell (bare app — calibration endpoints need the factory above)
uvicorn reconstruct.service.api:app --host 0.0.0.0 --port 8000
```

### 2. Connect from any client machine on the same network

```python
import requests, time

SERVER = "http://192.168.10.100:8000"   # lab PC IP
```

---

## Endpoints Reference

| Method | URL | Body / Params | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{ "status": "ok" }` |
| `POST` | `/experiments/run` | `RunRequest` JSON | `{ "job_id": "...", "status": "pending" }` |
| `GET` | `/experiments/{id}/status` | — | `{ "status": "pending\|running\|done\|error" }` |
| `GET` | `/experiments/{id}/result` | — | `ExperimentData` JSON |
| `GET` | `/experiments` | — | list of all jobs |
| `GET` | `/calibrations/{qubit}/params` | — | all stored params |
| `POST` | `/calibrations/{qubit}/set` | `{ "key": ..., "value": ... }` | echo |
| `GET` | `/calibrations/{qubit}/stale` | — | list of stale keys |
| `POST` | `/calibrate/{qubit}/run` | `{ "skip": [...] }` | `{ "job_id": ... }` |

### `RunRequest` schema

```json
{
  "experiment_type": "qubit_spec",
  "config": { "res_freq_ge": 6717, "qb_freq_ge": 2872, ... },
  "py_avg": 10,
  "kwargs": {}
}
```

Valid `experiment_type` strings:
`res_spec`, `qubit_spec`, `time_rabi`, `power_rabi`, `ramsey`, `spin_echo`, `t1`,
`res_spec_ef`, `qubit_spec_ef`, `power_rabi_ef`, `ramsey_ef`, `t1_ef`,
`allxy`, `rb`, `tomography`, `tof`

---

## Basic Client Workflow

```python
import requests, time

SERVER = "http://192.168.10.100:8000"

# 1. Submit
resp = requests.post(f"{SERVER}/experiments/run", json={
    "experiment_type": "qubit_spec",
    "config": run_cfg,       # flat dict from config_all.get_qubit("Q1")
    "py_avg": 10,
})
job_id = resp.json()["job_id"]
print("submitted:", job_id)

# 2. Poll
while True:
    status = requests.get(f"{SERVER}/experiments/{job_id}/status").json()["status"]
    if status == "done":
        break
    if status == "error":
        raise RuntimeError(requests.get(f"{SERVER}/experiments/{job_id}/status").json()["error"])
    time.sleep(3)

# 3. Fetch result
data = requests.get(f"{SERVER}/experiments/{job_id}/result").json()
print("fit_params:", data["fit_params"])
```

### Helper class for notebooks

```python
class RemoteExperiment:
    def __init__(self, server, exp_type, config, py_avg=10, poll_interval=3):
        self.server = server
        self.payload = {"experiment_type": exp_type, "config": config, "py_avg": py_avg}
        self.poll_interval = poll_interval

    def run(self):
        r = requests.post(f"{self.server}/experiments/run", json=self.payload)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        while True:
            s = requests.get(f"{self.server}/experiments/{job_id}/status").json()
            if s["status"] == "done":
                return requests.get(f"{self.server}/experiments/{job_id}/result").json()
            if s["status"] == "error":
                raise RuntimeError(s["error"])
            time.sleep(self.poll_interval)

# Usage
result = RemoteExperiment(SERVER, "t1", run_cfg, py_avg=20).run()
```

---

## Multi-User Coordination

The current service runs jobs **one-at-a-time in the background**. If two users submit simultaneously, their QICK programs will overlap on the hardware and both results will be corrupted.

### The problem

```
User A  POST /experiments/run  → job a1f3  (running on QICK)
User B  POST /experiments/run  → job 9c2e  (also starts on QICK immediately)
                                            ← hardware conflict, both results bad
```

### Solution 1 — Hardware lock (recommended, minimal change)

Add a `threading.Lock` around the QICK run on the server side. Edit `api.py`'s `_run_job`:

```python
# in server_start.py, before create_app()
import threading
_HW_LOCK = threading.Lock()

# monkey-patch into the app after creation, or pass as dependency
# simplest: subclass create_app and wrap _run_job
```

Or edit `reconstruct/service/api.py` directly — add one lock around the experiment `.run()` call:

```python
_HW_LOCK = threading.Lock()          # module-level, next to _JOBS

def _run_job(job_id, exp_type, cfg, py_avg, kwargs):
    _set_job(job_id, status="queued")
    with _HW_LOCK:                   # blocks until hardware is free
        _set_job(job_id, status="running", started_at=datetime.now().isoformat())
        try:
            cls = _resolve_experiment(exp_type)
            expt = cls(cfg, backend=backend)
            result = expt.run(py_avg, **kwargs)
            ...
```

Now jobs queue automatically. Clients see `"queued"` while waiting, `"running"` when on hardware.

### Solution 2 — Per-qubit locks

If users work on different qubits and the hardware is truly independent per-channel:

```python
_QB_LOCKS: dict[str, threading.Lock] = {}

def _get_qb_lock(cfg: dict) -> threading.Lock:
    qb = cfg.get("name", "default")
    if qb not in _QB_LOCKS:
        _QB_LOCKS[qb] = threading.Lock()
    return _QB_LOCKS[qb]
```

This allows Q1 and Q2 experiments to run concurrently while preventing two Q1 jobs from overlapping.

### Solution 3 — Priority booking via CalibrationStore

For labs where scheduled runs are preferred over first-come-first-served, add a simple reservation field to the store:

```python
# Client A reserves a slot
requests.post(f"{SERVER}/calibrations/Q1/set", json={
    "key": "__reserved_by__", "value": "Alice"
})

# Client B checks before submitting
params = requests.get(f"{SERVER}/calibrations/Q1/params").json()
if "__reserved_by__" in params:
    print("Q1 is reserved by", params["__reserved_by__"])
else:
    # safe to submit
    ...
```

This is a soft convention, not enforced — useful for team communication, not race-condition safety.

---

## Practical Lab Protocol (team of 2–4 people)

1. **One person starts the server** at the beginning of the day and keeps it running.
2. **Check `/health`** before submitting to confirm the server is up.
3. **Check `/experiments`** to see if a job is currently `running` — don't submit while someone else is active.
4. **Use descriptive qubit names** in your config (`"name": "Q1"`) so the job list shows who is using which qubit.
5. **Fetch your result immediately** after `done` — the in-memory job registry is lost on server restart.
6. After a run, **write key results back** to `CalibrationStore` via `POST /calibrations/{qubit}/set` so others see the latest params.

```python
# After fitting qubit spec
requests.post(f"{SERVER}/calibrations/Q1/set", json={
    "key": "qb_freq_ge", "value": 4502.3
})
```

---

## Checking Stale Parameters Before a Run

Before starting a long calibration, check what needs re-running:

```python
stale = requests.get(f"{SERVER}/calibrations/Q1/stale").json()["stale_keys"]
print("needs recal:", stale)
# → ['qb_freq_ge', 'pi_gain_ge']
```

Then run only the stale steps via AutoCalibrate:

```python
requests.post(f"{SERVER}/calibrate/Q1/run", json={
    "skip": ["spin_echo", "ss_opt"]   # skip steps that are still fresh
})
```
