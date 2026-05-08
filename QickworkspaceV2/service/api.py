"""
Service layer: FastAPI REST interface for remote experiment execution.

Endpoints
---------
POST  /experiments/run              — submit and run an experiment
GET   /experiments/{exp_id}/status  — poll job status
GET   /experiments/{exp_id}/result  — retrieve ExperimentData as JSON
GET   /calibrations/{qubit}/params  — fetch all calibration params for a qubit
POST  /calibrations/{qubit}/set     — update a single calibration param
GET   /calibrations/{qubit}/stale   — list stale param keys for a qubit
POST  /calibrate/{qubit}/run        — trigger AutoCalibrate pipeline

Usage
-----
    uvicorn QickworkspaceV2.service.api:app --host 0.0.0.0 --port 8000

    # or from within Python:
    import uvicorn
    from QickworkspaceV2.service.api import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# ── Job registry ──────────────────────────────────────────────────────────────

_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _new_job(exp_type: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    with _JOB_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "experiment_type": exp_type,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
        }
    return job_id


def _set_job(job_id: str, **kwargs):
    with _JOB_LOCK:
        _JOBS[job_id].update(kwargs)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(cal_store=None, config_all=None) -> "FastAPI":
    """
    Create the FastAPI application.

    Parameters
    ----------
    cal_store : CalibrationStore or None
        If provided, calibration endpoints will use this store.
    config_all : ExperimentConfig or None
        Required for /calibrate endpoints.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi and pydantic are required: pip install fastapi uvicorn pydantic")

    app = FastAPI(
        title="Reconstruct Experiment API",
        description="REST interface for QICK quantum calibration experiments",
        version="1.0.0",
    )

    # ── Request/response models ────────────────────────────────────────────────

    class RunRequest(BaseModel):
        experiment_type: str
        config: dict
        py_avg: int = 5
        kwargs: dict = {}

    class CalParamSet(BaseModel):
        key: str
        value: Any

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_experiment(exp_type: str):
        from ..experiments import (
            ResonatorSpec, QubitSpec, TimeRabi, PowerRabi,
            Ramsey, SpinEcho, T1, ResonatorSpec_ef,
            QubitSpecEf, PowerRabiEf, RamseyEf, T1Ef,
            AllXY, RandomizedBenchmarking, Tomography, TOF,
        )
        registry = {
            "res_spec":       ResonatorSpec,
            "qubit_spec":     QubitSpec,
            "time_rabi":      TimeRabi,
            "power_rabi":     PowerRabi,
            "ramsey":         Ramsey,
            "spin_echo":      SpinEcho,
            "t1":             T1,
            "res_spec_ef":    ResonatorSpec_ef,
            "qubit_spec_ef":  QubitSpecEf,
            "power_rabi_ef":  PowerRabiEf,
            "ramsey_ef":      RamseyEf,
            "t1_ef":          T1Ef,
            "allxy":          AllXY,
            "rb":             RandomizedBenchmarking,
            "tomography":     Tomography,
            "tof":            TOF,
        }
        cls = registry.get(exp_type)
        if cls is None:
            raise ValueError(f"Unknown experiment type: {exp_type!r}. Valid: {list(registry.keys())}")
        return cls

    def _run_job(job_id: str, exp_type: str, cfg: dict, py_avg: int, kwargs: dict):
        _set_job(job_id, status="running", started_at=datetime.now().isoformat())
        try:
            cls = _resolve_experiment(exp_type)
            expt = cls(cfg)
            result = expt.run(py_avg, **kwargs)
            from ..data.serializer import experiment_to_json
            _set_job(job_id, status="done", result=experiment_to_json(result),
                     finished_at=datetime.now().isoformat())
        except Exception as exc:
            _set_job(job_id, status="error", error=traceback.format_exc(),
                     finished_at=datetime.now().isoformat())

    # ── Experiment endpoints ──────────────────────────────────────────────────

    @app.post("/experiments/run", status_code=202)
    async def run_experiment(req: RunRequest, background_tasks: BackgroundTasks):
        """Submit an experiment; returns job_id immediately."""
        job_id = _new_job(req.experiment_type)
        background_tasks.add_task(
            _run_job, job_id, req.experiment_type, req.config, req.py_avg, req.kwargs
        )
        return {"job_id": job_id, "status": "pending"}

    @app.get("/experiments/{exp_id}/status")
    async def get_status(exp_id: str):
        """Poll job status: pending | running | done | error."""
        job = _JOBS.get(exp_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {exp_id!r} not found")
        return {"job_id": exp_id, "status": job["status"],
                "experiment_type": job["experiment_type"],
                "created_at": job["created_at"],
                "error": job.get("error")}

    @app.get("/experiments/{exp_id}/result")
    async def get_result(exp_id: str):
        """Retrieve the full ExperimentData JSON for a completed job."""
        job = _JOBS.get(exp_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {exp_id!r} not found")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not done")
        return JSONResponse(content=job["result"])

    @app.get("/experiments")
    async def list_experiments():
        """List all submitted jobs (id, type, status)."""
        with _JOB_LOCK:
            return [
                {"id": j["id"], "experiment_type": j["experiment_type"],
                 "status": j["status"], "created_at": j["created_at"]}
                for j in _JOBS.values()
            ]

    # ── Calibration endpoints ─────────────────────────────────────────────────

    @app.get("/calibrations/{qubit}/params")
    async def get_params(qubit: str):
        """Return all calibration parameters for *qubit*."""
        if cal_store is None:
            raise HTTPException(status_code=503, detail="CalibrationStore not configured")
        return cal_store.to_flat_dict(qubit)

    @app.post("/calibrations/{qubit}/set")
    async def set_param(qubit: str, body: CalParamSet):
        """Update a single calibration parameter for *qubit*."""
        if cal_store is None:
            raise HTTPException(status_code=503, detail="CalibrationStore not configured")
        cal_store.set(qubit, body.key, body.value)
        return {"qubit": qubit, "key": body.key, "value": body.value}

    @app.get("/calibrations/{qubit}/stale")
    async def get_stale(qubit: str):
        """Return list of stale parameter keys for *qubit*."""
        if cal_store is None:
            raise HTTPException(status_code=503, detail="CalibrationStore not configured")
        keys = cal_store.all_keys(qubit)
        return {"qubit": qubit, "stale_keys": [k for k in keys if cal_store.is_stale(qubit, k)]}

    # ── AutoCalibrate endpoint ────────────────────────────────────────────────

    class AutoCalibrateRequest(BaseModel):
        skip: list[str] = []

    @app.post("/calibrate/{qubit}/run", status_code=202)
    async def run_autocal(qubit: str, req: AutoCalibrateRequest, background_tasks: BackgroundTasks):
        """Trigger the full AutoCalibrate pipeline for *qubit* in the background."""
        if config_all is None:
            raise HTTPException(status_code=503, detail="config_all not configured")
        job_id = _new_job(f"AutoCalibrate/{qubit}")

        def _autocal_job():
            from ..calibration.pipeline import AutoCalibrate
            _set_job(job_id, status="running", started_at=datetime.now().isoformat())
            try:
                cal = AutoCalibrate(config_all, qubit, cal_store=cal_store)
                cal.run(skip=tuple(req.skip))
                _set_job(job_id, status="done", result=cal.results,
                         finished_at=datetime.now().isoformat())
            except Exception as exc:
                _set_job(job_id, status="error", error=traceback.format_exc(),
                         finished_at=datetime.now().isoformat())

        background_tasks.add_task(_autocal_job)
        return {"job_id": job_id, "qubit": qubit, "status": "pending"}

    # ── Health check ──────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    return app


# ── Default app instance (used by uvicorn) ────────────────────────────────────

if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
