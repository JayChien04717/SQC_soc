"""
FastAPI server for the QICK job queue system.

Provides HTTP endpoints for submitting, monitoring, and cancelling experiment jobs.

Run with:
    cd /Users/jay/Desktop/test/SQC_soc
    python -m uvicorn qick_job_server.server:app --host 0.0.0.0 --port 8585
"""

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_database, Database
from .models import Job, JobStatus
from .id_generator import IDGenerator

# Initialize FastAPI app
app = FastAPI(
    title="QICK Experiment Job Server",
    description="Job queue for multi-PC QICK experiment scheduling",
    version="1.0.0",
)

_db: Optional[Database] = None


def get_db() -> Session:
    """FastAPI dependency to get a database session."""
    global _db
    if _db is None:
        _db = get_database()
    session = _db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# Pydantic models for request/response
# ============================================================================


class JobSubmission(BaseModel):
    """Request body for submitting a new job."""
    experiment_class: str        # e.g., "ResonatorSpec"
    experiment_module: str       # e.g., "qick_workspace.newscrip.s002_res_spec_ge"
    experiment_config: Dict[str, Any]  # run_cfg overrides (sweeps encoded as dicts)
    qubit: str = "Q1"            # Qubit name
    py_avg: int = 1              # Software averages
    user: str = "anonymous"      # Username
    priority: int = 0            # Higher = runs sooner

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_class": "ResonatorSpec",
                "experiment_module": "qick_workspace.newscrip.s002_res_spec_ge",
                "experiment_config": {
                    "steps": 101,
                    "res_gain_ge": 0.1,
                    "res_freq_ge": {"__sweep__": True, "loop": "freqloop", "start": 5330, "stop": 5370},
                },
                "qubit": "Q1",
                "py_avg": 10,
                "user": "jay",
                "priority": 0,
            }
        }


class JobResponse(BaseModel):
    """Response after submitting a job."""
    job_id: str
    status: str
    created_at: datetime
    queue_position: Optional[int] = None

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    user: str
    experiment_class: str
    experiment_module: str
    qubit: str
    py_avg: int
    status: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None

    class Config:
        from_attributes = True


class QueueResponse(BaseModel):
    """Response for queue listing."""
    pending_jobs: List[JobStatusResponse]
    running_job: Optional[JobStatusResponse] = None
    total_pending: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pending_jobs: int
    running_jobs: int


# ============================================================================
# Helper: convert Job ORM → JobStatusResponse
# ============================================================================

def _job_to_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        user=job.user,
        experiment_class=job.experiment_class,
        experiment_module=job.experiment_module,
        qubit=job.qubit,
        py_avg=job.py_avg,
        status=job.status.value,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        result_path=job.result_path,
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "QICK Experiment Job Server",
        "version": "1.0.0",
        "endpoints": {
            "submit": "POST /jobs/submit",
            "status": "GET /jobs/{job_id}",
            "queue": "GET /jobs/queue",
            "history": "GET /jobs/history",
            "cancel": "DELETE /jobs/{job_id}",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(session: Session = Depends(get_db)):
    """Health check with queue statistics."""
    try:
        pending = session.query(Job).filter_by(status=JobStatus.PENDING).count()
        running = session.query(Job).filter_by(status=JobStatus.RUNNING).count()
        return HealthResponse(status="healthy", pending_jobs=pending, running_jobs=running)
    except Exception:
        return HealthResponse(status="unhealthy", pending_jobs=0, running_jobs=0)


@app.post("/jobs/submit", response_model=JobResponse, tags=["jobs"])
async def submit_job(submission: JobSubmission, session: Session = Depends(get_db)):
    """
    Submit a new experiment job to the queue.

    Jobs execute in priority order (higher first), then FIFO for same priority.
    """
    job_id = IDGenerator.generate_job_id(session)

    job = Job(
        job_id=job_id,
        user=submission.user,
        experiment_class=submission.experiment_class,
        experiment_module=submission.experiment_module,
        experiment_config=json.dumps(submission.experiment_config),
        qubit=submission.qubit,
        py_avg=submission.py_avg,
        status=JobStatus.PENDING,
        priority=submission.priority,
    )

    session.add(job)
    session.flush()

    # Calculate queue position
    queue_position = (
        session.query(Job)
        .filter(Job.status == JobStatus.PENDING)
        .filter(
            (Job.priority > submission.priority)
            | ((Job.priority == submission.priority) & (Job.created_at < job.created_at))
        )
        .count()
        + 1
    )

    print(f"[SERVER] Job submitted: {job_id} by {submission.user} ({submission.experiment_class})")

    return JobResponse(
        job_id=job_id,
        status=job.status.value,
        created_at=job.created_at,
        queue_position=queue_position,
    )


@app.get("/jobs/queue", response_model=QueueResponse, tags=["jobs"])
async def list_queue(session: Session = Depends(get_db)):
    """List all pending and running jobs."""
    running_job = session.query(Job).filter_by(status=JobStatus.RUNNING).first()
    pending_jobs = (
        session.query(Job)
        .filter_by(status=JobStatus.PENDING)
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .all()
    )

    return QueueResponse(
        pending_jobs=[_job_to_response(j) for j in pending_jobs],
        running_job=_job_to_response(running_job) if running_job else None,
        total_pending=len(pending_jobs),
    )


@app.get("/jobs/history", tags=["jobs"])
async def get_job_history(
    limit: int = 50,
    user: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_db),
):
    """Get recent job history (newest first)."""
    query = session.query(Job)

    if user:
        query = query.filter_by(user=user)
    if status:
        try:
            query = query.filter_by(status=JobStatus(status))
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()
    return [job.to_dict() for job in jobs]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, session: Session = Depends(get_db)):
    """Get detailed status of a specific job."""
    job = session.query(Job).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


from fastapi.responses import FileResponse
import os

@app.get("/jobs/{job_id}/result", tags=["jobs"])
async def get_job_result(job_id: str, session: Session = Depends(get_db)):
    """Download the serialized experiment result file (Pickle)."""
    job = session.query(Job).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job must be completed to get results. Current status: {job.status.value}")
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(404, "No result file available for this job")
    
    return FileResponse(
        path=job.result_path, 
        filename=f"{job_id}_result.pkl",
        media_type="application/octet-stream"
    )

@app.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(job_id: str, session: Session = Depends(get_db)):
    """Cancel a pending job. Running jobs cannot be cancelled."""
    job = session.query(Job).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(400, f"Cannot cancel: status is {job.status.value}")

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc)
    session.flush()

    print(f"[SERVER] Job cancelled: {job_id}")
    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting QICK Experiment Job Server...")
    print("API docs: http://localhost:8585/docs")
    uvicorn.run(app, host="0.0.0.0", port=8585)
