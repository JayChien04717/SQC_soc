"""
FastAPI server for the QICK job queue system.

Provides HTTP endpoints for:
- Submitting experiment jobs
- Checking job status
- Listing the job queue
- Cancelling pending jobs
- Streaming job output

Run with:
    cd c:/Users/cluster/Desktop/SQC_soc-jobserver
    python -m uvicorn qick_workspace.qick_job_server.server:app --host 0.0.0.0 --port 8585

Or for development with auto-reload:
    python -m uvicorn qick_workspace.qick_job_server.server:app --reload --port 8585
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_database, Database
from .models import Job, JobStatus, JobOutput
from .id_generator import IDGenerator

# Initialize FastAPI app
app = FastAPI(
    title="QICK Experiment Job Server",
    description="Central job queue for multi-user QICK experiment scheduling",
    version="1.0.0",
)

# Database instance (created on first request)
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
# Pydantic models for request/response validation
# ============================================================================


class JobSubmission(BaseModel):
    """Request body for submitting a new job."""

    experiment_class: str   # e.g., "ResonatorSpec"
    experiment_module: str  # e.g., "qick_workspace.newscrip.s002_res_spec_ge"
    run_cfg: Dict[str, Any] # Full experiment configuration
    qubit: str              # e.g., "Q1"
    py_avg: int             # Number of software averages
    user: str               # Username of submitter
    priority: int = 0       # Higher priority = runs sooner

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_class": "ResonatorSpec",
                "experiment_module": "qick_workspace.newscrip.s002_res_spec_ge",
                "run_cfg": {"res_freq_ge": 5351.559, "reps": 100},
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
    qubit: str
    py_avg: int
    status: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data_path: Optional[str] = None
    error_message: Optional[str] = None

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
    database_connected: bool
    pending_jobs: int
    running_jobs: int


class JobOutputResponse(BaseModel):
    """Response for job output streaming."""
    job_id: str
    output: str
    line_count: int
    is_complete: bool
    offset: int

    class Config:
        from_attributes = True


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
            "cancel": "DELETE /jobs/{job_id}",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(session: Session = Depends(get_db)):
    """Health check endpoint with queue statistics."""
    try:
        pending_count = session.query(Job).filter_by(status=JobStatus.PENDING).count()
        running_count = session.query(Job).filter_by(status=JobStatus.RUNNING).count()

        return HealthResponse(
            status="healthy",
            database_connected=True,
            pending_jobs=pending_count,
            running_jobs=running_count,
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            pending_jobs=0,
            running_jobs=0,
        )


def _job_to_status_response(job: Job) -> JobStatusResponse:
    """Convert a Job ORM object to JobStatusResponse."""
    return JobStatusResponse(
        job_id=job.job_id,
        user=job.user,
        experiment_class=job.experiment_class,
        qubit=job.qubit,
        py_avg=job.py_avg,
        status=job.status.value,
        priority=job.priority,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        data_path=job.data_path,
        error_message=job.error_message,
    )


@app.post("/jobs/submit", response_model=JobResponse, tags=["jobs"])
async def submit_job(submission: JobSubmission, session: Session = Depends(get_db)):
    """
    Submit a new experiment job to the queue.

    Jobs are executed in priority order (higher priority first),
    then by submission time (FIFO for same priority).
    """
    job_id = IDGenerator.generate_job_id(session)

    job = Job(
        job_id=job_id,
        user=submission.user,
        experiment_class=submission.experiment_class,
        experiment_module=submission.experiment_module,
        run_cfg=json.dumps(submission.run_cfg),
        qubit=submission.qubit,
        py_avg=submission.py_avg,
        status=JobStatus.PENDING,
        priority=submission.priority,
    )

    session.add(job)
    session.flush()

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

    print(f"[SERVER] Job submitted: {job_id} by {submission.user}")

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
        pending_jobs=[_job_to_status_response(j) for j in pending_jobs],
        running_job=_job_to_status_response(running_job) if running_job else None,
        total_pending=len(pending_jobs),
    )


@app.get("/jobs/history", tags=["jobs"])
async def get_job_history(
    limit: int = 50,
    user: Optional[str] = None,
    status: Optional[str] = None,
    session: Session = Depends(get_db),
):
    """Get recent job history."""
    query = session.query(Job)

    if user:
        query = query.filter_by(user=user)
    if status:
        try:
            status_enum = JobStatus(status)
            query = query.filter_by(status=status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid: {[s.value for s in JobStatus]}",
            )

    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()

    return [
        {
            "job_id": job.job_id,
            "user": job.user,
            "experiment_class": job.experiment_class,
            "qubit": job.qubit,
            "status": job.status.value,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "data_path": job.data_path,
        }
        for job in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, session: Session = Depends(get_db)):
    """Get the status of a specific job."""
    job = session.query(Job).filter_by(job_id=job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _job_to_status_response(job)


@app.delete("/jobs/{job_id}", tags=["jobs"])
async def cancel_job(job_id: str, session: Session = Depends(get_db)):
    """Cancel a pending job."""
    job = session.query(Job).filter_by(job_id=job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job {job_id}: status is {job.status.value}",
        )

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    session.flush()

    print(f"[SERVER] Job cancelled: {job_id}")
    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


@app.get("/jobs/{job_id}/output", response_model=JobOutputResponse, tags=["jobs"])
async def get_job_output(
    job_id: str,
    offset: int = 0,
    session: Session = Depends(get_db),
):
    """
    Get output from a running or completed job.

    Use offset for incremental polling:
    1. First call with offset=0
    2. Subsequent calls with offset=previous line_count
    3. Stop when is_complete=True
    """
    job = session.query(Job).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output = session.query(JobOutput).filter_by(job_id=job_id).first()

    if not output:
        is_complete = job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        return JobOutputResponse(
            job_id=job_id, output="", line_count=0,
            is_complete=is_complete, offset=0,
        )

    full_text = output.output_text or ""

    if offset > 0:
        lines = full_text.split('\n')
        partial_text = '\n'.join(lines[offset:]) if offset < len(lines) else ""
    else:
        partial_text = full_text

    return JobOutputResponse(
        job_id=job_id,
        output=partial_text,
        line_count=output.line_count,
        is_complete=output.is_complete,
        offset=offset,
    )


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting QICK Experiment Job Server...")
    print("API docs available at: http://localhost:8585/docs")
    uvicorn.run(app, host="0.0.0.0", port=8585)
