"""
Database models for the QICK job queue system.

Tables:
- Job: Tracks experiment jobs with status, config, and results
- IDCounter: Persistent counter for unique job ID generation
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import enum

Base = declarative_base()


class JobStatus(enum.Enum):
    """Status states for a job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """
    Represents an experiment job in the queue.

    Attributes:
        job_id: Unique identifier (format: JOB-YYYYMMDD-NNNNN)
        user: Username of submitter
        experiment_class: Name of experiment class (e.g., "ResonatorSpec")
        experiment_module: Python module path (e.g., "qick_workspace.newscrip.s002_res_spec_ge")
        experiment_config: JSON-serialized experiment configuration (run_cfg overrides)
        qubit: Qubit name (e.g., "Q1")
        py_avg: Number of software averages
        status: Current job status
        priority: Higher priority jobs run first (default 0)
        created_at: When job was submitted
        started_at: When job started executing
        completed_at: When job finished (success or failure)
        error_message: Error details if job failed
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(20), unique=True, nullable=False, index=True)
    user = Column(String(100), nullable=False)
    experiment_class = Column(String(200), nullable=False)
    experiment_module = Column(String(300), nullable=False)
    experiment_config = Column(Text, nullable=False)  # JSON
    qubit = Column(String(20), nullable=False, default="Q1")
    py_avg = Column(Integer, nullable=False, default=1)

    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    priority = Column(Integer, default=0, index=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    error_message = Column(Text, nullable=True)
    result_path = Column(String(500), nullable=True)

    def __repr__(self):
        return f"<Job({self.job_id}, {self.experiment_class}, {self.status.value})>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "user": self.user,
            "experiment_class": self.experiment_class,
            "experiment_module": self.experiment_module,
            "qubit": self.qubit,
            "py_avg": self.py_avg,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result_path": self.result_path,
        }


class IDCounter(Base):
    """
    Persistent counter for ID generation.

    Stores the last used counter value for each date prefix
    to ensure uniqueness across server restarts.
    """
    __tablename__ = "id_counters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prefix = Column(String(50), unique=True, nullable=False, index=True)
    counter = Column(Integer, default=0, nullable=False)

    def __repr__(self):
        return f"<IDCounter({self.prefix}, {self.counter})>"
