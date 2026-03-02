"""
Centralized ID generation for jobs.

Generates unique IDs in the format:
- Jobs: JOB-YYYYMMDD-NNNNN (e.g., JOB-20260302-00042)

IDs are thread-safe, persistent, and sortable by date.
"""

from datetime import datetime
from typing import Optional
import threading

from sqlalchemy.orm import Session

from .models import IDCounter


class IDGenerator:
    """Thread-safe ID generator using database-backed counters."""

    _lock = threading.Lock()

    @classmethod
    def generate_job_id(cls, session: Session) -> str:
        """
        Generate a unique job ID.

        Format: JOB-YYYYMMDD-NNNNN
        Example: JOB-20260302-00042
        """
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"JOB-{today}"
        counter = cls._get_next_counter(session, prefix)
        return f"{prefix}-{counter:05d}"

    @classmethod
    def generate_data_filename(cls, job_id: str, experiment_class: str) -> str:
        """
        Generate a data filename incorporating the job ID.

        Format: {job_id}_{ExperimentClass}.pkl
        Example: JOB-20260302-00042_ResonatorSpec.pkl
        """
        return f"{job_id}_{experiment_class}.pkl"

    @classmethod
    def _get_next_counter(cls, session: Session, prefix: str) -> int:
        """Get the next counter value for a prefix, creating if necessary."""
        with cls._lock:
            counter_row = session.query(IDCounter).filter_by(prefix=prefix).first()

            if counter_row is None:
                counter_row = IDCounter(prefix=prefix, counter=1)
                session.add(counter_row)
                session.flush()
                return 1
            else:
                counter_row.counter += 1
                session.flush()
                return counter_row.counter

    @classmethod
    def get_current_counter(cls, session: Session, prefix: str) -> Optional[int]:
        """Get the current counter value without incrementing."""
        counter_row = session.query(IDCounter).filter_by(prefix=prefix).first()
        return counter_row.counter if counter_row else None
