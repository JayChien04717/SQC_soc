"""
Unique job ID generator.

Format: JOB-YYYYMMDD-NNNNN (e.g., JOB-20260225-00001)
Uses a persistent counter in the database to survive restarts.
"""

from datetime import datetime, timezone
from sqlalchemy.orm import Session

from .models import IDCounter


class IDGenerator:
    """Generate unique, human-readable job IDs."""

    @staticmethod
    def generate_job_id(session: Session) -> str:
        """
        Generate the next job ID for today.

        Args:
            session: Active database session

        Returns:
            Unique job ID string like "JOB-20260225-00001"
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        prefix = f"JOB-{today}"

        # Get or create counter for today
        counter = session.query(IDCounter).filter_by(prefix=prefix).first()
        if counter is None:
            counter = IDCounter(prefix=prefix, counter=0)
            session.add(counter)
            session.flush()

        counter.counter += 1
        session.flush()

        return f"{prefix}-{counter.counter:05d}"
