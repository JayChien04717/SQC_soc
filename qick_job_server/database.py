"""
Database connection and session management for the QICK job queue.

Uses SQLite for simplicity. The database file is stored alongside the package.
"""

from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent / "jobs.db"


class Database:
    """
    Database connection manager.

    Usage:
        db = Database()
        with db.session() as session:
            session.add(job)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with auto commit/rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new session (caller responsible for closing)."""
        return self.SessionLocal()


# Global singleton
_db_instance: Optional[Database] = None


def get_database(db_path: Optional[Path] = None) -> Database:
    """Get the global database instance, creating it if necessary."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
        _db_instance.create_tables()
    return _db_instance


def reset_database():
    """Reset the global database instance (for testing)."""
    global _db_instance
    _db_instance = None
