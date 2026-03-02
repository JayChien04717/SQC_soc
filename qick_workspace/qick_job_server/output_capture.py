"""
Captures stdout/stderr during job execution and persists to file + database.

Usage:
    with OutputCapture(job_id, db, log_dir) as capture:
        # All print() calls in this block are captured
        expt.run(...)
"""

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
from io import StringIO


class OutputCapture:
    """
    Context manager that captures stdout/stderr and writes to:
    1. A log file (for persistence)
    2. The database (for serving to clients via polling)
    3. Original stdout/stderr (so worker terminal still shows output)
    """

    def __init__(
        self,
        job_id: str,
        db: "Database",
        log_dir: Path,
        flush_interval: float = 1.0,
    ):
        self.job_id = job_id
        self.db = db
        self.log_dir = Path(log_dir)
        self.flush_interval = flush_interval

        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[Path] = None
        self.buffer = StringIO()
        self.line_count = 0
        self.lock = threading.Lock()
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None
        self._flush_timer: Optional[threading.Timer] = None
        self._stopped = False

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.job_id}.log"
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        self._init_db_record()

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._original_stdout, self)
        sys.stderr = _TeeWriter(self._original_stderr, self)

        self._schedule_flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stopped = True
        if self._flush_timer:
            self._flush_timer.cancel()

        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

        self._flush_to_db(is_complete=True)

        if self.log_file:
            self.log_file.close()

        return False

    def write(self, text: str):
        """Write text to buffer and log file."""
        with self.lock:
            self.buffer.write(text)
            if self.log_file:
                self.log_file.write(text)
                self.log_file.flush()
            self.line_count += text.count('\n')

    def _init_db_record(self):
        """Create initial JobOutput record in database."""
        from .models import JobOutput

        try:
            with self.db.session() as session:
                existing = session.query(JobOutput).filter_by(job_id=self.job_id).first()
                if not existing:
                    output = JobOutput(
                        job_id=self.job_id,
                        output_text="",
                        line_count=0,
                        is_complete=False,
                    )
                    session.add(output)
        except Exception as e:
            if self._original_stderr:
                print(f"[OUTPUT_CAPTURE] Warning: Failed to init DB record: {e}",
                      file=self._original_stderr)

    def _schedule_flush(self):
        """Schedule next periodic DB flush."""
        if self._stopped:
            return
        self._flush_timer = threading.Timer(self.flush_interval, self._periodic_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _periodic_flush(self):
        """Periodic flush callback."""
        if self._stopped:
            return
        self._flush_to_db(is_complete=False)
        self._schedule_flush()

    def _flush_to_db(self, is_complete: bool = False):
        """Flush buffered output to database."""
        from .models import JobOutput

        with self.lock:
            text = self.buffer.getvalue()
            count = self.line_count

        if not text and not is_complete:
            return

        try:
            with self.db.session() as session:
                output = session.query(JobOutput).filter_by(job_id=self.job_id).first()
                if output:
                    output.output_text = text
                    output.line_count = count
                    output.is_complete = is_complete
                    output.last_updated = datetime.now()
        except Exception as e:
            if self._original_stderr:
                print(f"[OUTPUT_CAPTURE] Warning: Failed to flush to DB: {e}",
                      file=self._original_stderr)


class _TeeWriter:
    """Writes to both original stream (worker terminal) and capture buffer."""

    def __init__(self, original: Optional[TextIO], capture: OutputCapture):
        self.original = original
        self.capture = capture
        self._encoding = getattr(original, 'encoding', 'utf-8') if original else 'utf-8'

    def write(self, text: str):
        if self.original:
            try:
                self.original.write(text)
                self.original.flush()
            except Exception:
                pass
        try:
            self.capture.write(text)
        except Exception:
            pass

    def flush(self):
        if self.original:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        if self.original and hasattr(self.original, 'isatty'):
            try:
                return self.original.isatty()
            except Exception:
                pass
        return False

    def fileno(self):
        if self.original and hasattr(self.original, 'fileno'):
            return self.original.fileno()
        raise OSError("No fileno available")

    @property
    def encoding(self):
        return self._encoding

    @property
    def buffer(self):
        if self.original and hasattr(self.original, 'buffer'):
            return self.original.buffer
        return None

    def __getattr__(self, name):
        if self.original and hasattr(self.original, name):
            return getattr(self.original, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
