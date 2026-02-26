"""
Job worker daemon for executing queued QICK experiments.

This worker:
- Connects to the QICK instrument via Pyro4 proxy
- Polls the database for pending jobs
- Executes experiments one at a time (hardware exclusivity)
- Dynamically imports experiment classes from qick_workspace.newscrip

Run with:
    cd /Users/jay/Desktop/test/SQC_soc
    python -m qick_job_server.worker --ns-host 192.168.10.179 --ns-port 8887 --proxy-name myqick
"""

import argparse
import atexit
import importlib
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import Pyro4

from qick.pyro import make_proxy
from qick.asm_v2 import QickSweep1D

from qick_job_server.database import get_database
from qick_job_server.models import Job, JobStatus


# ============================================================================
# PID-based worker lock (prevent multiple workers)
# ============================================================================

DEFAULT_LOCK_FILE = Path(__file__).parent / "worker.lock"


class WorkerLock:
    """PID-based lock to prevent multiple workers from running simultaneously."""

    def __init__(self, lock_file: Path = DEFAULT_LOCK_FILE):
        self.lock_file = lock_file
        self._acquired = False

    def acquire(self) -> bool:
        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())
                try:
                    os.kill(old_pid, 0)
                    raise RuntimeError(
                        f"Another worker is already running (PID {old_pid}). "
                        f"Delete {self.lock_file} if this is an error."
                    )
                except OSError:
                    print(f"[WORKER] Removing stale lock (old PID {old_pid} is dead)")
                    self.lock_file.unlink()
            except (ValueError, IOError):
                print("[WORKER] Removing corrupted lock file")
                self.lock_file.unlink()

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))
        self._acquired = True
        atexit.register(self.release)
        return True

    def release(self):
        if self._acquired and self.lock_file.exists():
            try:
                with open(self.lock_file, "r") as f:
                    if int(f.read().strip()) == os.getpid():
                        self.lock_file.unlink()
                        print("[WORKER] Lock released")
            except (ValueError, IOError, FileNotFoundError):
                pass
            self._acquired = False


# ============================================================================
# QickSweep1D JSON serialization helpers
# ============================================================================

def encode_sweep(obj):
    """
    Recursively encode QickSweep1D objects to JSON-safe dicts.

    A QickSweep1D is encoded as:
        {"__sweep__": true, "loop": "freqloop", "start": 5330, "stop": 5370}
    """
    if isinstance(obj, QickSweep1D):
        return {
            "__sweep__": True,
            "loop": obj.label,
            "start": float(obj.minval),
            "stop": float(obj.maxval),
        }
    elif isinstance(obj, dict):
        return {k: encode_sweep(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [encode_sweep(v) for v in obj]
    return obj


def decode_sweep(obj):
    """Recursively reconstruct QickSweep1D objects from JSON dicts."""
    if isinstance(obj, dict):
        if obj.get("__sweep__"):
            return QickSweep1D(obj["loop"], obj["start"], obj["stop"])
        return {k: decode_sweep(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [decode_sweep(v) for v in obj]
    return obj


# ============================================================================
# Job Worker
# ============================================================================

class JobWorker:
    """
    Single-threaded worker that processes jobs from the queue.

    Connects to QICK via Pyro4 and runs experiments sequentially.
    """

    def __init__(
        self,
        ns_host: str = "192.168.10.179",
        ns_port: int = 8887,
        proxy_name: str = "myqick",
        poll_interval: float = 2.0,
    ):
        self.ns_host = ns_host
        self.ns_port = ns_port
        self.proxy_name = proxy_name
        self.poll_interval = poll_interval

        self.running = True
        self.current_job: Optional[Job] = None

        # Initialize database
        self.db = get_database()

        # Connect to QICK via Pyro4
        print(f"[WORKER] Connecting to QICK at {ns_host}:{ns_port} (proxy: {proxy_name})...")
        Pyro4.config.SERIALIZER = "pickle"
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
        self.soc, self.soccfg = make_proxy(
            ns_host=ns_host, ns_port=ns_port, proxy_name=proxy_name
        )
        print(f"[WORKER] Connected to QICK successfully")
        print(f"[WORKER] {self.soccfg}")

        # Load qubit config system
        from qick_workspace.tools.system_tool import ExperimentConfig
        from qick_workspace.tools.ncfg import config_list
        self.config_loader = ExperimentConfig(config_list)

        # Signal handlers
        self._interrupt_count = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Clean up jobs left in RUNNING state from previous crashes
        self._cleanup_incomplete_jobs()

        print("[WORKER] Initialized and ready")

    def _handle_interrupt(self, signum, frame):
        """Ctrl+C: cancel current job or shut down if idle."""
        self._interrupt_count += 1
        if self._interrupt_count >= 2:
            print("\n[WORKER] Force exit...")
            sys.exit(1)

        if self.current_job:
            print(f"\n[WORKER] Ctrl+C → cancelling job {self.current_job.job_id}")
            print("[WORKER] Worker continues. Press Ctrl+C again to stop.")
            raise KeyboardInterrupt("Job cancelled by user")
        else:
            print("\n[WORKER] Shutting down...")
            self.running = False

    def _handle_shutdown(self, signum, frame):
        print(f"\n[WORKER] Signal {signum} received, shutting down after current job...")
        self.running = False

    def run(self):
        """Main worker loop: poll for jobs and execute them."""
        print(f"[WORKER] Starting main loop (poll every {self.poll_interval}s)")
        print("[WORKER] Ctrl+C to cancel current job, Ctrl+C while idle to stop")

        while self.running:
            job = self._fetch_next_job()
            if job:
                self._execute_job(job)
            else:
                time.sleep(self.poll_interval)

        print("[WORKER] Shutdown complete")

    def _fetch_next_job(self) -> Optional[Job]:
        """Fetch and claim the next pending job (highest priority, then FIFO)."""
        with self.db.session() as session:
            job = (
                session.query(Job)
                .filter_by(status=JobStatus.PENDING)
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .first()
            )
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                session.flush()
                session.expunge(job)
                print(f"\n[WORKER] ▶ Claimed: {job.job_id} ({job.experiment_class}) by {job.user}")
                return job
        return None

    def _execute_job(self, job: Job):
        """Execute a single job."""
        self.current_job = job

        try:
            # 1. Decode the full run_cfg transmitted from the client
            run_cfg = decode_sweep(json.loads(job.experiment_config))

            # 2. Load experiment class dynamically
            ExptClass = self._load_experiment_class(job.experiment_module, job.experiment_class)

            # 3. Create experiment instance and run
            print(f"[WORKER] Creating {job.experiment_class}(soc, soccfg, run_cfg)")
            expt = ExptClass(self.soc, self.soccfg, run_cfg)

            print(f"[WORKER] Running with py_avg={job.py_avg}...")
            expt.run(job.py_avg)

            # 4. Extract data and serialize to disk
            result_dir = Path(__file__).parent / "results"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"{job.job_id}.pkl"
            
            result_data = {
                "job_id": job.job_id,
                "iqdata": getattr(expt, "iqdata", None),
                "sweep_vals_x": getattr(expt, "_sweep_vals_x", None),
                "sweep_vals_y": getattr(expt, "_sweep_vals_y", None),
            }
            
            import pickle
            with open(result_path, "wb") as f:
                pickle.dump(result_data, f)
            print(f"[WORKER] Result saved to {result_path}")

            # 5. Mark completed
            self._update_job_status(job.job_id, JobStatus.COMPLETED, result_path=str(result_path))
            print(f"[WORKER] ✓ Job completed: {job.job_id}")

        except KeyboardInterrupt:
            self._update_job_status(job.job_id, JobStatus.FAILED, "Cancelled by user (Ctrl+C)")
            print(f"[WORKER] ✗ Job cancelled: {job.job_id}")
            self._interrupt_count = 0

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self._update_job_status(job.job_id, JobStatus.FAILED, error_msg)
            print(f"[WORKER] ✗ Job failed: {job.job_id}")
            print(f"[WORKER]   Error: {e}")

        finally:
            self.current_job = None

    def _load_experiment_class(self, module_path: str, class_name: str):
        """Dynamically import an experiment class (with autoreload)."""
        print(f"[WORKER] Loading {class_name} from {module_path}")
        importlib.invalidate_caches()

        # Remove cached modules to pick up code changes
        modules_to_remove = [
            name for name in list(sys.modules.keys())
            if name == module_path
            or name.startswith("qick_workspace.newscrip.")
        ]
        for name in modules_to_remove:
            del sys.modules[name]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _update_job_status(self, job_id: str, status: JobStatus, error_message: str = None, result_path: str = None):
        """Update job status in database."""
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = status
                if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    job.completed_at = datetime.now(timezone.utc)
                if error_message:
                    job.error_message = error_message
                if result_path:
                    job.result_path = result_path

    def _cleanup_incomplete_jobs(self):
        """Mark any RUNNING jobs as FAILED on startup (crash recovery)."""
        with self.db.session() as session:
            running = session.query(Job).filter_by(status=JobStatus.RUNNING).all()
            for job in running:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = "Worker crashed or restarted during execution"
            if running:
                print(f"[WORKER] Cleaned up {len(running)} incomplete jobs")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="QICK experiment job worker")
    parser.add_argument("--ns-host", type=str, default="192.168.10.179",
                        help="Pyro4 nameserver host (default: 192.168.10.179)")
    parser.add_argument("--ns-port", type=int, default=8887,
                        help="Pyro4 nameserver port (default: 8887)")
    parser.add_argument("--proxy-name", type=str, default="myqick",
                        help="QICK proxy name (default: myqick)")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between DB polls (default: 2.0)")

    args = parser.parse_args()

    # Acquire lock
    lock = WorkerLock()
    try:
        lock.acquire()
        print(f"[WORKER] Lock acquired (PID {os.getpid()})")
    except RuntimeError as e:
        print(f"[WORKER] ERROR: {e}")
        sys.exit(1)

    worker = JobWorker(
        ns_host=args.ns_host,
        ns_port=args.ns_port,
        proxy_name=args.proxy_name,
        poll_interval=args.poll_interval,
    )

    try:
        worker.run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
