"""
Job worker daemon for executing queued QICK experiments.

This worker:
- Polls the database for pending jobs
- Executes experiments one at a time (hardware exclusivity)
- Updates job status and saves results
- Handles graceful shutdown on SIGINT/SIGTERM

Run with:
    cd c:/Users/cluster/Desktop/SQC_soc-jobserver

    # Mock mode (for testing without hardware):
    python -m qick_workspace.qick_job_server.worker --mock

    # Real hardware mode (requires soc/soccfg via Pyro or local):
    python -m qick_workspace.qick_job_server.worker
"""

import argparse
import atexit
import importlib
import json
import os
import pickle
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .database import get_database
from .id_generator import IDGenerator
from .models import Job, JobOutput, JobStatus
from .output_capture import OutputCapture

# Default lock file location
DEFAULT_LOCK_FILE = Path(__file__).parent / "worker.lock"

# Default data directory for results
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "job_results"


class WorkerLock:
    """
    PID-based lock to prevent multiple workers from running simultaneously.

    Checks if lock file exists and whether the owning process is alive.
    Stale lock files from crashed processes are cleaned up automatically.
    """

    def __init__(self, lock_file: Path = DEFAULT_LOCK_FILE):
        self.lock_file = lock_file
        self._acquired = False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        if sys.platform == "win32":
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_INFORMATION = 0x0400
                handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def acquire(self) -> bool:
        """Try to acquire the lock. Raises RuntimeError if another worker is running."""
        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())

                if self._is_process_running(old_pid):
                    raise RuntimeError(
                        f"Another worker is already running (PID {old_pid}). "
                        f"If you believe this is an error, delete {self.lock_file}"
                    )
                else:
                    print(f"[WORKER] Removing stale lock file (old PID {old_pid} is not running)")
                    self.lock_file.unlink()

            except (ValueError, IOError) as e:
                print(f"[WORKER] Removing corrupted lock file: {e}")
                self.lock_file.unlink()

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        self._acquired = True
        atexit.register(self.release)
        return True

    def release(self):
        """Release the lock by deleting the lock file."""
        if self._acquired and self.lock_file.exists():
            try:
                with open(self.lock_file, "r") as f:
                    file_pid = int(f.read().strip())
                if file_pid == os.getpid():
                    self.lock_file.unlink()
                    print("[WORKER] Lock released")
            except (ValueError, IOError, FileNotFoundError):
                pass
            self._acquired = False


class JobWorker:
    """
    Single-threaded worker that processes jobs from the queue.

    Only one worker should run at a time to ensure hardware exclusivity.
    The worker polls the database for pending jobs, executes them in
    priority order, and updates job status.
    """

    def __init__(
        self,
        soc=None,
        soccfg=None,
        mock_mode: bool = False,
        poll_interval: float = 2.0,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize the job worker.

        Args:
            soc: QICK soc object (None if mock_mode=True)
            soccfg: QICK soccfg object (None if mock_mode=True)
            mock_mode: If True, execute experiments with simulate=True
            poll_interval: Seconds between database polls when idle
            data_dir: Directory to save experiment results
        """
        self.soc = soc
        self.soccfg = soccfg
        self.mock_mode = mock_mode
        self.poll_interval = poll_interval
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.current_job: Optional[Job] = None

        # Initialize database
        self.db = get_database()

        # Setup signal handlers for graceful shutdown
        self._interrupt_count = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Clean up any jobs left in RUNNING state from previous crashes
        self._cleanup_incomplete_jobs()

        print(f"[WORKER] Initialized in {'MOCK' if mock_mode else 'REAL'} mode")
        print(f"[WORKER] Data directory: {self.data_dir}")

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C."""
        self._interrupt_count += 1

        if self._interrupt_count >= 2:
            print("\n[WORKER] Second Ctrl+C received, forcing exit...")
            sys.exit(1)

        if self.current_job:
            print(f"\n[WORKER] Ctrl+C received, cancelling current job {self.current_job.job_id}...")
            print("[WORKER] Worker will continue processing queue. Press Ctrl+C again to stop.")
            raise KeyboardInterrupt("Job cancelled by user")
        else:
            print("\n[WORKER] Ctrl+C received, shutting down...")
            self.running = False

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[WORKER] Received signal {signum}, shutting down after current job...")
        self.running = False

    def run(self):
        """Main worker loop. Continuously polls for jobs and executes them."""
        print(f"[WORKER] Starting main loop (poll interval: {self.poll_interval}s)")
        print("[WORKER] Press Ctrl+C to cancel current job, or Ctrl+C while idle to stop")

        while self.running:
            job = self._fetch_next_job()

            if job:
                self._execute_job(job)
            else:
                time.sleep(self.poll_interval)

        print("[WORKER] Shutdown complete")

    def _fetch_next_job(self) -> Optional[Job]:
        """Fetch the next pending job from the queue (highest priority, then FIFO)."""
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
                print(f"[WORKER] Claimed job: {job.job_id} ({job.experiment_class})")
                return job

        return None

    def _execute_job(self, job: Job):
        """
        Execute a single job with output capture.

        Steps:
        1. Parse run_cfg from JSON
        2. Capture stdout/stderr for streaming
        3. Load experiment class dynamically
        4. Create and run experiment
        5. Save expt object to pickle
        6. Update job status
        """
        self.current_job = job
        print(f"[WORKER] Executing job: {job.job_id}")
        print(f"[WORKER]   Experiment: {job.experiment_class}")
        print(f"[WORKER]   Qubit: {job.qubit}")
        print(f"[WORKER]   User: {job.user}")
        print(f"[WORKER]   py_avg: {job.py_avg}")

        # Create log directory
        log_dir = self.data_dir / "logs"

        try:
            with OutputCapture(job.job_id, self.db, log_dir) as capture:
                # Store log path in job record
                self._update_job_log_path(job.job_id, str(capture.log_path))

                # Load experiment class dynamically
                ExptClass = self._load_experiment_class(job.experiment_module, job.experiment_class)

                # Parse run_cfg
                run_cfg = json.loads(job.run_cfg)

                # Run the experiment
                data_path = self._run_experiment(ExptClass, run_cfg, job)

            # Update job as completed
            self._update_job_completed(job.job_id, str(data_path))

            print(f"[WORKER] Job completed: {job.job_id}")
            print(f"[WORKER]   Data saved to: {data_path}")

        except KeyboardInterrupt:
            error_msg = "Job cancelled by user (Ctrl+C)"
            self._update_job_failed(job.job_id, error_msg)
            print(f"[WORKER] Job cancelled: {job.job_id}")
            self._interrupt_count = 0

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._update_job_failed(job.job_id, error_msg)
            print(f"[WORKER] Job failed: {job.job_id}")
            print(f"[WORKER]   Error: {e}")

        finally:
            self.current_job = None

    def _load_experiment_class(self, module_path: str, class_name: str):
        """
        Dynamically load an experiment class with fresh code.

        Clears cached modules before importing to pick up source code changes.
        """
        print(f"[WORKER] Loading {class_name} from {module_path}")

        importlib.invalidate_caches()

        # Remove cached modules to get fresh code (like autoreload)
        # Preserve core modules to avoid pickle identity errors
        preserved_modules = {
            'qick_workspace.newscrip',
            'qick_workspace.newscrip.base_experiment',
            'qick_workspace.newscrip.base_program',
        }
        modules_to_remove = [
            name for name in list(sys.modules.keys())
            if name == module_path
            or (name.startswith('qick_workspace.newscrip.s') and name not in preserved_modules)
        ]
        for name in modules_to_remove:
            del sys.modules[name]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _run_experiment(self, ExptClass, run_cfg: dict, job: Job) -> Path:
        """
        Run an experiment using the loaded class.

        Follows the BaseExperiment pattern:
        1. Reconstruct QickSweep1D objects from JSON markers
        2. Create experiment instance with soc, soccfg, config
        3. Call expt.run(py_avg, simulate=mock_mode)
        4. Pickle the expt object for client retrieval
        5. Return pickle file path
        """
        # Reconstruct QickSweep1D objects from JSON markers
        run_cfg = self._reconstruct_sweeps(run_cfg)

        # Use plain dict instead of addict.Dict to avoid '__frozen' errors
        # when Qick internals interact with the configuration.
        config = run_cfg

        # Create experiment instance
        print(f"[WORKER] Creating experiment instance")

        if self.mock_mode:
            expt = ExptClass(soc=None, soccfg=None, config=config)
        else:
            expt = ExptClass(soc=self.soc, soccfg=self.soccfg, config=config)

        # Run experiment
        print(f"[WORKER] Running experiment (simulate={self.mock_mode})...")
        expt.run(py_avg=job.py_avg, simulate=self.mock_mode)

        # Save expt object to pickle
        pickle_filename = IDGenerator.generate_data_filename(job.job_id, job.experiment_class)
        pickle_path = self.data_dir / pickle_filename

        print(f"[WORKER] Saving expt object to: {pickle_path}")
        
        # We cannot pickle `expt` directly because QickParam has a lambda in __copy__
        # Instead, we extract the useful results into a plain dictionary and save that.
        result_dict = {
            "iqdata": getattr(expt, "iqdata", None),
            "fit_params": getattr(expt, "fit_params", None),
            "_sweep_vals_x": getattr(expt, "_sweep_vals_x", None),
            "_sweep_vals_y": getattr(expt, "_sweep_vals_y", None),
        }
        
        with open(pickle_path, "wb") as f:
            pickle.dump(result_dict, f)

        return pickle_path

    @staticmethod
    def _reconstruct_sweeps(cfg: dict) -> dict:
        """
        Recursively reconstruct QickSweep1D objects from JSON markers.

        JSON markers look like:
            {"__qick_sweep__": True, "type": "QickSweep1D", "loop_name": "freqloop",
             "start": 5331.559, "stop": 5371.559}
        """
        try:
            from qick.asm_v2 import QickSweep1D
        except ImportError:
            # If qick is not installed, return cfg as-is
            return cfg

        def reconstruct(obj):
            if isinstance(obj, dict):
                if obj.get("__qick_sweep__"):
                    return QickSweep1D(
                        obj["loop_name"],
                        obj["start"],
                        obj["stop"],
                    )
                return {k: reconstruct(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [reconstruct(item) for item in obj]
            return obj

        return reconstruct(cfg)

    def _update_job_completed(self, job_id: str, data_path: str):
        """Update job status to completed."""
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.data_path = data_path

    def _update_job_failed(self, job_id: str, error_message: str):
        """Update job status to failed."""
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = error_message

    def _update_job_log_path(self, job_id: str, log_path: str):
        """Update job record with output log file path."""
        with self.db.session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                job.output_log_path = log_path

    def _cleanup_incomplete_jobs(self):
        """Mark any RUNNING jobs as FAILED on startup (crash recovery)."""
        with self.db.session() as session:
            running_jobs = session.query(Job).filter_by(status=JobStatus.RUNNING).all()
            for job in running_jobs:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = "Worker crashed or was restarted during execution"

                output = session.query(JobOutput).filter_by(job_id=job.job_id).first()
                if output:
                    output.is_complete = True
                    output.output_text = (output.output_text or "") + "\n[WORKER CRASHED]"

            if running_jobs:
                print(f"[WORKER] Marked {len(running_jobs)} incomplete jobs as FAILED")


def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(
        description="QICK Job worker daemon for executing queued experiments"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (simulate experiments without hardware)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between database polls (default: 2.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--ns-host",
        type=str,
        default="192.168.10.82",
        help="Pyro4 nameserver host (default: 192.168.10.82)",
    )
    parser.add_argument(
        "--ns-port",
        type=int,
        default=8888,
        help="Pyro4 nameserver port (default: 8888)",
    )
    parser.add_argument(
        "--proxy-name",
        type=str,
        default="myqick",
        help="Pyro4 proxy name (default: myqick)",
    )

    args = parser.parse_args()

    # Acquire lock to prevent multiple workers
    lock = WorkerLock()
    try:
        lock.acquire()
        print(f"[WORKER] Lock acquired (PID {os.getpid()})")
    except RuntimeError as e:
        print(f"[WORKER] ERROR: {e}")
        sys.exit(1)

    # Initialize hardware connection (if not mock mode)
    soc = None
    soccfg = None

    if not args.mock:
        print(f"[WORKER] Connecting to QICK hardware at {args.ns_host}:{args.ns_port} ...")
        try:
            from qick import QickConfig
            import Pyro4

            Pyro4.config.SERIALIZER = "pickle"
            Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

            ns = Pyro4.locateNS(host=args.ns_host, port=args.ns_port)
            soc = Pyro4.Proxy(ns.lookup(args.proxy_name))
            soccfg = QickConfig(soc.get_cfg())
            print("[WORKER] QICK hardware connected successfully")
            print(soccfg)
        except Exception as e:
            print(f"[WORKER] ERROR: Failed to connect to QICK hardware: {e}")
            print("[WORKER] Use --mock flag to run without hardware")
            lock.release()
            sys.exit(1)

    worker = JobWorker(
        soc=soc,
        soccfg=soccfg,
        mock_mode=args.mock,
        poll_interval=args.poll_interval,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )

    try:
        worker.run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
