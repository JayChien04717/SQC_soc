"""
Client library for submitting and monitoring QICK jobs.

Usage in notebooks:
    from qick_job_server.client import JobClient

    client = JobClient()

    # Submit a job
    job_id = client.submit(
        experiment_class="ResonatorSpec",
        experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
        run_cfg=config,
        qubit="Q1",
        py_avg=10,
        user="jay",
    )

    # Wait for completion
    client.wait_for_completion(job_id)

    # Load result
    result = client.get_result(job_id)
    print(result.iqdata)
"""

import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests


@dataclass
class JobResult:
    """Result of a job query."""

    job_id: str
    status: str
    user: Optional[str] = None
    experiment_class: Optional[str] = None
    qubit: Optional[str] = None
    py_avg: Optional[int] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data_path: Optional[str] = None
    error_message: Optional[str] = None
    queue_position: Optional[int] = None

    # Loaded experiment object (populated by get_result)
    _expt: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: dict) -> "JobResult":
        """Create JobResult from API response dict."""
        return cls(
            job_id=data.get("job_id"),
            status=data.get("status"),
            user=data.get("user"),
            experiment_class=data.get("experiment_class"),
            qubit=data.get("qubit"),
            py_avg=data.get("py_avg"),
            created_at=_parse_datetime(data.get("created_at")),
            started_at=_parse_datetime(data.get("started_at")),
            completed_at=_parse_datetime(data.get("completed_at")),
            data_path=data.get("data_path"),
            error_message=data.get("error_message"),
            queue_position=data.get("queue_position"),
        )

    def is_done(self) -> bool:
        """Check if job has finished."""
        return self.status in ("completed", "failed", "cancelled")

    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"

    @property
    def iqdata(self):
        """Get IQ data from loaded experiment dictionary."""
        if isinstance(self._expt, dict):
            return self._expt.get("iqdata")
        elif self._expt is not None:
            return getattr(self._expt, "iqdata", None)
        return None

    @property
    def fit_params(self):
        """Get fit parameters from loaded experiment dictionary."""
        if isinstance(self._expt, dict):
            return self._expt.get("fit_params")
        elif self._expt is not None:
            return getattr(self._expt, "fit_params", None)
        return None

    def load_expt(self):
        """
        Load the experiment results from the saved pickle file.

        Returns a dictionary or object containing iqdata, fit_params, etc.
        """
        if not self.is_successful():
            raise ValueError(f"Cannot load expt: job status is {self.status}")
        if not self.data_path:
            raise ValueError("No data_path available for this job")

        with open(self.data_path, "rb") as f:
            self._expt = pickle.load(f)
            
        return self._expt

def _parse_datetime(value) -> Optional[datetime]:
    """Parse datetime from string or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


class JobClient:
    """
    Client for interacting with the QICK job queue server.

    Provides methods to submit jobs, check status, and wait for completion.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8585"):
        self.server_url = server_url.rstrip("/")

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the server."""
        url = f"{self.server_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        return response

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        response = self._request("GET", "/health")
        response.raise_for_status()
        return response.json()

    def submit(
        self,
        experiment_class: str,
        experiment_module: str,
        run_cfg: Dict[str, Any],
        qubit: str,
        py_avg: int,
        user: str,
        priority: int = 0,
    ) -> str:
        """
        Submit an experiment job to the queue.

        Args:
            experiment_class: Name of experiment class (e.g., "ResonatorSpec")
            experiment_module: Module path (e.g., "qick_workspace.newscrip.s002_res_spec_ge")
            run_cfg: Experiment configuration dict (addict.Dict or plain dict)
            qubit: Target qubit (e.g., "Q1")
            py_avg: Number of software averages
            user: Username of submitter
            priority: Job priority (higher = runs sooner)

        Returns:
            Unique job_id string
        """
        # Validate required parameters
        if not experiment_class:
            raise ValueError("experiment_class is required")
        if not experiment_module:
            raise ValueError("experiment_module is required")
        if not run_cfg:
            raise ValueError("run_cfg is required")
        if not qubit:
            raise ValueError("qubit is required")
        if not user:
            raise ValueError("user is required")

        # Convert addict.Dict or similar to plain dict for JSON serialization
        import json
        import numpy as np

        def convert_for_json(obj):
            """Recursively convert non-serializable types.
            
            QickSweep1D objects are encoded as special markers so the worker
            can reconstruct them. Other QICK types are converted to float/str.
            """
            # Handle addict.Dict -> plain dict
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Dict' and hasattr(obj, 'to_dict'):
                obj = obj.to_dict()

            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, complex):
                return {"__complex__": True, "real": obj.real, "imag": obj.imag}
                
            # Handle any QICK parameter type safely
            obj_module = getattr(type(obj), '__module__', '') or ''
            cls_name = type(obj).__name__
            if 'qick' in obj_module and cls_name not in ('QickConfig',):
                # QickSweep1D/QickParam has .start and a .spans dict for loops
                if hasattr(obj, 'start') and hasattr(obj, 'spans'):
                    if obj.spans:
                        # It's a swept param!
                        loop_name = list(obj.spans.keys())[0]
                        span_val = obj.spans[loop_name]
                        return {
                            "__qick_sweep__": True,
                            "type": "QickSweep1D",
                            "loop_name": str(loop_name),
                            "start": float(obj.start),
                            "stop": float(obj.start + span_val),
                        }
                    else:
                        # It's a constant QickParam, safe to serialize its start value
                        return float(obj.start)

                # Fallback for other Qick objects
                try:
                    return float(obj)
                except Exception:
                    return str(obj)

            return obj

        clean_cfg = convert_for_json(dict(run_cfg))

        payload = {
            "experiment_class": experiment_class,
            "experiment_module": experiment_module,
            "run_cfg": clean_cfg,
            "qubit": qubit,
            "py_avg": py_avg,
            "user": user,
            "priority": priority,
        }

        response = self._request("POST", "/jobs/submit", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Job submitted: {data['job_id']} (queue position: {data.get('queue_position', '?')})")
        return data["job_id"]

    def get_status(self, job_id: str) -> JobResult:
        """Get the current status of a job."""
        if not job_id:
            raise ValueError("job_id is required")

        response = self._request("GET", f"/jobs/{job_id}")
        response.raise_for_status()
        return JobResult.from_dict(response.json())

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
        stream_output: bool = True,
    ) -> JobResult:
        """
        Wait for a job to complete, optionally streaming output.

        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = forever)
            verbose: Print status updates
            stream_output: Stream job output in real-time
        """
        if not job_id:
            raise ValueError("job_id is required")

        start_time = time.time()
        last_status = None
        output_offset = 0
        output_fetch_failed = False

        try:
            while True:
                result = self.get_status(job_id)

                if verbose and result.status != last_status:
                    elapsed = time.time() - start_time
                    print(f"\n[{elapsed:.1f}s] Job {job_id}: {result.status}")
                    last_status = result.status

                # Stream output
                if stream_output and result.status in ("running", "completed", "failed"):
                    try:
                        output_result = self.get_output(job_id, offset=output_offset)
                        if output_result["output"]:
                            print(output_result["output"], end="", flush=True)
                        output_offset = output_result["line_count"]
                    except Exception as e:
                        if not output_fetch_failed:
                            print(f"\n[WARNING] Failed to fetch job output: {e}")
                            output_fetch_failed = True

                if result.is_done():
                    # Fetch remaining output
                    if stream_output and not output_fetch_failed:
                        try:
                            output_result = self.get_output(job_id, offset=output_offset)
                            if output_result["output"]:
                                print(output_result["output"], end="", flush=True)
                        except Exception:
                            pass

                    if verbose:
                        if result.is_successful():
                            print(f"\nJob completed! Data: {result.data_path}")
                        else:
                            print(f"\nJob {result.status}: {result.error_message or 'No details'}")
                    return result

                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Keyboard interrupt received for job {job_id}")
            result = self.get_status(job_id)

            if result.status == "pending":
                print("[INTERRUPT] Job is pending, cancelling...")
                try:
                    self.cancel(job_id)
                    print(f"[INTERRUPT] Job {job_id} cancelled successfully")
                except Exception as e:
                    print(f"[INTERRUPT] Failed to cancel job: {e}")
            elif result.status == "running":
                print("[INTERRUPT] Job is currently running on worker")
                print("[INTERRUPT] Cannot interrupt running jobs remotely")
                print(f"[INTERRUPT] Check status later: client.get_status('{job_id}')")
            else:
                print(f"[INTERRUPT] Job already in terminal state: {result.status}")

            raise

    def get_result(self, job_id: str) -> JobResult:
        """
        Get a completed job's result with loaded experiment object.

        Returns JobResult with populated _expt, iqdata, fit_params, etc.
        """
        result = self.get_status(job_id)
        if result.is_successful():
            result.load_expt()
        return result

    def get_handle(self, job_id: str, expt_instance=None) -> "JobHandle":
        """
        Get a handle for async job tracking.

        If expt_instance is provided, the handle can populate it with results
        when the job completes.
        """
        return JobHandle(self, job_id, expt_instance)

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        if not job_id:
            raise ValueError("job_id is required")

        response = self._request("DELETE", f"/jobs/{job_id}")
        response.raise_for_status()
        print(f"Job {job_id} cancelled")
        return True

    def list_queue(self) -> dict:
        """List all pending and running jobs."""
        response = self._request("GET", "/jobs/queue")
        response.raise_for_status()
        return response.json()

    def print_queue(self):
        """Print the current queue status in a readable format."""
        queue = self.list_queue()

        print("\n=== QICK Job Queue ===")

        if queue.get("running_job"):
            job = queue["running_job"]
            print(f"\nRunning: {job['job_id']}")
            print(f"  User: {job['user']}")
            print(f"  Experiment: {job['experiment_class']}")
            print(f"  Qubit: {job['qubit']}")
            print(f"  Started: {job.get('started_at', 'Unknown')}")
        else:
            print("\nNo job currently running")

        print(f"\nPending: {queue['total_pending']} jobs")
        for i, job in enumerate(queue.get("pending_jobs", [])[:10], 1):
            print(f"  {i}. {job['job_id']} - {job['experiment_class']} "
                  f"(user: {job['user']}, qubit: {job['qubit']}, priority: {job['priority']})")

        if queue["total_pending"] > 10:
            print(f"  ... and {queue['total_pending'] - 10} more")

        print()

    def get_history(
        self,
        limit: int = 50,
        user: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[dict]:
        """Get recent job history."""
        params = {"limit": limit}
        if user:
            params["user"] = user
        if status:
            params["status"] = status

        response = self._request("GET", "/jobs/history", params=params)
        response.raise_for_status()
        return response.json()

    def get_output(self, job_id: str, offset: int = 0) -> dict:
        """Get output from a running or completed job."""
        if not job_id:
            raise ValueError("job_id is required")

        response = self._request("GET", f"/jobs/{job_id}/output", params={"offset": offset})
        response.raise_for_status()
        return response.json()


class JobHandle:
    """
    Async handle for tracking a submitted job.

    Allows non-blocking job submission with later result retrieval.

    Usage:
        handle = client.get_handle(job_id, expt)
        # ... do other work ...
        handle.wait()  # blocks until complete, populates expt.iqdata
    """

    def __init__(self, client: JobClient, job_id: str, expt_instance=None):
        self.client = client
        self.job_id = job_id
        self.expt_instance = expt_instance
        self._result: Optional[JobResult] = None

    @property
    def status(self) -> str:
        """Get current job status."""
        result = self.client.get_status(self.job_id)
        return result.status

    @property
    def is_done(self) -> bool:
        """Check if the job is done."""
        return self.status in ("completed", "failed", "cancelled")

    def wait(self, poll_interval: float = 2.0, timeout: Optional[float] = None) -> JobResult:
        """Wait for job completion and populate expt_instance if provided."""
        self._result = self.client.wait_for_completion(
            self.job_id, poll_interval=poll_interval, timeout=timeout
        )

        # Populate the original experiment instance with results
        if self._result.is_successful() and self.expt_instance is not None:
            loaded = self._result.load_expt()
            # `loaded` is now a dict, but `JobResult` provides proxy properties
            self.expt_instance.iqdata = self._result.iqdata
            self.expt_instance.fit_params = self._result.fit_params
            
            if isinstance(loaded, dict):
                self.expt_instance._sweep_vals = loaded.get("_sweep_vals")
                self.expt_instance._sweep_vals_y = loaded.get("_sweep_vals_y")
            else:
                self.expt_instance._sweep_vals = getattr(loaded, "_sweep_vals", None)
                self.expt_instance._sweep_vals_y = getattr(loaded, "_sweep_vals_y", None)

        return self._result

    def cancel(self):
        """Cancel the job if it's still pending."""
        return self.client.cancel(self.job_id)

    def __repr__(self):
        return f"<JobHandle({self.job_id})>"
