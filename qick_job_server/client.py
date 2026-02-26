"""
Client library for submitting and monitoring QICK experiment jobs.

Usage in Jupyter notebooks:

    from qick_job_server.client import JobClient

    client = JobClient("http://192.168.10.100:8585")

    job_id = client.submit(
        experiment_class="ResonatorSpec",
        experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
        experiment_config={...},
        qubit="Q1",
        py_avg=10,
        user="jay",
    )

    result = client.wait_for_completion(job_id)
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests
def encode_sweep(obj):
    """
    Recursively encode QickSweep1D objects to JSON-safe dicts.

    A QickSweep1D is encoded as:
        {"__sweep__": true, "loop": "freqloop", "start": 5330, "stop": 5370}
    """
    if type(obj).__name__ == "QickSweep1D":
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
    error_message: Optional[str] = None
    queue_position: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "JobResult":
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
            error_message=data.get("error_message"),
            queue_position=data.get("queue_position"),
        )

    def is_done(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def is_successful(self) -> bool:
        return self.status == "completed"

@dataclass
class ExperimentResult:
    """Contains the serialized IQ data returned from the worker worker."""
    job_id: str
    iqdata: Any
    _sweep_vals_x: Any = None
    _sweep_vals_y: Any = None

def _parse_datetime(value) -> Optional[datetime]:
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
    Client for interacting with the QICK job server.

    Args:
        server_url: URL of the job server (default: http://127.0.0.1:8585)
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8585"):
        self.server_url = server_url.rstrip("/")

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.server_url}{endpoint}"
        return requests.request(method, url, **kwargs)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        r = self._request("GET", "/health")
        r.raise_for_status()
        return r.json()

    def submit(
        self,
        experiment_class: str,
        experiment_module: str,
        run_cfg: Dict[str, Any],
        qubit: str = "Q1",
        py_avg: int = 1,
        user: str = "anonymous",
        priority: int = 0,
    ) -> str:
        """
        Submit an experiment job to the queue.

        Args:
            experiment_class: Class name (e.g., "ResonatorSpec")
            experiment_module: Module path (e.g., "qick_workspace.newscrip.s002_res_spec_ge")
            run_cfg: Complete experiment configuration dictionary. QickSweep1D objects
                     will be automatically serialized.
            qubit: Qubit name (default: "Q1")
            py_avg: Software averages (default: 1)
            user: Your username
            priority: Higher = runs sooner (default: 0)

        Returns:
            Job ID string (e.g., "JOB-20260225-00001")
        """
        payload = {
            "experiment_class": experiment_class,
            "experiment_module": experiment_module,
            "experiment_config": encode_sweep(run_cfg),
            "qubit": qubit,
            "py_avg": py_avg,
            "user": user,
            "priority": priority,
        }

        r = self._request("POST", "/jobs/submit", json=payload)
        r.raise_for_status()
        data = r.json()
        print(f"✓ Job submitted: {data['job_id']} (queue position: {data.get('queue_position', '?')})")
        return data["job_id"]

    def get_status(self, job_id: str) -> JobResult:
        """Get the current status of a job."""
        r = self._request("GET", f"/jobs/{job_id}")
        r.raise_for_status()
        return JobResult.from_dict(r.json())

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> JobResult:
        """
        Wait for a job to complete (blocks until done).

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between checks (default: 2.0)
            timeout: Max seconds to wait (None = forever)
            verbose: Print status updates

        Returns:
            Final JobResult
        """
        start_time = time.time()
        last_status = None

        try:
            while True:
                result = self.get_status(job_id)

                if verbose and result.status != last_status:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] {job_id}: {result.status}")
                    last_status = result.status

                if result.is_done():
                    if verbose:
                        if result.is_successful():
                            print(f"✓ Job completed!")
                        else:
                            print(f"✗ Job {result.status}: {result.error_message or 'No details'}")
                    return result

                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            result = self.get_status(job_id)
            if result.status == "pending":
                print(f"Cancelling pending job {job_id}...")
                try:
                    self.cancel_job(job_id)
                except Exception:
                    pass
            elif result.status == "running":
                print(f"Job {job_id} is running on worker — cannot cancel remotely.")
                print(f"Check later with: client.get_status('{job_id}')")
            raise

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        r = self._request("DELETE", f"/jobs/{job_id}")
        r.raise_for_status()
        print(f"Job {job_id} cancelled")
        return True

    # ------------------------------------------------------------------
    # Queue & history
    # ------------------------------------------------------------------

    def list_queue(self) -> dict:
        """List pending and running jobs."""
        r = self._request("GET", "/jobs/queue")
        r.raise_for_status()
        return r.json()

    def print_queue(self):
        """Print the current queue status."""
        queue = self.list_queue()

        print("\n=== Job Queue ===")

        if queue.get("running_job"):
            job = queue["running_job"]
            print(f"\n▶ Running: {job['job_id']}")
            print(f"  User: {job['user']}  |  {job['experiment_class']}  |  {job['qubit']}")
        else:
            print("\nNo job currently running")

        print(f"\nPending: {queue['total_pending']} jobs")
        for i, job in enumerate(queue.get("pending_jobs", [])[:10], 1):
            print(f"  {i}. {job['job_id']} — {job['experiment_class']} "
                  f"(user: {job['user']}, qubit: {job['qubit']}, priority: {job['priority']})")

        if queue["total_pending"] > 10:
            print(f"  ... and {queue['total_pending'] - 10} more")
        print()

    def get_history(self, limit: int = 20, user: str = None, status: str = None) -> List[dict]:
        """Get recent job history."""
        params = {"limit": limit}
        if user:
            params["user"] = user
        if status:
            params["status"] = status
        r = self._request("GET", "/jobs/history", params=params)
        r.raise_for_status()
        return r.json()

    def get_result(self, job_id: str) -> "ExperimentResult":
        """
        Download and unpack the IQ data results of a completed experiment.

        Returns:
            ExperimentResult object with `.iqdata`, `._sweep_vals_x`, etc.
        """
        r = self._request("GET", f"/jobs/{job_id}/result", stream=True)
        r.raise_for_status()

        import pickle
        import io
        
        # Load pickle from response bytes
        buf = io.BytesIO(r.content)
        data = pickle.load(buf)

        return ExperimentResult(
            job_id=data["job_id"],
            iqdata=data.get("iqdata"),
            _sweep_vals_x=data.get("sweep_vals_x"),
            _sweep_vals_y=data.get("sweep_vals_y"),
        )

    # ------------------------------------------------------------------
    # Convenience: sweep helper
    # ------------------------------------------------------------------

    @staticmethod
    def sweep(loop: str, start: float, stop: float) -> dict:
        """
        Create a serialized QickSweep1D for use in experiment_config.

        Example:
            client.submit(
                ...,
                experiment_config={
                    "res_freq_ge": client.sweep("freqloop", 5330, 5370),
                    "steps": 101,
                },
            )
        """
        return {"__sweep__": True, "loop": loop, "start": start, "stop": stop}
