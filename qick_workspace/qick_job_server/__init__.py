"""
QICK Job Server Package for Multi-User Experiment Scheduling

This package provides:
- JobClient: Client library for submitting and monitoring jobs
- Job server (FastAPI): Central job queue on port 8585
- Job worker: Daemon that executes queued experiments on QICK hardware

Usage:
    from qick_job_server import JobClient

    client = JobClient()
    job_id = client.submit(
        experiment_class="ResonatorSpec",
        experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
        run_cfg=config,
        qubit="Q1",
        py_avg=10,
        user="jay",
    )
    result = client.wait_for_completion(job_id)
"""

from .client import JobClient

__all__ = ["JobClient"]
