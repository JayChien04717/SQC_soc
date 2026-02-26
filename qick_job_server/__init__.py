"""
QICK Experiment Job Server
===========================

A job queue system for scheduling QICK experiments from multiple PCs.

Quick start:
    # Server (Terminal 1):
    cd /Users/jay/Desktop/test/SQC_soc
    python -m uvicorn qick_job_server.server:app --host 0.0.0.0 --port 8585

    # Worker (Terminal 2):
    python -m qick_job_server.worker --ns-host 192.168.10.179 --ns-port 8887

    # Client (Jupyter notebook):
    from qick_job_server.client import JobClient
    client = JobClient("http://127.0.0.1:8585")
    job_id = client.submit(
        experiment_class="ResonatorSpec",
        experiment_module="qick_workspace.newscrip.s002_res_spec_ge",
        experiment_config={"steps": 101},
        qubit="Q1", py_avg=10, user="jay",
    )
"""

from .client import JobClient

__all__ = ["JobClient"]
