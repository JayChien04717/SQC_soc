"""
RB analysis — randomized benchmarking exponential decay fit.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag


class RBAnalysis(BaseAnalysis):
    """
    Exponential decay fit for randomized benchmarking.

    Extracts average Clifford gate fidelity (r) and error per Clifford (EPC).
    """

    thresholds = {
        "epc": {"max": 0.05},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitrb, rb_func

        x = data.x_axis  # Clifford lengths
        y = np.abs(data.raw_iq) if data.raw_iq.ndim == 1 else np.abs(data.raw_iq).mean(axis=-1)

        try:
            popt, pcov, _ = fitrb(x, y)
            err = np.sqrt(np.diag(pcov))
            # RB model: A * p^m + B  where p = 1 - EPC * d/(d-1), d=2 for single qubit
            A, p, B = popt
            # EPC = (1 - p) * (d-1)/d  for d=2 → EPC = (1-p)/2
            d = 2
            epc = (1 - p) * (d - 1) / d
            fidelity = 1 - epc

            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "epc": (round(epc, 6), None),
                "fidelity": (round(fidelity, 6), None),
                "p": (p, err[1]),
                "A": (A, err[0]),
                "B": (B, err[2]),
            }
            data.scalar_result = fidelity
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"RB fit failed: {exc}"


class AllXYAnalysis(BaseAnalysis):
    """AllXY pulse-sequence gate fidelity assessment."""

    thresholds = {
        "allxy_error": {"max": 0.1},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.raw_iq is None:
            return
        # AllXY ideal values for the 21 sequences: pattern of 0, 0.5, 1
        ideal = np.array([
            0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0
        ])
        measured = np.abs(data.raw_iq)
        # Normalize measured to [0, 1]
        measured_norm = (measured - measured.min()) / (measured.max() - measured.min() + 1e-12)

        if len(measured_norm) == len(ideal):
            error = float(np.mean(np.abs(measured_norm - ideal)))
            data.fit_result = {"allxy_error": (error, None)}
            data.scalar_result = error
        else:
            data.quality = QualityFlag.NO_INFORMATION
