"""
Resonator analysis classes — circle fit, Lorentzian, hanger.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag


class ResonatorSpecAnalysis(BaseAnalysis):
    """
    Analysis for resonator spectroscopy (s002).

    Applies circle fit (ABCD / hanger model) and extracts f0, Qi, Qc, Ql, κ.
    """

    thresholds = {
        "Qi": {"min": 1_000},
        "Ql": {"min": 500},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return

        freqs = data.x_axis  # MHz
        iq = data.raw_iq

        try:
            try:
                from abcd_rf_fit import analyze
            except ImportError:
                from ..tools.abcd_rf_fit.abcd_rf_fit import analyze

            solve_type = data.config.get("_solve_type", "hm")
            fit = analyze(freqs * 1e6, iq, solve_type, fit_edelay=True)
            p = fit.tolist()
            f0, kappa, kappa_c = p[0], p[1], p[2]

            Qi = round(f0 / (kappa - kappa_c)) if kappa > kappa_c else 0
            Qc = round(f0 / kappa_c) if kappa_c > 0 else 0
            Ql = round(f0 / kappa) if kappa > 0 else 0

            data.fit_result = {
                "f0_GHz": (round(f0 / 1e9, 6), None),
                "Qi": (Qi, None),
                "Qc": (Qc, None),
                "Ql": (Ql, None),
                "kappa_MHz": (round(kappa * 1e-6, 4), None),
            }
            data.scalar_result = f0 / 1e6  # MHz

        except Exception as exc:
            # Fall back to Lorentzian
            self._lorentzian_fallback(data, freqs, iq, exc)

    def _lorentzian_fallback(self, data, freqs, iq, original_exc):
        """Fit a Lorentzian if circle fit fails."""
        try:
            from ..tools.fitting import fitlor, lorfunc

            popt, pcov, _ = fitlor(freqs, np.abs(iq))
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "kappa_MHz": (abs(popt[1]), err[1]),
            }
            data.scalar_result = popt[2]
            data.quality_message = f"circle fit failed ({original_exc}); used Lorentzian"
        except Exception:
            data.quality = QualityFlag.BAD
            data.quality_message = f"all fits failed: {original_exc}"


class ResonatorPunchoutAnalysis(BaseAnalysis):
    """Analysis for resonator punchout (s002b) — detect critical power."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        # Punchout is primarily visual; store the 2D array summary
        if data.raw_iq is not None:
            data.fit_result = {"status": ("punchout_acquired", None)}


class LorentzianAnalysis(BaseAnalysis):
    """Generic Lorentzian analysis for qubit spectroscopy lines."""

    thresholds = {
        "linewidth_MHz": {"max": 100.0},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitlor, lorfunc

        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitlor(x, y)
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "linewidth_MHz": (abs(popt[1]), err[1]),
                "amplitude": (popt[0], err[0]),
            }
            data.scalar_result = popt[2]
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Lorentzian fit failed: {exc}"
