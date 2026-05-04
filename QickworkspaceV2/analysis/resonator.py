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
                "f_res[MHz]": (round(f0 / 1e6, 4), None),
                "Qi": (Qi, None),
                "Qc": (Qc, None),
                "Ql": (Ql, None),
                "kappa_MHz": (round(kappa * 1e-6, 4), None),
            }
            data.scalar_result = f0 / 1e6  # MHz

        except Exception as exc:
            # Fall back to Lorentzian
            self._lorentzian_fallback(data, freqs, iq, exc)

    def plot(self, data: ExperimentData) -> None:
        import matplotlib.pyplot as plt

        if data.x_axis is None or data.raw_iq is None:
            return

        f0    = (data.fit_result.get("f_res[MHz]") or data.fit_result.get("f0_MHz") or (None,))[0]
        kappa = data.fit_result.get("kappa_MHz", (None,))[0]
        Qi    = data.fit_result.get("Qi",  (None,))[0]
        Qc    = data.fit_result.get("Qc",  (None,))[0]
        Ql    = data.fit_result.get("Ql",  (None,))[0]

        title = "Resonator Spectroscopy"
        if f0:    title += f"  |  f_res = {f0:.4f} MHz"
        if kappa: title += f",  κ = {kappa:.3f} MHz"

        # ── Preferred: native ABCD circle-fit plot ────────────────────────────
        # Lorentzian fallback (data.fit_params set) → use our standard panel
        if data.fit_params is None:
            try:
                try:
                    from abcd_rf_fit import analyze
                except ImportError:
                    from ..tools.abcd_rf_fit.abcd_rf_fit import analyze

                solve_type = data.config.get("_solve_type", "hm")
                fit_obj = analyze(
                    data.x_axis * 1e6, data.raw_iq, solve_type, fit_edelay=True
                )
                fit_obj.plot(title=title)
                plt.tight_layout()
                plt.show()
                return
            except Exception:
                pass  # fall through to Lorentzian panel

        # ── Fallback: standard Lorentzian 2×3 panel ──────────────────────────
        from ..tools.fitting import lorfunc

        if data.fit_params is not None:
            fit_params = data.fit_params
        elif f0 is not None and kappa is not None:
            amp    = np.max(np.abs(data.raw_iq)) - np.min(np.abs(data.raw_iq))
            offset = np.min(np.abs(data.raw_iq))
            fit_params = np.array([offset, -amp, f0, kappa / 2])
        else:
            fit_params = None

        lines = []
        if f0:    lines.append(f"f_res  = {f0:.4f} MHz")
        if kappa: lines.append(f"κ      = {kappa:.3f} MHz")
        if Qi:    lines.append(f"Qi     = {int(Qi):,}")
        if Qc:    lines.append(f"Qc     = {int(Qc):,}")
        if Ql:    lines.append(f"Ql     = {int(Ql):,}")

        self._show_fit(
            data, lorfunc, fit_params,
            xlabel="Frequency (MHz)",
            title=title,
            result_text="\n".join(lines),
        )

    def _lorentzian_fallback(self, data, freqs, iq, original_exc):
        """Fit a Lorentzian if circle fit fails."""
        try:
            from ..tools.fitting import fitlor

            popt, pcov, _ = fitlor(freqs, np.abs(iq))
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "kappa_MHz": (abs(popt[1]), err[1]),
            }
            data.scalar_result = popt[2]
            data.quality_message = (
                f"circle fit failed ({original_exc}); used Lorentzian"
            )
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
        from ..tools.fitting import fitlor

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

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        from ..tools.fitting import lorfunc
        f0    = data.fit_result.get("f0_MHz",        (None,))[0]
        kappa = data.fit_result.get("linewidth_MHz",  (None,))[0]
        title = "Qubit Spectroscopy"
        if f0:    title += f"  |  f0 = {f0:.3f} MHz"
        if kappa: title += f",  κ = {kappa:.3f} MHz"
        lines = []
        if f0:    lines.append(f"f0     = {f0:.3f} MHz")
        if kappa: lines.append(f"κ      = {kappa:.3f} MHz")
        self._show_fit(
            data, lorfunc, data.fit_params,
            xlabel="Frequency (MHz)",
            title=title,
            result_text="\n".join(lines),
        )
