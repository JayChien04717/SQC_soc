"""
BaseAnalysis — abstract base class for all analysis classes.

Each experiment family has an Analysis subclass that:
  1. Accepts an ExperimentData with raw_iq populated.
  2. Runs fitting.
  3. Fills fit_params, fit_errors, fit_result, quality, figures.
  4. Returns the mutated ExperimentData.

This decouples analysis from acquisition: stored HDF5 data can be
re-analysed without re-running the hardware experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from .experiment_data import ExperimentData, QualityFlag


class BaseAnalysis(ABC):
    """
    Abstract analysis class.

    Subclass contract
    -----------------
    1. Set ``thresholds`` — dict mapping param-name → ``{"min": ..., "max": ...}``.
       Used by :meth:`_assess_quality` to auto-assign ``QualityFlag``.
    2. Override :meth:`_run` — perform fitting on ``data.raw_iq``, populate
       ``data.fit_params``, ``data.fit_errors``, ``data.fit_result``,
       ``data.scalar_result`` and append figures to ``data.figures``.

    Usage
    -----
    ::

        result = T1Analysis().run(exp_data)
        print(result.quality, result.fit_result["T1"])
    """

    thresholds: dict = {}

    def run(self, data: "ExperimentData") -> "ExperimentData":
        """
        Run analysis on *data* and return the annotated ExperimentData.

        Calls :meth:`_run`, then :meth:`_assess_quality`.
        """
        if data.raw_iq is None:
            from .experiment_data import QualityFlag

            data.quality = QualityFlag.BAD
            data.quality_message = "No raw IQ data present"
            return data

        self._run(data)
        data.quality = self._assess_quality(data)
        return data

    @abstractmethod
    def _run(self, data: "ExperimentData") -> None:
        """
        Perform fitting and populate ``data.fit_params``, ``data.fit_errors``,
        ``data.fit_result``, ``data.scalar_result``, and ``data.figures``.

        Mutates *data* in place; return value is ignored.
        """
        ...

    def _assess_quality(self, data: "ExperimentData") -> "QualityFlag":
        """
        Compare ``data.fit_result`` against ``self.thresholds``.

        Returns ``GOOD`` if all thresholds pass, ``BAD`` if any fail,
        ``WARNING`` if fit is valid but marginal, ``NO_INFORMATION`` if
        no thresholds are defined.
        """
        from .experiment_data import QualityFlag

        if not self.thresholds:
            return QualityFlag.NO_INFORMATION

        for param, bounds in self.thresholds.items():
            val = data.fit_result.get(param)
            if val is None:
                continue
            v = val[0] if isinstance(val, (tuple, list)) else val
            lo = bounds.get("min")
            hi = bounds.get("max")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                data.quality_message = f"{param}={v:.4g} out of [{lo}, {hi}]"
                return QualityFlag.BAD

        return QualityFlag.GOOD


class IdentityAnalysis(BaseAnalysis):
    """No-op analysis — passes data through unchanged."""

    def _run(self, data: "ExperimentData") -> None:
        pass
