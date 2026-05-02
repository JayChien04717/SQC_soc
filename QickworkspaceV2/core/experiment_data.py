"""
ExperimentData — unified result container for all experiments.

Backward compatibility
----------------------
Old code that unpacks ``(fit_params, error) = expt.run()`` still works via
``__iter__``.  Old code that does ``freq = expt.run()`` still works via
``__float__`` / ``__int__``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class QualityFlag(Enum):
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    NO_INFORMATION = "no_information"


@dataclass
class ExperimentData:
    """
    Unified result container returned by every ``BaseExperiment.run()`` call.

    Backward Compatibility
    ----------------------
    * ``fit_params, error = result``  — tuple unpacking via ``__iter__``
    * ``float(result)``               — scalar result via ``__float__``
    * ``result[0]``, ``result[1]``    — index access via ``__getitem__``

    New API
    -------
    * ``result.fit_result``     — named dict of fitted parameters + uncertainties
    * ``result.quality``        — ``QualityFlag`` enum
    * ``result.is_good()``      — convenience boolean
    * ``result.to_dict()``      — JSON-serialisable dict
    * ``result.save(path)``     — HDF5 save
    * ``ExperimentData.load(path)`` — HDF5 load
    """

    # Identity
    experiment_type: str = ""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # Raw data
    raw_iq: Optional[np.ndarray] = None
    x_axis: Optional[np.ndarray] = None
    y_axis: Optional[np.ndarray] = None

    # Fit results (backward-compat arrays)
    fit_params: Optional[np.ndarray] = None
    fit_errors: Optional[np.ndarray] = None

    # Named results dict — new API; keyed by param name, value is (val, err)
    fit_result: dict = field(default_factory=dict)

    # Scalar result for experiments that return a single number (e.g. frequency)
    scalar_result: Optional[float] = None

    # Figures (matplotlib Figure objects)
    figures: list = field(default_factory=list)

    # Quality
    quality: QualityFlag = QualityFlag.NO_INFORMATION
    quality_message: str = ""

    # Config snapshot
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # Composite / lineage
    parent_id: Optional[str] = None
    children: list = field(default_factory=list)

    # Axis save info
    x_name: str = ""
    x_unit: str = ""
    x_scale: float = 1.0
    y_name: str = ""
    y_unit: str = ""
    y_scale: float = 1.0

    # Acquisition info
    interrupted: bool = False
    avg_count: int = 0

    # ── Backward-compat dunder methods ──────────────────────────────────────

    def __iter__(self):
        """Support ``fit_params, error = result``."""
        yield self.fit_params
        yield self.fit_errors

    def __getitem__(self, idx):
        """Support ``result[0]`` (fit_params) and ``result[1]`` (fit_errors)."""
        return (self.fit_params, self.fit_errors)[idx]

    def __float__(self):
        """Support ``freq = float(result)`` for single-value experiments."""
        if self.scalar_result is not None:
            return float(self.scalar_result)
        if self.fit_params is not None and len(self.fit_params) > 0:
            return float(self.fit_params[0])
        raise TypeError(f"ExperimentData '{self.experiment_type}' has no scalar result")

    def __bool__(self):
        """True when data was acquired and fit succeeded."""
        return self.raw_iq is not None and self.fit_params is not None

    # ── Convenience ─────────────────────────────────────────────────────────

    def is_good(self) -> bool:
        return self.quality == QualityFlag.GOOD

    def get_param(self, name: str, default=None):
        """Return named fit result value, or default."""
        entry = self.fit_result.get(name)
        if entry is None:
            return default
        return entry[0] if isinstance(entry, (tuple, list)) else entry

    def get_error(self, name: str, default=None):
        """Return named fit result uncertainty, or default."""
        entry = self.fit_result.get(name)
        if isinstance(entry, (tuple, list)) and len(entry) > 1:
            return entry[1]
        return default

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (numpy arrays → lists)."""

        def _arr(v):
            return v.tolist() if isinstance(v, np.ndarray) else v

        return {
            "experiment_type": self.experiment_type,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "fit_params": _arr(self.fit_params),
            "fit_errors": _arr(self.fit_errors),
            "fit_result": {
                k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v)
                for k, v in self.fit_result.items()
            },
            "scalar_result": self.scalar_result,
            "quality": self.quality.value,
            "quality_message": self.quality_message,
            "config": self.config,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "x_name": self.x_name,
            "x_unit": self.x_unit,
            "x_scale": self.x_scale,
            "y_name": self.y_name,
            "y_unit": self.y_unit,
            "y_scale": self.y_scale,
            "interrupted": self.interrupted,
            "avg_count": self.avg_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentData":
        obj = cls(
            experiment_type=d.get("experiment_type", ""),
            experiment_id=d.get("experiment_id", str(uuid.uuid4())[:8]),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
            fit_params=np.array(d["fit_params"]) if d.get("fit_params") is not None else None,
            fit_errors=np.array(d["fit_errors"]) if d.get("fit_errors") is not None else None,
            fit_result=d.get("fit_result", {}),
            scalar_result=d.get("scalar_result"),
            quality=QualityFlag(d.get("quality", "no_information")),
            quality_message=d.get("quality_message", ""),
            config=d.get("config", {}),
            metadata=d.get("metadata", {}),
            parent_id=d.get("parent_id"),
            x_name=d.get("x_name", ""),
            x_unit=d.get("x_unit", ""),
            x_scale=d.get("x_scale", 1.0),
            y_name=d.get("y_name", ""),
            y_unit=d.get("y_unit", ""),
            y_scale=d.get("y_scale", 1.0),
            interrupted=d.get("interrupted", False),
            avg_count=d.get("avg_count", 0),
        )
        return obj

    def save(self, filepath: str) -> str:
        """Save ExperimentData to HDF5 (raw IQ + metadata)."""
        import os

        import h5py

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with h5py.File(filepath, "w") as f:
            # Root attrs: JSON snapshot of everything except arrays
            f.attrs["meta"] = json.dumps(self.to_dict())

            if self.raw_iq is not None:
                iq = np.asarray(self.raw_iq)
                dg = f.create_group("data")
                dg.create_dataset("avgi", data=np.real(iq).astype(np.float32))
                dg.create_dataset("avgq", data=np.imag(iq).astype(np.float32))
                dg.create_dataset("mag", data=np.abs(iq).astype(np.float32))
                dg.create_dataset("phase", data=np.degrees(np.arctan2(np.imag(iq), np.real(iq))).astype(np.float32))

            if self.x_axis is not None:
                xg = f.create_group("x")
                xg.create_dataset("values", data=(self.x_axis * self.x_scale))
                xg.attrs["name"] = self.x_name
                xg.attrs["unit"] = self.x_unit

            if self.y_axis is not None:
                yg = f.create_group("y")
                yg.create_dataset("values", data=(self.y_axis * self.y_scale))
                yg.attrs["name"] = self.y_name
                yg.attrs["unit"] = self.y_unit

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "ExperimentData":
        """Load ExperimentData from HDF5 file."""
        import h5py

        with h5py.File(filepath, "r") as f:
            obj = cls.from_dict(json.loads(f.attrs["meta"]))

            if "data" in f:
                dg = f["data"]
                obj.raw_iq = dg["avgi"][:] + 1j * dg["avgq"][:]

            if "x" in f:
                xg = f["x"]
                scale = obj.x_scale if obj.x_scale != 0 else 1.0
                obj.x_axis = xg["values"][:] / scale

            if "y" in f:
                yg = f["y"]
                scale = obj.y_scale if obj.y_scale != 0 else 1.0
                obj.y_axis = yg["values"][:] / scale

        return obj

    def __repr__(self) -> str:
        status = "interrupted" if self.interrupted else "complete"
        return (
            f"ExperimentData(type={self.experiment_type!r}, "
            f"id={self.experiment_id!r}, quality={self.quality.value}, "
            f"status={status})"
        )
