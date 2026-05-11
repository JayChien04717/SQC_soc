"""Small HDF5 data helpers used by the GUI data browser."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


def list_data_files(root_dir: str) -> list[dict]:
    """Return HDF5 files under ``root_dir`` with lightweight metadata."""
    files = []
    root = Path(root_dir)
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            continue
        meta = _read_meta(path)
        files.append({
            "path": str(path),
            "filename": path.name,
            "timestamp": meta.get("timestamp", ""),
            "experiment": meta.get("experiment_type", path.stem),
            "qubit": _qubit_index(meta),
            "tag": meta.get("metadata", {}).get("tag", ""),
        })
    files.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return files


def load_data(filepath: str) -> dict:
    """Load a saved experiment into the dict shape expected by the GUI."""
    with h5py.File(filepath, "r") as h5:
        meta = json.loads(h5.attrs.get("meta", "{}"))
        data_group = h5.get("data")
        if data_group is None:
            raise ValueError(f"No 'data' group found in {filepath}")

        avgi = np.asarray(data_group["avgi"][:])
        avgq = np.asarray(data_group["avgq"][:])
        mag = np.asarray(data_group["mag"][:]) if "mag" in data_group else np.abs(avgi + 1j * avgq)
        phase = (
            np.asarray(data_group["phase"][:])
            if "phase" in data_group
            else np.degrees(np.arctan2(avgq, avgi))
        )

        x = _axis_dict(h5, "x", fallback_size=avgi.shape[-1])
        y = _axis_dict(h5, "y", fallback_size=avgi.shape[0]) if "y" in h5 else None

    cfg = meta.get("config", {}) or {}
    return {
        "path": filepath,
        "filename": os.path.basename(filepath),
        "experiment": meta.get("experiment_type", Path(filepath).stem),
        "qubit": _qubit_index(meta),
        "timestamp": meta.get("timestamp", ""),
        "tag": cfg.get("tag", meta.get("metadata", {}).get("tag", "")),
        "config": cfg,
        "fit_result": meta.get("fit_result", {}),
        "x": x,
        "y": y,
        "avgi": avgi,
        "avgq": avgq,
        "mag": mag,
        "phase": phase,
    }


def _read_meta(path: Path) -> dict:
    try:
        with h5py.File(path, "r") as h5:
            return json.loads(h5.attrs.get("meta", "{}"))
    except Exception:
        return {}


def _axis_dict(h5, name: str, fallback_size: int) -> dict:
    if name in h5:
        group = h5[name]
        values = np.asarray(group["values"][:])
        axis_name = group.attrs.get("name", name)
        unit = group.attrs.get("unit", "")
    else:
        values = np.arange(fallback_size, dtype=float)
        axis_name = name
        unit = ""
    if isinstance(axis_name, bytes):
        axis_name = axis_name.decode()
    if isinstance(unit, bytes):
        unit = unit.decode()
    return {"name": axis_name or name, "unit": unit or "", "values": values}


def _qubit_index(meta: dict):
    cfg = meta.get("config", {}) or {}
    raw = cfg.get("name") or cfg.get("qubit") or meta.get("metadata", {}).get("qubit")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        match = re.search(r"(\d+)$", raw)
        if match:
            return int(match.group(1))
    return 0
