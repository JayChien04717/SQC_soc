"""
data_manager.py — HDF5 save/load for qick_workspace experiments.

Provides a Labber-independent save format using h5py.
Each file stores raw IQ data plus pre-computed mag/phase, axis arrays,
and a JSON config snapshot.

HDF5 layout
-----------
/
├── attrs: experiment, qubit, timestamp, tag, config (JSON)
├── x/     values + attrs: name, unit
├── y/     values + attrs: name, unit   (2-D experiments only)
└── data/
    ├── avgi     I component (averaged)
    ├── avgq     Q component (averaged)
    ├── mag      sqrt(I² + Q²)
    └── phase    arctan2(Q, I) in degrees

Usage
-----
from qick_workspace.tools.data_manager import save_data, load_data

save_data(
    expt,               # BaseExperiment instance after run()
    qb_idx=2,
    config_all=cfg,     # optional ExperimentConfig
    title="sweep1",     # optional filename suffix
)

data = load_data("path/to/file.h5")
# data["avgi"], data["mag"], data["x"]["values"], data["config"], ...
"""

import json
import os
from datetime import datetime

import h5py
import numpy as np

try:
    from .system_cfg import DATA_PATH
except Exception:
    DATA_PATH = ""


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_data(expt, qb_idx, config_all=None, yoko_value=None, title=None):
    """
    Save experiment data to a self-contained HDF5 file.

    Stores raw IQ data together with pre-computed mag and phase, the sweep
    axis arrays, and a JSON snapshot of the experiment config.  No Labber
    dependency is required.

    Parameters
    ----------
    expt : BaseExperiment
        Experiment instance after :meth:`~scrip.base_experiment.BaseExperiment.run`
        has been called.  Must have ``iqdata``, ``_sweep_vals_x``,
        ``_sweep_vals_y``, ``cfg``, ``EXPT_NAME``, ``TAG``,
        ``X_SAVE_NAME``, ``X_SAVE_UNIT``, ``X_SAVE_SCALE`` attributes.
    qb_idx : int
        Qubit index, used in the filename.
    config_all : ExperimentConfig, optional
        Full multi-qubit config object.  When provided,
        ``config_all.to_yaml(q_id=qb_idx)`` is called for the config
        snapshot.  Falls back to ``config_to_yaml(expt.cfg)`` when ``None``.
    yoko_value : float or dict or None, optional
        Yokogawa flux bias value, embedded in the filename via
        :func:`~tools.system_tool.get_next_filename_labber`.
    title : str, optional
        Optional filename suffix (e.g. ``"sweep1"``).

    Returns
    -------
    file_path : str
        Absolute path of the written ``.h5`` file.

    Raises
    ------
    RuntimeError
        If ``expt.iqdata`` is ``None`` (experiment was not run or was
        interrupted before any data was acquired).
    """
    if expt.iqdata is None:
        raise RuntimeError("No data to save — run the experiment first.")

    from .system_tool import get_next_filename_labber, config_to_yaml
    from .system_cfg import DATA_PATH as _DATA_PATH
    from scrip.base_experiment import BaseExperiment

    # ── Build filename ───────────────────────────────────────────────────────
    if title is not None:
        expt_name = f"{expt.EXPT_NAME}_{qb_idx}_{title}"
    else:
        expt_name = f"{expt.EXPT_NAME}_{qb_idx}"

    save_dir = BaseExperiment._data_path or _DATA_PATH
    base_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
    file_path = base_path + ".h5"

    # ── Config snapshot ──────────────────────────────────────────────────────
    if config_all is not None:
        cfg_dict = config_all.to_yaml(q_id=qb_idx)
    else:
        cfg_dict = config_to_yaml(expt.cfg)
    cfg_json = json.dumps(cfg_dict, default=str, indent=2)

    # ── Compute avgi / avgq / mag / phase ────────────────────────────────────
    iq = np.asarray(expt.iqdata)
    avgi = np.real(iq)
    avgq = np.imag(iq)
    mag = np.abs(iq)
    phase = np.degrees(np.arctan2(avgq, avgi))

    # ── Write HDF5 ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "w") as f:
        # Root attributes
        f.attrs["experiment"] = expt.EXPT_NAME
        f.attrs["qubit"] = int(qb_idx)
        f.attrs["timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["tag"] = expt.TAG
        f.attrs["config"] = cfg_json

        # x axis
        xg = f.create_group("x")
        xg.create_dataset("values",
                          data=(expt._sweep_vals_x * expt.X_SAVE_SCALE))
        xg.attrs["name"] = expt.X_SAVE_NAME
        xg.attrs["unit"] = expt.X_SAVE_UNIT

        # y axis (2-D experiments only)
        if expt._sweep_vals_y is not None:
            yg = f.create_group("y")
            yg.create_dataset("values",
                              data=(expt._sweep_vals_y * expt.Y_SAVE_SCALE))
            yg.attrs["name"] = expt.Y_SAVE_NAME
            yg.attrs["unit"] = expt.Y_SAVE_UNIT

        # Data channels
        dg = f.create_group("data")
        dg.create_dataset("avgi",  data=avgi,  dtype=np.float32)
        dg.create_dataset("avgq",  data=avgq,  dtype=np.float32)
        dg.create_dataset("mag",   data=mag,   dtype=np.float32)
        dg.create_dataset("phase", data=phase, dtype=np.float32)

    print(f"Data saved → {file_path}")
    return file_path


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_data(file_path):
    """
    Load experiment data from an HDF5 file written by :func:`save_data`.

    Parameters
    ----------
    file_path : str
        Path to the ``.h5`` file.

    Returns
    -------
    data : dict
        Dictionary with keys:

        ``"experiment"``
            Experiment name string.
        ``"qubit"``
            Qubit index (int).
        ``"timestamp"``
            ISO-format timestamp string.
        ``"tag"``
            Experiment tag string.
        ``"config"``
            Parsed config dict (from JSON).
        ``"x"``
            Dict with ``"values"`` (ndarray), ``"name"``, ``"unit"``.
        ``"y"``
            Dict (same structure as ``"x"``), or ``None`` for 1-D data.
        ``"avgi"``
            Averaged I component (ndarray).
        ``"avgq"``
            Averaged Q component (ndarray).
        ``"mag"``
            Magnitude ``sqrt(I² + Q²)`` (ndarray).
        ``"phase"``
            Phase in degrees (ndarray).
    """
    with h5py.File(file_path, "r") as f:
        data = {
            "experiment": f.attrs.get("experiment", ""),
            "qubit":      int(f.attrs.get("qubit", -1)),
            "timestamp":  f.attrs.get("timestamp", ""),
            "tag":        f.attrs.get("tag", ""),
            "config":     json.loads(f.attrs.get("config", "{}")),
        }

        # x axis
        xg = f["x"]
        data["x"] = {
            "values": xg["values"][:],
            "name":   xg.attrs.get("name", ""),
            "unit":   xg.attrs.get("unit", ""),
        }

        # y axis (optional)
        if "y" in f:
            yg = f["y"]
            data["y"] = {
                "values": yg["values"][:],
                "name":   yg.attrs.get("name", ""),
                "unit":   yg.attrs.get("unit", ""),
            }
        else:
            data["y"] = None

        # Data channels
        dg = f["data"]
        data["avgi"]  = dg["avgi"][:]
        data["avgq"]  = dg["avgq"][:]
        data["mag"]   = dg["mag"][:]
        data["phase"] = dg["phase"][:]

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Scan directory
# ──────────────────────────────────────────────────────────────────────────────

def list_data_files(directory=None):
    """
    Recursively list all ``.h5`` data files in a directory.

    Parameters
    ----------
    directory : str, optional
        Root directory to scan.  Defaults to ``DATA_PATH``.

    Returns
    -------
    files : list of dict
        Each entry contains:

        ``"path"``
            Absolute file path.
        ``"filename"``
            Basename without extension.
        ``"experiment"``
            Experiment name from HDF5 root attrs.
        ``"qubit"``
            Qubit index from HDF5 root attrs.
        ``"timestamp"``
            Timestamp string from HDF5 root attrs.
        ``"tag"``
            Tag string from HDF5 root attrs.
        ``"mtime"``
            File modification time (float, Unix timestamp).
    """
    root = directory or DATA_PATH
    files = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".h5"):
                continue
            full = os.path.join(dirpath, fname)
            entry = {
                "path":     full,
                "filename": os.path.splitext(fname)[0],
                "experiment": "",
                "qubit":    -1,
                "timestamp": "",
                "tag":      "",
                "mtime":    os.path.getmtime(full),
            }
            try:
                with h5py.File(full, "r") as f:
                    entry["experiment"] = f.attrs.get("experiment", "")
                    entry["qubit"]      = int(f.attrs.get("qubit", -1))
                    entry["timestamp"]  = f.attrs.get("timestamp", "")
                    entry["tag"]        = f.attrs.get("tag", "")
            except Exception:
                pass
            files.append(entry)

    files.sort(key=lambda e: e["mtime"], reverse=True)
    return files
