"""
BaseExperiment — IBM/IQM-style base class for all experiment wrappers.

Key differences from qick_workspace BaseExperiment
----------------------------------------------------
* ``run()`` returns :class:`~core.experiment_data.ExperimentData` instead of
  raw fit tuples.
* Accepts an optional ``backend`` argument for dependency injection
  (:class:`~backend.base_backend.BaseBackend` subclass).  Falls back to the
  legacy ``setup(soc, soccfg, data_path)`` class method.
* Each subclass may declare an ``Analysis`` class attribute (a
  :class:`~core.base_analysis.BaseAnalysis` subclass) that is run
  automatically after ``_post_fit``.
* ``saveLabber`` and ``save`` both work; ``save`` uses the new HDF5 layout.

Backward Compatibility
----------------------
Old notebooks using ``BaseExperiment.setup(soc, soccfg, data_path)`` and
then calling ``expt.run()`` still work.  The returned ``ExperimentData``
supports tuple unpacking::

    fit_params, error = expt.run(py_avg)   # unchanged
    freq = float(expt.run(py_avg))         # unchanged

New API::

    result = expt.run(py_avg)
    result.is_good()
    result.fit_result["T1"]
    result.save("path/file.h5")
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np
import matplotlib.pyplot as plt

from .experiment_data import ExperimentData, QualityFlag
from .base_analysis import BaseAnalysis


class BaseExperiment:
    """
    Base class for all experiment wrappers.

    Session initialisation (legacy, call once per notebook)::

        BaseExperiment.setup(soc, soccfg, data_path)

    Or inject a Backend::

        expt = MyExperiment(cfg, backend=QICKBackend(soc, soccfg))

    Subclass contract
    -----------------
    1. Set class-level metadata (``EXPT_NAME``, ``TAG``, ``X_LABEL``, etc.).
    2. Optionally set ``Analysis`` to a :class:`BaseAnalysis` subclass.
    3. Override :meth:`_create_program` — return the QICK program instance.
    4. Override :meth:`_extract_sweep_axis` — return the x-axis array.
    5. Optionally override :meth:`_post_fit` — perform fitting; populate
       ``self.fit_params``, ``self.fit_errors``, and return the old-style
       value (tuple or scalar) for backward compat.
    """

    # ── Legacy session state ─────────────────────────────────────────────────
    _soc = None
    _soccfg = None
    _data_path = None

    @classmethod
    def setup(cls, soc, soccfg, data_path: str):
        """Initialise shared QICK session (call once at notebook startup)."""
        cls._soc = soc
        cls._soccfg = soccfg
        cls._data_path = data_path

    @classmethod
    def set_data_path(cls, data_path: str):
        cls._data_path = data_path

    # ── Subclass metadata ────────────────────────────────────────────────────
    EXPT_NAME: str = ""
    TAG: str = ""
    X_LABEL: str = ""
    Y_LABEL: str = "ADC Units"
    TITLE_PREFIX: str = ""
    SWEEP_KEYS_TO_REMOVE: list = []

    IQ_PROCESS: str = "abs"

    YOKO_VOLTAGE_RAMP_STEP: float = 1e-5
    YOKO_CURRENT_RAMP_STEP: float = 1e-8
    YOKO_RAMP_INTERVAL: float = 0.01

    X_SAVE_NAME: str = ""
    X_SAVE_UNIT: str = ""
    X_SAVE_SCALE: float = 1.0

    Y_SAVE_NAME: str = ""
    Y_SAVE_UNIT: str = ""
    Y_SAVE_SCALE: float = 1.0

    # ── NEW: link to analysis class ──────────────────────────────────────────
    Analysis: Optional[Type[BaseAnalysis]] = None

    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, config, backend=None):
        """
        Parameters
        ----------
        config : dict or ExperimentConfig
            Experiment configuration.
        backend : BaseBackend, optional
            Hardware backend.  When ``None``, falls back to the legacy
            class-level ``_soc`` / ``_soccfg`` set via ``setup()``.
        """
        if backend is not None:
            self.soc = backend.soc
            self.soccfg = backend.soccfg
        else:
            if BaseExperiment._soc is None:
                raise RuntimeError(
                    "QICK session not initialised. "
                    "Call BaseExperiment.setup(soc, soccfg, data_path) "
                    "or pass backend=QICKBackend(soc, soccfg)."
                )
            self.soc = BaseExperiment._soc
            self.soccfg = BaseExperiment._soccfg

        self.cfg = config
        self.iqdata = None
        self.fit_params = None
        self.fit_errors = None
        self.result: Optional[ExperimentData] = None
        self._sweep_vals_x = None
        self._sweep_vals_y = None
        self._yoko_mode = None

    # ══════════════════════════════════════════════════════════════════════════
    # Unified entry point
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        py_avg: int,
        iq_process: Optional[str] = None,
        show_final_plot: bool = False,
        **kwargs,
    ) -> ExperimentData:
        """
        Execute the experiment with live plotting and optional post-fit.

        Returns
        -------
        ExperimentData
            Populated result object.  Supports backward-compat unpacking::

                fit_params, error = expt.run(py_avg)
                freq = float(expt.run(py_avg))
        """
        from ..plotter.liveplot import liveplotfun

        if iq_process is not None:
            self.IQ_PROCESS = iq_process

        self._yoko_mode = kwargs.get("yoko_mode", None)
        prog = self._create_program()
        self._sweep_vals_x = self._extract_sweep_axis(prog)
        self._sweep_vals_y = self._extract_sweep_axis_y(prog)

        yoko_value_kwarg = kwargs.get("yoko_value")
        if yoko_value_kwarg is not None:
            self._sweep_vals_y = np.asarray(yoko_value_kwarg)

        yoko_addr = kwargs.get("yoko_inst_addr") or kwargs.get("yoko_inst")

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self._sweep_vals_x,
            y_axis_vals=self._sweep_vals_y,
            x_label=self.X_LABEL,
            y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX,
            yoko_inst_addr=yoko_addr,
            yoko_mode=kwargs.get("yoko_mode", "current"),
            yoko_voltage_ramp_step=self.YOKO_VOLTAGE_RAMP_STEP,
            yoko_current_ramp_step=self.YOKO_CURRENT_RAMP_STEP,
            yoko_ramp_interval=self.YOKO_RAMP_INTERVAL,
            show_final_plot=show_final_plot,
            iq_process=self.IQ_PROCESS,
        )

        if self.iqdata is None:
            print("No data was acquired.")
            result = ExperimentData(
                experiment_type=self.EXPT_NAME,
                quality=QualityFlag.BAD,
                quality_message="No data acquired",
                interrupted=True,
                avg_count=0,
            )
            self.result = result
            return result

        if interrupted:
            print(
                f"Experiment interrupted at {avg_count} averages. "
                "Fit is based on partial data."
            )

        # Run subclass post-fit (sets self.fit_params and returns old-style value)
        old_result = self._post_fit(self._sweep_vals_x)

        # ── Build ExperimentData ─────────────────────────────────────────────
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            fit_params=self.fit_params,
            fit_errors=self.fit_errors,
            config=dict(self.cfg) if hasattr(self.cfg, "__iter__") else {},
            interrupted=interrupted,
            avg_count=avg_count,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )

        # Scalar result (for experiments that return a single float/dict)
        if old_result is not None:
            if isinstance(old_result, (int, float)):
                result.scalar_result = float(old_result)
            elif isinstance(old_result, (tuple, list)) and len(old_result) == 2:
                # (fit_params_arr, errors_arr) — already in result
                pass
            elif isinstance(old_result, dict):
                result.fit_result = {k: (v, None) for k, v in old_result.items()}

        # Populate fit_result from fit_params for standard single-param experiments
        if result.fit_result == {} and self.fit_params is not None:
            result.fit_result = self._build_fit_result()

        # ── Run registered Analysis class ────────────────────────────────────
        if self.Analysis is not None:
            result = self.Analysis().run(result)

        self.result = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, qb_idx, yoko_value=None, config_all=None, title=None) -> str:
        """Save to HDF5 using the new ExperimentData format."""
        if self.result is None:
            raise RuntimeError("Run the experiment before saving.")

        from ..tools.system_tool import get_next_filename_labber

        data_path = BaseExperiment._data_path or ""
        suffix = f"_{title}" if title else ""
        expt_name = f"{self.EXPT_NAME}_{qb_idx}{suffix}"
        base = get_next_filename_labber(data_path, expt_name, yoko_value)
        filepath = base + ".h5"

        return self.result.save(filepath)

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """Legacy Labber-format HDF5 save (unchanged from original)."""
        from ..tools.system_tool import (
            get_next_filename_labber,
            hdf5_generator,
            config_to_yaml,
        )
        from ..config.system_cfg import DATA_PATH

        if title is not None:
            expt_name = f"{self.EXPT_NAME}_{qb_idx}_{title}"
        else:
            expt_name = f"{self.EXPT_NAME}_{qb_idx}"

        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)

        if config_all is not None:
            dict_val = config_all.to_yaml(q_id=qb_idx)
        else:
            dict_val = config_to_yaml(self.cfg)

        comment = self._save_comment(dict_val)

        x_info = {
            "name": self.X_SAVE_NAME,
            "unit": self.X_SAVE_UNIT,
            "values": self._sweep_vals_x * self.X_SAVE_SCALE,
        }
        y_info = None
        if self._sweep_vals_y is not None:
            y_info = {
                "name": self.Y_SAVE_NAME,
                "unit": self.Y_SAVE_UNIT,
                "values": self._sweep_vals_y * self.Y_SAVE_SCALE,
            }

        hdf5_generator(
            filepath=file_path,
            x_info=x_info,
            y_info=y_info,
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=comment,
            tag=self.TAG,
        )
        print(f"Data saved to {file_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Subclass MUST override
    # ══════════════════════════════════════════════════════════════════════════

    def _create_program(self):
        raise NotImplementedError("Subclass must implement _create_program()")

    def _extract_sweep_axis(self, prog) -> np.ndarray:
        raise NotImplementedError("Subclass must implement _extract_sweep_axis()")

    def _extract_sweep_axis_y(self, prog) -> Optional[np.ndarray]:
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Subclass MAY override
    # ══════════════════════════════════════════════════════════════════════════

    def _post_fit(self, x_vals):
        """Optional: fit and return old-style value. Should set self.fit_params."""
        return None

    def _save_comment(self, dict_val: str) -> str:
        return str(dict_val)

    def _build_fit_result(self) -> dict:
        """Build named fit_result dict from self.fit_params. Override for clarity."""
        return {}
