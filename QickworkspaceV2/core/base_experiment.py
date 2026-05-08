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

from .base_analysis import BaseAnalysis
from .experiment_data import ExperimentData, QualityFlag


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
        self._last_prog = None

    def prog_asm(self, use_last: bool = False):
        """
        Build and print the QICK program for this experiment.

        Useful in notebooks before acquisition:

            prog = expt.prog_asm()

        Parameters
        ----------
        use_last : bool, default False
            When True, print the most recently built program if available.

        Returns
        -------
        object
            The program object that was printed.
        """
        if use_last and self._last_prog is not None:
            prog = self._last_prog
        else:
            prog = self._create_program()

        self._last_prog = prog
        print(prog)
        return prog

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
        # Snapshot config BEFORE _create_program: QICK mutates the dict in-place,
        # replacing QickSweep1D (MHz) with compiled QickParam (register units).
        try:
            self._cfg_snap = dict(self.cfg)
        except Exception:
            self._cfg_snap = None
        prog = self._create_program()
        self._last_prog = prog

        threshold = self._get_readout_threshold()
        if threshold is not None:
            return self._run_threshold_acquire(prog, threshold, py_avg=py_avg)

        self._sweep_vals_x = BaseExperiment._resolve_axis(
            self._extract_sweep_axis(prog), self.cfg.get("steps")
        )
        self._sweep_vals_y = BaseExperiment._resolve_axis(
            self._extract_sweep_axis_y(prog), self.cfg.get("steps")
        )

        yoko_value_kwarg = kwargs.get("yoko_value")
        if yoko_value_kwarg is not None:
            self._sweep_vals_y = np.asarray(yoko_value_kwarg, dtype=float)

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
            analysis_inst = self.Analysis()
            result = analysis_inst.run(result)
            analysis_inst.plot(result)

        self.result = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════════════

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """Legacy Labber-format HDF5 save (unchanged from original)."""
        from ..config.system_cfg import DATA_PATH
        from ..tools.system_tool import (
            config_to_yaml,
            get_next_filename_labber,
            hdf5_generator,
        )

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
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_axis(vals, steps=None):
        """
        Convert whatever _extract_sweep_axis returns into a plain float array.

        get_pulse_param / get_time_param may return a QickParam sweep object
        instead of a numpy array, depending on the QICK version.  This method
        tries every known extraction path and catches RuntimeError from
        QickParam.__float__.

        Always call as  BaseExperiment._resolve_axis(vals, steps)  (not via
        self._resolve_axis) to avoid Python descriptor ambiguity.
        """
        if vals is None:
            return None

        # Already a plain numpy numeric array — fast path
        if isinstance(vals, np.ndarray) and np.issubdtype(vals.dtype, np.number):
            return vals.astype(float)

        # QickParam: try every known array-extraction method
        for method in ("to_array", "sweep_vals", "get_array"):
            fn = getattr(vals, method, None)
            if callable(fn):
                try:
                    return np.asarray(fn(), dtype=float)
                except Exception:
                    pass

        # QickParam with is_sweep(): extract from start/stop/step/expts/steps
        if hasattr(vals, "is_sweep") and callable(vals.is_sweep) and vals.is_sweep():
            start = getattr(vals, "start", None)
            stop = getattr(vals, "stop", None)
            step = getattr(vals, "step", None)
            # 'expts' or 'steps' for count
            n_raw = getattr(vals, "expts", None) or getattr(vals, "steps", None)
            n = int(n_raw) if n_raw is not None else (int(steps) if steps else 100)
            if start is not None and stop is not None:
                try:
                    return np.linspace(float(start), float(stop), n)
                except (TypeError, ValueError, RuntimeError):
                    pass
            if start is not None and step is not None:
                try:
                    return float(start) + np.arange(n) * float(step)
                except (TypeError, ValueError, RuntimeError):
                    pass

        # QickSweep1D-like (start + stop present, no is_sweep)
        start = getattr(vals, "start", None)
        stop = getattr(vals, "stop", None)
        if start is not None and stop is not None:
            n = int(steps) if steps is not None else 100
            try:
                return np.linspace(float(start), float(stop), n)
            except (TypeError, ValueError, RuntimeError):
                pass

        # Plain Python scalar
        if isinstance(vals, (int, float)):
            return np.array([float(vals)])

        # Try direct numpy cast — catches RuntimeError from QickParam.__float__
        try:
            return np.asarray(vals, dtype=float)
        except (TypeError, ValueError, RuntimeError):
            pass

        # Object array: resolve element-wise
        try:
            obj_arr = np.asarray(vals)
            resolved = []
            for v in obj_arr.flat:
                for method in ("to_array", "sweep_vals"):
                    fn = getattr(v, method, None)
                    if callable(fn):
                        try:
                            resolved.extend(np.asarray(fn(), dtype=float).tolist())
                            break
                        except Exception:
                            pass
                else:
                    s = getattr(v, "start", None)
                    e = getattr(v, "stop", None)
                    if s is not None and e is not None and steps:
                        resolved.extend(
                            np.linspace(float(s), float(e), int(steps)).tolist()
                        )
                    elif s is not None:
                        try:
                            resolved.append(float(s))
                        except (TypeError, ValueError, RuntimeError):
                            pass
                    else:
                        try:
                            resolved.append(float(v))
                        except (TypeError, ValueError, RuntimeError):
                            pass
            if resolved:
                return np.array(resolved)
        except Exception:
            pass

        raise ValueError(f"Cannot resolve sweep axis from {type(vals).__name__}")

    def _get_readout_threshold(self):
        """Return configured readout threshold, or None when disabled."""
        if not hasattr(self.cfg, "get"):
            return None
        return self.cfg.get("threshold")

    def _run_threshold_acquire(
        self, prog, threshold=None, py_avg: int = 1
    ) -> ExperimentData:
        """
        Acquire with QICK's threshold discriminator and skip live plotting.

        QICK returns already-discriminated I/population values when threshold
        is supplied, so the result stores the returned I channel directly.
        """
        self._sweep_vals_x = BaseExperiment._resolve_axis(
            self._extract_sweep_axis(prog), self.cfg.get("steps")
        )
        self._sweep_vals_y = BaseExperiment._resolve_axis(
            self._extract_sweep_axis_y(prog), self.cfg.get("steps")
        )

        try:
            acquired = prog.acquire(
                self.soc,
                rounds=py_avg,
                threshold=threshold,
                progress=True,
            )
        except TypeError:
            acquired = prog.acquire(
                self.soc,
                threshold=threshold,
                progress=True,
            )

        try:
            i_values = acquired[0][0].dot([1, 1j]).real
        except (AttributeError, IndexError, TypeError, ValueError):
            i_values = np.asarray(acquired, dtype=float).squeeze()
        self.iqdata = i_values

        scalar = None
        if np.size(i_values) == 1:
            scalar = float(np.asarray(i_values).reshape(-1)[0])

        fit_result = {"population": (i_values, None)}
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=i_values,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            fit_params=np.array([scalar]) if scalar is not None else None,
            fit_errors=None,
            fit_result=fit_result,
            scalar_result=scalar,
            quality=QualityFlag.NO_INFORMATION,
            quality_message="Threshold discrimination used; live plot skipped.",
            config=dict(self.cfg) if hasattr(self.cfg, "__iter__") else {},
            metadata={
                "threshold": threshold,
                "threshold_discrimination": True,
            },
            interrupted=False,
            avg_count=py_avg,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )
        self.result = result
        return result

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
