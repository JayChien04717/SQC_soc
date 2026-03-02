"""
BaseExperiment: Base class for all experiment wrappers.
========================================================
Unifies the run (liveplot-first) and saveLabber logic.
Subclasses only need to override a few methods + declare metadata.
Supports `simulate=True` mode for hardware-free testing.
"""
import numpy as np
import matplotlib.pyplot as plt

from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml
from ..plotter.liveplot import liveplotfun
from ..plotter.plot_utils import plot_final


class BaseExperiment:
    """
    Base class for all experiment wrappers.

    Subclasses need to:
    1. Set class-level metadata attributes (EXPT_NAME, TAG, X_LABEL, etc.)
    2. Override _create_program() — return the Program instance
    3. Override _extract_sweep_axis(prog) — return the sweep axis array
    4. Optionally override _post_fit() — perform fitting after liveplot
    5. Optionally override _simulate() — return mock IQ data for simulate mode

    Example:
        class QubitSpec(BaseExperiment):
            EXPT_NAME = "s003_qubit_spec_ge"
            TAG = "TwoTone"
            X_LABEL = "Frequency (MHz)"
            TITLE_PREFIX = "Qubit ge Spectrum"
            SWEEP_KEYS_TO_REMOVE = ["qb_freq_ge"]
            ...
    """

    # ── Subclass must set these metadata ──
    EXPT_NAME: str = ""
    TAG: str = ""
    X_LABEL: str = ""
    Y_LABEL: str = "ADC Units"
    TITLE_PREFIX: str = ""
    SWEEP_KEYS_TO_REMOVE: list = []

    # ── x-axis save info (for saveLabber) ──
    X_SAVE_NAME: str = ""      # e.g. "Frequency", "Times", "Gain"
    X_SAVE_UNIT: str = ""      # e.g. "Hz", "us", "DAC unit"
    X_SAVE_SCALE: float = 1.0  # multiply factor for unit conversion (e.g. 1e6 for MHz→Hz)

    # ── y-axis save info (optional, for 2D experiments) ──
    Y_SAVE_NAME: str = ""
    Y_SAVE_UNIT: str = ""
    Y_SAVE_SCALE: float = 1.0

    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config
        self.iqdata = None
        self.fit_params = None
        self._sweep_vals = None    # sweep axis values (x)
        self._sweep_vals_y = None  # sweep axis values (y, for 2D)

    # ══════════════════════════════════════════════
    # Unified entry point
    # ══════════════════════════════════════════════

    def run(self, py_avg, simulate=False, **kwargs):
        """
        Execute the experiment with liveplot, or simulate without hardware.

        Args:
            py_avg: number of software averages
            simulate: if True, bypass hardware and generate mock data
            **kwargs: extra args passed to subclass hooks

        Returns:
            Fitting result (if any), or None.
        """
        # ── Simulate mode: no hardware needed ──
        if simulate:
            self._sweep_vals = self._mock_sweep_axis(**kwargs)
            self._sweep_vals_y = self._mock_sweep_axis_y(**kwargs)

            if self._sweep_vals_y is not None:
                # 2D Simulation
                self.iqdata = self._simulate(self._sweep_vals, self._sweep_vals_y)
                
                # Show static plot (2D)
                fig, ax = plt.subplots(figsize=(6, 5))
                pcm = ax.pcolormesh(
                    self._sweep_vals, 
                    self._sweep_vals_y, 
                    np.abs(self.iqdata), 
                    shading='auto'
                )
                ax.set_xlabel(self.X_LABEL)
                ax.set_ylabel(self.Y_LABEL)
                ax.set_title(f"{self.TITLE_PREFIX} [SIMULATED]")
                fig.colorbar(pcm, ax=ax, label="ADC Units (Abs)")
                fig.tight_layout()
                plt.show()
                
            else:
                # 1D Simulation
                self.iqdata = self._simulate(self._sweep_vals)

                # Show static plot (1D)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(self._sweep_vals, np.abs(self.iqdata), "o-", markersize=4, alpha=0.7)
                ax.set_xlabel(self.X_LABEL)
                ax.set_ylabel("ADC Units (Abs)")
                ax.set_title(f"{self.TITLE_PREFIX} [SIMULATED]")
                fig.tight_layout()
                plt.show()

            return self._post_fit(self._sweep_vals)

        # ── Normal hardware mode ──
        prog = self._create_program()
        self._sweep_vals_x = self._extract_sweep_axis(prog)
        self._sweep_vals_y = self._extract_sweep_axis_y(prog)

        self.iqdata, interrupted, avg_count = liveplotfun(
            prog=prog,
            soc=self.soc,
            py_avg=py_avg,
            x_axis_vals=self._sweep_vals_x,
            y_axis_vals=self._sweep_vals_y,
            x_label=self.X_LABEL,
            y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX,
            yoko_inst_addr=kwargs.get("yoko_inst_addr"),
            yoko_mode=kwargs.get("yoko_mode", "current"),
            show_final_plot=False,
        )

        if self.iqdata is None:
            print("No data was acquired.")
            return None

        if interrupted:
            print(
                f"Experiment interrupted at {avg_count} averages. "
                "Fit is based on partial data."
            )

        return self._post_fit(self._sweep_vals_x)

    # ══════════════════════════════════════════════
    # Unified save
    # ══════════════════════════════════════════════

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None):
        """
        Save data to HDF5 (Labber) format.

        Args:
            qb_idx: qubit index/name for filename and config extraction.
            yoko_value: optional yoko current value for filename.
            config_all: optional ExperimentConfig instance.
                        If provided → nested YAML via config_all.to_yaml(q_id).
                        If None → flat YAML via config_to_yaml(self.cfg).
        """
        expt_name = f"{self.EXPT_NAME}_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        if config_all is not None:
            dict_val = config_all.to_yaml(q_id=qb_idx)
        else:
            dict_val = config_to_yaml(self.cfg)

        comment = self._save_comment(dict_val)

        # Construct x_info
        x_info = {
            "name": self.X_SAVE_NAME,
            "unit": self.X_SAVE_UNIT,
            "values": self._sweep_vals_x * self.X_SAVE_SCALE,
        }

        # Construct y_info (optional)
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

    # ══════════════════════════════════════════════
    # Subclass MUST override
    # ══════════════════════════════════════════════

    def _create_program(self):
        """Subclass must implement: create and return the Program instance."""
        raise NotImplementedError("Subclass must implement _create_program()")

    def _extract_sweep_axis(self, prog):
        """Subclass must implement: extract and return sweep axis values from prog."""
        raise NotImplementedError("Subclass must implement _extract_sweep_axis()")

    def _extract_sweep_axis_y(self, prog):
        """
        Optional: extract and return y-axis sweep values (for 2D experiments).
        Default: returns None (indicating 1D experiment).
        """
        return None

    # ══════════════════════════════════════════════
    # Subclass MAY override (hooks)
    # ══════════════════════════════════════════════

    def _post_fit(self, x_vals):
        """
        Optional hook: perform fitting after liveplot and return the result.
        Default: no fitting.
        """
        return None

    def _simulate(self, x_pts):
        """
        Optional hook: generate mock IQ data for simulate mode.
        Subclass should override to provide experiment-specific signals.
        Default: raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support simulate mode. "
            "Override _simulate(x_pts) or _simulate(x_pts, y_pts) to enable it."
        )

    def _mock_sweep_axis_y(self, **kwargs):
        """
        Optional: Generate y-axis sweep values without hardware.
        Default: returns None.
        """
        # Allow explicit override via kwargs
        if "y_pts" in kwargs:
            return np.asarray(kwargs["y_pts"])
        return None

    def _mock_sweep_axis(self, **kwargs):
        """
        Generate sweep axis values without hardware.
        Tries to extract from QickSweep1D objects in cfg, or from kwargs.
        """
        # Allow explicit override via kwargs
        if "x_pts" in kwargs:
            return np.asarray(kwargs["x_pts"])

        # Try to extract from QickSweep1D objects in cfg
        for key in self.SWEEP_KEYS_TO_REMOVE:
            val = self.cfg.get(key)
            if val is not None and hasattr(val, "start"):
                try:
                    span = list(val.spans.values())[0] if hasattr(val, "spans") and val.spans else 0
                    stop = val.start + span
                except Exception:
                    # Fallback if spans dict isn't available
                    stop = getattr(val, "stop", getattr(val, "maxval", val.start))
                    
                steps = self.cfg.get("steps", 101)
                return np.linspace(val.start, stop, steps)

        raise ValueError(
            f"Cannot determine sweep axis for {self.__class__.__name__} in simulate mode. "
            "Pass x_pts=np.linspace(...) explicitly."
        )

    def subjob(self, py_avg: int, qubit: str = "Q1", priority: int = 0, user: str = "jay", server_url: str = "http://127.0.0.1:8585", wait: bool = True, analyze: bool = True):
        """
        Submit this experiment to the QICK Job Server.

        If ``wait`` is ``True`` (default), blocks until the job completes and populates
        ``self.iqdata``, ``self._sweep_vals_x`` etc. from the worker's result.
        If ``wait`` is ``False``, returns a ``JobHandle`` for asynchronous tracking.

        The optional ``analyze`` flag (default ``True``) determines whether the
        post‑fit routine ``_post_fit`` is applied after the data is loaded.

        Args:
            py_avg: Number of software averages
            qubit: Target qubit identifier (e.g., "Q1")
            priority: Higher = runs sooner (default 0)
            user: Username for the job queue
            server_url: URL of the QICK Job Server
            wait: If ``True``, block until completion
            analyze: If ``True``, run ``_post_fit`` on the loaded sweep values.

        Returns:
            If ``wait`` is ``True``:
                - ``self.iqdata`` (numpy array) when ``analyze`` is ``False``
                - The result of ``_post_fit`` when ``analyze`` is ``True``
            If ``wait`` is ``False``: ``JobHandle`` instance
        """
        from qick_workspace.qick_job_server.client import JobClient
        client = JobClient(server_url)

        # Submit using the instance's own class, module, and cfg
        job_id = client.submit(
            experiment_class=self.__class__.__name__,
            experiment_module=self.__module__,
            run_cfg=dict(self.cfg) if hasattr(self.cfg, 'items') else self.cfg,
            qubit=qubit,
            py_avg=py_avg,
            user=user,
            priority=priority,
        )

        if wait:
            # Wait for completion and load result
            client.wait_for_completion(job_id)
            result = client.get_result(job_id)

            # Populate this instance with worker results
            if result._expt is not None:
                self.iqdata = result.iqdata
                self.fit_params = result.fit_params
                
                # Check if it was saved as dict (new format) or object (old format)
                if isinstance(result._expt, dict):
                    self._sweep_vals_x = result._expt.get("_sweep_vals_x", result._expt.get("_sweep_vals"))
                    self._sweep_vals_y = result._expt.get("_sweep_vals_y")
                else:
                    self._sweep_vals_x = getattr(result._expt, "_sweep_vals_x", getattr(result._expt, "_sweep_vals", None))
                    self._sweep_vals_y = getattr(result._expt, "_sweep_vals_y", None)

            print(f"[{self.__class__.__name__}] Job {job_id} complete. Data loaded.")

            if analyze:
                # Run post‑fit on the loaded sweep values and store the result
                fit_result = self._post_fit(self._sweep_vals_x)
                self.fit_params = fit_result
                return fit_result
            else:
                return self.iqdata
        else:
            print(f"[{self.__class__.__name__}] Job {job_id} submitted to background.")
            return client.get_handle(job_id, self)

    def _save_comment(self, dict_val):
        """
        Optional hook: customize the comment string for saveLabber.
        Default: just the cfg dump.
        """
        return f"{dict_val}"
