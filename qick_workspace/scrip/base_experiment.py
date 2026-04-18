"""
BaseExperiment: Base class for all experiment wrappers.
========================================================
Unifies the run (liveplot-first) and saveLabber logic.
Subclasses only need to override a few methods + declare metadata.
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

    Provides a unified ``run`` → liveplot → ``_post_fit`` → ``saveLabber``
    pipeline.  Subclasses declare class-level metadata and override a small
    set of hook methods.

    Session initialisation (call once per notebook)::

        BaseExperiment.setup(soc, soccfg, data_path)

    Subclass contract
    -----------------
    1. Set class-level metadata attributes (``EXPT_NAME``, ``TAG``,
       ``X_LABEL``, etc.).
    2. Override :meth:`_create_program` — return the ``Program`` instance.
    3. Override :meth:`_extract_sweep_axis` — return the x-axis array.
    4. Optionally override :meth:`_post_fit` — perform fitting after liveplot.

    Examples
    --------
    >>> class QubitSpec(BaseExperiment):
    ...     EXPT_NAME = "s003_qubit_spec_ge"
    ...     TAG = "TwoTone"
    ...     X_LABEL = "Frequency (MHz)"
    ...     TITLE_PREFIX = "Qubit ge Spectrum"
    ...     SWEEP_KEYS_TO_REMOVE = ["qb_freq_ge"]
    """

    # ── Session state — set once per notebook via BaseExperiment.setup() ──
    _soc = None
    _soccfg = None
    _data_path = None   # overrides DATA_PATH from system_cfg when set

    @classmethod
    def setup(cls, soc, soccfg, data_path):
        """
        Initialise the shared QICK session (call once at notebook startup).

        Parameters
        ----------
        soc : object
            QICK SoC proxy object.
        soccfg : object
            QICK SoC configuration object.
        data_path : str
            Root directory for experiment data storage.
        """
        cls._soc = soc
        cls._soccfg = soccfg
        cls._data_path = data_path

    @classmethod
    def set_data_path(cls, data_path):
        """
        Change the save directory at runtime without re-connecting.

        Parameters
        ----------
        data_path : str
            New root directory for experiment data storage.
        """
        cls._data_path = data_path

    # ── Subclass must set these metadata ──
    EXPT_NAME: str = ""
    TAG: str = ""
    X_LABEL: str = ""
    Y_LABEL: str = "ADC Units"
    TITLE_PREFIX: str = ""
    SWEEP_KEYS_TO_REMOVE: list = []

    # ── IQ processing mode (class-level default, overridable in run()) ──
    # "abs"  → np.abs(iqdata)  — default, works without readout optimization
    # "real" → np.real(iqdata) — use after readout optimization (best SNR on I axis)
    IQ_PROCESS: str = "abs"

    # ── x-axis save info (for saveLabber) ──
    X_SAVE_NAME: str = ""  # e.g. "Frequency", "Times", "Gain"
    X_SAVE_UNIT: str = ""  # e.g. "Hz", "us", "DAC unit"
    X_SAVE_SCALE: float = (
        1.0  # multiply factor for unit conversion (e.g. 1e6 for MHz→Hz)
    )

    # ── y-axis save info (optional, for 2D experiments) ──
    Y_SAVE_NAME: str = ""
    Y_SAVE_UNIT: str = ""
    Y_SAVE_SCALE: float = 1.0

    def __init__(self, config):
        if BaseExperiment._soc is None:
            raise RuntimeError(
                "QICK session not initialised. "
                "Call BaseExperiment.setup(soc, soccfg, data_path) first."
            )
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config
        self.iqdata = None
        self.fit_params = None
        self._sweep_vals = None  # sweep axis values (x)
        self._sweep_vals_y = None  # sweep axis values (y, for 2D)

    # ══════════════════════════════════════════════
    # Unified entry point
    # ══════════════════════════════════════════════

    def run(self, py_avg, iq_process=None, show_final_plot=False, **kwargs):
        """
        Execute the experiment with live plotting and optional post-fit.

        Compiles the program via :meth:`_create_program`, runs
        :func:`~plotter.liveplot.liveplotfun` for live data display, and
        calls :meth:`_post_fit` once acquisition completes.

        Parameters
        ----------
        py_avg : int
            Number of software averages.
        iq_process : str, optional
            ``"abs"`` or ``"real"`` — overrides the class-level
            :attr:`IQ_PROCESS` for this run only.  Use ``"real"`` after
            readout optimisation for best SNR on the I axis.
        show_final_plot : bool, optional
            Whether to display a final static plot after acquisition.
            Default is ``False``.
        **kwargs
            Extra keyword arguments forwarded to :func:`liveplotfun`
            (e.g. ``yoko_inst_addr``, ``yoko_mode``).

        Returns
        -------
        fit_result : any
            Return value of :meth:`_post_fit`, or ``None`` if no data was
            acquired or if the subclass does not override ``_post_fit``.
        """
        if iq_process is not None:
            self.IQ_PROCESS = iq_process

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
            show_final_plot=show_final_plot,
            iq_process=self.IQ_PROCESS,
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

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """
        Save acquired data to an HDF5 file in Labber format.

        Parameters
        ----------
        qb_idx : int or str
            Qubit index or name used for the filename and config extraction.
        yoko_value : dict, optional
            Yokogawa output value dict (``{"value": ..., "unit": ...}``) for
            encoding the flux bias in the filename.
        config_all : ExperimentConfig, optional
            When provided, YAML is generated via
            ``config_all.to_yaml(q_id=qb_idx)`` (nested, full config).
            When ``None``, :func:`~tools.system_tool.config_to_yaml` is used
            on the flat ``self.cfg`` dict.
        title : str, optional
            Optional suffix appended to the experiment name in the filename.
        """
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
        """
        Create and return the QICK program instance for this experiment.

        Returns
        -------
        prog : AveragerProgramV2
            Compiled QICK program ready for acquisition.

        Raises
        ------
        NotImplementedError
            Always raised if the subclass does not override this method.
        """
        raise NotImplementedError("Subclass must implement _create_program()")

    def _extract_sweep_axis(self, prog):
        """
        Extract and return the x-axis sweep values from the compiled program.

        Parameters
        ----------
        prog : AveragerProgramV2
            Compiled QICK program.

        Returns
        -------
        x_vals : np.ndarray
            Sweep axis values for the x (fast) axis.

        Raises
        ------
        NotImplementedError
            Always raised if the subclass does not override this method.
        """
        raise NotImplementedError("Subclass must implement _extract_sweep_axis()")

    def _extract_sweep_axis_y(self, prog):
        """
        Extract and return the y-axis sweep values for 2-D experiments.

        Parameters
        ----------
        prog : AveragerProgramV2
            Compiled QICK program.

        Returns
        -------
        y_vals : np.ndarray or None
            Sweep axis values for the y (slow) axis, or ``None`` for 1-D
            experiments (default).
        """
        return None

    # ══════════════════════════════════════════════
    # Subclass MAY override (hooks)
    # ══════════════════════════════════════════════

    def _post_fit(self, x_vals):
        """
        Optional hook: perform fitting after liveplot and return the result.

        Parameters
        ----------
        x_vals : np.ndarray
            Sweep axis values (same as returned by :meth:`_extract_sweep_axis`).

        Returns
        -------
        fit_result : any
            Fitting result (subclass-defined), or ``None`` (default).
        """
        return None

    def _save_comment(self, dict_val):
        """
        Optional hook: build the comment string written to the HDF5 file.

        Parameters
        ----------
        dict_val : str
            YAML string of the experiment configuration.

        Returns
        -------
        comment : str
            Comment string to embed in the log file.
        """
        return f"{dict_val}"
