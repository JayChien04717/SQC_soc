import importlib
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QStatusBar, QLabel,
    QTabWidget, QWidget, QVBoxLayout, QTextEdit,
    QProgressBar, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QProcess, QSettings, QThread, Signal, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QColor

from .panels.setup_panel      import SetupPanel
from .panels.experiment_panel import ExperimentPanel
from .panels.plot_panel       import PlotPanel


# ── Sweep config builder ──────────────────────────────────────────────────────

def _build_run_cfg(class_path: str, qubit_cfg: dict, params: dict) -> dict:
    """
    Merge qubit hardware config with GUI form params.

    Simple overrides (reps, relax_delay) are applied first.  Per-experiment
    sweep axes are converted to QickSweep1D objects as the notebook does.
    """
    from qick.asm_v2 import QickSweep1D

    cfg = dict(qubit_cfg)
    for key in ("reps", "relax_delay"):
        if key in params:
            cfg[key] = params[key]

    cp = class_path

    if cp == "s001_time_of_flight.TOF":
        cfg["reps"] = 1  # TOF always acquires reps=1; averaging done in software
        for k in ("ro_length", "res_length", "res_gain_ge", "res_ch"):
            if k in params:
                cfg[k] = params[k]

    elif cp == "s002_res_spec_ge.ResonatorSpec":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({"steps": params["steps"], "res_gain_ge": params["res_gain_ge"],
                    "res_freq_ge": QickSweep1D("freqloop", center - span, center + span)})

    elif cp == "s002b_res_punchout_ge.Punchout":
        fc, fs = params["freq_center"], params["freq_span"]
        cfg.update({"f_steps": params["freq_steps"], "g_steps": params["gain_steps"],
                    "res_freq_ge": QickSweep1D("freqloop", fc - fs, fc + fs),
                    "res_gain_ge": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"])})

    elif cp == "s002c_res_spec_ge_flux.ResonatorSpecFlux":
        fc, fs = params["freq_center"], params["freq_span"]
        cfg.update({"res_gain_ge": params["res_gain_ge"],
                    "freq_steps": params["freq_steps"],
                    "flux_ch": params["flux_ch"], "flux_length": params["flux_length"],
                    "res_freq_ge": QickSweep1D("freqloop", fc - fs, fc + fs),
                    "flux_gain": QickSweep1D("fluxloop", params["flux_gain_start"], params["flux_gain_stop"]),
                    "steps_flux": params["flux_steps"]})

    elif cp == "s003_qubit_spec_ge.QubitSpec":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({"steps": params["steps"], "qb_gain_ge": params["qb_gain_ge"],
                    "qb_flat_top_length_ge": params["qb_flat_top_length_ge"],
                    "nqz_qb": params["nqz_qb"], "qb_mixer": center,
                    "qb_freq_ge": QickSweep1D("freqloop", center - span, center + span)})

    elif cp == "s003a_qubit_flux_spec_ge.QubitSpecFlux":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({"freq_steps": params["freq_steps"], "qb_gain_ge": params["qb_gain_ge"],
                    "qb_flat_top_length_ge": params["qb_flat_top_length_ge"],
                    "flux_ch": params["flux_ch"], "flux_length": params["flux_length"],
                    "qb_mixer": center,
                    "qb_freq_ge": QickSweep1D("freqloop", center - span, center + span),
                    "flux_gain": QickSweep1D("fluxloop", params["flux_gain_start"], params["flux_gain_stop"]),
                    "steps_flux": params["flux_steps"]})

    elif cp == "s004_time_rabi_ge.TimeRabi":
        cfg.update({"sigma_ge": params["sigma_ge"], "qb_gain_ge": params["qb_gain_ge"],
                    "steps": params["steps"],
                    "qb_flat_top_length_ge": QickSweep1D("lenloop", params["length_start"], params["length_stop"])})

    elif cp == "s005_power_rabi_ge.PowerRabi":
        cfg.update({"sigma_ge": params["sigma_ge"], "nqz_qb": params["nqz_qb"],
                    "steps": params["steps"],
                    "qb_gain_ge": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"])})

    elif cp == "s005a_drag.DragCalibration":
        cfg.update({k: params[k] for k in
                    ("alpha_start", "alpha_stop", "alpha_steps",
                     "iter_start", "iter_stop", "iter_step")})

    elif cp == "s005a_AAE.PowerRabiChevron":
        focus = cfg.get("pi_gain_ge", 0.5)
        half  = params["gain_half_span"]
        cfg.update({"steps": params["steps"],
                    "iter_start": params["iter_start"], "iter_stop": params["iter_stop"],
                    "iter_step": params["iter_step"],
                    "qb_gain_ge": QickSweep1D("gainloop", focus - half, focus + half)})

    elif cp in ("s006_Ramsey_ge.Ramsey", "s012_Ramsey_ef.Ramsey_ef"):
        cfg.update({"steps": params["steps"], "ramsey_freq": params["ramsey_freq"],
                    "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"])})
        if "ge_ref" in params:
            cfg["ge_ref"] = params["ge_ref"]

    elif cp == "s007_SpinEcho_ge.SpinEcho":
        cfg.update({"steps": params["steps"], "ramsey_freq": params["ramsey_freq"],
                    "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"])})

    elif cp in ("s008_T1_ge.T1", "s013_T1_ef.T1_ef"):
        cfg.update({"steps": params["steps"],
                    "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"])})

    elif cp == "s009_res_spec_ef.ResonatorSpec_ef":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({"res_gain_ge": params["res_gain_ge"], "steps": params["steps"],
                    "res_freq_ge": QickSweep1D("freqloop", center - span, center + span)})

    elif cp == "s010_qubit_spec_ef.QubitSpec_ef":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({"steps": params["steps"], "qb_gain_ef": params["qb_gain_ef"],
                    "qb_flat_top_length_ef": params["qb_flat_top_length_ef"],
                    "ge_ref": params["ge_ref"],
                    "qb_freq_ef": QickSweep1D("freqloop", center - span, center + span)})

    elif cp == "s011_power_rabi_ef.PowerRabi_ef":
        cfg.update({"sigma_ef": params["sigma_ef"], "steps": params["steps"],
                    "ge_ref": params["ge_ref"],
                    "qb_gain_ef": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"])})

    elif cp == "s013_qubit_temp.QubitTemperatureEf":
        cfg.update({"sigma_ef": params["sigma_ef"], "steps": params["steps"],
                    "ge_ref": params["ge_ref"],
                    "qb_gain_ef": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"])})

    elif cp == "s014_AllXY.AllXY":
        cfg["pulse_type"] = params["pulse_type"]

    elif cp == "s000_SingleShot_prog.SingleShot_gef":
        cfg["shot_f"] = params["shot_f"]

    elif cp in ("s015_Single_qubit_RB.RandomizedBenchmarking",
                "s015_Auto_RB.AutoRB", "s015_RB_asm.RandomizedBenchmarkingAsm"):
        cfg.update({k: params[k] for k in
                    ("max_circuit_depth", "delta_clifford", "number_sample")
                    if k in params})
        if "pulse_type" in params:
            cfg["pulse_type"] = params["pulse_type"]

    elif cp == "s016_state_tomography.Tomography":
        cfg["prep_pulse_name"] = params["prep_pulse_name"]

    return cfg


# ── Background worker ─────────────────────────────────────────────────────────

class AcquireWorker(QThread):
    """
    Run an experiment off the GUI thread.

    Parameters
    ----------
    class_path : str
        ``"module.ClassName"`` from EXPERIMENT_REGISTRY.
    params : dict
        Parameter dict from the experiment form.
    qubit_cfg : dict
        Flat hardware config from ExperimentConfig.get_qubit().
    debug : bool
        When ``True`` use synthetic data; no hardware calls are made.
    parent : QObject, optional
        Qt parent object.
    """

    data_ready  = Signal(object, object, str, str)  # x, iq, xlabel, title
    log_message = Signal(str, str)                   # msg, level
    finished    = Signal()

    def __init__(self, class_path: str, params: dict, qubit_cfg: dict,
                 debug: bool = False, parent=None):
        super().__init__(parent)
        self.class_path = class_path
        self.params     = params
        self.qubit_cfg  = qubit_cfg
        self.debug      = debug

    # Experiments that bypass the standard sw_avg loop (custom run() signatures)
    _SPECIAL_EXPTS = frozenset({
        "s000_SingleShot_prog.SingleShot_gef",
        "s000_SingleShot_opt.SingleShot_ge_opt",
        "s015_Single_qubit_RB.RandomizedBenchmarking",
        "s015_Auto_RB.AutoRB",
        "s015_RB_asm.RandomizedBenchmarkingAsm",
    })

    # Experiments that use acquire_decimated (time-domain waveform, not sweep IQ)
    _DECIMATED_EXPTS = frozenset({
        "s001_time_of_flight.TOF",
    })

    def run(self):
        name = self.class_path.split(".")[-1]
        self.log_message.emit(f"Starting {name}…", "info")
        try:
            if self.debug:
                from gui.mock.runner import generate_mock_data
                x, iq, xlabel, title = generate_mock_data(self.class_path, self.params)
                self.data_ready.emit(x, iq, xlabel, title)
            elif self.class_path in self._SPECIAL_EXPTS:
                self._run_special()
            else:
                self._run_liveplot()
            self.log_message.emit("Experiment complete.", "success")
        except Exception as exc:
            self.log_message.emit(f"Error: {exc}", "error")
            self.log_message.emit(traceback.format_exc(), "error")
        finally:
            self.finished.emit()

    def _make_expt(self):
        """Instantiate the experiment class with the merged run_cfg."""
        run_cfg = _build_run_cfg(self.class_path, self.qubit_cfg, self.params)
        mod_name, cls_name = self.class_path.rsplit(".", 1)
        mod  = importlib.import_module(f"qick_workspace.scrip.{mod_name}")
        return getattr(mod, cls_name)(run_cfg)

    def _run_liveplot(self):
        """
        Standard sw_avg loop with per-step Qt signal updates (liveplot).

        Emits data_ready after every report_every averages so PlotPanel
        refreshes in real time.  The IPython-based liveplotfun is bypassed.
        """
        import numpy as np
        expt   = self._make_expt()
        py_avg = self.params.get("py_avg", self.params.get("shots", 10))

        prog   = expt._create_program()
        x_vals = expt._extract_sweep_axis(prog)
        expt._sweep_vals_x = x_vals
        expt._sweep_vals_y = expt._extract_sweep_axis_y(prog)
        soc    = expt.soc

        is_decimated = self.class_path in self._DECIMATED_EXPTS

        # Throttle: at most ~60 plot refreshes regardless of py_avg
        report_every = max(1, py_avg // 60)

        iq_sum  = 0
        iqdata  = None
        for i in range(py_avg):
            if is_decimated:
                # TOF: acquire_decimated returns time-domain waveform
                # iq_list[0] has shape (ro_length_samples, 2); combine I+jQ
                iq_list = prog.acquire_decimated(soc, rounds=1, progress=False)
                iq_data = iq_list[0].dot([1, 1j])
            else:
                # Sweep experiments: iq_list[0][0] has shape (sweep_steps, 2)
                iq_list = prog.acquire(soc, rounds=1, progress=False)
                iq_data = iq_list[0][0].dot([1, 1j])
            iq_sum  = iq_data if i == 0 else iq_sum + iq_data
            iqdata  = iq_sum / (i + 1)

            if (i + 1) % report_every == 0 or i == py_avg - 1:
                self.data_ready.emit(
                    x_vals, iqdata,
                    expt.X_LABEL,
                    f"{expt.TITLE_PREFIX}  [{i+1}/{py_avg}]",
                )

        expt.iqdata = iqdata
        expt._post_fit(x_vals)

    def _run_special(self):
        """Run experiments with non-standard run() signatures (no per-step liveplot)."""
        expt   = self._make_expt()
        py_avg = self.params.get("py_avg", self.params.get("shots", 10))
        cp     = self.class_path

        if cp == "s000_SingleShot_prog.SingleShot_gef":
            expt.run(py_avg, shot_f=self.params.get("shot_f", False))
        elif cp in ("s015_Single_qubit_RB.RandomizedBenchmarking",
                    "s015_RB_asm.RandomizedBenchmarkingAsm"):
            expt.run(py_avg,
                     max_circuit_depth=self.params.get("max_circuit_depth", 400),
                     delta_clifford=self.params.get("delta_clifford", 40),
                     number_sample=self.params.get("number_sample", 30))
        elif cp == "s015_Auto_RB.AutoRB":
            expt.run(py_avg,
                     max_circuit_depth=self.params.get("max_circuit_depth", 600),
                     delta_clifford=self.params.get("delta_clifford", 50),
                     number_sample=self.params.get("number_sample", 50))
        else:
            expt.run(py_avg)

        if expt.iqdata is not None and hasattr(expt, "_sweep_vals_x"):
            self.data_ready.emit(
                expt._sweep_vals_x, expt.iqdata,
                expt.X_LABEL, expt.TITLE_PREFIX,
            )


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window with dockable panels."""

    def __init__(self, debug: bool = False):
        super().__init__()
        self._debug = debug
        title = "qick_workspace  ·  Control Panel"
        if debug:
            title += "  [SIMULATION]"
        self.setWindowTitle(title)
        self.resize(1440, 860)
        self._worker: AcquireWorker | None = None
        self._config: dict = {}
        self._config_list: list = []
        self._browser_procs: list[QProcess] = []
        self._build_layout()
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_geometry()
        if debug:
            self._lbl_debug.setVisible(True)
            self.log("Simulation mode active — no hardware required.", "warn")
            self._auto_connect_mock()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Central: live plot
        self.plot_panel = PlotPanel()
        self.setCentralWidget(self.plot_panel)

        # Left dock: Setup
        self.setup_panel = SetupPanel()
        self._add_dock("Setup", self.setup_panel, Qt.LeftDockWidgetArea,
                       movable=True, floatable=True, closable=False, width=260)

        # Right dock: Experiment
        self.expt_panel = ExperimentPanel()
        self._add_dock("Experiment", self.expt_panel, Qt.RightDockWidgetArea,
                       movable=True, floatable=True, closable=False, width=280)

        # Bottom dock: Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        self._log.document().setMaximumBlockCount(500)
        self._add_dock("Log", self._log, Qt.BottomDockWidgetArea,
                       movable=True, floatable=True, closable=False)

        # ── Menu bar ──────────────────────────────────────────────────────────
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        fm.addAction("&Save Plot…",   self.plot_panel.export_figure,
                     QKeySequence("Ctrl+S"))
        fm.addSeparator()
        fm.addAction("&Quit",         self.close, QKeySequence("Ctrl+Q"))

        tm = mb.addMenu("&Tools")
        tm.addAction("&Data Browser", self._launch_browser,
                     QKeySequence("Ctrl+B"))
        tm.addAction("Clear &Log",    self._log.clear)

        hm = mb.addMenu("&Help")
        hm.addAction("&About",        self._show_about)

        # ── Status bar ────────────────────────────────────────────────────────
        self._led       = QLabel("●")
        self._led.setStyleSheet(f"color: #f0883e; font-size: 14px;")
        self._lbl_conn  = QLabel("Disconnected")
        self._lbl_qubit = QLabel("Q0")
        self._lbl_state = QLabel("Ready")
        self._lbl_debug = QLabel("⬡ SIM")
        self._lbl_debug.setStyleSheet("color: #f0d050; font-weight: bold; font-size: 11px;")
        self._lbl_debug.setVisible(False)
        self._progress  = QProgressBar()
        self._progress.setRange(0, 0)           # indeterminate
        self._progress.setFixedWidth(140)
        self._progress.setFixedHeight(14)
        self._progress.setVisible(False)

        sb = QStatusBar()
        sb.addWidget(self._led)
        sb.addWidget(self._lbl_conn)
        sb.addWidget(_sep())
        sb.addWidget(self._lbl_qubit)
        sb.addWidget(_sep())
        sb.addWidget(self._lbl_state)
        sb.addWidget(_sep())
        sb.addWidget(self._lbl_debug)
        sb.addPermanentWidget(self._progress)
        self.setStatusBar(sb)

    def _add_dock(self, title, widget, area, *,
                  movable=True, floatable=True, closable=True, width=None):
        dock = QDockWidget(title, self)
        feat = QDockWidget.NoDockWidgetFeatures
        if movable:   feat |= QDockWidget.DockWidgetMovable
        if floatable: feat |= QDockWidget.DockWidgetFloatable
        if closable:  feat |= QDockWidget.DockWidgetClosable
        dock.setFeatures(feat)
        dock.setWidget(widget)
        if width:
            dock.setMinimumWidth(width)
        self.addDockWidget(area, dock)
        return dock

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self.expt_panel.trigger_run)
        QShortcut(QKeySequence("Escape"), self, self.expt_panel.trigger_stop)
        QShortcut(QKeySequence("Ctrl+B"), self, self._launch_browser)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.setup_panel.connection_changed.connect(self._on_connection)
        self.setup_panel.soc_ready.connect(self._on_soc_ready)
        self.setup_panel.qubit_changed.connect(self._on_qubit)
        self.setup_panel.config_loaded.connect(self._on_config)
        self.expt_panel.run_requested.connect(self._on_run)
        self.expt_panel.stop_requested.connect(self._on_stop)
        self.expt_panel.save_requested.connect(self._on_save)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_connection(self, connected: bool):
        if connected:
            self._led.setStyleSheet("color: #7ee787; font-size: 14px;")
            self._lbl_conn.setText("Connected")
            self.log("Connected to SOC.", "success")
        else:
            self._led.setStyleSheet("color: #f0883e; font-size: 14px;")
            self._lbl_conn.setText("Disconnected")
            self.log("Disconnected.", "warn")

    def _on_soc_ready(self, soc, soccfg):
        """Call BaseExperiment.setup() once soc/soccfg are available."""
        # Log board info (same as print(soccfg) in the notebook)
        board_info = str(soccfg).strip()
        for line in board_info.splitlines()[:6]:   # first 6 lines is enough
            self.log(line, "info")

        data_path = self.setup_panel.data_folder or "data"
        try:
            from qick_workspace.scrip.base_experiment import BaseExperiment
            BaseExperiment.setup(soc, soccfg, data_path)
            self.log(f"BaseExperiment ready  —  data path: {data_path}", "success")
        except Exception as exc:
            self.log(f"BaseExperiment.setup() failed: {exc}", "error")

    def _on_qubit(self, qidx: int):
        name = self.setup_panel.active_qubit_name
        self._lbl_qubit.setText(name)
        self.log(f"Active qubit → {name}", "info")

    def _on_config(self, path: str):
        try:
            cfg, qubit_names, config_list = self._parse_config(path)
        except Exception as exc:
            self.log(f"Config load failed: {exc}", "error")
            return
        self._config = cfg
        self._config_list = config_list
        if qubit_names:
            self.setup_panel.update_qubits(qubit_names)
            self.log(f"Qubits detected: {', '.join(qubit_names)}", "info")
            name = self.setup_panel.active_qubit_name
            self._lbl_qubit.setText(name)
        keys = list(cfg.keys()) if cfg else []
        suffix = (f"  ({len(keys)} keys)" if keys
                  else f"  ({len(qubit_names)} qubits)" if qubit_names
                  else "")
        self.log(f"Config loaded: {path}{suffix}", "info")

    def _parse_config(self, path: str) -> tuple:
        """Return (cfg_dict, qubit_names, config_list)."""
        p = Path(path)
        qubit_names: list = []
        config_list: list = []
        if p.suffix == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("_gui_cfg", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg_list = getattr(mod, "config_list", None)
            if isinstance(cfg_list, list):
                config_list = cfg_list
                qubit_names = [
                    item.get("name", f"Q{i}")
                    for i, item in enumerate(cfg_list)
                    if isinstance(item, dict)
                ]
            for name in ("config", "cfg", "hw_cfg", "expt_cfg"):
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    if isinstance(val, dict):
                        return val, qubit_names, config_list
            return ({k: v for k, v in vars(mod).items() if not k.startswith("_")},
                    qubit_names, config_list)
        else:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, list):
                config_list = data
                qubit_names = [
                    item.get("name", f"Q{i}")
                    for i, item in enumerate(data)
                    if isinstance(item, dict)
                ]
                return {}, qubit_names, config_list
            return data, qubit_names, config_list

    def _auto_connect_mock(self):
        """Fire a simulated connection immediately on startup in debug mode."""
        from gui.mock.hardware import MockSoc, MockSoccfg
        QTimer.singleShot(300, lambda: (
            self.setup_panel._on_success(MockSoc(), MockSoccfg()),
        ))

    def _on_run(self, class_path: str, params: dict):
        if self._worker and self._worker.isRunning():
            self.log("Already running — stop first.", "warn")
            return

        qubit_cfg: dict = {}
        if not self._debug and self._config_list:
            try:
                from qick_workspace.tools.system_tool import ExperimentConfig
                config_all = ExperimentConfig(self._config_list)
                active_qubit = self.setup_panel.active_qubit_name
                qubit_cfg = dict(config_all.get_qubit(active_qubit))
            except Exception as exc:
                self.log(f"Could not build qubit config: {exc}", "warn")
        elif not self._debug and not self._config_list:
            self.log("No config loaded — Browse and Load a config file first.", "warn")

        self._worker = AcquireWorker(class_path, params, qubit_cfg, self._debug, self)
        self._worker.log_message.connect(self.log)
        self._worker.data_ready.connect(self.plot_panel.update_data)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        name = class_path.split(".")[-1]
        self._lbl_state.setText(f"Running  {name}…")
        self._progress.setVisible(True)
        self.expt_panel.set_running(True)
        self.log(f"Run → {name}  {params}", "info")

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._on_worker_done()
        self.log("Stopped.", "warn")

    def _on_worker_done(self):
        self._progress.setVisible(False)
        self._lbl_state.setText("Ready")
        self.expt_panel.set_running(False)

    def _on_save(self):
        self.log("Save requested (wire to save_data()).", "info")

    # ── Data Browser ──────────────────────────────────────────────────────────

    def _launch_browser(self):
        script = str(Path(__file__).parent / "data_browser_app.py")
        proc = QProcess(self)
        proc.start(sys.executable, [script])
        self._browser_procs.append(proc)
        self.log("Data Browser opened.", "info")

    # ── About ─────────────────────────────────────────────────────────────────

    def _show_about(self):
        QMessageBox.about(
            self, "About qick_workspace GUI",
            "<h3>qick_workspace GUI</h3>"
            "<p>Superconducting qubit control interface<br>"
            "built on <b>PySide6</b> + <b>matplotlib</b> + <b>QICK</b>.</p>"
            "<p style='color:#8b949e;font-size:11px;'>"
            "Use <b>Ctrl+R</b> to run · <b>Ctrl+B</b> for Data Browser · "
            "<b>Esc</b> to stop</p>",
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str, level: str = "info"):
        color = {
            "info":    "#8b949e",
            "success": "#7ee787",
            "warn":    "#f0883e",
            "error":   "#ff7b72",
        }.get(level, "#8b949e")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(
            f'<span style="color:{color};">'
            f'[{ts}] {msg}'
            f'</span>'
        )

    # ── Geometry persistence ──────────────────────────────────────────────────

    def _restore_geometry(self):
        s = QSettings("qick_workspace", "GUI")
        geom = s.value("geometry")
        state = s.value("windowState")
        if geom:
            self.restoreGeometry(geom)
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        s = QSettings("qick_workspace", "GUI")
        s.setValue("geometry",    self.saveGeometry())
        s.setValue("windowState", self.saveState())
        for proc in self._browser_procs:
            proc.kill()
        super().closeEvent(event)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep():
    lbl = QLabel(" | ")
    lbl.setStyleSheet("color: #30363d;")
    return lbl
