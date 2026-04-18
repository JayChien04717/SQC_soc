import sys
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


# ── Background worker (stub — wire to real experiment later) ──────────────────

class AcquireWorker(QThread):
    """Run an experiment off the GUI thread."""

    data_ready  = Signal(object, object, str, str)  # x, iq, xlabel, title
    log_message = Signal(str, str)                   # msg, level
    finished    = Signal()

    def __init__(self, class_path: str, params: dict, parent=None):
        super().__init__(parent)
        self.class_path = class_path
        self.params     = params

    def run(self):
        self.log_message.emit(f"Starting {self.class_path.split('.')[-1]}…", "info")
        try:
            # TODO: import and instantiate the actual experiment class, call .run()
            # import importlib
            # mod_name, cls_name = self.class_path.rsplit(".", 1)
            # mod = importlib.import_module(f"qick_workspace.scrip.{mod_name}")
            # expt = getattr(mod, cls_name)(params=self.params)
            # expt.run()
            # self.data_ready.emit(expt._sweep_vals_x, expt.iqdata, "x", cls_name)
            self.log_message.emit("Experiment complete.", "success")
        except Exception as exc:
            self.log_message.emit(f"Error: {exc}", "error")
        finally:
            self.finished.emit()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window with dockable panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("qick_workspace  ·  Control Panel")
        self.resize(1440, 860)
        self._worker: AcquireWorker | None = None
        self._browser_procs: list[QProcess] = []
        self._build_layout()
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_geometry()

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
        self._lbl_qubit.setText(f"Q{qidx}")
        self.log(f"Active qubit → Q{qidx}", "info")

    def _on_config(self, path: str):
        self.log(f"Config loaded: {path}", "info")

    def _on_run(self, class_path: str, params: dict):
        if self._worker and self._worker.isRunning():
            self.log("Already running — stop first.", "warn")
            return
        self._worker = AcquireWorker(class_path, params, self)
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
