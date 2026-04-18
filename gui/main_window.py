from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QStatusBar, QLabel,
    QTabWidget, QWidget, QVBoxLayout,
)
from PySide6.QtCore import Qt

from .panels.setup_panel      import SetupPanel
from .panels.experiment_panel import ExperimentPanel
from .panels.plot_panel       import PlotPanel
from .panels.data_browser     import DataBrowser


class MainWindow(QMainWindow):
    """Main application window with dockable panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("qick_workspace GUI")
        self.resize(1400, 800)
        self._build_layout()
        self._connect_signals()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # ── Central: Live Plot ──────────────────────────────────────────────
        self.plot_panel = PlotPanel()
        self.setCentralWidget(self.plot_panel)

        # ── Left dock: Setup ────────────────────────────────────────────────
        self.setup_panel = SetupPanel()
        left_dock = QDockWidget("Setup", self)
        left_dock.setWidget(self.setup_panel)
        left_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # ── Right dock: Experiment ──────────────────────────────────────────
        self.expt_panel = ExperimentPanel()
        right_dock = QDockWidget("Experiment", self)
        right_dock.setWidget(self.expt_panel)
        right_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # ── Bottom dock: Result + Log tabs ─────────────────────────────────
        from PySide6.QtWidgets import QTextEdit
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)

        bottom_tabs = QTabWidget()
        bottom_tabs.addTab(self._log, "Log")
        bottom_dock = QDockWidget("Output", self)
        bottom_dock.setWidget(bottom_tabs)
        bottom_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom_dock)

        # ── Floating Data Browser ───────────────────────────────────────────
        self.data_browser = DataBrowser()
        self._browser_dock = QDockWidget("Data Browser", self)
        self._browser_dock.setWidget(self.data_browser)
        self._browser_dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self._browser_dock)
        self._browser_dock.hide()

        # ── Menu bar ───────────────────────────────────────────────────────
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Quit", self.close)

        tools_menu = menu.addMenu("Tools")
        tools_menu.addAction("Data Browser", self._show_browser)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_conn  = QLabel("✗ Disconnected")
        self._status_qubit = QLabel("Q0")
        self._status_last  = QLabel("Ready")
        bar = QStatusBar()
        bar.addWidget(self._status_conn)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_qubit)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_last)
        self.setStatusBar(bar)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.setup_panel.connection_changed.connect(self._on_connection)
        self.setup_panel.qubit_changed.connect(self._on_qubit)
        self.setup_panel.config_loaded.connect(self._on_config_loaded)
        self.expt_panel.run_requested.connect(self._on_run)
        self.expt_panel.stop_requested.connect(self._on_stop)
        self.expt_panel.save_requested.connect(self._on_save)
        self.data_browser.file_selected.connect(
            lambda p: self.log(f"Loaded: {p}")
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_connection(self, connected):
        if connected:
            self._status_conn.setText("● Connected")
            self._status_conn.setStyleSheet("color: green;")
            self.log("Connected to SOC")
        else:
            self._status_conn.setText("✗ Disconnected")
            self._status_conn.setStyleSheet("color: red;")
            self.log("Disconnected")

    def _on_qubit(self, qidx):
        self._status_qubit.setText(f"Q{qidx}")
        self.log(f"Active qubit → Q{qidx}")

    def _on_config_loaded(self, path):
        self.log(f"Config loaded: {path}")
        # Sync data browser folder to setup panel data path
        data_dir = self.setup_panel.data_folder
        if data_dir:
            self.data_browser.set_folder(data_dir)

    def _on_run(self, class_path, params):
        self.log(f"Run: {class_path.split('.')[-1]}  params={params}")
        self._status_last.setText(f"Running {class_path.split('.')[-1]}…")
        # TODO: instantiate experiment via AcquireWorker(QThread)
        # worker = AcquireWorker(class_path, params)
        # worker.data_ready.connect(self.plot_panel.update_data)
        # worker.finished.connect(lambda: self._status_last.setText("Done"))
        # worker.start()

    def _on_stop(self):
        self.log("Stop requested")
        self._status_last.setText("Stopped")

    def _on_save(self):
        self.log("Save requested")
        # TODO: call save_data(expt, qb_idx) here

    def _show_browser(self):
        self._browser_dock.show()
        self._browser_dock.raise_()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")
