import socket

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal, QThread

# Ports tried in order — first one that accepts a TCP connection wins
_PROBE_PORTS = [8888, 22, 80]
_TIMEOUT_S   = 3


class _ConnectWorker(QThread):
    """Try opening a TCP socket to the host; emit success or failure."""

    succeeded = Signal()
    failed    = Signal(str)

    def __init__(self, host: str, parent=None):
        super().__init__(parent)
        self.host = host

    def run(self):
        for port in _PROBE_PORTS:
            try:
                s = socket.create_connection((self.host, port),
                                             timeout=_TIMEOUT_S)
                s.close()
                self.succeeded.emit()
                return
            except OSError:
                continue
        self.failed.emit(
            f"Could not reach {self.host} on ports {_PROBE_PORTS}.\n"
            "Check the IP address and that the board is powered on."
        )


class SetupPanel(QWidget):
    """[1] Connection, config file, and qubit selector."""

    connection_changed = Signal(bool)
    qubit_changed      = Signal(int)
    config_loaded      = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
        self._worker: _ConnectWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)
        root.addWidget(self._connection_group())
        root.addWidget(self._config_group())
        root.addWidget(self._qubit_group())
        root.addStretch()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connection_group(self):
        grp = QGroupBox("Connection")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("SOC IP:"))
        self.ip_edit = QLineEdit("192.168.1.1")
        self.ip_edit.setToolTip("IP address of the QICK SoC board")
        row.addWidget(self.ip_edit)
        lay.addLayout(row)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(False)          # managed manually
        self.connect_btn.setToolTip("Connect / disconnect from the SoC")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        lay.addWidget(self.connect_btn)

        led_row = QHBoxLayout()
        self._led = QLabel("●")
        self._led.setStyleSheet("color: #f0883e; font-size: 16px;")
        self._status_lbl = QLabel("Disconnected")
        led_row.addWidget(self._led)
        led_row.addWidget(self._status_lbl)
        led_row.addStretch()
        lay.addLayout(led_row)
        return grp

    def _on_connect_clicked(self):
        if self._connected:
            # Disconnect
            self._connected = False
            self.connect_btn.setText("Connect")
            self._set_led(False, "Disconnected")
            self.connection_changed.emit(False)
        else:
            # Start probing
            host = self.ip_edit.text().strip()
            if not host:
                QMessageBox.warning(self, "No IP", "Please enter the SOC IP address.")
                return
            self.connect_btn.setEnabled(False)
            self.connect_btn.setText("Connecting…")
            self._set_led(None, f"Probing {host}…")

            self._worker = _ConnectWorker(host, self)
            self._worker.succeeded.connect(self._on_probe_success)
            self._worker.failed.connect(self._on_probe_failure)
            self._worker.start()

    def _on_probe_success(self):
        self._connected = True
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Disconnect")
        self._set_led(True, "Connected")
        self.connection_changed.emit(True)

    def _on_probe_failure(self, msg: str):
        self._connected = False
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self._set_led(False, "Unreachable")
        QMessageBox.warning(self, "Connection failed", msg)
        self.connection_changed.emit(False)

    def _set_led(self, state, text: str):
        """state: True=green, False=orange, None=yellow (probing)."""
        colors = {True: "#7ee787", False: "#f0883e", None: "#f0d050"}
        self._led.setStyleSheet(
            f"color: {colors[state]}; font-size: 16px;")
        self._status_lbl.setText(text)

    # ── Config ────────────────────────────────────────────────────────────────

    def _config_group(self):
        grp = QGroupBox("Config")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.config_path = QLineEdit()
        self.config_path.setPlaceholderText("config.yaml")
        self.config_path.setToolTip("Path to experiment config YAML file")
        row.addWidget(self.config_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_config)
        row.addWidget(browse_btn)
        lay.addLayout(row)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Config")
        load_btn.setToolTip("Load and apply the YAML config")
        load_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("Save Config")
        save_btn.setToolTip("Save current config to YAML")
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Data path:"))
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("D:/data")
        self.data_path.setToolTip("Root folder where HDF5 files are saved")
        row3.addWidget(self.data_path)
        browse2 = QPushButton("…")
        browse2.setFixedWidth(26)
        browse2.clicked.connect(self._browse_data)
        row3.addWidget(browse2)
        lay.addLayout(row3)
        return grp

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Config", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if path:
            self.config_path.setText(path)

    def _browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        if path:
            self.data_path.setText(path)

    def _load_config(self):
        path = self.config_path.text().strip()
        if path:
            self.config_loaded.emit(path)

    # ── Qubit ─────────────────────────────────────────────────────────────────

    def _qubit_group(self):
        grp = QGroupBox("Qubit")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self._qubit_bg = QButtonGroup(self)
        qrow = QHBoxLayout()
        for i in range(4):
            rb = QRadioButton(f"Q{i}")
            rb.setToolTip(f"Set active qubit to Q{i}")
            if i == 0:
                rb.setChecked(True)
            self._qubit_bg.addButton(rb, i)
            qrow.addWidget(rb)
        self._qubit_bg.idToggled.connect(
            lambda qid, checked: self.qubit_changed.emit(qid) if checked else None
        )
        lay.addLayout(qrow)

        row = QHBoxLayout()
        row.addWidget(QLabel("Yoko (mA):"))
        self.yoko_spin = QDoubleSpinBox()
        self.yoko_spin.setRange(-100, 100)
        self.yoko_spin.setDecimals(3)
        self.yoko_spin.setSingleStep(0.1)
        self.yoko_spin.setToolTip("Yokogawa flux bias current in mA")
        row.addWidget(self.yoko_spin)
        lay.addLayout(row)
        return grp

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_qubit(self) -> int:
        return self._qubit_bg.checkedId()

    @property
    def data_folder(self) -> str:
        return self.data_path.text().strip()
