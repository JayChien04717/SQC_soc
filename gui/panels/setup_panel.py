from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QFileDialog,
    QSpinBox, QMessageBox,
)
from PySide6.QtCore import Signal, QThread


class _ConnectWorker(QThread):
    """Call make_proxy in a background thread; emit soc+soccfg or an error."""

    succeeded = Signal(object, object)   # soc, soccfg
    failed    = Signal(str)

    def __init__(self, ns_host: str, ns_port: int, proxy_name: str, parent=None):
        super().__init__(parent)
        self.ns_host    = ns_host
        self.ns_port    = ns_port
        self.proxy_name = proxy_name

    def run(self):
        try:
            import Pyro4
            from qick.pyro import make_proxy
            Pyro4.config.SERIALIZER = "pickle"
            Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
            soc, soccfg = make_proxy(
                ns_host=self.ns_host,
                ns_port=self.ns_port,
                proxy_name=self.proxy_name,
            )
            self.succeeded.emit(soc, soccfg)
        except Exception as exc:
            self.failed.emit(str(exc))


class SetupPanel(QWidget):
    """[1] Pyro connection, config file, and qubit selector."""

    connection_changed = Signal(bool)
    soc_ready          = Signal(object, object)   # soc, soccfg
    qubit_changed      = Signal(int)
    config_loaded      = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
        self._worker: _ConnectWorker | None = None
        self.soc     = None
        self.soccfg  = None
        self._qubit_names: list[str] = [f"Q{i}" for i in range(4)]
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

        # ns_host
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("NS host:"))
        self.ip_edit = QLineEdit("192.168.10.82")
        self.ip_edit.setToolTip("Pyro4 nameserver host (ns_host)")
        row1.addWidget(self.ip_edit)
        lay.addLayout(row1)

        # ns_port + proxy_name on same row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8888)
        self.port_spin.setFixedWidth(70)
        self.port_spin.setToolTip("Pyro4 nameserver port (ns_port)")
        row2.addWidget(self.port_spin)
        row2.addSpacing(8)
        row2.addWidget(QLabel("Name:"))
        self.proxy_edit = QLineEdit("myqick")
        self.proxy_edit.setToolTip("Pyro4 proxy name (proxy_name)")
        row2.addWidget(self.proxy_edit)
        lay.addLayout(row2)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setToolTip("Connect via Pyro4 make_proxy")
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
            self._connected = False
            self.soc = self.soccfg = None
            self.connect_btn.setText("Connect")
            self._set_led(False, "Disconnected")
            self.connection_changed.emit(False)
            return

        ns_host    = self.ip_edit.text().strip()
        ns_port    = self.port_spin.value()
        proxy_name = self.proxy_edit.text().strip()
        if not ns_host or not proxy_name:
            QMessageBox.warning(self, "Missing fields",
                                "Please fill in NS host and proxy name.")
            return

        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting…")
        self._set_led(None, f"Connecting to {ns_host}:{ns_port}…")

        self._worker = _ConnectWorker(ns_host, ns_port, proxy_name, self)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.start()

    def _on_success(self, soc, soccfg):
        self._connected  = True
        self.soc         = soc
        self.soccfg      = soccfg
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Disconnect")
        self._set_led(True, "Connected")
        self.connection_changed.emit(True)
        self.soc_ready.emit(soc, soccfg)

    def _on_failure(self, msg: str):
        self._connected = False
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self._set_led(False, "Failed")
        QMessageBox.critical(self, "Connection failed", msg)
        self.connection_changed.emit(False)

    def _set_led(self, state, text: str):
        colors = {True: "#7ee787", False: "#f0883e", None: "#f0d050"}
        self._led.setStyleSheet(f"color: {colors[state]}; font-size: 16px;")
        self._status_lbl.setText(text)

    # ── Config ────────────────────────────────────────────────────────────────

    def _config_group(self):
        grp = QGroupBox("Config")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.config_path = QLineEdit()
        self.config_path.setPlaceholderText("config.yaml / config.py")
        self.config_path.setToolTip("Path to experiment config file (.yaml or .py)")
        row.addWidget(self.config_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_config)
        row.addWidget(browse_btn)
        lay.addLayout(row)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Config")
        load_btn.setToolTip("Load and apply config (.yaml or .py)")
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
            "Config files (*.yaml *.yml *.py);;YAML files (*.yaml *.yml);;Python files (*.py);;All files (*)")
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
        self._qubit_row = QHBoxLayout()
        lay.addLayout(self._qubit_row)
        self._rebuild_qubit_buttons()
        self._qubit_bg.idToggled.connect(
            lambda qid, checked: self.qubit_changed.emit(qid) if checked else None
        )

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

    def _rebuild_qubit_buttons(self):
        for btn in list(self._qubit_bg.buttons()):
            self._qubit_bg.removeButton(btn)
            self._qubit_row.removeWidget(btn)
            btn.setParent(None)
        for i, name in enumerate(self._qubit_names):
            rb = QRadioButton(name)
            rb.setToolTip(f"Set active qubit to {name}")
            if i == 0:
                rb.setChecked(True)
            self._qubit_bg.addButton(rb, i)
            self._qubit_row.addWidget(rb)

    def update_qubits(self, names: list):
        """Rebuild qubit radio buttons from loaded config qubit names."""
        self._qubit_names = list(names)
        self._rebuild_qubit_buttons()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_qubit(self) -> int:
        return self._qubit_bg.checkedId()

    @property
    def active_qubit_name(self) -> str:
        qid = self._qubit_bg.checkedId()
        if 0 <= qid < len(self._qubit_names):
            return self._qubit_names[qid]
        return f"Q{qid}"

    @property
    def data_folder(self) -> str:
        return self.data_path.text().strip()
