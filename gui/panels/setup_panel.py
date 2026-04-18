from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QFileDialog,
)
from PySide6.QtCore import Signal


class SetupPanel(QWidget):
    """[1] Connection, config file, and qubit selector."""

    connection_changed = Signal(bool)
    qubit_changed      = Signal(int)
    config_loaded      = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(self._connection_group())
        root.addWidget(self._config_group())
        root.addWidget(self._qubit_group())
        root.addStretch()

    def _connection_group(self):
        grp = QGroupBox("Connection")
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        row.addWidget(QLabel("SOC IP:"))
        self.ip_edit = QLineEdit("192.168.1.1")
        row.addWidget(self.ip_edit)
        lay.addLayout(row)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._on_connect)
        lay.addWidget(self.connect_btn)
        self.status_label = QLabel("● Disconnected")
        self.status_label.setStyleSheet("color: red;")
        lay.addWidget(self.status_label)
        return grp

    def _config_group(self):
        grp = QGroupBox("Config")
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.config_path = QLineEdit()
        self.config_path.setPlaceholderText("config.yaml")
        row.addWidget(self.config_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_config)
        row.addWidget(browse_btn)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("Save")
        row2.addWidget(load_btn)
        row2.addWidget(save_btn)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Data path:"))
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("D:/data")
        row3.addWidget(self.data_path)
        browse2 = QPushButton("…")
        browse2.setFixedWidth(28)
        browse2.clicked.connect(self._browse_data)
        row3.addWidget(browse2)
        lay.addLayout(row3)
        return grp

    def _qubit_group(self):
        grp = QGroupBox("Qubit")
        lay = QVBoxLayout(grp)
        self._qubit_bg = QButtonGroup(self)
        qrow = QHBoxLayout()
        for i in range(4):
            rb = QRadioButton(f"Q{i}")
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
        row.addWidget(self.yoko_spin)
        lay.addLayout(row)
        return grp

    def _on_connect(self, checked):
        if checked:
            self.connect_btn.setText("Disconnect")
            self.status_label.setText("● Connected")
            self.status_label.setStyleSheet("color: green;")
            self.connection_changed.emit(True)
        else:
            self.connect_btn.setText("Connect")
            self.status_label.setText("● Disconnected")
            self.status_label.setStyleSheet("color: red;")
            self.connection_changed.emit(False)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Config", "", "YAML (*.yaml *.yml)")
        if path:
            self.config_path.setText(path)

    def _browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Data folder")
        if path:
            self.data_path.setText(path)

    def _load_config(self):
        path = self.config_path.text().strip()
        if path:
            self.config_loaded.emit(path)

    @property
    def active_qubit(self):
        return self._qubit_bg.checkedId()

    @property
    def data_folder(self):
        return self.data_path.text().strip()
