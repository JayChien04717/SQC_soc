from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QFormLayout, QScrollArea,
)
from PySide6.QtCore import Signal

EXPERIMENT_REGISTRY = {
    "Setup":     [("Time of Flight",   "s001_time_of_flight.TimeOfFlight")],
    "Resonator": [("Res Spec GE",      "s002_res_spec_ge.ResonatorSpec"),
                  ("Res Punchout",     "s002b_res_punchout_ge.ResonatorPunchout"),
                  ("Res Spec Flux",    "s002c_res_spec_ge_flux.ResonatorSpecFlux")],
    "Qubit GE":  [("Qubit Spec",       "s003_qubit_spec_ge.QubitSpec"),
                  ("Qubit Flux Spec",  "s003a_qubit_flux_spec_ge.QubitFluxSpec"),
                  ("Time Rabi",        "s004_time_rabi_ge.TimeRabi"),
                  ("Power Rabi",       "s005_power_rabi_ge.PowerRabi"),
                  ("DRAG",             "s005a_drag.DRAG"),
                  ("AAE",              "s005a_AAE.AAE")],
    "Coherence": [("Ramsey",           "s006_Ramsey_ge.Ramsey"),
                  ("Spin Echo",        "s007_SpinEcho_ge.SpinEcho"),
                  ("T1",               "s008_T1_ge.T1")],
    "Qubit EF":  [("Res Spec EF",      "s009_res_spec_ef.ResonatorSpecEF"),
                  ("Qubit Spec EF",    "s010_qubit_spec_ef.QubitSpecEF"),
                  ("Power Rabi EF",    "s011_power_rabi_ef.PowerRabiEF"),
                  ("Ramsey EF",        "s012_Ramsey_ef.RamseyEF"),
                  ("T1 EF",           "s013_T1_ef.T1EF")],
    "Advanced":  [("AllXY",           "s014_AllXY.AllXY"),
                  ("SingleShot Opt",  "s000_SingleShot_opt.SingleShotOpt"),
                  ("Qubit Temp",      "s013_qubit_temp.QubitTemp"),
                  ("AC Stark",        "s006_ac_stark.ACStark")],
    "RB":        [("Single Qubit RB", "s015_Single_qubit_RB.RandomizedBenchmarking"),
                  ("Auto RB",         "s015_Auto_RB.AutoRB"),
                  ("RB ASM",          "s015_RB_asm.RandomizedBenchmarkingAsm")],
    "Tomography":[("State Tomography","s016_state_tomography.Tomography")],
}

_COMMON_PARAMS = {
    "py_avg":      ("int",   10,    1,     500),
    "reps":        ("int",   1000,  100,   50000),
    "relax_delay": ("float", 50.0,  0.1,   5000.0),
    "steps":       ("int",   100,   10,    2000),
    "span":        ("float", 50.0,  0.1,   2000.0),
}


class ExperimentPanel(QWidget):
    """[2] Experiment selector + parameter form + run controls."""

    run_requested  = Signal(str, dict)
    stop_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._param_widgets = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        sel_grp = QGroupBox("Experiment")
        sel_lay = QFormLayout(sel_grp)
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(EXPERIMENT_REGISTRY.keys())
        self.cat_combo.currentTextChanged.connect(self._on_category)
        sel_lay.addRow("Category:", self.cat_combo)
        self.expt_combo = QComboBox()
        sel_lay.addRow("Experiment:", self.expt_combo)
        root.addWidget(sel_grp)

        self._params_form = QFormLayout()
        params_widget = QWidget()
        params_widget.setLayout(self._params_form)
        scroll = QScrollArea()
        scroll.setWidget(params_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        params_grp = QGroupBox("Parameters")
        QVBoxLayout(params_grp).addWidget(scroll)
        root.addWidget(params_grp)

        btn_row = QHBoxLayout()
        self.run_btn  = QPushButton("▶ Run")
        self.stop_btn = QPushButton("■ Stop")
        self.save_btn = QPushButton("Save")
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.save_btn.clicked.connect(self.save_requested)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.save_btn)
        root.addLayout(btn_row)
        root.addStretch()

        self._on_category(self.cat_combo.currentText())

    def _on_category(self, category):
        self.expt_combo.blockSignals(True)
        self.expt_combo.clear()
        for display, _ in EXPERIMENT_REGISTRY.get(category, []):
            self.expt_combo.addItem(display)
        self.expt_combo.blockSignals(False)
        self._rebuild_params()

    def _rebuild_params(self):
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_widgets.clear()
        for name, (kind, default, lo, hi) in _COMMON_PARAMS.items():
            if kind == "int":
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setValue(default)
            else:
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(2)
                w.setValue(default)
            self._params_form.addRow(name + ":", w)
            self._param_widgets[name] = w

    def _on_run(self):
        cat = self.cat_combo.currentText()
        idx = self.expt_combo.currentIndex()
        _, class_path = EXPERIMENT_REGISTRY[cat][idx]
        params = {k: w.value() for k, w in self._param_widgets.items()}
        self.run_requested.emit(class_path, params)

    def get_params(self):
        return {k: w.value() for k, w in self._param_widgets.items()}
