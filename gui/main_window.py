import importlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .panels.experiment_panel import ExperimentPanel
from .panels.plot_panel import PlotPanel
from .panels.setup_panel import SetupPanel
from QickworkspaceV2.core.experiment_registry import (
    canonical_class_path,
    fit_updates_from_result,
)


def _build_run_cfg(class_path: str, qubit_cfg: dict, params: dict) -> dict:
    """Merge active qubit config with GUI form values."""
    from qick.asm_v2 import QickSweep1D

    cfg = dict(qubit_cfg)
    for key in ("reps", "relax_delay"):
        if key in params:
            cfg[key] = params[key]

    cp = class_path
    if cp == "s001_time_of_flight.TOF":
        cfg["reps"] = 1
        for key in ("ro_length", "res_length", "res_gain_ge", "res_ch"):
            if key in params:
                cfg[key] = params[key]
    elif cp == "s002_res_spec_ge.ResonatorSpec":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "steps": params["steps"],
            "res_gain_ge": params["res_gain_ge"],
            "res_freq_ge": QickSweep1D("freqloop", center - span, center + span),
        })
    elif cp == "s002b_res_punchout_ge.Punchout":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "f_steps": params["freq_steps"],
            "g_steps": params["gain_steps"],
            "res_freq_ge": QickSweep1D("freqloop", center - span, center + span),
            "res_gain_ge": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"]),
        })
    elif cp == "s002c_res_spec_ge_flux.ResonatorSpecFlux":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "res_gain_ge": params["res_gain_ge"],
            "freq_steps": params["freq_steps"],
            "flux_ch": params["flux_ch"],
            "flux_length": params["flux_length"],
            "res_freq_ge": QickSweep1D("freqloop", center - span, center + span),
            "flux_gain": QickSweep1D("fluxloop", params["flux_gain_start"], params["flux_gain_stop"]),
            "steps_flux": params["flux_steps"],
        })
    elif cp == "s003_qubit_spec_ge.QubitSpec":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "steps": params["steps"],
            "qb_gain_ge": params["qb_gain_ge"],
            "qb_flat_top_length_ge": params["qb_flat_top_length_ge"],
            "nqz_qb": params["nqz_qb"],
            "qb_mixer": center,
            "qb_freq_ge": QickSweep1D("freqloop", center - span, center + span),
        })
    elif cp == "s003a_qubit_flux_spec_ge.QubitSpecFlux":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "freq_steps": params["freq_steps"],
            "qb_gain_ge": params["qb_gain_ge"],
            "qb_flat_top_length_ge": params["qb_flat_top_length_ge"],
            "flux_ch": params["flux_ch"],
            "flux_length": params["flux_length"],
            "qb_mixer": center,
            "qb_freq_ge": QickSweep1D("freqloop", center - span, center + span),
            "flux_gain": QickSweep1D("fluxloop", params["flux_gain_start"], params["flux_gain_stop"]),
            "steps_flux": params["flux_steps"],
        })
    elif cp == "s004_time_rabi_ge.TimeRabi":
        cfg.update({
            "sigma_ge": params["sigma_ge"],
            "qb_gain_ge": params["qb_gain_ge"],
            "steps": params["steps"],
            "qb_flat_top_length_ge": QickSweep1D("lenloop", params["length_start"], params["length_stop"]),
        })
    elif cp == "s005_power_rabi_ge.PowerRabi":
        cfg.update({
            "sigma_ge": params["sigma_ge"],
            "nqz_qb": params["nqz_qb"],
            "steps": params["steps"],
            "qb_gain_ge": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"]),
        })
    elif cp == "s005a_drag.DragCalibration":
        cfg.update({k: params[k] for k in ("alpha_start", "alpha_stop", "alpha_steps", "iter_start", "iter_stop", "iter_step")})
    elif cp == "s005a_AAE.PowerRabiChevron":
        focus = float(cfg.get("pi_gain_ge", 0.5))
        half = params["gain_half_span"]
        cfg.update({
            "steps": params["steps"],
            "iter_start": params["iter_start"],
            "iter_stop": params["iter_stop"],
            "iter_step": params["iter_step"],
            "qb_gain_ge": QickSweep1D("gainloop", focus - half, focus + half),
        })
    elif cp in ("s006_Ramsey_ge.Ramsey", "s012_Ramsey_ef.Ramsey_ef"):
        cfg.update({
            "steps": params["steps"],
            "ramsey_freq": params["ramsey_freq"],
            "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"]),
        })
        if "ge_ref" in params:
            cfg["ge_ref"] = params["ge_ref"]
    elif cp == "s007_SpinEcho_ge.SpinEcho":
        cfg.update({
            "steps": params["steps"],
            "ramsey_freq": params["ramsey_freq"],
            "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"]),
        })
    elif cp in ("s008_T1_ge.T1", "s013_T1_ef.T1_ef"):
        cfg.update({
            "steps": params["steps"],
            "wait_time": QickSweep1D("waitloop", params["time_start"], params["time_stop"]),
        })
    elif cp == "s009_res_spec_ef.ResonatorSpec_ef":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "res_gain_ge": params["res_gain_ge"],
            "steps": params["steps"],
            "res_freq_ge": QickSweep1D("freqloop", center - span, center + span),
        })
    elif cp == "s010_qubit_spec_ef.QubitSpec_ef":
        center, span = params["freq_center"], params["freq_span"]
        cfg.update({
            "steps": params["steps"],
            "qb_gain_ef": params["qb_gain_ef"],
            "qb_flat_top_length_ef": params["qb_flat_top_length_ef"],
            "ge_ref": params["ge_ref"],
            "qb_freq_ef": QickSweep1D("freqloop", center - span, center + span),
        })
    elif cp in ("s011_power_rabi_ef.PowerRabi_ef", "s013_qubit_temp.QubitTemperatureEf"):
        cfg.update({
            "sigma_ef": params["sigma_ef"],
            "steps": params["steps"],
            "ge_ref": params["ge_ref"],
            "qb_gain_ef": QickSweep1D("gainloop", params["gain_start"], params["gain_stop"]),
        })
    elif cp == "s014_AllXY.AllXY":
        cfg["pulse_type"] = params["pulse_type"]
    elif cp == "s000_SingleShot_prog.SingleShot_gef":
        cfg["shot_f"] = params["shot_f"]
    elif cp in ("s015_Single_qubit_RB.RandomizedBenchmarking", "s015_Auto_RB.AutoRB", "s015_RB_asm.RandomizedBenchmarkingAsm"):
        cfg.update({k: params[k] for k in ("max_circuit_depth", "delta_clifford", "number_sample") if k in params})
        if "pulse_type" in params:
            cfg["pulse_type"] = params["pulse_type"]
    elif cp == "s016_state_tomography.Tomography":
        cfg["prep_pulse_name"] = params["prep_pulse_name"]
    elif cp == "s006_ac_stark.AcStarkCalib":
        cfg.update({k: params[k] for k in ("steps",) if k in params})

    return cfg


class ConfigViewer(QWidget):
    """Floating config and fit-result viewer."""

    update_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumWidth(360)
        self.text.setMinimumHeight(420)
        self.update_btn = QPushButton("Update Config From Fit")
        self.update_btn.clicked.connect(self.update_requested)
        layout.addWidget(self.text)
        layout.addWidget(self.update_btn)

    def set_payload(self, payload: dict):
        self.text.setPlainText(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


class AcquireWorker(QThread):
    data_ready = Signal(object, object, str, str)
    result_ready = Signal(object)
    log_message = Signal(str, str)
    finished = Signal()

    _SPECIAL_EXPTS = frozenset({
        "s000_SingleShot_prog.SingleShot_gef",
        "s000_SingleShot_opt.SingleShot_ge_opt",
        "s015_Single_qubit_RB.RandomizedBenchmarking",
        "s015_Auto_RB.AutoRB",
        "s015_RB_asm.RandomizedBenchmarkingAsm",
        "s014_AllXY.AllXY",
        "s016_state_tomography.Tomography",
    })
    _DECIMATED_EXPTS = frozenset({"s001_time_of_flight.TOF"})

    def __init__(self, class_path: str, params: dict, qubit_cfg: dict, debug: bool = False, parent=None):
        super().__init__(parent)
        self.class_path = class_path
        self.params = params
        self.qubit_cfg = qubit_cfg
        self.debug = debug
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        self.log_message.emit("Stop requested; current acquisition will finish cleanly.", "warn")

    def run(self):
        name = self.class_path.split(".")[-1]
        self.log_message.emit(f"Starting {name}.", "info")
        try:
            if self.debug:
                self._run_debug()
            elif self.class_path in self._SPECIAL_EXPTS:
                self._run_special()
            else:
                self._run_liveplot()
            if self._stop_requested:
                self.log_message.emit("Experiment interrupted.", "warn")
            else:
                self.log_message.emit("Experiment complete.", "success")
        except Exception as exc:
            self.log_message.emit(f"Error: {exc}", "error")
            self.log_message.emit(traceback.format_exc(), "error")
        finally:
            self.finished.emit()

    def _make_expt(self):
        run_cfg = _build_run_cfg(self.class_path, self.qubit_cfg, self.params)
        mod_name, cls_name = canonical_class_path(self.class_path).rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)(run_cfg)

    def _run_debug(self):
        from gui.mock.runner import generate_mock_data
        from QickworkspaceV2.core.experiment_data import ExperimentData

        py_avg = int(self.params.get("py_avg", self.params.get("shots", 10)))
        py_avg = max(py_avg, 1)
        x, final_iq, xlabel, title = generate_mock_data(self.class_path, self.params, sleep=0)
        acc = np.zeros_like(final_iq, dtype=complex)
        report_every = max(1, py_avg // 60)
        avg_count = 0
        for i in range(py_avg):
            if self._stop_requested:
                break
            noise = (np.random.normal(size=final_iq.shape) + 1j * np.random.normal(size=final_iq.shape)) * 25
            acc += final_iq + noise
            avg_count = i + 1
            if avg_count % report_every == 0 or avg_count == py_avg:
                self.data_ready.emit(x, acc / avg_count, xlabel, f"{title} [{avg_count}/{py_avg}]")
            self.msleep(30)

        iq = acc / max(avg_count, 1)
        result = ExperimentData(
            experiment_type=title,
            raw_iq=iq,
            x_axis=x,
            fit_result={},
            config=dict(self.qubit_cfg),
            interrupted=self._stop_requested,
            avg_count=avg_count,
        )
        self._fit_debug_result(result)
        self.result_ready.emit(result)

    def _fit_debug_result(self, result):
        """Best-effort fit for synthetic data so GUI-only testing exercises overlays."""
        if result.raw_iq is None or result.x_axis is None:
            return
        x = np.asarray(result.x_axis, dtype=float)
        y = np.abs(result.raw_iq)
        cls = self.class_path.split(".")[-1].lower()
        try:
            if "resonatorspec" in cls or "qubitspec" in cls:
                from QickworkspaceV2.tools.fitting import fitlor
                popt, pcov, _ = fitlor(x, y)
                err = np.sqrt(np.diag(pcov))
                result.fit_params = np.asarray(popt)
                result.fit_errors = err
                key = "f_res[MHz]" if "resonator" in cls else "f0_MHz"
                width_key = "kappa_MHz" if "resonator" in cls else "linewidth_MHz"
                result.fit_result = {key: (float(popt[2]), float(err[2])), width_key: (abs(float(popt[3])), float(err[3]))}
                result.scalar_result = float(popt[2])
            elif "t1" in cls:
                from QickworkspaceV2.tools.fitting import fitexp
                popt, pcov, _ = fitexp(x, y)
                err = np.sqrt(np.diag(pcov))
                result.fit_params = np.asarray(popt)
                result.fit_errors = err
                result.fit_result = {"T1_us": (abs(float(popt[2])), float(err[2]))}
                result.scalar_result = abs(float(popt[2]))
            elif any(name in cls for name in ("ramsey", "spinecho", "powerrabi", "timerabi")):
                from QickworkspaceV2.tools.fitting import fitdecaysin, fix_phase
                popt, pcov, _ = fitdecaysin(x, y)
                err = np.sqrt(np.diag(pcov))
                result.fit_params = np.asarray(popt)
                result.fit_errors = err
                if "powerrabi" in cls:
                    pi_gain, pi2_gain = fix_phase(popt)
                    result.fit_result = {"pi_gain": (float(pi_gain), None), "pi2_gain": (float(pi2_gain), None)}
                    result.scalar_result = float(pi_gain)
                elif "timerabi" in cls:
                    pi_length = 1.0 / (2.0 * abs(float(popt[1]))) if popt[1] else 0.0
                    result.fit_result = {"pi_length_us": (pi_length, None)}
                    result.scalar_result = pi_length
                elif "ramsey" in cls:
                    result.fit_result = {"T2r_us": (abs(float(popt[3])), float(err[3])), "detune_MHz": (float(popt[1]), float(err[1]))}
                    result.scalar_result = abs(float(popt[3]))
                else:
                    result.fit_result = {"T2e_us": (abs(float(popt[3])), float(err[3]))}
                    result.scalar_result = abs(float(popt[3]))
        except Exception as exc:
            result.quality_message = f"Debug fit failed: {exc}"

    def _run_liveplot(self):
        from QickworkspaceV2.core.base_experiment import BaseExperiment
        from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag

        expt = self._make_expt()
        py_avg = int(self.params.get("py_avg", self.params.get("shots", 10)))
        prog = expt._create_program()
        expt._last_prog = prog
        x_vals = BaseExperiment._resolve_axis(expt._extract_sweep_axis(prog), expt.cfg.get("steps"))
        y_vals = BaseExperiment._resolve_axis(expt._extract_sweep_axis_y(prog), expt.cfg.get("steps"))
        expt._sweep_vals_x = x_vals
        expt._sweep_vals_y = y_vals

        is_decimated = self.class_path in self._DECIMATED_EXPTS
        report_every = max(1, py_avg // 60)
        iq_sum = 0
        iqdata = None
        avg_count = 0
        for i in range(py_avg):
            if self._stop_requested:
                break
            if is_decimated:
                iq_list = prog.acquire_decimated(expt.soc, rounds=1, progress=False)
                iq_data = iq_list[0].dot([1, 1j])
            else:
                iq_list = prog.acquire(expt.soc, rounds=1, progress=False)
                iq_data = iq_list[0][0].dot([1, 1j])

            iq_sum = iq_data if i == 0 else iq_sum + iq_data
            avg_count = i + 1
            iqdata = iq_sum / avg_count
            if avg_count % report_every == 0 or avg_count == py_avg:
                self.data_ready.emit(x_vals, iqdata, expt.X_LABEL, f"{expt.TITLE_PREFIX} [{avg_count}/{py_avg}]")

        if iqdata is None:
            result = ExperimentData(
                experiment_type=expt.EXPT_NAME,
                quality=QualityFlag.BAD,
                quality_message="No data acquired",
                interrupted=True,
                avg_count=0,
                config=dict(expt.cfg),
            )
            self.result_ready.emit(result)
            return

        expt.iqdata = iqdata
        old_result = expt._post_fit(x_vals)
        result = ExperimentData(
            experiment_type=expt.EXPT_NAME,
            raw_iq=iqdata,
            x_axis=x_vals,
            y_axis=y_vals,
            fit_params=expt.fit_params,
            fit_errors=expt.fit_errors,
            config=dict(expt.cfg),
            interrupted=self._stop_requested,
            avg_count=avg_count,
            x_name=expt.X_SAVE_NAME,
            x_unit=expt.X_SAVE_UNIT,
            x_scale=expt.X_SAVE_SCALE,
            y_name=expt.Y_SAVE_NAME,
            y_unit=expt.Y_SAVE_UNIT,
            y_scale=expt.Y_SAVE_SCALE,
        )
        if isinstance(old_result, (int, float)):
            result.scalar_result = float(old_result)
        elif isinstance(old_result, dict):
            result.fit_result = {k: (v, None) for k, v in old_result.items()}
        elif result.fit_result == {} and expt.fit_params is not None:
            result.fit_result = expt._build_fit_result()
        if expt.Analysis is not None:
            result = expt.Analysis().run(result)
        expt.result = result
        self.result_ready.emit(result)

    def _run_special(self):
        expt = self._make_expt()
        py_avg = int(self.params.get("py_avg", self.params.get("shots", 10)))
        cp = self.class_path
        self.log_message.emit("This experiment uses a single blocking run; stop takes effect after the current call.", "warn")
        if cp == "s000_SingleShot_prog.SingleShot_gef":
            result = expt.run(py_avg, shot_f=self.params.get("shot_f", False))
        elif cp in ("s015_Single_qubit_RB.RandomizedBenchmarking", "s015_RB_asm.RandomizedBenchmarkingAsm"):
            result = expt.run(
                py_avg,
                max_circuit_depth=self.params.get("max_circuit_depth", 400),
                delta_clifford=self.params.get("delta_clifford", 40),
                number_sample=self.params.get("number_sample", 30),
            )
        elif cp == "s015_Auto_RB.AutoRB":
            result = expt.run(
                py_avg,
                max_circuit_depth=self.params.get("max_circuit_depth", 600),
                delta_clifford=self.params.get("delta_clifford", 50),
                number_sample=self.params.get("number_sample", 50),
            )
        elif cp == "s016_state_tomography.Tomography":
            result = expt.run(py_avg, prep_pulse_name=self.params.get("prep_pulse_name"))
        else:
            result = expt.run(py_avg)
        result.interrupted = result.interrupted or self._stop_requested
        if result.raw_iq is not None and result.x_axis is not None:
            self.data_ready.emit(result.x_axis, result.raw_iq, expt.X_LABEL, expt.TITLE_PREFIX)
        self.result_ready.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, debug: bool = False):
        super().__init__()
        self._debug = debug
        self.setWindowTitle("QickworkspaceV2 Control Panel" + (" [SIMULATION]" if debug else ""))
        self.resize(1440, 860)
        self._worker: AcquireWorker | None = None
        self._config: dict = {}
        self._config_list: list = []
        self._config_all = None
        self._config_overrides: dict[str, dict] = {}
        self._last_result = None
        self._last_result_path = None
        self._browser_procs: list[QProcess] = []
        self._build_layout()
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_geometry()
        self._refresh_config_view()
        if debug:
            self._lbl_debug.setVisible(True)
            self.log("Simulation mode active; no hardware required.", "warn")
            self._auto_connect_mock()

    def _build_layout(self):
        self.plot_panel = PlotPanel()
        self.setCentralWidget(self.plot_panel)

        self.setup_panel = SetupPanel()
        self._add_dock("Setup", self.setup_panel, Qt.LeftDockWidgetArea, closable=False, width=280)

        self.expt_panel = ExperimentPanel()
        self._add_dock("Experiment", self.expt_panel, Qt.RightDockWidgetArea, closable=False, width=300)

        self.config_viewer = ConfigViewer()
        self._config_dock = self._add_dock("Config", self.config_viewer, Qt.RightDockWidgetArea, width=380)
        self._config_dock.setFloating(True)
        self._config_dock.hide()

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        self._log.document().setMaximumBlockCount(700)
        self._add_dock("Log", self._log, Qt.BottomDockWidgetArea, closable=False)

        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("&Save Plot", self.plot_panel.export_figure, QKeySequence("Ctrl+S"))
        fm.addAction("Save &Result", self._on_save, QKeySequence("Ctrl+Shift+S"))
        fm.addSeparator()
        fm.addAction("&Quit", self.close, QKeySequence("Ctrl+Q"))

        tm = mb.addMenu("&Tools")
        tm.addAction("&Config Viewer", self._show_config_viewer, QKeySequence("Ctrl+I"))
        tm.addAction("&Update Config From Fit", self._update_config_from_fit, QKeySequence("Ctrl+U"))
        tm.addAction("&Data Browser", self._launch_browser, QKeySequence("Ctrl+B"))
        tm.addAction("Clear &Log", self._log.clear)

        hm = mb.addMenu("&Help")
        hm.addAction("&About", self._show_about)

        self._led = QLabel("●")
        self._led.setStyleSheet("color: #f0883e; font-size: 14px;")
        self._lbl_conn = QLabel("Disconnected")
        self._lbl_qubit = QLabel("Q0")
        self._lbl_state = QLabel("Ready")
        self._lbl_debug = QLabel("SIM")
        self._lbl_debug.setStyleSheet("color: #f0d050; font-weight: bold; font-size: 11px;")
        self._lbl_debug.setVisible(False)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
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

    def _add_dock(self, title, widget, area, *, movable=True, floatable=True, closable=True, width=None):
        dock = QDockWidget(title, self)
        features = QDockWidget.NoDockWidgetFeatures
        if movable:
            features |= QDockWidget.DockWidgetMovable
        if floatable:
            features |= QDockWidget.DockWidgetFloatable
        if closable:
            features |= QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        dock.setWidget(widget)
        if width:
            dock.setMinimumWidth(width)
        self.addDockWidget(area, dock)
        return dock

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self.expt_panel.trigger_run)
        QShortcut(QKeySequence("Escape"), self, self.expt_panel.trigger_stop)
        QShortcut(QKeySequence("Ctrl+B"), self, self._launch_browser)
        QShortcut(QKeySequence("Ctrl+I"), self, self._show_config_viewer)

    def _connect_signals(self):
        self.setup_panel.connection_changed.connect(self._on_connection)
        self.setup_panel.soc_ready.connect(self._on_soc_ready)
        self.setup_panel.qubit_changed.connect(self._on_qubit)
        self.setup_panel.config_loaded.connect(self._on_config)
        self.config_viewer.update_requested.connect(self._update_config_from_fit)
        self.expt_panel.run_requested.connect(self._on_run)
        self.expt_panel.stop_requested.connect(self._on_stop)
        self.expt_panel.save_requested.connect(self._on_save)

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
        board_info = str(soccfg).strip()
        for line in board_info.splitlines()[:6]:
            self.log(line, "info")
        data_path = self.setup_panel.data_folder or "data"
        try:
            from QickworkspaceV2.core.base_experiment import BaseExperiment
            BaseExperiment.setup(soc, soccfg, data_path)
            self.log(f"BaseExperiment ready; data path: {data_path}", "success")
        except Exception as exc:
            self.log(f"BaseExperiment.setup() failed: {exc}", "error")

    def _on_qubit(self, qidx: int):
        del qidx
        name = self.setup_panel.active_qubit_name
        self._lbl_qubit.setText(name)
        self.log(f"Active qubit: {name}", "info")
        self._refresh_config_view()

    def _on_config(self, path: str):
        try:
            cfg, qubit_names, config_list = self._parse_config(path)
            if config_list:
                from QickworkspaceV2.tools.system_tool import ExperimentConfig
                self._config_all = ExperimentConfig(config_list)
            else:
                self._config_all = None
        except Exception as exc:
            self.log(f"Config load failed: {exc}", "error")
            return
        self._config = cfg
        self._config_list = config_list
        self._config_overrides.clear()
        if qubit_names:
            self.setup_panel.update_qubits(qubit_names)
            self.log(f"Qubits detected: {', '.join(qubit_names)}", "info")
            self._lbl_qubit.setText(self.setup_panel.active_qubit_name)
        suffix = f" ({len(config_list)} qubits)" if config_list else f" ({len(cfg)} keys)"
        self.log(f"Config loaded: {path}{suffix}", "info")
        self._refresh_config_view()

    def _parse_config(self, path: str) -> tuple[dict, list, list]:
        p = Path(path)
        qubit_names: list = []
        config_list: list = []
        if p.suffix == ".py":
            mod = self._load_python_config_module(p)
            cfg_list = getattr(mod, "config_list", None)
            if isinstance(cfg_list, list):
                config_list = cfg_list
                qubit_names = [item.get("name", f"Q{i}") for i, item in enumerate(cfg_list) if isinstance(item, dict)]
            for name in ("config", "cfg", "hw_cfg", "expt_cfg"):
                val = getattr(mod, name, None)
                if isinstance(val, dict):
                    return val, qubit_names, config_list
            return {k: v for k, v in vars(mod).items() if not k.startswith("_")}, qubit_names, config_list

        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, list):
            config_list = data
            qubit_names = [item.get("name", f"Q{i}") for i, item in enumerate(data) if isinstance(item, dict)]
            return {}, qubit_names, config_list
        return data, qubit_names, config_list

    @staticmethod
    def _load_python_config_module(path: Path):
        """Load Python configs with package context when possible."""
        import importlib.util

        resolved = path.resolve()
        repo_root = Path.cwd().resolve()

        try:
            rel = resolved.relative_to(repo_root)
        except ValueError:
            rel = None

        if rel is not None and rel.suffix == ".py":
            parts = rel.with_suffix("").parts
            if parts and all(part.isidentifier() for part in parts):
                module_name = ".".join(parts)
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                module = importlib.import_module(module_name)
                return importlib.reload(module)

        spec = importlib.util.spec_from_file_location("_gui_cfg", resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config module from {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _auto_connect_mock(self):
        from gui.mock.hardware import MockSoc, MockSoccfg
        QTimer.singleShot(300, lambda: self.setup_panel._on_success(MockSoc(), MockSoccfg()))

    def _active_qubit_cfg(self) -> dict:
        name = self.setup_panel.active_qubit_name
        cfg = {}
        if self._config_all is not None:
            cfg = dict(self._config_all.get_qubit(name))
        elif self._config:
            cfg = dict(self._config)
        cfg.update(self._config_overrides.get(name, {}))
        return cfg

    def _on_run(self, class_path: str, params: dict):
        if self._worker and self._worker.isRunning():
            self.log("Already running; stop first.", "warn")
            return
        qubit_cfg = self._active_qubit_cfg()
        if not self._debug and not qubit_cfg:
            self.log("No config loaded; browse and load a config file first.", "warn")

        self._last_result = None
        self.plot_panel.clear_fit()
        self._worker = AcquireWorker(class_path, params, qubit_cfg, self._debug, self)
        self._worker.log_message.connect(self.log)
        self._worker.data_ready.connect(self.plot_panel.update_data)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        name = class_path.split(".")[-1]
        self._lbl_state.setText(f"Running {name}")
        self._progress.setVisible(True)
        self.expt_panel.set_running(True)
        self.log(f"Run {name}: {params}", "info")

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._lbl_state.setText("Stopping")
            self.expt_panel.set_running(False)
            return
        self.log("No active experiment to stop.", "warn")

    def _on_result_ready(self, result):
        self._last_result = result
        self.plot_panel.set_result(result)
        self._refresh_config_view()
        if getattr(result, "fit_result", None):
            self.log(f"Fit result: {self._format_fit_result(result.fit_result)}", "success")
        if getattr(result, "interrupted", False):
            self.log(f"Partial result kept after {getattr(result, 'avg_count', 0)} averages.", "warn")

    def _on_worker_done(self):
        self._progress.setVisible(False)
        self._lbl_state.setText("Ready")
        self.expt_panel.set_running(False)
        self._worker = None

    def _on_save(self):
        if self._last_result is None:
            self.log("No result to save yet.", "warn")
            return
        data_dir = Path(self.setup_panel.data_folder or "data")
        stem = self._last_result.experiment_type or "experiment"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = data_dir / f"{stem}_{stamp}.h5"
        try:
            saved = self._last_result.save(str(path))
            self._last_result_path = saved
            self.log(f"Result saved: {saved}", "success")
        except Exception as exc:
            self.log(f"Save failed: {exc}", "error")

    def _show_config_viewer(self):
        self._refresh_config_view()
        self._config_dock.show()
        self._config_dock.raise_()

    def _refresh_config_view(self):
        payload = {
            "active_qubit": self.setup_panel.active_qubit_name if hasattr(self, "setup_panel") else "",
            "config": self._active_qubit_cfg() if hasattr(self, "setup_panel") else {},
            "overrides": self._config_overrides.get(self.setup_panel.active_qubit_name, {}) if hasattr(self, "setup_panel") else {},
            "last_fit": getattr(self._last_result, "fit_result", {}) or {},
            "last_result": {
                "experiment_type": getattr(self._last_result, "experiment_type", None),
                "avg_count": getattr(self._last_result, "avg_count", None),
                "interrupted": getattr(self._last_result, "interrupted", None),
                "saved_path": self._last_result_path,
            },
        }
        if hasattr(self, "config_viewer"):
            self.config_viewer.set_payload(payload)

    def _update_config_from_fit(self):
        if self._last_result is None or not getattr(self._last_result, "fit_result", None):
            self.log("No fit result available for config update.", "warn")
            return
        updates = fit_updates_from_result(self._last_result)
        if not updates:
            self.log("Fit result has no known config mapping.", "warn")
            return
        qubit = self.setup_panel.active_qubit_name
        self._config_overrides.setdefault(qubit, {}).update(updates)
        if self._config_all is not None:
            for key, value in updates.items():
                try:
                    self._config_all.update(key, value, q_index=qubit)
                except Exception as exc:
                    self.log(f"Config update failed for {key}: {exc}", "warn")
        self.log(f"Updated {qubit}: {updates}", "success")
        self._refresh_config_view()

    @staticmethod
    def _format_fit_result(fit_result: dict) -> str:
        parts = []
        for key, raw in fit_result.items():
            value = raw[0] if isinstance(raw, (tuple, list)) and raw else raw
            if isinstance(value, (int, float, np.number)):
                parts.append(f"{key}={float(value):.6g}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _launch_browser(self):
        script = str(Path(__file__).parent / "data_browser_app.py")
        proc = QProcess(self)
        proc.start(sys.executable, [script])
        self._browser_procs.append(proc)
        self.log("Data Browser opened.", "info")

    def _show_about(self):
        QMessageBox.about(
            self,
            "About QickworkspaceV2 GUI",
            "<h3>QickworkspaceV2 GUI</h3>"
            "<p>PySide6 control interface for QICK experiments.</p>"
            "<p style='color:#8b949e;font-size:11px;'>Ctrl+R run, Esc stop, Ctrl+I config, Ctrl+U update config.</p>",
        )

    def log(self, msg: str, level: str = "info"):
        color = {
            "info": "#8b949e",
            "success": "#7ee787",
            "warn": "#f0883e",
            "error": "#ff7b72",
        }.get(level, "#8b949e")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f'<span style="color:{color};">[{ts}] {msg}</span>')

    def _restore_geometry(self):
        settings = QSettings("QickworkspaceV2", "GUI")
        geom = settings.value("geometry")
        state = settings.value("windowState")
        if geom:
            self.restoreGeometry(geom)
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        settings = QSettings("QickworkspaceV2", "GUI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        for proc in self._browser_procs:
            proc.kill()
        super().closeEvent(event)


def _sep():
    label = QLabel(" | ")
    label.setStyleSheet("color: #30363d;")
    return label
