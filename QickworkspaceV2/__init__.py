"""
QickworkspaceV2 — IBM/IQM-style automated quantum calibration framework.

Quick start
-----------
    from .core.base_experiment import BaseExperiment
    from .backend.qick_backend import QICKBackend
    from .config.system_cfg import ExperimentConfig
    from .calibration import CalibrationStore, AutoCalibrate
    from .experiments import ResonatorSpec, QubitSpec, T1

    # --- Hardware setup ---
    backend = QICKBackend.from_pyro4("192.168.1.100", 8888)
    backend.activate()                     # sets BaseExperiment._soc/_soccfg

    # --- Config ---
    from .config.system_cfg import ExperimentConfig
    cfg_all = ExperimentConfig()

    # --- Single experiment ---
    cfg = cfg_all.get_qubit("Q1")
    result = ResonatorSpec(cfg, backend=backend).run(py_avg=5)
    print(result.fit_result)

    # --- Automated calibration ---
    store = CalibrationStore("data/cal_Q1.json")
    auto  = AutoCalibrate(cfg_all, "Q1", cal_store=store)
    auto.run()
    auto.summary()

    # --- REST service ---
    from .service import create_app
    app = create_app(cal_store=store, config_all=cfg_all, backend=backend)
    # uvicorn .service.api:app --host 0.0.0.0 --port 8000
"""

from .core.experiment_data import ExperimentData, QualityFlag
from .core.base_analysis import BaseAnalysis
from .core.base_experiment import BaseExperiment
from .core.composite import BatchExperiment, ParallelExperiment
from .backend.base_backend import BaseBackend
from .backend.qick_backend import QICKBackend
from .backend.simulated_backend import SimulatedBackend
from .calibration import CalibrationStore, CalibrationGraph, CalibrationNode, CalibrationMonitor, AutoCalibrate

_LAZY_EXPORTS = {
    "ExperimentConfig": ".config.system_cfg",
    "SingleShot_gef": ".experiments.setup",
    "SingleShot_ge_opt": ".experiments.setup",
    "hist": ".experiments.setup",
    "TOF": ".experiments.setup",
    "ResonatorSpec": ".experiments.resonator",
    "Punchout": ".experiments.resonator",
    "ResonatorSpecFlux": ".experiments.resonator",
    "QubitSpec": ".experiments.qubit_ge",
    "QubitSpecFlux": ".experiments.qubit_ge",
    "TimeRabi": ".experiments.qubit_ge",
    "PowerRabi": ".experiments.qubit_ge",
    "PowerRabiReset": ".experiments.qubit_ge",
    "Ramsey": ".experiments.coherence",
    "ACStark": ".experiments.coherence",
    "SpinEcho": ".experiments.coherence",
    "T1": ".experiments.coherence",
    "RamseyEf": ".experiments.coherence",
    "T1Ef": ".experiments.coherence",
    "ResonatorSpec_ef": ".experiments.qubit_ef",
    "QubitSpecEf": ".experiments.qubit_ef",
    "PowerRabiEf": ".experiments.qubit_ef",
    "QubitTemp": ".experiments.qubit_ef",
    "AllXY": ".experiments.characterization",
    "RandomizedBenchmarking": ".experiments.characterization",
    "AutoRB": ".experiments.characterization",
    "Tomography": ".experiments.characterization",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__version__ = "1.0.0"

__all__ = [
    # core
    "ExperimentData", "QualityFlag",
    "BaseAnalysis", "BaseExperiment",
    "BatchExperiment", "ParallelExperiment",
    # backend
    "BaseBackend", "QICKBackend", "SimulatedBackend",
    # calibration
    "CalibrationStore", "CalibrationGraph", "CalibrationNode",
    "CalibrationMonitor", "AutoCalibrate",
    # config
    "ExperimentConfig",
    # experiments — setup
    "SingleShot_gef", "SingleShot_ge_opt", "hist", "TOF",
    # experiments — resonator
    "ResonatorSpec", "Punchout", "ResonatorSpecFlux",
    # experiments — qubit ge
    "QubitSpec", "QubitSpecFlux", "TimeRabi", "PowerRabi", "PowerRabiReset",
    # experiments — coherence
    "Ramsey", "ACStark", "SpinEcho", "T1", "RamseyEf", "T1Ef",
    # experiments — qubit ef
    "ResonatorSpec_ef", "QubitSpecEf", "PowerRabiEf", "QubitTemp",
    # experiments — characterization
    "AllXY", "RandomizedBenchmarking", "AutoRB", "Tomography",
]
