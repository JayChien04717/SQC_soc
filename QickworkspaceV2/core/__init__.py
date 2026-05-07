from .experiment_data import ExperimentData, QualityFlag
from .base_analysis import BaseAnalysis, IdentityAnalysis
from .base_experiment import BaseExperiment
from .composite import BatchExperiment, ParallelExperiment


def __getattr__(name):
    if name in {"BaseProgram", "GATE_ALIAS", "resolve_gate"}:
        from .base_program import BaseProgram, GATE_ALIAS, resolve_gate

        exports = {
            "BaseProgram": BaseProgram,
            "GATE_ALIAS": GATE_ALIAS,
            "resolve_gate": resolve_gate,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExperimentData",
    "QualityFlag",
    "BaseAnalysis",
    "IdentityAnalysis",
    "BaseProgram",
    "GATE_ALIAS",
    "resolve_gate",
    "BaseExperiment",
    "BatchExperiment",
    "ParallelExperiment",
]
