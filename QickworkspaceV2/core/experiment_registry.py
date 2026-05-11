"""Shared experiment registry for GUI and service clients.

This module is deliberately free of Qt/FastAPI imports. It is the stable
contract layer that maps user-facing experiment ids to Python classes and
known fit-result to config-update rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    display: str
    category: str
    class_path: str
    legacy_id: str = ""
    supports_liveplot: bool = True
    supports_stop: bool = True


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("tof", "Time of Flight", "Setup", "QickworkspaceV2.experiments.setup.TOF", "s001_time_of_flight.TOF"),
    ExperimentSpec("res_spec_ge", "Res Spec GE", "Resonator", "QickworkspaceV2.experiments.resonator.ResonatorSpec", "s002_res_spec_ge.ResonatorSpec"),
    ExperimentSpec("res_punchout", "Res Punchout", "Resonator", "QickworkspaceV2.experiments.resonator.Punchout", "s002b_res_punchout_ge.Punchout"),
    ExperimentSpec("res_spec_flux", "Res Spec Flux", "Resonator", "QickworkspaceV2.experiments.resonator.ResonatorSpecFlux", "s002c_res_spec_ge_flux.ResonatorSpecFlux"),
    ExperimentSpec("qubit_spec_ge", "Qubit Spec", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.QubitSpec", "s003_qubit_spec_ge.QubitSpec"),
    ExperimentSpec("qubit_flux_spec_ge", "Qubit Flux Spec", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.QubitSpecFlux", "s003a_qubit_flux_spec_ge.QubitSpecFlux"),
    ExperimentSpec("time_rabi_ge", "Time Rabi", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.TimeRabi", "s004_time_rabi_ge.TimeRabi"),
    ExperimentSpec("power_rabi_ge", "Power Rabi", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.PowerRabi", "s005_power_rabi_ge.PowerRabi"),
    ExperimentSpec("drag", "DRAG", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.DragCalibration", "s005a_drag.DragCalibration", supports_liveplot=False),
    ExperimentSpec("aae", "AAE", "Qubit GE", "QickworkspaceV2.experiments.qubit_ge.AAE", "s005a_AAE.PowerRabiChevron", supports_liveplot=False),
    ExperimentSpec("ramsey_ge", "Ramsey", "Coherence", "QickworkspaceV2.experiments.coherence.Ramsey", "s006_Ramsey_ge.Ramsey"),
    ExperimentSpec("spin_echo_ge", "Spin Echo", "Coherence", "QickworkspaceV2.experiments.coherence.SpinEcho", "s007_SpinEcho_ge.SpinEcho"),
    ExperimentSpec("t1_ge", "T1", "Coherence", "QickworkspaceV2.experiments.coherence.T1", "s008_T1_ge.T1"),
    ExperimentSpec("res_spec_ef", "Res Spec EF", "Qubit EF", "QickworkspaceV2.experiments.qubit_ef.ResonatorSpec_ef", "s009_res_spec_ef.ResonatorSpec_ef"),
    ExperimentSpec("qubit_spec_ef", "Qubit Spec EF", "Qubit EF", "QickworkspaceV2.experiments.qubit_ef.QubitSpecEf", "s010_qubit_spec_ef.QubitSpec_ef"),
    ExperimentSpec("power_rabi_ef", "Power Rabi EF", "Qubit EF", "QickworkspaceV2.experiments.qubit_ef.PowerRabiEf", "s011_power_rabi_ef.PowerRabi_ef"),
    ExperimentSpec("ramsey_ef", "Ramsey EF", "Qubit EF", "QickworkspaceV2.experiments.coherence.RamseyEf", "s012_Ramsey_ef.Ramsey_ef"),
    ExperimentSpec("t1_ef", "T1 EF", "Qubit EF", "QickworkspaceV2.experiments.coherence.T1Ef", "s013_T1_ef.T1_ef"),
    ExperimentSpec("allxy", "AllXY", "Advanced", "QickworkspaceV2.experiments.characterization.AllXY", "s014_AllXY.AllXY", supports_liveplot=False),
    ExperimentSpec("single_shot", "SingleShot", "Advanced", "QickworkspaceV2.experiments.setup.SingleShot_gef", "s000_SingleShot_prog.SingleShot_gef", supports_liveplot=False),
    ExperimentSpec("single_shot_opt", "SingleShot Opt", "Advanced", "QickworkspaceV2.experiments.setup.SingleShot_ge_opt", "s000_SingleShot_opt.SingleShot_ge_opt", supports_liveplot=False),
    ExperimentSpec("qubit_temp", "Qubit Temp", "Advanced", "QickworkspaceV2.experiments.qubit_ef.QubitTemp", "s013_qubit_temp.QubitTemperatureEf"),
    ExperimentSpec("ac_stark", "AC Stark", "Advanced", "QickworkspaceV2.experiments.coherence.ACStark", "s006_ac_stark.AcStarkCalib"),
    ExperimentSpec("rb", "Single Qubit RB", "RB", "QickworkspaceV2.experiments.characterization.RandomizedBenchmarking", "s015_Single_qubit_RB.RandomizedBenchmarking", supports_liveplot=False),
    ExperimentSpec("auto_rb", "Auto RB", "RB", "QickworkspaceV2.experiments.characterization.AutoRB", "s015_Auto_RB.AutoRB", supports_liveplot=False),
    ExperimentSpec("rb_asm", "RB ASM", "RB", "QickworkspaceV2.experiments.characterization.RandomizedBenchmarkingAsm", "s015_RB_asm.RandomizedBenchmarkingAsm", supports_liveplot=False),
    ExperimentSpec("tomography", "State Tomography", "Tomography", "QickworkspaceV2.experiments.characterization.Tomography", "s016_state_tomography.Tomography", supports_liveplot=False),
)

_BY_ID = {spec.id: spec for spec in EXPERIMENT_SPECS}
_BY_LEGACY = {spec.legacy_id: spec for spec in EXPERIMENT_SPECS if spec.legacy_id}
_BY_CLASS_PATH = {spec.class_path: spec for spec in EXPERIMENT_SPECS}


def experiment_schema() -> dict[str, Any]:
    """Return a JSON-serialisable experiment catalog."""
    categories: dict[str, list[dict[str, Any]]] = {}
    for spec in EXPERIMENT_SPECS:
        categories.setdefault(spec.category, []).append(asdict(spec))
    return {"categories": categories, "experiments": [asdict(spec) for spec in EXPERIMENT_SPECS]}


def resolve_experiment_spec(identifier: str) -> ExperimentSpec:
    """Resolve public id, legacy GUI id, or full class path to a spec."""
    spec = _BY_ID.get(identifier) or _BY_LEGACY.get(identifier) or _BY_CLASS_PATH.get(identifier)
    if spec is None:
        valid = sorted(_BY_ID)
        raise ValueError(f"Unknown experiment {identifier!r}. Valid ids: {valid}")
    return spec


def canonical_class_path(identifier: str) -> str:
    return resolve_experiment_spec(identifier).class_path


def resolve_experiment_class(identifier: str):
    class_path = canonical_class_path(identifier)
    module_name, class_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def fit_updates_from_result(result) -> dict[str, Any]:
    """Convert known fit_result entries into config update suggestions."""
    fit_result = getattr(result, "fit_result", {}) or {}
    experiment_type = (getattr(result, "experiment_type", "") or "").lower()

    def value_of(name):
        item = fit_result.get(name)
        if item is None:
            return None
        if isinstance(item, (tuple, list)):
            item = item[0]
        try:
            if isinstance(item, np.generic):
                item = item.item()
            return float(item)
        except (TypeError, ValueError):
            return item

    mapping = {
        "f_res[MHz]": "res_freq_ge",
        "corrected_freq_MHz": "qb_freq_ge",
        "pi_gain": "pi_gain_ge",
        "pi2_gain": "pi2_gain_ge",
        "pi_length_us": "qb_flat_top_length_ge",
        "T1_us": "T1_ge",
        "T2r_us": "T2r_ge",
        "T2e_us": "T2e_ge",
        "optimal_alpha": "drag_alpha",
        "optimal_gain": "pi_gain_ge",
        "trig_time_us": "trig_time",
        "linewidth_MHz": "kappa",
        "kappa_MHz": "kappa",
    }
    mapping["f0_MHz"] = "res_freq_ge" if "res" in experiment_type and "qubit" not in experiment_type else "qb_freq_ge"

    updates = {}
    for result_key, config_key in mapping.items():
        value = value_of(result_key)
        if value is None:
            continue
        updates[config_key] = value
        if config_key == "qb_freq_ge":
            updates.setdefault("qb_mixer", value)

    f0_ghz = value_of("f0_GHz")
    if f0_ghz is not None:
        updates["res_freq_ge"] = f0_ghz * 1000.0 if f0_ghz < 100 else f0_ghz
    return updates
