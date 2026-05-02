# Reconstruct — Checkpoint Tracker

Resume from the first `[ ]` checkpoint if session is interrupted.

## Checkpoint Status

- [x] CP1 — Core layer (`core/experiment_data.py`, `base_analysis.py`, `base_experiment.py`, `base_program.py`, `composite.py`)
- [x] CP2 — Backend layer (`backend/base_backend.py`, `qick_backend.py`, `simulated_backend.py`)
- [x] CP3 — Infrastructure (tools re-exports, plotter re-exports, config, data, instruments)
- [x] CP4 — Analysis layer (`analysis/resonator.py`, `qubit.py`, `rb.py`)
- [x] CP5 — Experiments batch 1: Setup (s000–s001) + Resonator (s002–s002g)
- [x] CP6 — Experiments batch 2: QubitGE (s003–s005b) + Coherence (s006–s008)
- [x] CP7 — Experiments batch 3: QubitEF (s009–s013) + Characterization (s014–s016)
- [x] CP8 — Calibration layer (`calibration/store.py`, `graph.py`, `monitor.py`, `pipeline.py`)
- [x] CP9 — Service layer (`service/api.py` FastAPI REST)
- [x] CP10 — Integration (`reconstruct/__init__.py`, all sub-`__init__.py`, CHECKPOINT.md final)
- [x] CP11 — Independence refactor: zero live `qick_workspace` imports in any `.py` file

**All checkpoints complete. Package is fully self-contained. Ready for hardware testing.**

## Directory Layout

```text
reconstruct/
├── __init__.py              ← top-level public API
├── core/                    ← ExperimentData, BaseAnalysis, BaseExperiment, BaseProgram, Composite
├── backend/                 ← BaseBackend, QICKBackend, SimulatedBackend
├── tools/                   ← standalone copies (fitting, system_tool, scoring, abcd_rf_fit, electrical_length, rb_generator, …)
├── plotter/                 ← standalone copies (liveplot, plot_utils)
├── instruments/             ← standalone copies (YOKOGS200, SGS100A, MG3692C)
├── config/                  ← ExperimentConfig (CalibrationStore-aware)
├── data/                    ← save/load HDF5 + JSON serializer
├── analysis/                ← BaseAnalysis subclasses (one per experiment family)
├── experiments/
│   ├── setup/               s000 SingleShot, s001 TOF
│   ├── resonator/           s002 ResonatorSpec, s002b Punchout, s002c Flux, TWPA
│   ├── qubit_ge/            s003 QubitSpec, s004 TimeRabi, s005 PowerRabi, DRAG, AAE
│   ├── coherence/           s006 Ramsey, s007 SpinEcho, s008 T1
│   ├── qubit_ef/            s009 ResSpecEf, s010 QubitSpecEf, s011 PowerRabiEf, s012 RamseyEf, s013 T1Ef, QubitTemp
│   └── characterization/    s014 AllXY, s015 RB/AutoRB, s016 Tomography
├── calibration/             ← CalibrationStore, CalibrationGraph, CalibrationMonitor, AutoCalibrate
└── service/                 ← FastAPI REST API (create_app, app)
```

## Key Design Decisions

- `BaseExperiment.run()` returns `ExperimentData` (not raw tuples).
- `ExperimentData.__iter__` yields `(fit_params, fit_errors)` → backward-compat tuple unpacking.
- `ExperimentData.__float__` exposes `scalar_result` → backward-compat single-value returns.
- Tools/plotter/instruments are **standalone copies** — `reconstruct/` has zero live imports from `qick_workspace`.
- Each experiment class has an `Analysis` class attribute linking to its `BaseAnalysis` subclass.
- `CalibrationStore` replaces static `system_cfg` qubit dicts with live, timestamped entries.
- `BatchExperiment` composes multiple experiments into an ordered, dependency-aware pipeline.
- `AutoCalibrate` uses GP-guided Ramsey correction and Bayesian-optimised single-shot readout.
- FastAPI service enables remote experiment submission via REST (async background jobs).

## Hardware Test Checklist

Before running on hardware:

1. `python -c "import reconstruct; print(reconstruct.__version__)"` — verify import
2. `python -c "from reconstruct import QICKBackend; b = QICKBackend.from_pyro4(IP, PORT)"` — check Pyro4 connection
3. Run `SimulatedBackend` smoke tests first (no hardware required)
4. `ResonatorSpec` → `QubitSpec` → `PowerRabi` → `Ramsey` in order
5. Call `AutoCalibrate.run(skip=("spin_echo","t1","ss_opt"))` for a fast first pass
