# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

`reconstruct/` is a fully self-contained automated quantum calibration framework built on QICK (Quantum Instruction Control Kit). It replaces the original `qick_workspace` codebase. **No file inside `reconstruct/` may import from `qick_workspace`** — all shared code is maintained as standalone copies inside `reconstruct/tools/`, `reconstruct/plotter/`, and `reconstruct/instruments/`.

## Verifying the Package

```bash
python -c "import reconstruct; print(reconstruct.__version__)"
python -c "from reconstruct import QICKBackend; b = QICKBackend.from_pyro4('IP', PORT)"
```

There are no automated test suites or lint scripts. Validation is notebook-driven + hardware-in-the-loop. Use `SimulatedBackend` for offline smoke testing.

## Architecture

The framework is layered:

```
experiments/ → core/ → backend/ → (hardware)
                ↓
           analysis/ + tools/ + plotter/
                ↓
          calibration/ (store, graph, pipeline)
                ↓
           service/ (FastAPI REST)
```

### Core Run/Fit/Save Flow

Every experiment inherits from `BaseExperiment` (`core/base_experiment.py`). The subclass contract:

- Set class attributes: `EXPT_NAME`, `TAG`, `X_LABEL`, `TITLE_PREFIX`, `SWEEP_KEYS_TO_REMOVE`
- Optionally bind `Analysis = SomeAnalysis` (a `BaseAnalysis` subclass)
- Override `_create_program()` → return a `BaseProgram` instance
- Override `_extract_sweep_axis(prog)` → return the x-axis numpy array
- Optionally override `_post_fit(x_vals)` → run fitting, populate `self.fit_params` / `self.fit_errors`

`run(py_avg, iq_process="abs")` compiles the QICK program, streams live plots, acquires IQ data, calls `_post_fit`, runs `Analysis`, and returns `ExperimentData`.

`ExperimentData` supports backward-compatible tuple unpacking (`fit_params, err = result`) and scalar coercion (`float(result)`), so old notebook code still works unchanged.

### QICK Programs

`BaseProgram` (`core/base_program.py`) wraps `AveragerProgramV2`. Key helper methods:
- `setup_resonator(cfg, prefix="ge")` / `setup_qubit_gen(cfg, prefix)`
- `setup_qb_pulse(cfg, shape, name, gain_key)` — declares a named pulse
- `setup_standard_gates(cfg, prefix)` — registers `x180_ge`, `y180_ge`, `x90_ge`, etc.
- `apply_cool(cfg)` + `cooling_body()` — active-reset cooling
- `measure(cfg)` — fire readout, collect ADC

In `_body()`, call `self.pulse(ch=..., name=..., t=0)` then `self.delay_auto(dt)` then `self.measure(cfg)`.

### Configuration

`ExperimentConfig` (`config/system_cfg.py`) wraps the static `config_list` dict and flattens nested sub-dicts (`ch`, `res`, `qb`, `cooling`) into a single flat dict per qubit. Always call `.get_qubit("Q1")` to get a **copy** — mutation is safe and expected.

Critical flat config keys: `ro_ch`, `res_ch`, `qb_ch`, `reps`, `relax_delay`, `steps`, `res_freq_ge`, `qb_freq_ge`, `pi_gain_ge`, `qb_mixer`.

### Calibration Store

`CalibrationStore` (`calibration/store.py`) persists parameters as timestamped JSON. It is the live replacement for editing config dicts by hand:

```python
store = CalibrationStore("cal_Q1.json")
store.set("Q1", "qb_freq_ge", 4500.0)          # auto-saves
store.get("Q1", "qb_freq_ge")                   # → 4500.0
store.is_stale("Q1", "qb_freq_ge", max_age_hours=24)
store.update_from_dict("Q1", {...})
```

`AutoCalibrate` (`calibration/pipeline.py`) runs a 7-step ge-transition pipeline (res_spec → qubit_spec → power_rabi → ramsey → spin_echo → t1 → ss_opt) and writes results back into both `ExperimentConfig` and `CalibrationStore` automatically.

### IQ Data Convention

Raw hardware data is always complex. `iq_process` controls what gets stored in `ExperimentData.y_axis`:
- `"abs"` (default) → `np.abs(iq)` — works without readout optimization
- `"real"` → `np.real(iq)` — use after single-shot optimization rotates IQ to I-axis

### Import Paths Inside `reconstruct/`

Depth-relative dot-count from package root:
- `experiments/<family>/` → 3 levels deep → `from ...core.base_program import BaseProgram`
- `analysis/` or `plotter/` → 2 levels → `from ..tools.fitting import fitlor`
- `calibration/` → 2 levels → `from ..config.system_cfg import ExperimentConfig`

Never use `from qick_workspace...` anywhere in `reconstruct/`. The fallback for `abcd_rf_fit` is `from ...tools.abcd_rf_fit.abcd_rf_fit import analyze`, not the qick_workspace path.

### Adding a New Experiment

1. Create `reconstruct/experiments/<family>/<name>.py`
2. Define `class MyProgram(BaseProgram)` with `_initialize` + `_body`
3. Define `class MyExperiment(BaseExperiment)` with the class attributes above
4. Export from `reconstruct/experiments/<family>/__init__.py`
5. Optionally add an `Analysis` subclass in `reconstruct/analysis/`

### REST Service

```bash
uvicorn reconstruct.service.api:app --host 0.0.0.0 --port 8000
```

Jobs run async in background threads. Key endpoints: `POST /experiments/run`, `GET /experiments/{id}/result`, `POST /calibrate/{qubit}/run`.
