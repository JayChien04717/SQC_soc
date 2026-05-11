# QICK Qubit Measurement GUI — Product Design Plan

**Role context:** This document is written from the dual perspective of a superconducting qubit measurement engineer (who knows what the physics demands) and a product manager (who knows what ships vs. what sinks). Every decision here is grounded in how we actually use the hardware, not in how we wish we used it.

---

## 1. Product Goal

A **single-window desktop application** that lets a lab member go from power-on to a calibrated qubit in one session, without ever touching a terminal or a Jupyter notebook. The output is correct science: reliable fits, properly named HDF5 files, and a qubit parameter set ready for gate experiments.

Secondary goal: the app is **self-documenting** — every run is logged with its parameters, every file is searchable, and no result is silently lost.

---

## 2. User Persona

| Persona | Description | Pain today |
|---|---|---|
| **Lab student** | First year PhD, knows the physics, hates debugging Python imports | Runs the wrong script, wrong config, saves over yesterday's data |
| **Lab engineer** | Knows the system well, runs 50 experiments a day | Wastes time copy-pasting notebook cells, no live progress |
| **PI / visitor** | Wants a summary of qubit parameters, not raw IQ data | Has to open a notebook to see T1/T2/frequency |

Primary target: **lab engineer**. If the engineer is happy, the student can follow, and the PI gets their summary.

---

## 3. What the App Must Do (Core — non-negotiable)

### 3.1 Hardware Panel
- Connect via Pyro4 `make_proxy(ns_host, ns_port, proxy_name)` → real `soc, soccfg`
- Show board info (first 6 lines of `str(soccfg)`) in log after connect
- Call `BaseExperiment.setup(soc, soccfg, data_path)` immediately
- LED: orange = disconnected, yellow = connecting, green = connected
- Disconnect button that cleanly releases the proxy
- **Fields:** NS host, NS port (default 8888), proxy name (default "myqick")

### 3.2 Configuration Panel
- Load a YAML config file (browse + drag-drop)
- Parse with `ExperimentConfig(config_list)` into `config_all`
- Show parsed qubit list (Q0, Q1, Q2 …) as radio buttons or tab selector
- Active qubit selection drives `config_all.get_qubit("Q{n}")` for all runs
- Save current parameter edits back to YAML
- Data path field: where HDF5 files go (persisted in QSettings)

### 3.3 Experiment Execution Panel
- Category + experiment combo (current registry is correct)
- Parameter form: common params (py_avg, reps, relax_delay, steps, span) + experiment-specific overrides
- **Run** (Ctrl+R) → `AcquireWorker` thread → real experiment class → `.run()` → `.saveLabber()`
- **Stop** (Esc) → `worker.terminate()` + graceful cleanup
- Live progress: indeterminate spinner while running, "Ready" when done
- State label: "● Running QubitSpec…" with experiment name

### 3.4 Live Plot Panel
- matplotlib canvas, dark theme, NavigationToolbar
- Channel selector: mag / phase / avgi / avgq
- **Auto-update every sweep point** (not just at the end) — requires `data_ready` signal emitted per-point in worker
- Autoscale checkbox
- Export button (PNG/PDF/SVG)
- Clear button

### 3.5 Log Panel
- Timestamped, color-coded (info=grey, success=green, warn=orange, error=red)
- Shows: connection events, experiment start/stop, fit results, save paths, errors
- 500-line rolling buffer
- "Clear Log" menu item

### 3.6 Data Browser (separate process, Ctrl+B)
- Tree: date → file list with (name, size, timestamp)
- File count per date shown in tree
- Preview on selection: auto-detect 1D or 2D, plot accordingly
- Plot modes: Scatter 1D, Imshow 2D, Hist 1D (with KDE), Hist 2D (IQ cloud)
- Channel selector: avgi / avgq / mag / phase
- Export plot button (Ctrl+S)
- Delete file with confirmation (Del key)
- Keyboard shortcuts: F5 refresh, Del delete, Ctrl+S export
- Independent QProcess — browser stays open if measurement window crashes

---

## 4. What the App Should Do (Important — high value, not blocking v1)

### 4.1 Qubit Parameter Dashboard
A persistent panel (or tab) showing the current "known good" values for the active qubit:
- `f_ge`, `f_ef`, `T1`, `T2*`, `T2_echo`
- `pi_gain`, `pi_half_gain`, `pi_ef_gain`
- `resonator_freq`, `readout_length`, `adc_trig_offset`
- Values are populated by fit results after each experiment, stored in a per-qubit dict in memory
- "Copy to clipboard as YAML" button so the user can paste into their config

**Why important:** Right now these values live scattered across notebook outputs. A running tally is essential for a productive calibration session.

### 4.2 Calibration Sequence Runner
A "one-click calibration" wizard that runs a fixed sequence:
1. Time of Flight → sets `adc_trig_offset`
2. Res Spec GE → finds `resonator_freq`
3. Qubit Spec GE → finds `f_ge`
4. Power Rabi → sets `pi_gain`
5. T1 → measures T1
6. Ramsey → measures T2*, corrects `f_ge`

Each step: auto-fits, writes result to the parameter dashboard, proceeds to next. Pause/resume. Skip individual steps. **This is the killer feature** for daily calibration.

### 4.3 Experiment-Specific Parameter Injection
When an experiment is selected, load the recommended parameters for the current qubit from the YAML config automatically — not just the generic `_COMMON_PARAMS` defaults. Example: selecting "Time Rabi" should pre-fill `start`, `stop`, `step` from `cfg["time_rabi"]` in the config.

### 4.4 Fit Results Overlay on Plot
After each experiment, if a fitter exists (e.g., `_post_fit` in the experiment class), overlay the fit curve on the live plot and annotate the extracted value (e.g., "f_ge = 4.821 GHz", "T1 = 42 µs"). This is already partially implemented in `base_experiment.py`.

### 4.5 Yokogawa Flux Bias Control
- Current field (mA) → `yoko.set_current(mA)` via a driver
- Sweep mode: set a range, run an experiment at each flux point (for flux-tunable qubits)
- Live 2D map: x = flux, y = frequency sweep, color = mag

---

## 5. What the App Might Do (Lower Priority — defer unless easy)

### 5.1 Multi-Qubit Support
Running the same experiment on Q0…Q3 in sequence, comparing results side-by-side. Not needed until multi-qubit gate experiments start. The current radio-button qubit selector is sufficient.

### 5.2 Automated Report Generation
After a calibration session, export a PDF summary with all fit results and plots. Nice for the PI, but can be done in a notebook for now.

### 5.3 Remote Monitoring
A web dashboard (Flask + Plotly Dash) that shows live experiment state from another machine. Useful for noisy lab environments where you don't want to be physically at the computer. Defer — adds significant complexity.

### 5.4 Experiment Scheduling
A queue system: submit multiple experiments, run overnight. Useful for long T1 sweeps or RB sequences. Not needed until the core loop is reliable.

### 5.5 Parameter History / Version Control
Track how `f_ge` changed over the past week. Interesting science, but a CSV log + notebook is good enough for now.

---

## 6. What the App Will NOT Do (Explicitly Cut)

| Feature | Why cut |
|---|---|
| Cloud sync / remote storage | Lab instruments are air-gapped or firewalled; adds security risk |
| Multi-user login | One person controls the instrument at a time; no conflict |
| In-app notebook editor | That's what Jupyter is for; don't rebuild it |
| Real-time collaboration | Complexity far exceeds the benefit in a 3-person lab |
| Waveform editor / pulse designer | QICK firmware handles this; expose via config YAML only |
| Drag-and-drop experiment ordering in tree | The calibration sequence covers 95% of ordering needs |
| Theming / color customization | One dark theme. Period. |
| Plugin system / extensibility framework | Over-engineering; new experiments = new Python file + one registry entry |

---

## 7. UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Menu: File | Tools | Help                                          │
├──────────────┬──────────────────────────────────┬──────────────────┤
│  SETUP       │         LIVE PLOT                │  EXPERIMENT      │
│  ─────────   │                                  │  ──────────      │
│  Connection  │   matplotlib canvas              │  Category        │
│    NS host   │   (dark, autoscale)              │  Experiment      │
│    Port      │                                  │  ──────────      │
│    Name      │   [Channel▼] [✓Autoscale]        │  Parameters      │
│    [Connect] │   [Clear] [Export…]              │  (scrollable     │
│    ● status  │                                  │   form)          │
│  ─────────   │   NavToolbar                     │  ──────────      │
│  Config      │                                  │  ● Idle          │
│    YAML path │                                  │  [▶ Run] [■ Stop]│
│    [Load]    │                                  │  [Save]          │
│    [Save]    │                                  │                  │
│    Data path │                                  │  QUBIT PARAMS    │
│  ─────────   │                                  │  ──────────      │
│  Qubit       │                                  │  f_ge: –         │
│    Q0 Q1 Q2  │                                  │  T1:   –         │
│    Q3        │                                  │  T2*:  –         │
│    Yoko(mA)  │                                  │  π gain: –       │
│              │                                  │  [Copy YAML]     │
├──────────────┴──────────────────────────────────┴──────────────────┤
│  LOG   [HH:MM:SS] message …                                 [Clear] │
└─────────────────────────────────────────────────────────────────────┘
  ● Connected  |  Q1  |  Ready                          [████░░░░░░]
```

**Key layout decisions:**
- Setup left, Experiment right — mirrors how you think: "am I connected?" → "what am I running?"
- Qubit Parameters below Experiment — naturally fills with data as calibration progresses
- Log at bottom, always visible — errors must never be hidden
- Central plot fills remaining space — this is the primary output
- Status bar carries only: connection LED, active qubit, state label, progress bar

---

## 8. Data Model

### HDF5 file structure (already established in `hdf5_generator`)
```
/{EXPT_NAME}_{qubit}_{NNN}.h5
  /x             — 1D sweep axis (float64)
  /y             — 2D sweep axis if applicable (float64)
  /data/
    avgi         — I quadrature (float64, shape [reps] or [Ny, Nx])
    avgq         — Q quadrature
    mag          — |IQ|
    phase        — angle(IQ) in degrees
  /config        — serialized YAML config as string attribute
  /meta
    .attrs["expt_name"]    — "s003_qubit_spec_ge"
    .attrs["qubit"]        — "Q1"
    .attrs["timestamp"]    — ISO8601
    .attrs["py_avg"]       — int
    .attrs["fit_result"]   — JSON string of fit outputs (f_ge, T1, etc.)
```

`/meta/fit_result` is the key addition — fits are stored in the file, not just plotted. This is what feeds the Qubit Parameter Dashboard in §4.1.

### Qubit state dict (in memory, lost on close — intentional)
```python
qubit_state: dict[str, dict] = {
    "Q0": {"f_ge": 4.821e9, "T1": 42e-6, "pi_gain": 14200, ...},
    "Q1": {...},
}
```
Populated by fit hooks. "Copy to YAML" serializes it. No database.

---

## 9. AcquireWorker — Real Implementation Contract

The current `AcquireWorker.run()` is a stub. The real implementation must:

```python
def run(self):
    try:
        # 1. Import the experiment class
        mod_name, cls_name = self.class_path.rsplit(".", 1)
        mod = importlib.import_module(f"qick_workspace.scrip.{mod_name}")
        ExptClass = getattr(mod, cls_name)

        # 2. Build run config from active qubit config
        run_cfg = self.config_all.get_qubit(self.qubit_label)
        run_cfg.update(self.params)          # GUI overrides

        # 3. Instantiate and run
        expt = ExptClass(run_cfg)
        # Emit per-point updates via a progress callback if the class supports it
        expt.run(py_avg=self.params["py_avg"])

        # 4. Emit data for live plot
        self.data_ready.emit(expt._sweep_vals_x, expt.iqdata,
                             ExptClass.X_LABEL, ExptClass.TITLE_PREFIX)

        # 5. Save
        expt.saveLabber(self.qubit_label, config_all=self.config_all)

        # 6. Emit fit results if available
        if hasattr(expt, "fit_result"):
            self.fit_ready.emit(self.qubit_label, expt.fit_result)

        self.log_message.emit("Done.", "success")
    except Exception as exc:
        self.log_message.emit(f"Error: {exc}", "error")
    finally:
        self.finished.emit()
```

`AcquireWorker` needs these constructor parameters added:
- `config_all` — the `ExperimentConfig` object (set after YAML load)
- `qubit_label` — "Q0", "Q1", etc. (from active qubit radio button)

New signal: `fit_ready = Signal(str, dict)` — carries qubit label + fit dict → updates Parameter Dashboard.

---

## 10. What Is Broken or Missing Right Now

| Issue | Severity | Fix |
|---|---|---|
| `AcquireWorker.run()` is a stub — no real experiment runs | **Critical** | Implement per §9 |
| No `config_all` passed to worker — qubit config never used | **Critical** | Pass after YAML load |
| Experiment-specific params not loaded from config | High | Load from `run_cfg` on experiment select |
| No fit results displayed or stored | High | Hook `fit_ready` signal to dashboard + HDF5 |
| `_COMMON_PARAMS` are generic defaults, not qubit-aware | High | Override from loaded config |
| `Save Config` button is unimplemented | Medium | `config_to_yaml()` call |
| `_on_save` in main_window is a log stub | Medium | Call `expt.saveLabber()` explicitly or confirm auto-save |
| Data browser launched as script path — breaks if CWD changes | Medium | Use `sys.executable -m gui.data_browser_app` |
| No Qubit Parameter Dashboard | Medium | Add to right panel below Experiment |
| Yoko driver not wired — spin box does nothing | Low | Wire after hardware confirmed present |

---

## 11. Development Phases

### Phase 1 — Make It Actually Run Experiments (1–2 weeks)
1. Wire `AcquireWorker` to real experiment classes (§9)
2. Pass `config_all` through the signal chain: YAML load → SetupPanel → MainWindow → Worker
3. Emit `data_ready` per sweep point (not just at end) — requires a liveplot callback in `BaseExperiment`
4. Implement `fit_ready` signal → log the fit result string

Deliverable: Can run QubitSpec, see live sweep, get T1 fit result in log.

### Phase 2 — Parameter Integrity (1 week)
5. Load experiment-specific params from `run_cfg` when experiment is selected
6. Implement "Save Config" button → `config_to_yaml()`
7. Write fit results into HDF5 `/meta/fit_result` attribute

Deliverable: Parameters are always correct and persisted.

### Phase 3 — Qubit Parameter Dashboard (1 week)
8. Add `QubitParamsPanel` to right dock below Experiment
9. Connect `fit_ready` signal → update dashboard fields
10. "Copy YAML" button

Deliverable: Running calibration session produces a live summary.

### Phase 4 — Calibration Sequence Runner (2 weeks)
11. `CalibrationWizard` dialog: ordered step list, checkboxes to skip
12. Runs steps sequentially, auto-populates config from each fit result
13. Pause / resume / re-run single step

Deliverable: Daily calibration in one click.

### Phase 5 — Polish (ongoing)
- Keyboard shortcut help overlay (press ?)
- Parameter tooltips showing units and typical ranges
- "What changed?" diff when loading a new config
- File drag-drop onto data browser tree

---

## 12. What Stays Simple On Purpose

- **One config file at a time.** No project system, no config history. Open YAML → run → done.
- **No undo.** Experiment parameters are ephemeral; config is a file you can version with git.
- **No in-app fitting controls.** Fits run automatically via `_post_fit`. If the fit fails, re-run with better parameters.
- **No custom color themes.** The dark theme is chosen for low-light lab environments. One theme, no settings page.
- **No tooltips on every widget.** Only where the value is non-obvious (e.g., `adc_trig_offset`, not `reps`).
- **Subprocess for data browser.** Not a tab, not a panel. Separate process = independent crash domain.

---

## 13. Success Criteria

The GUI is "done" when a lab member can:

1. Open the app, connect to the QICK board, load a config — in under 30 seconds
2. Run Time of Flight → Res Spec → Qubit Spec → Power Rabi → T1 without touching the terminal
3. See live sweep data updating during each run
4. Find yesterday's T1 file in the data browser and export the plot
5. Copy the current qubit parameters as YAML and paste them into a new config

If all five work reliably, the app is production-ready for a superconducting qubit lab.

---

## Appendix A. Adding a New Experiment to the GUI

The GUI works best when every experiment returns a clean `ExperimentData`. Add GUI-specific glue only when the experiment cannot follow the normal `BaseExperiment` liveplot flow.

### 1. Make the experiment importable

Place the experiment under `QickworkspaceV2/experiments/<family>/` and export it from that family's `__init__.py`.

For a normal sweep experiment, prefer the standard `BaseExperiment` contract:

```python
class MyExperiment(BaseExperiment):
    EXPT_NAME = "s017_MyExperiment_ge"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "My Experiment"

    def _create_program(self):
        return MyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return self.cfg["freq_pts"]
```

Special experiments can override `run()`, but should still return `ExperimentData`.

### 2. Return plottable `ExperimentData`

The GUI mainly needs:

```python
ExperimentData(
    experiment_type=self.EXPT_NAME,
    raw_iq=data,
    x_axis=x_vals,
    fit_result={...},
    config=dict(self.cfg),
)
```

Common result shapes:

| Experiment type | `raw_iq` shape | GUI plot |
| --- | --- | --- |
| 1D sweep | `(N,)` | line/scatter plus fit line |
| 2D sweep | `(Ny, Nx)` | pcolormesh |
| SingleShot | `(states, shots)` complex | IQ 2D histogram plus rotated-I histogram |
| AllXY | `(21,)` | gate-set x labels plus reference line |
| Tomography | `(3,)`, with `y_axis=rho_mle` | 3D density-matrix bar |

Do not store primary plot data only in `y_axis`; use `raw_iq` for the measured values. Use `y_axis` for a second sweep axis or extra structured data such as `rho_mle`.

### 3. Register the experiment

Edit `gui/panels/experiment_panel.py`.

Add a display entry:

```python
EXPERIMENT_REGISTRY = {
    "Qubit GE": [
        ("My Experiment", "s017_my_experiment.MyExperiment"),
    ],
}
```

Add parameter metadata using the same class path:

```python
"s017_my_experiment.MyExperiment": [
    ("py_avg", "int", 10, 1, 500, "Python-level averages"),
    ("reps", "int", 100, 100, 50000, "Hardware reps per point"),
    ("relax_delay", "float", 50.0, 0.1, 5000.0, "Relax delay (us)"),
    ("steps", "int", 101, 2, 2000, "Sweep points"),
]
```

Supported parameter kinds are `"int"`, `"float"`, `"bool"`, and `"combo"`.

### 4. Choose liveplot or special run mode

Normal `BaseExperiment` sweep experiments usually need no changes in `gui/main_window.py`; they use `_run_liveplot()`.

If the experiment owns a blocking multi-step flow, add it to `AcquireWorker._SPECIAL_EXPTS` and handle any custom `run()` arguments in `_run_special()`:

```python
elif cp == "s017_my_experiment.MyExperiment":
    result = expt.run(py_avg, my_option=self.params.get("my_option"))
```

Special runs usually update the plot only after completion unless the experiment provides its own streaming callback.

### 5. Add a custom plot only when needed

Most experiments should use the default `PlotPanel` behavior. Add a special renderer in `gui/panels/plot_panel.py` only for nonstandard plots such as AllXY gate labels, SingleShot histograms, or Tomography density matrices.

Pattern:

```python
if self._looks_like_my_experiment(self._iq, self._experiment_type or self._title):
    self._plot_my_experiment()
    return
```

When drawing reference lines, do not store them in `_fit_line`; `_draw_fit()` clears `_fit_line` before drawing final fit summaries.

### 6. Add mock data for GUI debug mode

Edit `gui/mock/runner.py` so the GUI can test the new experiment without QICK hardware:

```python
elif "myexperiment" in cls:
    x = np.linspace(-10, 10, 101)
    avgi = np.exp(-(x / 3) ** 2) * 500
    avgq = _noise(len(x), 10)
    xlabel = "Detuning (MHz)"
    title = "My Experiment"
```

### 7. Minimum checks

Run:

```bash
python -m py_compile gui/main_window.py gui/panels/experiment_panel.py gui/panels/plot_panel.py
python -m py_compile QickworkspaceV2/experiments/<family>/<name>.py
```

Manual checks:

- Experiment appears in the category/experiment combo
- Parameters have the right widget types and units
- Run logs start and complete
- Plot is not blank
- Fit summary appears when available
- Save result does not fail
- Config update changes only intended fit keys

---

## Appendix B. API Productization Direction

Long term, the GUI should become a thin client over a stable experiment service:

```text
GUI / Notebook / Web client
    -> QickworkspaceV2 API service
    -> Experiment job manager
    -> BaseExperiment / QICK hardware
    -> ExperimentData + CalibrationStore
```

Core service endpoints should cover:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status |
| `GET` | `/experiments/schema` | Experiment catalog and parameter metadata |
| `POST` | `/experiments/run` | Submit experiment job |
| `GET` | `/experiments/{job_id}/status` | Poll job status |
| `GET` | `/experiments/{job_id}/result` | Fetch `ExperimentData` |
| `POST` | `/experiments/{job_id}/stop` | Request graceful stop |
| `GET` | `/experiments/{job_id}/stream` | Live data/log stream |

Product rules:

- The GUI should not own hardware lifecycle forever.
- Every run should have a job id, config snapshot, status, result, and audit metadata.
- Experiment metadata should become schema-driven instead of hardcoded separately in the UI.
- Fit results should update config through preview/apply, not silent mutation.
- Persistent calibration values should go through `CalibrationStore`.
- Live data should stream from the service instead of being pulled from GUI-owned acquisition loops.
