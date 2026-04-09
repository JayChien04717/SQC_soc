# CLAUDE.md — qick_workspace Coding Conventions

This file governs all code under `qick_workspace/` **except** the legacy `scrip/` directory.

---

## 1. Directory layout

```
qick_workspace/
├── newscrip/        # Experiment scripts (s000_*.py)
├── plotter/         # Live plotting and final-plot utilities
├── tools/           # Fitting, hardware drivers, system config
└── CLAUDE.md
```

---

## 2. Experiment file naming

| Pattern | Example |
|---------|---------|
| `s{NNN}_{description}_{transition}.py` | `s003_qubit_spec_ge.py` |

- `NNN` is zero-padded to 3 digits.
- Suffix indicates transition: `ge`, `ef`, `gef`, `flux`, etc.
- Variant letters (`a`, `b`, `c`) go before the transition: `s002b_res_punchout_ge.py`.

---

## 3. Session initialisation

Call **once** at the top of every notebook, right after `make_proxy`:

```python
datafile = r"D:\Labber_Data\Jay\..."
soc, soccfg = make_proxy(ns_host=..., ns_port=..., proxy_name=...)
BaseExperiment.setup(soc, soccfg, datafile)
```

To change the save directory mid-session:
```python
BaseExperiment.set_data_path(r"D:\other\path")
```

`BaseExperiment.setup()` stores `soc`, `soccfg`, and `data_path` as class-level
state shared by **all** experiment classes (both `BaseExperiment` subclasses and
standalone classes).  `data_path` overrides `DATA_PATH` in `system_cfg.py`.

---

## 4. Class structure

### 4a. Standard experiments — inherit `BaseExperiment`

```python
class MyExperiment(BaseExperiment):
    # ── Required metadata ──────────────────────────────────────
    EXPT_NAME   = "s003_qubit_spec_ge"
    TAG         = "TwoTone"
    X_LABEL     = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ge Spectrum"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6          # MHz → Hz

    # ── IQ processing mode (class-level default) ───────────────
    # "abs"  → np.abs(iqdata)   default; no readout optimisation needed
    # "real" → np.real(iqdata)  set this after readout optimisation
    IQ_PROCESS = "abs"

    def _create_program(self): ...
    def _extract_sweep_axis(self, prog): ...
    def _post_fit(self, x_vals): ...          # optional
```

Instantiate with config only (no soc/soccfg):
```python
expt = MyExperiment(run_cfg)
expt.run(py_avg=10)                       # uses class IQ_PROCESS
expt.run(py_avg=10, iq_process="real")    # override for this run
```

**Rules:**
- `__init__` takes only `config` — soc/soccfg come from the session.
- Override `iq_process` in `run()` rather than patching the class attribute.
- Never call `prog.acquire()` directly inside `BaseExperiment` subclasses.
- `_post_fit` must return the primary numerical result (e.g. qubit frequency), or `None`.

### 4b. Standalone experiments (AllXY, RB, SingleShot)

These do **not** inherit `BaseExperiment` because they have custom multi-program or
multi-trigger data flows.  They follow the same session pattern:

```python
class MyStandalone:
    def __init__(self, config):
        from .base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(...) first.")
        self.soc     = BaseExperiment._soc
        self.soccfg  = BaseExperiment._soccfg
        self.cfg     = config

    def run(self, ..., iq_process="abs"): ...   # store self._iq_process
    def plot(self, ...): ...                     # reads self._iq_process
    def saveLabber(self, qb_idx, ...): ...
```

In `saveLabber`, always resolve the path via the session:
```python
from .base_experiment import BaseExperiment
save_dir = BaseExperiment._data_path or DATA_PATH
file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
```

Pass `iq_process` in `run()` and store it as `self._iq_process`.  All amplitude
conversions inside `plot()` and internal helpers must use:

```python
_proc = np.real if self._iq_process == "real" else np.abs
```

---

## 5. Config dict conventions

| Key | Type | Description |
|-----|------|-------------|
| `ro_ch` | int | Readout ADC channel |
| `res_ch` | int | Resonator DAC channel |
| `qb_ch` | int | Qubit drive DAC channel (ge) |
| `qb_ch_ef` | int | Qubit drive DAC channel (ef) |
| `reps` | int | Hardware averages per acquire call |
| `relax_delay` | float | Qubit reset delay (µs) |
| `shots` | int | Single-shot count (SingleShot only) |
| `steps` | int | Sweep points |
| `res_freq_ge` | float | Resonator frequency (MHz) |
| `qb_freq_ge` | float | Qubit ge frequency (MHz) |
| `pi_gain_ge` | float | π-pulse gain (DAC units) |
| `pi_gain_ef` | float | ef π-pulse gain (DAC units) |

Pulse names inside programs: `res_pulse`, `qb_pulse`, `qb_ge_pulse`, `qb_ef_pulse`.

---

## 6. IQ data conventions

- Raw hardware data is always accumulated as **complex**: `iq_list[0][0].dot([1, 1j])`.
- `self.iqdata` (on `BaseExperiment`) stores the **complex** averaged IQ array.
- Never convert to real/abs before saving — always save raw complex data.
- Conversion to scalar (for plotting and fitting) is controlled by `IQ_PROCESS` / `iq_process`:
  - `"abs"`  → `np.abs(data)`   — works without readout optimisation
  - `"real"` → `np.real(data)`  — use after readout optimisation (max SNR on I-axis)

---

## 7. Plotting

| Use case | Function | Location |
|----------|----------|----------|
| Live acquisition loop | `liveplotfun(...)` | `plotter/liveplot.py` |
| Post-acquisition fit panel | `plot_final(...)` | `plotter/plot_utils.py` |
| Single-shot histogram + confusion matrix | `hist(data)` / `general_hist(...)` | `newscrip/singleshot_utils.py` |

**Do not** import single-shot histogram utilities from `plotter/plot_utils.py` — they were removed.  Use `singleshot_utils` instead.

---

## 8. Single-shot analysis

Always use `newscrip/singleshot_utils.py`:

```python
from .singleshot_utils import hist, general_hist

# Two-state
results = hist(data)          # data = {"Ig", "Qg", "Ie", "Qe"}

# Three-state
results = hist(data)          # data += {"If", "Qf"}

# Returns: [fids, thresholds, angle_deg, conf_matrix_pct]
```

**Conventions:**
- Fitting: Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`), fit on combined I projections.
- Fidelity: mean of confusion-matrix diagonal — generalises correctly to any number of states.
- Thresholds: derived from GMM posterior crossings, not histogram CDF contrast.
- Do **not** add `fit`, `gauss_overlap`, or `fitparams` parameters — the GMM approach replaces all of them.

---

## 9. Fitting

All fit functions live in `tools/fitting.py`.  Do **not** call `scipy.optimize.curve_fit`
directly in experiment scripts.

```python
from ..tools.fitting import fitlor, lorfunc          # Lorentzian
from ..tools.fitting import fitdecaysin, decaysin    # Decaying sinusoid
from ..tools.fitting import fitrb, rb_func           # RB decay
```

---

## 10. Data saving

Always use `hdf5_generator` from `tools/system_tool.py`.  Resolve the save
directory via the session (never hard-code `DATA_PATH` directly in experiments):

```python
from .base_experiment import BaseExperiment
from ..tools.system_cfg import DATA_PATH

save_dir  = BaseExperiment._data_path or DATA_PATH
file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)

hdf5_generator(
    filepath=file_path,
    x_info={"name": "Frequency", "unit": "Hz", "values": freqs_hz},
    z_info={"name": "Signal",    "unit": "ADC unit", "values": self.iqdata},
    comment=str(dict_val),
    tag=self.TAG,
)
```

---

## 11. What NOT to do

- Do not pass `soc` or `soccfg` to experiment `__init__` — they come from the session.
- Do not instantiate any experiment class before calling `BaseExperiment.setup()`.
- Do not hard-code `DATA_PATH` in `saveLabber` — always use `BaseExperiment._data_path or DATA_PATH`.
- Do not patch `IQ_PROCESS` on the class — pass `iq_process=` to `run()` instead.
- Do not define `plot_hist`, `gaussian`, `hist`, or single-shot helpers outside `singleshot_utils.py`.
- Do not duplicate functions between files — if a utility is needed in more than one place, add it to the appropriate shared module.
- Do not call `np.abs` or `np.real` on `iqdata` in `run()` — always store raw complex data and convert at plot/fit time via `iq_process`.
- Do not add `fit`, `gauss_overlap`, `plotoverlap`, `fitparams`, or `check_qnd` parameters to singleshot functions.
- Do not import from the legacy `scrip/` directory.
