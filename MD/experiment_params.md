# Experiment Parameters Reference
*Extracted from `new_single_qb_cal.ipynb` — ground truth for GUI parameter forms*

All sweep axes use `QickSweep1D(loop_name, start, stop)`.  
Config-level updates (`config_all.update(...)`) are applied before `run_cfg = config_all.get_qubit(qubit)`.

---

## Setup

### Time of Flight (TOF)
```python
run_cfg.update({
    "ro_length":      2.5,      # us — readout window length
    "res_length":     1.7,      # us — resonator drive length
    "res_gain_ge":    0.18,     # DAC — readout tone gain
    "res_ch":         0,        # readout channel index
    "cheek_e":        False,    # also check e state
    "check_f":        True,     # also check f state
    "relax_delay":    100,      # us
})
tof.run(1000)                   # reps, no py_avg
```
**After run:** `config_all.update("trig_time", 0.49, q_index=qubit)` — set manually from plot.

---

## Resonator (GE)

### Resonator Spec GE (OneTone)
```python
# Config update first
config_all.update("res.res_gain_ge", 0.05, q_index=qubit)
run_cfg = config_all.get_qubit(qubit)

START_FREQ = res_freq_ge - 10   # MHz — center - span
STOP_FREQ  = res_freq_ge + 10   # MHz — center + span
STEPS = 101

run_cfg.update([
    ("steps",      STEPS),
    ("res_freq_ge", QickSweep1D("freqloop", START_FREQ, STOP_FREQ)),
    ("relax_delay", 0),
])
fres = onetone.run(py_avg=5, solve_type="hm")
config_all.update("res_freq_ge", round(fres[0]/1e6, 4), q_index=qubit)
```
**GUI needs:** `res_freq_ge` (center), span (default ±10 MHz), steps, res_gain_ge, relax_delay.

---

### Punchout (Resonator Gain × Frequency 2D)
```python
START_FREQ = res_freq_ge - 20;  STOP_FREQ  = res_freq_ge + 20   # MHz
STEPS_freq = 101
START_gain = 0.01;  STOP_gain = 0.3;  STEPS_gain = 11

run_cfg.update([
    ("f_steps",    STEPS_freq),
    ("res_freq_ge", QickSweep1D("freqloop", START_FREQ, STOP_FREQ)),
    ("g_steps",    STEPS_gain),
    ("res_gain_ge", QickSweep1D("gainloop", START_gain, STOP_gain)),
    ("relax_delay", 0),
])
punchout.run(py_avg=100)
```
**GUI needs:** freq center/span/steps, gain start/stop/steps, relax_delay.

---

## Qubit GE

### Qubit Spec GE
```python
center = 3620           # MHz — LO/mixer center, roughly where qubit is
SPAN   = 50             # MHz — half-span each side
START_FREQ = center - SPAN;  STOP_FREQ = center + SPAN
STEPS = 100

run_cfg.update([
    ("steps",                STEPS),
    ("qb_freq_ge",           QickSweep1D("freqloop", START_FREQ, STOP_FREQ)),
    ("qb_mixer",             center),          # LO frequency
    ("qb_gain_ge",           0.1),             # DAC gain
    ("qb_flat_top_length_ge", 1),              # us — flat-top pulse length
    ("relax_delay",          1),               # us
    ("nqz_qb",               2),               # Nyquist zone
])
f_ge = spectrum_ge.run(5)
config_all.update("qb_freq_ge", f_ge, q_index=qubit)
config_all.update("qb_mixer",   f_ge, q_index=qubit)
```
**GUI needs:** center freq, span, steps, qb_gain_ge, qb_flat_top_length_ge, relax_delay, nqz_qb.

---

### Qubit Flux Spec (2D: freq × flux)
```python
center = 5970;  SPAN = 50;  STEPS = 201
START_GAIN = -1;  STOP_GAIN = 1;  STEPS_GAIN = 11

config_all.update("qb_ch", 2, q_index=qubit)
run_cfg.update([
    ("steps",                STEPS),
    ("qb_freq_ge",           QickSweep1D("freqloop", center-SPAN, center+SPAN)),
    ("qb_mixer",             center),
    ("qb_gain_ge",           0.3),
    ("qb_flat_top_length_ge", 0.5),
    ("relax_delay",          1),
    ("nqz_qb",               2),
    ("res_length",           3),
    ("ro_length",            2.478),
    ("res_gain_ge",          0.95),
    ("flux_ch",              15),
    ("flux_gain",            QickSweep1D("fluxloop", START_GAIN, STOP_GAIN)),
    ("steps_flux",           STEPS_GAIN),
    ("flux_length",          0.7),             # us — flux pulse duration
])
spectrumff_ge.run(500, swap_xy=True)
```
**GUI needs:** freq center/span/steps, qb_gain_ge, qb_flat_top_length_ge, flux_ch, flux gain start/stop/steps, flux_length.

---

### Time Rabi GE
```python
# Config-level (persisted)
config_all.update("sigma_ge",    0.01, q_index=qubit)   # us — Gaussian sigma
config_all.update("relax_delay", 50,   q_index=qubit)

run_cfg.update([
    ("cooling",              False),
    ("qb_gain_ge",           0.5),
    ("qb_flat_top_length_ge", QickSweep1D("lenloop", 0.01, 0.3)),  # us sweep
    ("steps",                81),
])
Trabi.run(100)
```
**GUI needs:** sigma_ge, qb_gain_ge, length start/stop (default 0.01→0.3 µs), steps, relax_delay.

---

### Power Rabi GE
```python
# Config-level
config_all.update("sigma_ge",    0.01, q_index=qubit)
config_all.update("nqz_qb",      2,    q_index=qubit)
config_all.update("relax_delay", 200,  q_index=qubit)

START_GAIN = 0.0;  STOP_GAIN = 1.0;  STEPS = 100

run_cfg.update([
    ("steps",      STEPS),
    ("qb_gain_ge", QickSweep1D("gainloop", START_GAIN, STOP_GAIN)),
])
pi_gain, pi2_gain = prabi.run(5)
config_all.update("pi_gain_ge",  pi_gain,  q_index=qubit)
config_all.update("pi2_gain_ge", pi2_gain, q_index=qubit)
```
**GUI needs:** sigma_ge, gain start/stop/steps, relax_delay, nqz_qb.

---

### DRAG Calibration
```python
run_cfg.update([
    ("alpha_start", -1),
    ("alpha_stop",  1.5),
    ("alpha_steps", 41),
    ("iter_start",  2),
    ("iter_stop",   20),
    ("iter_step",   2),
])
fit_results = drag_cal.run(py_avg=5)
config_all.update("drag_alpha", fit_results['optimal_alpha'], q_index=qubit)
```
**GUI needs:** alpha_start, alpha_stop, alpha_steps, iter_start, iter_stop, iter_step.

---

### AAE (Amplified Amplitude Error / Power Rabi Chevron)
```python
focus = run_cfg['pi_gain_ge']   # centered on current pi_gain

run_cfg.update([
    ("steps",      100),
    ("qb_gain_ge", QickSweep1D("gainloop", focus-0.1, focus+0.1)),
    ("iter_start", 1),
    ("iter_stop",  50),
    ("iter_step",  7),
])
result = prabichevron.run(10)
config_all.update("pi_gain_ge",  result,   q_index=qubit)
config_all.update("pi2_gain_ge", result/2, q_index=qubit)
```
**GUI needs:** gain half-span (±0.1 around pi_gain_ge), steps, iter_start/stop/step.

---

## Coherence

### Ramsey GE
```python
START_TIME = 0.0;  STOP_TIME = 2;  STEPS = 100   # us

run_cfg.update([
    ("steps",       STEPS),
    ("wait_time",   QickSweep1D("waitloop", START_TIME, STOP_TIME)),
    ("ramsey_freq", 2),    # MHz — artificial detuning
])
t2r.run(20)
# After run:
config_all.update("qb_freq_ge", t2r.correct_detune(), qubit)
```
**GUI needs:** wait_time start/stop (default 0→2 µs), steps, ramsey_freq (default 2 MHz).

---

### Spin Echo GE
```python
START_TIME = 0.0;  STOP_TIME = 150;  STEPS = 100   # us

run_cfg.update([
    ("steps",       STEPS),
    ("wait_time",   QickSweep1D("waitloop", START_TIME, STOP_TIME)),
    ("ramsey_freq", 0.05),    # MHz — small detuning for echo fringes
])
t2e.run(100)
```
**GUI needs:** wait_time start/stop (default 0→150 µs), steps, ramsey_freq (default 0.05 MHz).

---

### T1 GE
```python
START_TIME = 0.0;  STOP_TIME = 400;  STEPS = 100   # us

run_cfg.update([
    ("steps",     STEPS),
    ("wait_time", QickSweep1D("waitloop", START_TIME, STOP_TIME)),
])
t1.run(50)
```
**GUI needs:** wait_time start/stop (default 0→400 µs), steps. No ramsey_freq needed.

---

## Single Shot

### SingleShot GE (Readout fidelity)
```python
config_all.update('res_phase', 0, q_index=qubit)
run_cfg = config_all.get_qubit(qubit)

SHOT = 5000
ssh = SingleShot_gef(run_cfg)
ssh.run(SHOT, shot_f=False)     # shot_f=False → only g/e, no f
ssh_result = ssh.plot(fid_avg=True, verbose=True)
config_all.update('res_phase', ssh_result[2], q_index=qubit)
```
**GUI needs:** shots (default 5000), shot_f toggle (g/e only vs g/e/f).

---

### SingleShot Opt (Optimize readout parameters)
```python
freq_axis   = np.linspace(res_freq_ge - 1, res_freq_ge + 8, 11)
gain_axis   = np.linspace(0.07, 0.1, 5)
length_axis = np.linspace(2, 4, 4)
sweep_para  = {"freq": res_freq_ge, "gain": gain_axis, "length": length_axis}

SHOT = 2000
ssh_opt.run(SHOT, sweep_para=sweep_para)
length, gain, freq = ssh_opt.analyze()
# Updates: ro_length, res_gain_ge, res_freq_ge
```
**GUI needs:** shots, freq range/steps, gain range/steps, length range/steps.

---

## Qubit EF

### Resonator Spec EF
Same structure as Res Spec GE.
```python
config_all.update("res.res_gain_ge", 0.1, q_index=qubit)
run_cfg.update([
    ("steps",      101),
    ("res_freq_ge", QickSweep1D("freqloop", res_freq_ge-10, res_freq_ge+10)),
])
onetone_ef.run(py_avg=20, solve_type="hm")
```
**GUI needs:** same as Res Spec GE (res_freq_ge center, span, steps, gain).

---

### Qubit Spec EF
```python
center = qb_freq_ge - 200    # MHz — EF is ~200 MHz below GE
SPAN   = 100;  STEPS = 101

config_all.update("qb_ch_ef",  0, q_index=qubit)
config_all.update("nqz_qb_ef", 2, q_index=qubit)

run_cfg.update([
    ("steps",                 STEPS),
    ("qb_freq_ef",            QickSweep1D("freqloop", center-SPAN, center+SPAN)),
    ("qb_gain_ef",            0.3),
    ("qb_flat_top_length_ef", 2),     # us
    ("ge_ref",                True),  # use GE rotation as reference
])
f_ef = spectrum_ef.run(10)
config_all.update("qb_freq_ef",   f_ef, q_index=qubit)
config_all.update("qb_mixer_ef",  f_ef, q_index=qubit)
```
**GUI needs:** EF center (default qb_freq_ge - 200), span (default ±100), steps, qb_gain_ef, qb_flat_top_length_ef, ge_ref toggle.

---

### Power Rabi EF
```python
config_all.update("sigma_ef", 0.01, q_index=qubit)

run_cfg.update([
    ("steps",      100),
    ("qb_gain_ef", QickSweep1D("gainloop", 0.0, 1.0)),
    ("ge_ref",     True),
])
pi_gain_ef, pi2_gain_ef = prabi_ef.run(50)
config_all.update("pi_gain_ef",  pi_gain_ef,  q_index=qubit)
config_all.update("pi2_gain_ef", pi2_gain_ef, q_index=qubit)
```
**GUI needs:** sigma_ef, gain start/stop/steps, ge_ref toggle.

---

### Ramsey EF
```python
START_TIME = 0.0;  STOP_TIME = 2;  STEPS = 100   # us

run_cfg.update([
    ("steps",       STEPS),
    ("wait_time",   QickSweep1D("waitloop", START_TIME, STOP_TIME)),
    ("ramsey_freq", 2),     # MHz
    ("ge_ref",      True),
])
t2r_ef.run(100)
config_all.update("qb_freq_ef", t2r_ef.correct_detune(), qubit)
```
**GUI needs:** wait_time start/stop, steps, ramsey_freq, ge_ref toggle.

---

### T1 EF
```python
START_TIME = 0.0;  STOP_TIME = 400;  STEPS = 100   # us

run_cfg.update([
    ("steps",     STEPS),
    ("wait_time", QickSweep1D("waitloop", START_TIME, STOP_TIME)),
])
t1_ef.run(50)
```
**GUI needs:** wait_time start/stop (default 0→400 µs), steps.

---

## Advanced

### AllXY
```python
run_cfg.update([
    # ("pulse_type", "drag"),   # optional
])
allxy.run(10)
allxy.plot()
```
**GUI needs:** py_avg, pulse_type toggle (flat_top / drag). No sweep parameters.

---

### State Tomography
```python
tomo.run(py_avg=5, prep_pulse_name="x180")
tomo.plot(plot_type="2d", qb_idx=3)
```
**GUI needs:** py_avg, prep_pulse_name (dropdown: x180, x90, y180, y90, identity).

---

## Randomized Benchmarking

### Auto RB
```python
run_cfg['pulse_type'] = "drag"

auto.run(
    py_avg=5,
    max_circuit_depth=600,
    delta_clifford=50,
    number_sample=50,
    interleaved_gates=["X", "X/2"],   # list of gates to interleave
    iq_process='real',
)
```
**GUI needs:** py_avg, max_circuit_depth, delta_clifford, number_sample, pulse_type, interleaved_gates (text field or checkboxes).

---

### Single Qubit RB (with Interleaved RB)
```python
# Standard RB
rb_exp_ref.run(
    py_avg=5,
    max_circuit_depth=400,
    delta_clifford=40,
    number_sample=30,
)
# Interleaved RB (runs per gate in a loop)
rb_exp_irb.run(..., interleaved_gate="X/2")
```
**GUI needs:** py_avg, max_circuit_depth, delta_clifford, number_sample, interleaved_gate (single gate string or "none").

---

## Summary: What the GUI Parameter Form Is Missing

| Experiment | Missing Parameters |
|---|---|
| **TOF** | ro_length, res_length, res_gain_ge, res_ch, check_f |
| **Res Spec GE** | freq center, sweep span (instead of just "span"), res_gain_ge |
| **Punchout** | freq center/span/steps, gain start/stop/steps |
| **Qubit Spec GE** | freq center, sweep span, qb_gain_ge, qb_flat_top_length_ge, qb_mixer, nqz_qb |
| **Time Rabi** | sigma_ge, qb_gain_ge, **length start/stop** (not "span") |
| **Power Rabi** | sigma_ge, **gain start/stop**, nqz_qb |
| **DRAG** | alpha_start/stop/steps, iter_start/stop/step |
| **AAE** | gain half-span, iter_start/stop/step |
| **Ramsey** | **wait_time start/stop**, **ramsey_freq** |
| **Spin Echo** | **wait_time start/stop**, ramsey_freq |
| **T1** | **wait_time start/stop** (not "span") |
| **SingleShot** | shots count, shot_f toggle |
| **Qubit Spec EF** | EF center, span, qb_gain_ef, qb_flat_top_length_ef, ge_ref |
| **Power Rabi EF** | sigma_ef, gain start/stop, ge_ref |
| **Ramsey EF** | wait_time start/stop, ramsey_freq, ge_ref |
| **T1 EF** | wait_time start/stop |
| **AllXY** | pulse_type toggle |
| **Tomography** | prep_pulse_name |
| **RB** | max_circuit_depth, delta_clifford, number_sample, interleaved_gate |

**Root cause:** The current `_COMMON_PARAMS` dict (`py_avg, reps, relax_delay, steps, span`) is a generic set that fits no experiment exactly. Each experiment needs its own parameter spec, driven by the table above.

**Fix strategy:** Replace `_COMMON_PARAMS` with a per-experiment registry that maps `class_path → list of (name, kind, default, min, max, tooltip)`. When the user selects an experiment, rebuild the form from its specific spec.
