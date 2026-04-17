# Auto-Calibration (`AutoCalibrate`)

## Input

- `config_all` : live config store (`ExperimentConfig`)
- `qubit` : qubit label, e.g. `"Q1"`
- Pre-calibration data: rough `res_freq_ge`, `qb_freq_ge`

## Usage

```python
cal = AutoCalibrate(config_all, qubit="Q1")
cal.run()                          # full pipeline
cal.run(skip=("spin_echo", "t1")) # skip coherence steps
cal.summary()                      # print results table
```

---

## Calibration Pipeline

### 1. Resonator Spectrum (`res_spec`)

- Sweep ±10 MHz around current `res_freq_ge` (101 pts, py_avg=5)
- Fit with circle-fit (`solve_type="hm"`)
- If fit fails: expand to ±20 MHz → ±40 MHz (3 attempts max)
- **Updates:** `res_freq_ge`

---

### 2. Qubit ge Spectrum (`qubit_spec`)

- Sweep ±50 MHz around current `qb_freq_ge` (101 pts, py_avg=5)
- Fit Lorentzian peak
- If fit fails: expand to ±100 MHz → ±200 MHz (3 attempts max)
- Seeds T2\* estimate from linewidth: `T2* ≈ 1 / (π × FWHM)`
- **Updates:** `qb_freq_ge`, `qb_mixer`

---

### 3. Power Rabi (`power_rabi`)

- Sweep gain 0 → 1 (100 pts, py_avg=10)
- Acceptable range: `0.15 ≤ pi_gain ≤ 0.90`
- If out of range: rescale `sigma_ge` by `pi_gain / TARGET_PI_GAIN` (target = 0.5), retry
- Max retries: 3
- **Updates:** `pi_gain_ge`, `pi2_gain_ge`

---

### 4. Ramsey — frequency fine-tune + T2\* (`ramsey`)

- Sweep 0 → `max(2, 3 × T2*)` µs (100 pts, py_avg=20, ramsey_freq=2 MHz)
- Convergence criterion: `|detuning error| < 50 kHz`
- Correction strategy (after ≥2 observations): GP zero-crossing prediction
  - GP kernel: RBF + WhiteKernel; finds freq where predicted error ≈ 0
  - Falls back to naive `correct_detune()` if GP uncertainty > 5× limit
- Max retries: 3 (+ 2 extra budget for GP)
- Sweep window narrows to `min(3×T2*, 15)` µs on each retry
- **Updates:** `qb_freq_ge`, records `T2r_us`

---

### 5. Spin Echo — T2E (`spin_echo`)

- Adaptive sweep range:
  - `T2E_estimate = 3 × T2*`
  - `stop = max(20, 5 × T2E_estimate)` µs
  - `ramsey_freq = min(0.5, 1.5 / stop)` MHz → keeps 2–3 fringes in window
- 100 pts, py_avg=100
- Fit: decaysin (if ramsey_freq ≠ 0) or expfunc (if ramsey_freq = 0)
- **Records:** `T2e_us`

---

### 6. T1 (`t1`)

- Adaptive sweep range:
  - `T1_estimate = T2E` (or T2\* if T2E unavailable)
  - `stop = max(50, 3 × T1_estimate)` µs
  - `relax_delay = max(current_relax, 5 × T1_estimate)` set before scan
- 100 pts, py_avg=50
- Fit: expfunc → `[amplitude, offset, tau]`; tau = T1
- Post-scan: `relax_delay = max(100, 5 × T1)` µs
- **Updates:** `relax_delay`, records `T1_us`

---

### 7. Single-Shot Readout Optimization (`ss_opt`)

Two-phase Bayesian-optimized search over (freq, gain, length):

**Phase 1 — Coarse grid** (45 pts = 5×3×3, 1000 shots each)

```python
freq   = np.linspace(res_freq_ge - 1.0, res_freq_ge + 8.0, 5)   # MHz
gain   = np.linspace(0.07, 0.10, 3)
length = np.linspace(2.0, 4.0, 3)                                 # µs
```

GP surrogate (Matern-2.5) fitted offline to find coarse optimum.

**Phase 2 — Online BO refinement** (15 extra hardware runs)

- Expected-Improvement (EI) proposes next point; GP refitted after each run
- Infeasible points (leakage / thermal penalty) auto-rejected

#### Phase 3 — IQ rotation angle

- `res_phase = 0` reset, then 5000-shot `SingleShot_gef` to measure rotation angle
- **Updates:** `ro_length`, `res_gain_ge`, `res_freq_ge`, `res_phase`
- **Records:** `readout_fidelity`, `res_phase_deg`

---

## Constants

| Parameter                | Value    |
|--------------------------|----------|
| `TARGET_PI_GAIN`         | 0.5      |
| `MAX_RABI_RETRIES`       | 3        |
| `MAX_RAMSEY_RETRIES`     | 3        |
| `RAMSEY_DETUNE_LIMIT`    | 50 kHz   |
