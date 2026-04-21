from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QFormLayout, QScrollArea, QFrame,
    QCheckBox,
)
from PySide6.QtCore import Signal, Qt


# ── Experiment registry ───────────────────────────────────────────────────────
# (display_name, module.ClassName)  — class names verified against script files

EXPERIMENT_REGISTRY = {
    "Setup": [
        ("Time of Flight",     "s001_time_of_flight.TOF"),
    ],
    "Resonator": [
        ("Res Spec GE",        "s002_res_spec_ge.ResonatorSpec"),
        ("Res Punchout",       "s002b_res_punchout_ge.Punchout"),
        ("Res Spec Flux",      "s002c_res_spec_ge_flux.ResonatorSpecFlux"),
    ],
    "Qubit GE": [
        ("Qubit Spec",         "s003_qubit_spec_ge.QubitSpec"),
        ("Qubit Flux Spec",    "s003a_qubit_flux_spec_ge.QubitSpecFlux"),
        ("Time Rabi",          "s004_time_rabi_ge.TimeRabi"),
        ("Power Rabi",         "s005_power_rabi_ge.PowerRabi"),
        ("DRAG",               "s005a_drag.DragCalibration"),
        ("AAE",                "s005a_AAE.PowerRabiChevron"),
    ],
    "Coherence": [
        ("Ramsey",             "s006_Ramsey_ge.Ramsey"),
        ("Spin Echo",          "s007_SpinEcho_ge.SpinEcho"),
        ("T1",                 "s008_T1_ge.T1"),
    ],
    "Qubit EF": [
        ("Res Spec EF",        "s009_res_spec_ef.ResonatorSpec_ef"),
        ("Qubit Spec EF",      "s010_qubit_spec_ef.QubitSpec_ef"),
        ("Power Rabi EF",      "s011_power_rabi_ef.PowerRabi_ef"),
        ("Ramsey EF",          "s012_Ramsey_ef.Ramsey_ef"),
        ("T1 EF",              "s013_T1_ef.T1_ef"),
    ],
    "Advanced": [
        ("AllXY",              "s014_AllXY.AllXY"),
        ("SingleShot",         "s000_SingleShot_prog.SingleShot_gef"),
        ("SingleShot Opt",     "s000_SingleShot_opt.SingleShot_ge_opt"),
        ("Qubit Temp",         "s013_qubit_temp.QubitTemperatureEf"),
        ("AC Stark",           "s006_ac_stark.AcStarkCalib"),
    ],
    "RB": [
        ("Single Qubit RB",    "s015_Single_qubit_RB.RandomizedBenchmarking"),
        ("Auto RB",            "s015_Auto_RB.AutoRB"),
        ("RB ASM",             "s015_RB_asm.RandomizedBenchmarkingAsm"),
    ],
    "Tomography": [
        ("State Tomography",   "s016_state_tomography.Tomography"),
    ],
}


# ── Per-experiment parameter specs ────────────────────────────────────────────
# Each entry: (key, kind, default, lo, hi, tooltip)
#   kind "int"   → QSpinBox,       lo/hi = int bounds
#   kind "float" → QDoubleSpinBox, lo/hi = float bounds
#   kind "bool"  → QCheckBox,      lo/hi ignored
#   kind "combo" → QComboBox,      lo = list[str], hi ignored
# Sweep axes use explicit start/stop pairs; the worker builds QickSweep1D from them.

_P = {
    # ── reusable building blocks ──────────────────────────────────────────────
    "_py_avg":      ("py_avg",      "int",   10,    1,      500,   "Python-level averages"),
    "_reps":        ("reps",        "int",   100,   100,  50000,   "Hardware reps per point"),
    "_relax":       ("relax_delay", "float", 50.0,  0.1,  5000.0,  "Relax delay between shots (µs)"),
    "_steps":       ("steps",       "int",   100,   10,   2000,    "Sweep points"),
    "_sigma_ge":    ("sigma_ge",    "float", 0.01,  0.001, 1.0,   "Gaussian σ of GE π-pulse (µs)"),
    "_sigma_ef":    ("sigma_ef",    "float", 0.01,  0.001, 1.0,   "Gaussian σ of EF π-pulse (µs)"),
    "_ge_ref":      ("ge_ref",      "bool",  True,  None,  None,  "Use GE π-pulse as reference"),
}

def _p(*keys):
    """Return pre-built param tuples by key."""
    return [_P[k] for k in keys]


EXPT_PARAMS: dict[str, list[tuple]] = {

    # ── Setup ─────────────────────────────────────────────────────────────────

    "s001_time_of_flight.TOF": [
        ("py_avg",       "int",   1000,  1,    10000, "Software averages (hardware reps forced to 1)"),
        ("relax_delay",  "float", 100.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("ro_length",    "float", 2.5,   0.1,  20.0,  "Readout window length (µs)"),
        ("res_length",   "float", 1.7,   0.1,  10.0,  "Resonator drive length (µs)"),
        ("res_gain_ge",  "float", 0.18,  0.0,  1.0,   "Readout tone gain [0–1]"),
        ("res_ch",       "int",   0,     0,    9,     "Readout channel index"),
    ],

    # ── Resonator ─────────────────────────────────────────────────────────────

    "s002_res_spec_ge.ResonatorSpec": [
        ("py_avg",       "int",   5,     1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 0.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",  "float", 6500., 1000.,12000.,"Resonator center frequency (MHz)"),
        ("freq_span",    "float", 10.0,  0.1,  500.0, "Sweep half-span each side (MHz)"),
        ("steps",        "int",   101,   10,   2000,  "Sweep points"),
        ("res_gain_ge",  "float", 0.05,  0.0,  1.0,   "Readout tone gain [0–1]"),
    ],

    "s002b_res_punchout_ge.Punchout": [
        ("py_avg",       "int",   100,   1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 0.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",  "float", 6500., 1000.,12000.,"Resonator center frequency (MHz)"),
        ("freq_span",    "float", 20.0,  0.1,  500.0, "Frequency half-span each side (MHz)"),
        ("freq_steps",   "int",   101,   10,   2000,  "Frequency sweep points"),
        ("gain_start",   "float", 0.01,  0.0,  1.0,   "Gain sweep start [0–1]"),
        ("gain_stop",    "float", 0.30,  0.0,  1.0,   "Gain sweep stop [0–1]"),
        ("gain_steps",   "int",   11,    2,    100,   "Gain sweep points"),
    ],

    "s002c_res_spec_ge_flux.ResonatorSpecFlux": [
        ("py_avg",           "int",   10,    1,    500,   "Python-level averages"),
        ("reps",             "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",      "float", 0.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",      "float", 6500., 1000.,12000.,"Resonator center frequency (MHz)"),
        ("freq_span",        "float", 10.0,  0.1,  500.0, "Frequency half-span each side (MHz)"),
        ("freq_steps",       "int",   101,   10,   2000,  "Frequency sweep points"),
        ("res_gain_ge",      "float", 0.05,  0.0,  1.0,   "Readout tone gain [0–1]"),
        ("flux_ch",          "int",   15,    0,    15,    "Flux DAC channel"),
        ("flux_gain_start",  "float", -1.0, -1.0,  0.0,  "Flux gain sweep start [−1–0]"),
        ("flux_gain_stop",   "float",  1.0,  0.0,  1.0,  "Flux gain sweep stop [0–1]"),
        ("flux_steps",       "int",   11,    2,    100,   "Flux sweep points"),
        ("flux_length",      "float", 0.7,   0.01, 10.0,  "Flux pulse length (µs)"),
    ],

    # ── Qubit GE ──────────────────────────────────────────────────────────────

    "s003_qubit_spec_ge.QubitSpec": [
        ("py_avg",               "int",   5,     1,    500,   "Python-level averages"),
        ("reps",                 "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",          "float", 1.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",          "float", 3620., 500., 9000., "Qubit center frequency (MHz)"),
        ("freq_span",            "float", 50.0,  1.0,  500.0, "Sweep half-span each side (MHz)"),
        ("steps",                "int",   100,   10,   2000,  "Sweep points"),
        ("qb_gain_ge",           "float", 0.1,   0.0,  1.0,   "Qubit drive gain [0–1]"),
        ("qb_flat_top_length_ge","float", 1.0,   0.01, 20.0,  "Flat-top pulse length (µs)"),
        ("nqz_qb",               "int",   2,     1,    3,     "Nyquist zone of qubit channel"),
    ],

    "s003a_qubit_flux_spec_ge.QubitSpecFlux": [
        ("py_avg",               "int",   500,   1,    2000,  "Python-level averages"),
        ("reps",                 "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",          "float", 1.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",          "float", 5970., 500., 9000., "Qubit center frequency (MHz)"),
        ("freq_span",            "float", 50.0,  1.0,  500.0, "Frequency half-span each side (MHz)"),
        ("freq_steps",           "int",   201,   10,   2000,  "Frequency sweep points"),
        ("qb_gain_ge",           "float", 0.3,   0.0,  1.0,   "Qubit drive gain [0–1]"),
        ("qb_flat_top_length_ge","float", 0.5,   0.01, 20.0,  "Flat-top pulse length (µs)"),
        ("flux_ch",              "int",   15,    0,    15,    "Flux DAC channel"),
        ("flux_gain_start",      "float",-1.0,  -1.0,  0.0,  "Flux gain sweep start"),
        ("flux_gain_stop",       "float", 1.0,   0.0,  1.0,  "Flux gain sweep stop"),
        ("flux_steps",           "int",   11,    2,    100,   "Flux sweep points"),
        ("flux_length",          "float", 0.7,   0.01, 10.0,  "Flux pulse length (µs)"),
    ],

    "s004_time_rabi_ge.TimeRabi": [
        ("py_avg",       "int",   100,   1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("sigma_ge",     "float", 0.01,  0.001, 1.0,  "Gaussian σ of GE π-pulse (µs)"),
        ("qb_gain_ge",   "float", 0.5,   0.0,  1.0,   "Qubit drive gain [0–1]"),
        ("length_start", "float", 0.01,  0.001, 1.0,  "Flat-top length sweep start (µs)"),
        ("length_stop",  "float", 0.30,  0.01,  20.0, "Flat-top length sweep stop (µs)"),
        ("steps",        "int",   81,    10,   2000,  "Sweep points"),
    ],

    "s005_power_rabi_ge.PowerRabi": [
        ("py_avg",       "int",   5,     1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 200.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("sigma_ge",     "float", 0.01,  0.001, 1.0,  "Gaussian σ of GE π-pulse (µs)"),
        ("nqz_qb",       "int",   2,     1,    3,     "Nyquist zone of qubit channel"),
        ("gain_start",   "float", 0.0,   0.0,  1.0,   "Gain sweep start [0–1]"),
        ("gain_stop",    "float", 1.0,   0.0,  1.0,   "Gain sweep stop [0–1]"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
    ],

    "s005a_drag.DragCalibration": [
        ("py_avg",       "int",   5,     1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 200.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("alpha_start",  "float",-1.0,  -5.0,  0.0,  "DRAG α sweep start"),
        ("alpha_stop",   "float", 1.5,   0.0,  5.0,  "DRAG α sweep stop"),
        ("alpha_steps",  "int",   41,    5,    200,   "α sweep points"),
        ("iter_start",   "int",   2,     1,    50,    "First repetition count"),
        ("iter_stop",    "int",   20,    1,    200,   "Last repetition count"),
        ("iter_step",    "int",   2,     1,    50,    "Repetition count step"),
    ],

    "s005a_AAE.PowerRabiChevron": [
        ("py_avg",           "int",   10,    1,    500,   "Python-level averages"),
        ("reps",             "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",      "float", 200.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("gain_half_span",   "float", 0.1,   0.001, 0.5,  "Gain sweep ± around π-gain"),
        ("steps",            "int",   100,   10,   2000,  "Sweep points"),
        ("iter_start",       "int",   1,     1,    50,    "First repetition count"),
        ("iter_stop",        "int",   50,    1,    500,   "Last repetition count"),
        ("iter_step",        "int",   7,     1,    50,    "Repetition count step"),
    ],

    # ── Coherence ─────────────────────────────────────────────────────────────

    "s006_Ramsey_ge.Ramsey": [
        ("py_avg",       "int",   20,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("time_start",   "float", 0.0,   0.0,  100.0, "Wait-time sweep start (µs)"),
        ("time_stop",    "float", 2.0,   0.01, 2000.0,"Wait-time sweep stop (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
        ("ramsey_freq",  "float", 2.0,   0.0,  100.0, "Artificial detuning (MHz)"),
    ],

    "s007_SpinEcho_ge.SpinEcho": [
        ("py_avg",       "int",   100,   1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("time_start",   "float", 0.0,   0.0,  100.0, "Wait-time sweep start (µs)"),
        ("time_stop",    "float", 150.0, 1.0,  5000.0,"Wait-time sweep stop (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
        ("ramsey_freq",  "float", 0.05,  0.0,  10.0,  "Small detuning for echo fringes (MHz)"),
    ],

    "s008_T1_ge.T1": [
        ("py_avg",       "int",   50,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("time_start",   "float", 0.0,   0.0,  100.0, "Wait-time sweep start (µs)"),
        ("time_stop",    "float", 400.0, 1.0,  5000.0,"Wait-time sweep stop (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
    ],

    # ── Qubit EF ──────────────────────────────────────────────────────────────

    "s009_res_spec_ef.ResonatorSpec_ef": [
        ("py_avg",       "int",   20,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 0.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",  "float", 6500., 1000.,12000.,"Resonator center frequency (MHz)"),
        ("freq_span",    "float", 10.0,  0.1,  500.0, "Sweep half-span each side (MHz)"),
        ("steps",        "int",   101,   10,   2000,  "Sweep points"),
        ("res_gain_ge",  "float", 0.1,   0.0,  1.0,   "Readout tone gain [0–1]"),
    ],

    "s010_qubit_spec_ef.QubitSpec_ef": [
        ("py_avg",               "int",   10,    1,    500,   "Python-level averages"),
        ("reps",                 "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",          "float", 1.0,   0.0,  5000.0,"Relax delay (µs)"),
        ("freq_center",          "float", 3420., 500., 9000., "EF center freq (MHz); ~f_ge − 200"),
        ("freq_span",            "float", 100.0, 1.0,  500.0, "Sweep half-span each side (MHz)"),
        ("steps",                "int",   101,   10,   2000,  "Sweep points"),
        ("qb_gain_ef",           "float", 0.3,   0.0,  1.0,   "EF drive gain [0–1]"),
        ("qb_flat_top_length_ef","float", 2.0,   0.01, 20.0,  "EF flat-top pulse length (µs)"),
        ("ge_ref",               "bool",  True,  None, None,  "Use GE π-pulse as reference"),
    ],

    "s011_power_rabi_ef.PowerRabi_ef": [
        ("py_avg",       "int",   50,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 200.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("sigma_ef",     "float", 0.01,  0.001, 1.0,  "Gaussian σ of EF π-pulse (µs)"),
        ("gain_start",   "float", 0.0,   0.0,  1.0,   "Gain sweep start [0–1]"),
        ("gain_stop",    "float", 1.0,   0.0,  1.0,   "Gain sweep stop [0–1]"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
        ("ge_ref",       "bool",  True,  None, None,  "Use GE π-pulse as reference"),
    ],

    "s012_Ramsey_ef.Ramsey_ef": [
        ("py_avg",       "int",   100,   1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("time_start",   "float", 0.0,   0.0,  100.0, "Wait-time sweep start (µs)"),
        ("time_stop",    "float", 2.0,   0.01, 2000.0,"Wait-time sweep stop (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
        ("ramsey_freq",  "float", 2.0,   0.0,  100.0, "Artificial detuning (MHz)"),
        ("ge_ref",       "bool",  True,  None, None,  "Use GE π-pulse as reference"),
    ],

    "s013_T1_ef.T1_ef": [
        ("py_avg",       "int",   50,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("time_start",   "float", 0.0,   0.0,  100.0, "Wait-time sweep start (µs)"),
        ("time_stop",    "float", 400.0, 1.0,  5000.0,"Wait-time sweep stop (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
    ],

    # ── Advanced ──────────────────────────────────────────────────────────────

    "s014_AllXY.AllXY": [
        ("py_avg",       "int",   10,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("pulse_type",   "combo", "flat_top", ["flat_top", "drag"], None,
         "Qubit pulse envelope shape"),
    ],

    "s000_SingleShot_prog.SingleShot_gef": [
        ("shots",        "int",   5000,  100,  50000, "Number of single shots"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("shot_f",       "bool",  False, None, None,  "Include f-state (g/e/f) measurement"),
    ],

    "s000_SingleShot_opt.SingleShot_ge_opt": [
        ("shots",         "int",   2000,  100,  20000, "Single shots per setting"),
        ("relax_delay",   "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("freq_span",     "float", 4.5,   0.1,  50.0,  "Frequency sweep half-span (MHz)"),
        ("freq_steps",    "int",   11,    2,    50,    "Frequency sweep points"),
        ("gain_start",    "float", 0.07,  0.0,  1.0,   "Readout gain sweep start"),
        ("gain_stop",     "float", 0.10,  0.0,  1.0,   "Readout gain sweep stop"),
        ("gain_steps",    "int",   5,     2,    30,    "Gain sweep points"),
        ("length_start",  "float", 2.0,   0.1,  20.0,  "Readout length sweep start (µs)"),
        ("length_stop",   "float", 4.0,   0.1,  20.0,  "Readout length sweep stop (µs)"),
        ("length_steps",  "int",   4,     2,    20,    "Length sweep points"),
        ("shot_f",        "bool",  False, None, None,  "Include f-state measurement"),
    ],

    "s013_qubit_temp.QubitTemperatureEf": [
        ("py_avg",       "int",   5,     1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 100.0, 0.1,  5000.0,"Relax delay (µs)"),
        ("sigma_ef",     "float", 0.01,  0.001, 1.0,  "Gaussian σ of EF π-pulse (µs)"),
        ("gain_start",   "float", 0.0,   0.0,  1.0,   "EF gain sweep start [0–1]"),
        ("gain_stop",    "float", 1.0,   0.0,  1.0,   "EF gain sweep stop [0–1]"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
        ("ge_ref",       "bool",  True,  None, None,  "Use GE π-pulse as reference"),
    ],

    "s006_ac_stark.AcStarkCalib": [
        ("py_avg",       "int",   10,    1,    500,   "Python-level averages"),
        ("reps",         "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",  "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("steps",        "int",   100,   10,   2000,  "Sweep points"),
    ],

    # ── RB ────────────────────────────────────────────────────────────────────

    "s015_Single_qubit_RB.RandomizedBenchmarking": [
        ("py_avg",            "int",   5,     1,    500,   "Python-level averages"),
        ("reps",              "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",       "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("max_circuit_depth", "int",   400,   10,   5000,  "Maximum Clifford circuit depth"),
        ("delta_clifford",    "int",   40,    1,    500,   "Step size between circuit depths"),
        ("number_sample",     "int",   30,    1,    200,   "Random circuit samples per depth"),
        ("pulse_type",        "combo", "flat_top", ["flat_top", "drag"], None,
         "Qubit pulse envelope shape"),
    ],

    "s015_Auto_RB.AutoRB": [
        ("py_avg",            "int",   5,     1,    500,   "Python-level averages"),
        ("reps",              "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",       "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("max_circuit_depth", "int",   600,   10,   5000,  "Maximum Clifford circuit depth"),
        ("delta_clifford",    "int",   50,    1,    500,   "Step size between circuit depths"),
        ("number_sample",     "int",   50,    1,    200,   "Random circuit samples per depth"),
        ("pulse_type",        "combo", "drag", ["flat_top", "drag"], None,
         "Qubit pulse envelope shape"),
    ],

    "s015_RB_asm.RandomizedBenchmarkingAsm": [
        ("py_avg",            "int",   5,     1,    500,   "Python-level averages"),
        ("reps",              "int",   100,   100,  50000, "Hardware reps per point"),
        ("relax_delay",       "float", 50.0,  0.1,  5000.0,"Relax delay (µs)"),
        ("max_circuit_depth", "int",   400,   10,   5000,  "Maximum Clifford circuit depth"),
        ("delta_clifford",    "int",   40,    1,    500,   "Step size between circuit depths"),
        ("number_sample",     "int",   30,    1,    200,   "Random circuit samples per depth"),
    ],

    # ── Tomography ────────────────────────────────────────────────────────────

    "s016_state_tomography.Tomography": [
        ("py_avg",          "int",   5,    1,   500,   "Python-level averages"),
        ("reps",            "int",   100,  100, 50000, "Hardware reps per point"),
        ("relax_delay",     "float", 50.0, 0.1, 5000.0,"Relax delay (µs)"),
        ("prep_pulse_name", "combo", "x180",
         ["x180", "x90", "y180", "y90", "identity"], None,
         "Prep-pulse applied before tomography"),
    ],
}

_FALLBACK_PARAMS: list[tuple] = [
    ("py_avg",      "int",   10,    1,      500,    "Python-level averages"),
    ("reps",        "int",   100,   100,  50000,    "Hardware reps per point"),
    ("relax_delay", "float", 50.0,  0.1,  5000.0,  "Relax delay (µs)"),
    ("steps",       "int",   100,   10,   2000,     "Sweep points"),
]


# ── Panel ─────────────────────────────────────────────────────────────────────

class ExperimentPanel(QWidget):
    """
    Experiment selector, per-experiment parameter form, and run controls.

    Signals
    -------
    run_requested : Signal(str, dict)
        Emitted when Run is clicked.  Carries (class_path, params).
    stop_requested : Signal()
        Emitted when Stop is clicked or Esc shortcut fires.
    save_requested : Signal()
        Emitted when Save is clicked.
    """

    run_requested  = Signal(str, dict)
    stop_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._param_widgets: dict[str, QWidget] = {}
        self._running = False
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        # Experiment selector
        sel_grp = QGroupBox("Experiment")
        sel_lay = QFormLayout(sel_grp)
        sel_lay.setSpacing(6)

        self.cat_combo = QComboBox()
        self.cat_combo.addItems(EXPERIMENT_REGISTRY.keys())
        self.cat_combo.setToolTip("Experiment category")
        self.cat_combo.currentTextChanged.connect(self._on_category)
        sel_lay.addRow("Category:", self.cat_combo)

        self.expt_combo = QComboBox()
        self.expt_combo.setToolTip("Select experiment")
        self.expt_combo.currentTextChanged.connect(self._rebuild_params)
        sel_lay.addRow("Experiment:", self.expt_combo)
        root.addWidget(sel_grp)

        # Parameters (scrollable)
        self._params_form = QFormLayout()
        self._params_form.setSpacing(5)
        params_widget = QWidget()
        params_widget.setLayout(self._params_form)
        scroll = QScrollArea()
        scroll.setWidget(params_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(200)
        params_grp = QGroupBox("Parameters")
        QVBoxLayout(params_grp).addWidget(scroll)
        root.addWidget(params_grp)

        # State label
        self._state_lbl = QLabel("Idle")
        self._state_lbl.setAlignment(Qt.AlignCenter)
        self._state_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        root.addWidget(self._state_lbl)

        # Run / Stop / Save buttons
        btn_row = QHBoxLayout()
        self.run_btn  = QPushButton("▶  Run")
        self.stop_btn = QPushButton("■  Stop")
        self.save_btn = QPushButton("Save")
        self.run_btn.setToolTip("Start experiment  (Ctrl+R)")
        self.stop_btn.setToolTip("Abort experiment  (Esc)")
        self.save_btn.setToolTip("Save last result to HDF5")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.save_btn.clicked.connect(self.save_requested)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.save_btn)
        root.addLayout(btn_row)
        root.addStretch()

        self._on_category(self.cat_combo.currentText())

    # ── Category / experiment selection ───────────────────────────────────────

    def _on_category(self, category: str):
        """
        Repopulate the experiment combo when the category changes.

        Parameters
        ----------
        category : str
            Key into EXPERIMENT_REGISTRY.
        """
        self.expt_combo.blockSignals(True)
        self.expt_combo.clear()
        for display, _ in EXPERIMENT_REGISTRY.get(category, []):
            self.expt_combo.addItem(display)
        self.expt_combo.blockSignals(False)
        self._rebuild_params()

    def _current_class_path(self) -> str:
        """
        Return the class path string for the currently selected experiment.

        Returns
        -------
        class_path : str
            e.g. ``"s003_qubit_spec_ge.QubitSpec"``
        """
        cat = self.cat_combo.currentText()
        idx = self.expt_combo.currentIndex()
        entries = EXPERIMENT_REGISTRY.get(cat, [])
        if not entries or idx < 0:
            return ""
        return entries[idx][1]

    def _rebuild_params(self, *_):
        """
        Clear and rebuild the parameter form for the current experiment.

        Looks up EXPT_PARAMS by class path; falls back to _FALLBACK_PARAMS
        for any experiment not yet listed.
        """
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_widgets.clear()

        class_path = self._current_class_path()
        specs = EXPT_PARAMS.get(class_path, _FALLBACK_PARAMS)

        for spec in specs:
            key, kind, default, lo, hi, tip = spec
            w = self._make_widget(kind, default, lo, hi, tip)
            label = key.replace("_", " ").title() + ":"
            self._params_form.addRow(label, w)
            self._param_widgets[key] = w

    @staticmethod
    def _make_widget(kind: str, default, lo, hi, tip: str) -> QWidget:
        """
        Instantiate the appropriate Qt widget for a parameter spec.

        Parameters
        ----------
        kind : str
            One of ``"int"``, ``"float"``, ``"bool"``, ``"combo"``.
        default : int | float | bool | str
            Initial widget value.
        lo : int | float | list | None
            Lower bound (int/float) or option list (combo).
        hi : int | float | None
            Upper bound (int/float).
        tip : str
            Tooltip text shown on hover.

        Returns
        -------
        widget : QWidget
            Ready-to-use Qt widget.
        """
        if kind == "int":
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setValue(int(default))
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setRange(float(lo), float(hi))
            w.setDecimals(3)
            w.setSingleStep(max((hi - lo) / 100, 0.001))
            w.setValue(float(default))
        elif kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
        elif kind == "combo":
            w = QComboBox()
            w.addItems(lo)          # lo holds the list of options
            idx = lo.index(default) if default in lo else 0
            w.setCurrentIndex(idx)
        else:
            w = QSpinBox()
        w.setToolTip(tip)
        return w

    # ── Slot helpers ──────────────────────────────────────────────────────────

    def _widget_value(self, w: QWidget):
        """
        Extract the current value from any parameter widget.

        Parameters
        ----------
        w : QWidget
            A widget created by :meth:`_make_widget`.

        Returns
        -------
        value : int | float | bool | str
            The widget's current value in its native Python type.
        """
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.value()

    def _on_run(self):
        class_path = self._current_class_path()
        if not class_path:
            return
        params = {k: self._widget_value(w) for k, w in self._param_widgets.items()}
        self.run_requested.emit(class_path, params)

    def _on_stop(self):
        self.stop_requested.emit()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_running(self, running: bool):
        """
        Toggle the panel between running and idle states.

        Parameters
        ----------
        running : bool
            ``True`` while an experiment is executing.
        """
        self._running = running
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.save_btn.setEnabled(not running)
        if running:
            self._state_lbl.setText("● Running…")
            self._state_lbl.setStyleSheet("color: #00d4ff; font-size: 11px;")
        else:
            self._state_lbl.setText("Idle")
            self._state_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")

    def trigger_run(self):
        """
        Programmatically fire the Run action (used by keyboard shortcut).
        """
        if not self._running:
            self._on_run()

    def trigger_stop(self):
        """
        Programmatically fire the Stop action (used by keyboard shortcut).
        """
        if self._running:
            self._on_stop()

    def get_params(self) -> dict:
        """
        Return the current parameter dict without emitting any signal.

        Returns
        -------
        params : dict
            Mapping of parameter key → current widget value.
        """
        return {k: self._widget_value(w) for k, w in self._param_widgets.items()}
