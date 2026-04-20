"""
Synthetic data generator for --debug mode.

Each experiment class gets a realistic-looking signal so the live plot
and fit overlays can be exercised without real hardware.

All generators return (x, iq, xlabel, title) matching the
``AcquireWorker.data_ready`` signal contract:
  x      -- 1-D numpy array (sweep axis values)
  iq     -- complex numpy array; real=avgi, imag=avgq
  xlabel -- axis label string
  title  -- plot title string
"""

import time
import numpy as np


# ── Noise helper ──────────────────────────────────────────────────────────────

def _noise(n: int, scale: float = 30.0) -> np.ndarray:
    return np.random.randn(n) * scale


# ── Per-pattern generators ────────────────────────────────────────────────────

def _lorentzian_dip(x: np.ndarray, depth: float = 900.,
                    lw: float = 2.) -> np.ndarray:
    """Lorentzian dip centred randomly within the sweep range."""
    cx = np.random.uniform(x[len(x)//4], x[3*len(x)//4])
    return -depth / (1 + ((x - cx) / lw) ** 2) + _noise(len(x))


def _lorentzian_peak(x: np.ndarray, height: float = 800.,
                     lw: float = 2.) -> np.ndarray:
    """Lorentzian peak centred randomly within the sweep range."""
    cx = np.random.uniform(x[len(x)//4], x[3*len(x)//4])
    return height / (1 + ((x - cx) / lw) ** 2) + _noise(len(x))


def _decaying_cosine(x: np.ndarray, period: float | None = None,
                     tau: float | None = None,
                     amp: float = 700.) -> np.ndarray:
    span = x[-1] - x[0]
    T    = period if period else span / 2.5
    t    = tau    if tau    else span / 1.5
    return amp * np.cos(2 * np.pi * (x - x[0]) / T) * np.exp(-(x - x[0]) / t) + _noise(len(x))


def _exp_decay(x: np.ndarray, tau: float | None = None,
               amp: float = 800.) -> np.ndarray:
    t = tau if tau else (x[-1] - x[0]) / 2.5
    return amp * np.exp(-(x - x[0]) / t) + _noise(len(x))


def _allxy_pattern(n_points: int = 21) -> np.ndarray:
    """Ideal AllXY sequence (21 gate pairs → expected I values)."""
    ideal = np.array([
        0, 0,                    # II, XX
        0, 0,                    # YY, XY
        0,                       # YX
        1, 1, 1, 1, 1,           # X/2 I, Y/2 I, X/2 Y/2, Y/2 X/2, X/2 Y
        1, 1, 1, 1, 1,           # Y/2 X, X/2 X, X/2 Y/2, Y/2 X/2, X/2 −X/2
        0.5, 0.5, 0.5, 0.5, 0.5, # Y/2 −Y/2, X/2 −Y/2, Y/2 −X/2, −X/2 Y/2, −Y/2 X/2
        0.5,                     # −X/2 −Y/2
    ], dtype=float)
    # trim or pad to n_points
    ideal = ideal[:n_points]
    return 800. * ideal + _noise(n_points, 20)


def _rb_decay(x: np.ndarray) -> np.ndarray:
    """Exponential decay for RB survival probability vs Clifford depth."""
    epc = np.random.uniform(0.003, 0.015)
    p   = 1 - epc
    return 400. * p ** x + 400. + _noise(len(x), 15)


def _tof_pulse(x: np.ndarray) -> np.ndarray:
    """Step-function readout pulse shape for Time of Flight."""
    pulse_start = x[len(x) // 3]
    pulse_end   = x[2 * len(x) // 3]
    y = np.where((x >= pulse_start) & (x <= pulse_end), 800., 0.)
    return y + _noise(len(x), 25)


def _iq_cloud_1d(n: int) -> np.ndarray:
    """Flat I projection of two IQ blobs (g/e states)."""
    ng  = n // 2
    ne  = n - ng
    g   = np.random.randn(ng) * 80 + 500
    e   = np.random.randn(ne) * 80 - 300
    return np.concatenate([g, e])


# ── Dispatch table ────────────────────────────────────────────────────────────

def generate_mock_data(class_path: str, params: dict,
                       sleep: float = 1.2) -> tuple:
    """
    Generate synthetic experiment data for the requested class.

    Parameters
    ----------
    class_path : str
        ``"module.ClassName"`` from EXPERIMENT_REGISTRY.
    params : dict
        GUI parameter dict; used for sweep axis construction.
    sleep : float, optional
        Simulated acquisition time in seconds (default 1.2).

    Returns
    -------
    x : numpy.ndarray
        Sweep axis values.
    iq : numpy.ndarray
        Complex IQ data: ``real=avgi``, ``imag=avgq``.
    xlabel : str
        Sweep axis label.
    title : str
        Plot title.
    """
    time.sleep(sleep)

    cls = class_path.split(".")[-1].lower()
    rng = np.random.default_rng()

    # ── Time of Flight ────────────────────────────────────────────────────────
    if "tof" in cls:
        x      = np.linspace(0, 2.5, 250)
        avgi   = _tof_pulse(x)
        avgq   = _noise(len(x), 15)
        xlabel = "Time (µs)"
        title  = "Time of Flight"

    # ── Resonator spec (1-tone) ───────────────────────────────────────────────
    elif "resonatorspec" in cls or ("resonator" in cls and "spec" in cls):
        fc     = params.get("freq_center", 6500.)
        span   = params.get("freq_span",   10.)
        steps  = int(params.get("steps", params.get("freq_steps", 101)))
        x      = np.linspace(fc - span, fc + span, steps)
        avgi   = _lorentzian_dip(x, depth=900, lw=1.5)
        avgq   = _noise(steps, 20)
        xlabel = "Frequency (MHz)"
        title  = "Resonator Spec"

    # ── Punchout (2-D → return 1-D mag slice) ────────────────────────────────
    elif "punchout" in cls:
        fc     = params.get("freq_center", 6500.)
        span   = params.get("freq_span",   20.)
        steps  = int(params.get("freq_steps", 101))
        x      = np.linspace(fc - span, fc + span, steps)
        avgi   = _lorentzian_dip(x, depth=700, lw=2.)
        avgq   = _noise(steps, 20)
        xlabel = "Frequency (MHz)"
        title  = "Punchout (gain mid-slice)"

    # ── Qubit spec ────────────────────────────────────────────────────────────
    elif "qubitspec" in cls or ("qubit" in cls and "spec" in cls):
        fc     = params.get("freq_center", 3620.)
        span   = params.get("freq_span",   50.)
        steps  = int(params.get("steps", params.get("freq_steps", 100)))
        x      = np.linspace(fc - span, fc + span, steps)
        avgi   = _lorentzian_peak(x, height=800, lw=2.)
        avgq   = _noise(steps, 20)
        xlabel = "Frequency (MHz)"
        title  = "Qubit Spec"

    # ── Time Rabi ─────────────────────────────────────────────────────────────
    elif "timerabi" in cls:
        start  = params.get("length_start", 0.01)
        stop   = params.get("length_stop",  0.30)
        steps  = int(params.get("steps", 81))
        x      = np.linspace(start, stop, steps)
        avgi   = _decaying_cosine(x, period=(stop-start)/3.5)
        avgq   = _noise(steps, 20)
        xlabel = "Pulse length (µs)"
        title  = "Time Rabi"

    # ── Power Rabi ────────────────────────────────────────────────────────────
    elif "powerrabi" in cls or "powerrabi_ef" in cls:
        start  = params.get("gain_start", 0.0)
        stop   = params.get("gain_stop",  1.0)
        steps  = int(params.get("steps", 100))
        x      = np.linspace(start, stop, steps)
        avgi   = _decaying_cosine(x, period=(stop-start)/2.5)
        avgq   = _noise(steps, 20)
        xlabel = "Gain (DAC)"
        title  = "Power Rabi"

    # ── DRAG calibration ──────────────────────────────────────────────────────
    elif "dragcalibration" in cls:
        start  = params.get("alpha_start", -1.0)
        stop   = params.get("alpha_stop",   1.5)
        steps  = int(params.get("alpha_steps", 41))
        x      = np.linspace(start, stop, steps)
        avgi   = 700. * np.exp(-((x - 0.2) ** 2) / 0.3) + _noise(steps)
        avgq   = _noise(steps, 20)
        xlabel = "DRAG α"
        title  = "DRAG Calibration"

    # ── AAE (Power Rabi Chevron) ───────────────────────────────────────────────
    elif "powerrabichevron" in cls:
        span   = params.get("gain_half_span", 0.1)
        steps  = int(params.get("steps", 100))
        x      = np.linspace(-span, span, steps)
        avgi   = 700. * np.cos(np.pi * x / span * 3) * np.exp(-abs(x) / span) + _noise(steps)
        avgq   = _noise(steps, 20)
        xlabel = "ΔGain around π-gain"
        title  = "AAE"

    # ── Ramsey (GE and EF) ────────────────────────────────────────────────────
    elif "ramsey" in cls:
        start  = params.get("time_start", 0.0)
        stop   = params.get("time_stop",  2.0)
        steps  = int(params.get("steps", 100))
        fdet   = params.get("ramsey_freq", 2.0)
        x      = np.linspace(start, stop, steps)
        tau    = (stop - start) / 2.0
        avgi   = 700. * np.cos(2*np.pi*fdet*x) * np.exp(-x/tau) + _noise(steps)
        avgq   = 700. * np.sin(2*np.pi*fdet*x) * np.exp(-x/tau) + _noise(steps)
        xlabel = "Wait time (µs)"
        title  = "Ramsey"

    # ── Spin Echo ─────────────────────────────────────────────────────────────
    elif "spinecho" in cls:
        start  = params.get("time_start", 0.0)
        stop   = params.get("time_stop",  150.)
        steps  = int(params.get("steps", 100))
        fdet   = params.get("ramsey_freq", 0.05)
        x      = np.linspace(start, stop, steps)
        tau    = (stop - start) / 2.5
        avgi   = 700. * np.cos(2*np.pi*fdet*x) * np.exp(-x/tau) + _noise(steps)
        avgq   = _noise(steps, 20)
        xlabel = "Wait time (µs)"
        title  = "Spin Echo"

    # ── T1 (GE and EF) ────────────────────────────────────────────────────────
    elif "t1" in cls:
        start  = params.get("time_start", 0.0)
        stop   = params.get("time_stop",  400.)
        steps  = int(params.get("steps", 100))
        x      = np.linspace(start, stop, steps)
        avgi   = _exp_decay(x, tau=(stop-start)/2.5)
        avgq   = _noise(steps, 20)
        xlabel = "Wait time (µs)"
        title  = "T1"

    # ── SingleShot ────────────────────────────────────────────────────────────
    elif "singleshot" in cls:
        shots  = int(params.get("shots", 5000))
        x      = np.arange(shots, dtype=float)
        avgi   = _iq_cloud_1d(shots)
        avgq   = _iq_cloud_1d(shots) * 0.3 + _noise(shots, 60)
        xlabel = "Shot index"
        title  = "Single Shot"

    # ── AllXY ─────────────────────────────────────────────────────────────────
    elif "allxy" in cls:
        x      = np.arange(21, dtype=float)
        avgi   = _allxy_pattern(21)
        avgq   = _noise(21, 15)
        xlabel = "Gate pair index"
        title  = "AllXY"

    # ── Tomography ────────────────────────────────────────────────────────────
    elif "tomography" in cls:
        x      = np.arange(6, dtype=float)
        # |+X⟩ state: expect [Ix Iy Iz Xx Xy Xz] ≈ [0,0,0,1,0,0]
        avgi   = np.array([0, 0, 0, 800, 50, 30], dtype=float) + _noise(6, 30)
        avgq   = _noise(6, 20)
        xlabel = "Tomography basis"
        title  = "State Tomography"

    # ── Randomized Benchmarking ───────────────────────────────────────────────
    elif "randomizedbenchmarking" in cls or "autorb" in cls or "rbasm" in cls:
        depth  = int(params.get("max_circuit_depth", 400))
        delta  = int(params.get("delta_clifford",    40))
        x      = np.arange(0, depth + 1, delta, dtype=float)
        avgi   = _rb_decay(x)
        avgq   = _noise(len(x), 10)
        xlabel = "Clifford depth"
        title  = "Randomized Benchmarking"

    # ── Qubit Temperature ─────────────────────────────────────────────────────
    elif "qubittemperature" in cls:
        start  = params.get("gain_start", 0.0)
        stop   = params.get("gain_stop",  1.0)
        steps  = int(params.get("steps", 100))
        x      = np.linspace(start, stop, steps)
        avgi   = _decaying_cosine(x, period=(stop-start)/2.5, amp=400.)
        avgq   = _noise(steps, 20)
        xlabel = "EF Gain (DAC)"
        title  = "Qubit Temperature"

    # ── AC Stark ──────────────────────────────────────────────────────────────
    elif "acstark" in cls:
        steps  = int(params.get("steps", 100))
        x      = np.linspace(0, 1, steps)
        avgi   = 500. * x + _noise(steps, 30)
        avgq   = _noise(steps, 20)
        xlabel = "Drive amplitude"
        title  = "AC Stark Shift"

    # ── Flux Spec (ResonatorSpecFlux / QubitSpecFlux) ─────────────────────────
    elif "flux" in cls and "spec" in cls:
        fc     = params.get("freq_center", 6500.)
        span   = params.get("freq_span",   50.)
        steps  = int(params.get("freq_steps", params.get("steps", 101)))
        x      = np.linspace(fc - span, fc + span, steps)
        # Anticrossing-like dip
        cx     = fc - span * 0.2
        avgi   = _lorentzian_dip(x, depth=800, lw=3.)
        avgq   = _noise(steps, 20)
        xlabel = "Frequency (MHz)"
        title  = "Flux Spec"

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        steps  = int(params.get("steps", 100))
        x      = np.linspace(0, 1, steps)
        avgi   = _noise(steps, 50)
        avgq   = _noise(steps, 50)
        xlabel = "Sweep axis"
        title  = cls

    iq = avgi.astype(complex) + 1j * avgq
    return x, iq, xlabel, title
