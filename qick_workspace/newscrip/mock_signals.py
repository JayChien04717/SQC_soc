"""
Mock Signal Generators for QICK Experiment Simulation
=====================================================
Pure-numpy functions that produce realistic IQ data for each experiment type.
All functions return complex arrays: I + jQ.
"""
import numpy as np


def _add_noise(signal, noise_sigma=0.02):
    """Add complex Gaussian noise to a signal."""
    noise = noise_sigma * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


# ── Spectroscopy ──

def mock_lorentzian(x, f0=None, gamma=2, amp=1.0, offset=0.5, noise=0.02):
    """
    Lorentzian dip for resonator/qubit spectroscopy.
    Returns complex IQ with a dip centered at f0.
    """
    if f0 is None:
        f0 = (x[0] + x[-1]) / 2  # center of sweep
    mag = offset - amp * gamma**2 / ((x - f0)**2 + gamma**2)
    phase = np.arctan2(gamma, x - f0)
    signal = mag * np.exp(1j * phase)
    return _add_noise(signal, noise)


def mock_lorentzian_2d(x_freq, y_gain, f0=None, gamma_base=2, noise=0.02):
    """
    2D Lorentzian for punchout: dip narrows/shifts with gain.
    Returns complex 2D array [len(y_gain), len(x_freq)].
    """
    if f0 is None:
        f0 = (x_freq[0] + x_freq[-1]) / 2
    result = np.zeros((len(y_gain), len(x_freq)), dtype=complex)
    for i, g in enumerate(y_gain):
        gamma = gamma_base * (1 + 0.5 * g)
        shift = -0.5 * g
        result[i] = mock_lorentzian(x_freq, f0=f0 + shift, gamma=gamma,
                                     amp=g * 5, offset=0.5 + g, noise=noise)
    return result


# ── Rabi / oscillation ──

def mock_decaysin(x, amp=0.5, freq=2.0, decay=3.0, phase=0, offset=0.5, noise=0.02):
    """
    Decaying sinusoid for Power Rabi, Ramsey, Spin Echo.
    """
    mag = offset + amp * np.sin(2 * np.pi * freq * x + phase) * np.exp(-x / decay)
    signal = mag * (1 + 0.1j)  # slight IQ angle
    return _add_noise(signal, noise)


def mock_rabi_time(x, amp=0.5, freq=5.0, decay=2.0, offset=0.5, noise=0.02):
    """
    Time Rabi: decaying sinusoid vs pulse length.
    """
    return mock_decaysin(x, amp=amp, freq=freq, decay=decay, offset=offset, noise=noise)


# ── Exponential decay ──

def mock_exp_decay(x, amp=0.5, tau=20.0, offset=0.0, noise=0.02):
    """
    Exponential decay for T1.
    """
    mag = offset + amp * np.exp(-x / tau)
    signal = mag * (1 + 0.1j)
    return _add_noise(signal, noise)


# ── AllXY ──

def mock_allxy(noise=0.02):
    """
    21-gate AllXY staircase pattern.
    Returns complex array of 21 points.
    Ideal: [0,0, 0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5, 1,1,1,1,1, 0.5,0.5,0.5,0.5,0.5, 0.5]
    """
    ideal = np.array([
        0.0, 0.0,                                  # II, XX
        0.5, 0.5, 0.5,                              # XY, YX, XI
        0.5, 0.5, 0.5, 0.5, 0.5,                    # YI, X/2·I, Y/2·I, X/2·Y/2, Y/2·X/2
        1.0, 1.0, 1.0, 1.0, 1.0,                    # X/2·Y, Y/2·X, X·Y/2, Y·X/2, XI
        0.5, 0.5, 0.5, 0.5, 0.5,                    # X/2, Y/2, ...
        0.5,                                         # last
    ])
    mag = ideal + noise * np.random.randn(len(ideal))
    return mag * (1 + 0.1j)


# ── Randomized Benchmarking ──

def mock_rb(depths, p=0.99, amp=0.5, offset=0.5, noise=0.02):
    """
    RB exponential decay: F(m) = A * p^m + B.
    """
    mag = offset + amp * p**np.array(depths)
    signal = mag * (1 + 0.1j)
    return _add_noise(signal, noise)


# ── Single Shot ──

def mock_singleshot(n_shots, separation=3.0, sigma=1.0, include_f=False, noise=0.0):
    """
    Gaussian blobs in IQ plane for single-shot readout.
    Returns dict with 'ig', 'qg', 'ie', 'qe' (and 'if', 'qf' if include_f).
    """
    result = {}
    # Ground state blob
    result['ig'] = np.random.normal(0, sigma, n_shots)
    result['qg'] = np.random.normal(0, sigma, n_shots)
    # Excited state blob
    result['ie'] = np.random.normal(separation, sigma, n_shots)
    result['qe'] = np.random.normal(0, sigma, n_shots)
    if include_f:
        result['if'] = np.random.normal(separation * 0.7, sigma, n_shots)
        result['qf'] = np.random.normal(separation * 0.7, sigma, n_shots)
    return result


# ── Time of Flight ──

def mock_tof(t_pts, pulse_start=0.3, pulse_end=0.7, amp=1.0, noise=0.02):
    """
    Rectangular pulse envelope for TOF measurement.
    """
    t_norm = (t_pts - t_pts[0]) / (t_pts[-1] - t_pts[0])
    mask = (t_norm >= pulse_start) & (t_norm <= pulse_end)
    mag = np.where(mask, amp, 0.05)
    # Add rising/falling edge smoothing
    signal = mag * np.exp(1j * 0.3)
    return _add_noise(signal, noise)


# ── State Tomography ──

def mock_tomography(prep_pulse="x180", noise=0.02):
    """
    Mock tomography measurement for 6 measurement bases.
    Returns complex array of 6 values: [X+, X-, Y+, Y-, Z+, Z-].
    """
    # Simulate ideal Bloch vector for common prep pulses
    states = {
        "x180": [0, 0, -1],    # |1⟩
        "x90":  [0, 0, 0],     # |+⟩ in XZ plane
        "y90":  [0, 0, 0],     # |+i⟩ in YZ plane
        "idle": [0, 0, 1],     # |0⟩
    }
    bv = np.array(states.get(prep_pulse, [0, 0, 1]), dtype=float)
    # 6 bases: +X, -X, +Y, -Y, +Z, -Z
    projections = np.array([
        0.5 * (1 + bv[0]),   # X+
        0.5 * (1 - bv[0]),   # X-
        0.5 * (1 + bv[1]),   # Y+
        0.5 * (1 - bv[1]),   # Y-
        0.5 * (1 + bv[2]),   # Z+
        0.5 * (1 - bv[2]),   # Z-
    ])
    signal = projections * (1 + 0.1j)
    return _add_noise(signal, noise)


# ── AAE ──

def mock_aae(n_iterations, optimal_gain=0.5, noise=0.02):
    """
    Amplified Amplitude Error: parabolic error vs iteration.
    Error accumulates as a function of gate count.
    """
    gate_counts = np.arange(1, n_iterations + 1) * 2  # 2 gates per iteration
    error = 0.5 - 0.5 * np.cos(2 * np.pi * 0.01 * gate_counts)
    signal = error * (1 + 0.1j)
    return _add_noise(signal, noise)
