"""
SimulatedBackend — software mock backend for offline development and testing.

Generates realistic synthetic IQ data for all experiment types without
requiring QICK hardware.  Replaces ``gui/mock/hardware.py`` and
``gui/mock/runner.py`` but is also usable in notebooks and CLI scripts.

Usage
-----
::

    backend = SimulatedBackend()
    expt = T1(cfg, backend=backend)
    result = expt.run(py_avg=10)   # runs offline, returns realistic synthetic data
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base_backend import BaseBackend

# ── Mock SoC objects ──────────────────────────────────────────────────────────


class _MockSoCCfg:
    """Minimal mock of QickConfig needed by BaseProgram helpers."""

    def __init__(self):
        self._gens = [{"type": "axis_signal_gen_v6", "maxlen": 65536} for _ in range(8)]
        self._readouts = [{"tproc_ctrl": 0} for _ in range(4)]

    def __getitem__(self, key):
        if key == "gens":
            return self._gens
        if key == "readouts":
            return self._readouts
        return {}

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default


class _MockSoC:
    """Minimal mock of QickSoc — provides acquire() interface."""

    is_simulated = (
        True  # sentinel checked by BaseExperiment to skip real program creation
    )

    def __init__(self, noise_level: float = 0.02):
        self.noise_level = noise_level
        self._cfg = _MockSoCCfg()

    def get_cfg(self):
        return {}


# ── Synthetic data generators ─────────────────────────────────────────────────


def _make_lorentzian(freqs, f0, kappa, amp=1.0, noise=0.02):
    """Complex Lorentzian peak used for spectroscopy-like mock traces."""
    signal = amp / (1 + 2j * (freqs - f0) / kappa)
    return signal + noise * (
        np.random.randn(*freqs.shape) + 1j * np.random.randn(*freqs.shape)
    )


def _make_notch_resonator(freqs, f0, kappa, depth=0.75, phi=0.25, noise=0.02):
    """Complex hanger/notch-style S21 response with a dip at f0."""
    detuning = 2.0 * (freqs - f0) / kappa
    notch = 1.0 - depth * np.exp(1j * phi) / (1.0 + 1j * detuning)
    # Add a small cable-delay-like phase slope so IQ traces look realistic.
    phase_slope = np.exp(1j * 0.015 * (freqs - np.mean(freqs)))
    signal = notch * phase_slope
    return signal + noise * (
        np.random.randn(*freqs.shape) + 1j * np.random.randn(*freqs.shape)
    )


def _make_decaysin(times, amp, freq, phase, tau, offset=1.0, noise=0.02):
    """Positive IQ-amplitude damped sinusoid for Ramsey/SpinEcho mock traces."""
    signal = offset + amp * np.exp(-times / tau) * np.sin(2 * np.pi * freq * times + phase)
    return signal + noise * np.random.randn(*times.shape)


def _make_expfunc(times, amp, offset, tau, noise=0.02):
    """Positive exponential decay (T1)."""
    signal = amp * np.exp(-times / tau) + offset
    return signal + noise * np.random.randn(*times.shape)


def _make_rabi(gains, pi_gain, amp=0.45, offset=0.6, decay=3.0, noise=0.02):
    """Positive IQ-amplitude Rabi oscillation compatible with abs(IQ) analysis."""
    signal = offset + amp * np.cos(np.pi * gains / pi_gain) * np.exp(-gains / decay)
    return signal + noise * np.random.randn(*gains.shape)


# ── Mock program wrapper ───────────────────────────────────────────────────────


class _MockProgram:
    """
    Mock QICK program that generates synthetic IQ data on acquire().

    The experiment type is inferred from the config to select the correct
    synthetic data model.
    """

    def __init__(self, cfg: dict, experiment_hint: str = ""):
        self.cfg = cfg
        self.experiment_hint = experiment_hint.lower()
        self._x_array: Optional[np.ndarray] = None

    @staticmethod
    def _scalar(value, default: float) -> float:
        """Return a representative scalar for plain values or QICK sweeps."""
        if value is None:
            return float(default)
        for attr in ("start", "stop"):
            if not hasattr(value, attr):
                break
        else:
            try:
                return 0.5 * (float(value.start) + float(value.stop))
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return float(default)

    @classmethod
    def _axis_from_cfg(cls, cfg: dict, key: str, default_center: float, span: float, steps_key: str = "steps"):
        value = cfg.get(key)
        steps = int(cfg.get(steps_key, cfg.get("steps", 101)))
        if value is not None:
            for method in ("to_array", "sweep_vals", "get_array"):
                fn = getattr(value, method, None)
                if callable(fn):
                    try:
                        return np.asarray(fn(), dtype=float)
                    except Exception:
                        pass
            start = getattr(value, "start", None)
            stop = getattr(value, "stop", None)
            if start is not None and stop is not None:
                try:
                    return np.linspace(float(start), float(stop), steps)
                except Exception:
                    pass
            try:
                center = float(value)
                return np.linspace(center - span, center + span, steps)
            except Exception:
                pass
        return np.linspace(default_center - span, default_center + span, steps)

    def acquire(self, soc, rounds: int = 1, progress: bool = False):
        """Return synthetic IQ data in QICK acquire() format."""
        x = self._x_array if self._x_array is not None else np.linspace(0, 1, 101)
        n = len(x)
        noise = getattr(soc, "noise_level", 0.02)

        hint = self.experiment_hint

        if "res_spec" in hint or "onetone" in hint:
            f0 = self._scalar(self.cfg.get("res_freq_ge"), 6700.0)
            kappa = self.cfg.get("kappa", 5.0)
            depth = self.cfg.get("notch_depth", 0.75)
            phi = self.cfg.get("notch_phi", 0.25)
            iq = _make_notch_resonator(x, f0, kappa, depth=depth, phi=phi, noise=noise)

        elif "t1" in hint:
            tau = self.cfg.get("T1_true", 40.0)
            iq = _make_expfunc(x, 0.8, 0.2, tau, noise=noise).astype(complex)

        elif "ramsey" in hint or "spin_echo" in hint:
            tau = self.cfg.get("T2_true", 15.0)
            freq = self.cfg.get("ramsey_freq", 2.0)
            iq = _make_decaysin(x, 0.35, freq, 0.0, tau, offset=0.7, noise=noise).astype(complex)

        elif "rabi" in hint or "power_rabi" in hint:
            pi_gain = self.cfg.get("pi_gain_true", 0.5)
            iq = _make_rabi(x, pi_gain, noise=noise).astype(complex)

        elif "qubit_spec" in hint or "twotone" in hint:
            f0 = self._scalar(self.cfg.get("qb_freq_ge"), 5000.0)
            kappa = self.cfg.get("qb_kappa", 2.0)
            iq = _make_lorentzian(x, f0, kappa, noise=noise)

        else:
            # Generic: flat noisy signal
            iq = noise * (np.random.randn(n) + 1j * np.random.randn(n))

        # Pack into QICK acquire() format: list of [array([I, Q])]
        iq_iq = np.stack([iq.real, iq.imag], axis=-1)
        return [[iq_iq]]

    def get_pulse_param(self, pulse_name: str, param: str, as_array: bool = False):
        """Return a plausible sweep axis for the given pulse parameter."""
        cfg = self.cfg
        if param == "freq":
            if "res" in pulse_name:
                arr = self._axis_from_cfg(cfg, "res_freq_ge", 6700.0, 10.0)
            else:
                arr = self._axis_from_cfg(cfg, "qb_freq_ge", 5000.0, 50.0)
        elif param == "gain":
            key = "qb_gain_ge" if "qb" in pulse_name else "res_gain_ge"
            arr = self._axis_from_cfg(cfg, key, 0.5, 0.5)
        elif param == "length":
            arr = self._axis_from_cfg(cfg, "qb_flat_top_length_ge", 0.5, 0.5)
        else:
            arr = self._axis_from_cfg(cfg, param, 0.5, 0.5)
        self._x_array = arr
        return arr if as_array else arr

    def get_time_param(self, tag: str, attr: str, as_array: bool = False):
        """Return a plausible time axis."""
        cfg = self.cfg
        arr = self._axis_from_cfg(cfg, "wait_time", 50.0, 50.0)
        self._x_array = arr
        return arr if as_array else arr


# ── SimulatedBackend ──────────────────────────────────────────────────────────


class SimulatedBackend(BaseBackend):
    """
    Software mock backend — no QICK hardware required.

    Parameters
    ----------
    noise_level : float, optional
        Gaussian noise amplitude (default 0.02).
    """

    def __init__(self, noise_level: float = 0.02):
        soc = _MockSoC(noise_level=noise_level)
        soccfg = _MockSoCCfg()
        super().__init__(soc, soccfg, name="SimulatedBackend")
        self.noise_level = noise_level

    def run_program(self, prog, rounds: int = 1) -> Any:
        """Return synthetic IQ data."""
        return prog.acquire(self.soc, rounds=rounds, progress=False)

    def is_simulated(self) -> bool:
        return True

    def make_mock_program(self, cfg: dict, experiment_hint: str = "") -> _MockProgram:
        """
        Create a ``_MockProgram`` that generates synthetic data.

        Called by experiment classes that override ``_create_program`` to
        support the simulated backend.
        """
        return _MockProgram(cfg, experiment_hint)

    def activate(self):
        """Set as global legacy session."""
        from QickworkspaceV2.core.base_experiment import BaseExperiment

        BaseExperiment.setup(self.soc, self.soccfg, "")
        print("[SimulatedBackend] Session activated (no hardware required)")
