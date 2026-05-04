"""
Qubit analysis classes — T1, Ramsey, SpinEcho, PowerRabi, etc.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag


# ── T1 ────────────────────────────────────────────────────────────────────────

class T1Analysis(BaseAnalysis):
    """Exponential decay fit → T1 relaxation time."""

    thresholds = {
        "T1_us": {"min": 0.5, "max": 2000.0},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitexp, expfunc

        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitexp(x, y)
            err = np.sqrt(np.diag(pcov))
            T1 = abs(float(popt[2]))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T1_us": (T1, err[2]),
                "amplitude": (popt[0], err[0]),
                "offset": (popt[1], err[1]),
            }
            data.scalar_result = T1
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"T1 fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        from ..tools.fitting import expfunc
        x_fit = np.linspace(data.x_axis[0], data.x_axis[-1], 400)
        T1 = data.fit_result.get("T1_us", (None,))[0]
        self._show_fit(
            data, expfunc(x_fit, *data.fit_params),
            xlabel="Wait time (µs)", ylabel="ADC (Abs)",
            title=f"T1 Relaxation  |  T1 = {T1:.2f} µs" if T1 else "T1 Relaxation",
            fit_label=f"T1 = {T1:.2f} µs" if T1 else "fit",
        )


# ── Ramsey ────────────────────────────────────────────────────────────────────

class RamseyAnalysis(BaseAnalysis):
    """Damped sinusoid fit → T2*, frequency detuning."""

    thresholds = {
        "T2r_us": {"min": 0.1, "max": 1000.0},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return

        ramsey_freq = data.config.get("ramsey_freq", 0.0)

        if ramsey_freq != 0:
            self._fit_decaysin(data)
        else:
            self._fit_exp(data)

    def _fit_decaysin(self, data: ExperimentData) -> None:
        from ..tools.fitting import fitdecaysin, decaysin

        x = data.x_axis
        # Use best channel (abs works for most cases)
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitdecaysin(x, y)
            err = np.sqrt(np.diag(pcov))
            T2r = abs(float(popt[3]))
            detune = float(popt[1])
            ramsey_freq = data.config.get("ramsey_freq", 0.0)
            qb_freq = data.config.get("qb_freq_ge", 0.0)
            corrected_freq = qb_freq - round(detune - ramsey_freq, 2)

            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T2r_us": (T2r, err[3]),
                "detune_MHz": (detune, err[1]),
                "corrected_freq_MHz": (corrected_freq, None),
                "amplitude": (popt[0], err[0]),
            }
            data.scalar_result = T2r
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Ramsey decaysin fit failed: {exc}"

    def _fit_exp(self, data: ExperimentData) -> None:
        from ..tools.fitting import fitexp, expfunc

        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitexp(x, y)
            err = np.sqrt(np.diag(pcov))
            T2r = abs(float(popt[2]))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {"T2r_us": (T2r, err[2])}
            data.scalar_result = T2r
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Ramsey exp fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        x_fit = np.linspace(data.x_axis[0], data.x_axis[-1], 400)
        T2r = data.fit_result.get("T2r_us", (None,))[0]
        detune = data.fit_result.get("detune_MHz", (None,))[0]
        title = "Ramsey"
        if T2r:
            title += f"  |  T2* = {T2r:.2f} µs"
        if detune:
            title += f",  detune = {detune:.3f} MHz"
        ramsey_freq = data.config.get("ramsey_freq", 0.0)
        if ramsey_freq != 0:
            from ..tools.fitting import decaysin
            fit_y = decaysin(x_fit, *data.fit_params)
        else:
            from ..tools.fitting import expfunc
            fit_y = expfunc(x_fit, *data.fit_params)
        self._show_fit(
            data, fit_y,
            xlabel="Free evolution time (µs)", ylabel="ADC (Abs)",
            title=title,
            fit_label=f"T2* = {T2r:.2f} µs" if T2r else "fit",
        )


# ── SpinEcho ──────────────────────────────────────────────────────────────────

class SpinEchoAnalysis(BaseAnalysis):
    """Hahn echo fit → T2E."""

    thresholds = {
        "T2e_us": {"min": 0.1, "max": 5000.0},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return

        ramsey_freq = data.config.get("ramsey_freq", 0.0)
        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            if ramsey_freq != 0:
                from ..tools.fitting import fitdecaysin

                popt, pcov, _ = fitdecaysin(x, y)
                err = np.sqrt(np.diag(pcov))
                T2e = abs(float(popt[3]))
                data.fit_params = np.array(popt)
                data.fit_errors = err
            else:
                from ..tools.fitting import fitexp

                popt, pcov, _ = fitexp(x, y)
                err = np.sqrt(np.diag(pcov))
                T2e = abs(float(popt[2]))
                data.fit_params = np.array(popt)
                data.fit_errors = err

            data.fit_result = {"T2e_us": (T2e, None)}
            data.scalar_result = T2e
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"SpinEcho fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        x_fit = np.linspace(data.x_axis[0], data.x_axis[-1], 400)
        T2e = data.fit_result.get("T2e_us", (None,))[0]
        ramsey_freq = data.config.get("ramsey_freq", 0.0)
        if ramsey_freq != 0:
            from ..tools.fitting import decaysin
            fit_y = decaysin(x_fit, *data.fit_params)
        else:
            from ..tools.fitting import expfunc
            fit_y = expfunc(x_fit, *data.fit_params)
        self._show_fit(
            data, fit_y,
            xlabel="Echo time (µs)", ylabel="ADC (Abs)",
            title=f"Spin Echo  |  T2E = {T2e:.2f} µs" if T2e else "Spin Echo",
            fit_label=f"T2E = {T2e:.2f} µs" if T2e else "fit",
        )


# ── PowerRabi ─────────────────────────────────────────────────────────────────

class PowerRabiAnalysis(BaseAnalysis):
    """Decaying sinusoid → π gain and π/2 gain."""

    thresholds = {
        "pi_gain": {"min": 0.05, "max": 0.95},
    }

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitdecaysin, fix_phase

        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitdecaysin(x, y)
            err = np.sqrt(np.diag(pcov))
            pi_gain, pi2_gain = fix_phase(popt)
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "pi_gain": (round(pi_gain, 6), None),
                "pi2_gain": (round(pi2_gain, 6), None),
            }
            data.scalar_result = pi_gain
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"PowerRabi fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        from ..tools.fitting import decaysin
        x_fit = np.linspace(data.x_axis[0], data.x_axis[-1], 400)
        pi_gain = data.fit_result.get("pi_gain", (None,))[0]
        pi2_gain = data.fit_result.get("pi2_gain", (None,))[0]
        extra = []
        if pi_gain:
            extra.append({"x": pi_gain, "color": "r", "ls": "--", "lw": 1, "label": f"π={pi_gain:.4f}"})
        if pi2_gain:
            extra.append({"x": pi2_gain, "color": "g", "ls": "--", "lw": 1, "label": f"π/2={pi2_gain:.4f}"})
        self._show_fit(
            data, decaysin(x_fit, *data.fit_params),
            xlabel="Gain (a.u.)", ylabel="ADC (Abs)",
            title=f"Power Rabi  |  π gain = {pi_gain:.4f}" if pi_gain else "Power Rabi",
            fit_label="fit", extra_lines=extra,
        )


# ── TimeRabi ──────────────────────────────────────────────────────────────────

class TimeRabiAnalysis(BaseAnalysis):
    """Sinusoidal fit to time-domain Rabi → pi_length."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitdecaysin

        x = data.x_axis
        y = np.abs(data.raw_iq)

        try:
            popt, pcov, _ = fitdecaysin(x, y)
            err = np.sqrt(np.diag(pcov))
            # pi time = 1 / (2 * frequency)
            pi_length = 1.0 / (2.0 * abs(popt[1])) if popt[1] != 0 else 0.0
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {"pi_length_us": (pi_length, None)}
            data.scalar_result = pi_length
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"TimeRabi fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        if data.fit_params is None:
            return
        from ..tools.fitting import decaysin
        x_fit = np.linspace(data.x_axis[0], data.x_axis[-1], 400)
        pi_len = data.fit_result.get("pi_length_us", (None,))[0]
        self._show_fit(
            data, decaysin(x_fit, *data.fit_params),
            xlabel="Pulse length (µs)", ylabel="ADC (Abs)",
            title=f"Time Rabi  |  π time = {pi_len:.3f} µs" if pi_len else "Time Rabi",
            fit_label=f"π = {pi_len:.3f} µs" if pi_len else "fit",
        )


# ── QubitTemp ─────────────────────────────────────────────────────────────────

class QubitTempAnalysis(BaseAnalysis):
    """Qubit temperature estimation from |e⟩ population."""

    thresholds = {
        "T_mK": {"max": 300.0},
    }

    def _run(self, data: ExperimentData) -> None:
        # Population ratio → temperature via Boltzmann
        n_e = data.fit_result.get("n_excited", (None, None))[0]
        f_ge = data.config.get("qb_freq_ge", 5000.0)  # MHz

        if n_e is not None and 0 < n_e < 1:
            import scipy.constants as const

            f_hz = f_ge * 1e6
            ratio = n_e / (1 - n_e)
            if ratio > 0:
                T_K = (const.h * f_hz) / (const.k * np.log(1.0 / ratio))
                T_mK = T_K * 1000
                data.fit_result["T_mK"] = (round(T_mK, 1), None)
                data.scalar_result = T_mK


# ── SingleShot ────────────────────────────────────────────────────────────────

class SingleShotAnalysis(BaseAnalysis):
    """Single-shot readout quality metrics — fidelity, SNR, angle."""

    thresholds = {
        "fidelity": {"min": 0.85},
    }

    def _run(self, data: ExperimentData) -> None:
        # Actual analysis is handled inside SingleShot_gef.plot()
        # This class just wraps whatever the experiment already computed.
        if "fidelity" not in data.fit_result:
            data.quality = QualityFlag.NO_INFORMATION
