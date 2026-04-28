"""
Fitting module for quantum experiments.

This module provides functions for fitting various types of data from quantum experiments,
including exponential decays, sinusoids, Lorentzians, and more specialized functions like
Hanger resonator fits and randomized benchmarking.
"""

import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy as sp

# ====================================================== #
# Utility Functions
# ====================================================== #


def get_r2(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitfunc: Callable,
    fit_params: List[float],
) -> float:
    """
    Compute the coefficient of determination (R²) for a fit.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data points.
    ydata : np.ndarray
        Dependent variable data points.
    fitfunc : Callable
        Model function with signature f(x, *params).
    fit_params : list of float
        Fitted model parameters.

    Returns
    -------
    r2 : float
        R² value; 1.0 is a perfect fit, negative values indicate a worse-than-mean
        fit. Returns -inf when ss_tot is zero.
    """
    ss_res = np.sum((fitfunc(xdata, *fit_params) - ydata) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf


def fix_phase(p: List[float]) -> Tuple[float, float]:
    """
    Normalize phase and calculate pi and pi/2 gains from sinusoidal fit parameters.

    Wraps p[2] into [-180, 180] degrees and then computes the gain values
    corresponding to a pi rotation and a pi/2 rotation.

    Parameters
    ----------
    p : list of float
        Fit parameters ``[yscale, freq, phase_deg, y0]`` from a sinusoidal fit.

    Returns
    -------
    pi_gain : float
        Gain value for a pi rotation.
    pi2_gain : float
        Gain value for a pi/2 rotation.
    """
    phase = p[2]
    if phase > 180:
        phase -= 360
    elif phase < -180:
        phase += 360

    if phase < 0:
        pi_gain  = (1 / 2 - phase / 180) / 2 / p[1]
        pi2_gain = (0     - phase / 180) / 2 / p[1]
    else:
        pi_gain  = (3 / 2 - phase / 180) / 2 / p[1]
        pi2_gain = (1     - phase / 180) / 2 / p[1]
    return pi_gain, pi2_gain


def fourier_init(
    xdata: np.ndarray,
    ydata: np.ndarray,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Estimate oscillation frequency and phase using a real-valued FFT.

    Subtracts the mean from ydata before computing the FFT, skips the DC bin,
    and returns the frequency and phase of the dominant spectral component.

    Parameters
    ----------
    xdata : np.ndarray
        Uniformly spaced x-axis values (used to compute the sample spacing).
    ydata : np.ndarray
        Signal values corresponding to xdata.
    debug : bool, optional
        If True, displays a two-panel plot of FFT magnitude and phase.
        Default is False.

    Returns
    -------
    max_freq : float
        Frequency of the dominant spectral component.
    max_phase : float
        Phase (radians) of the dominant spectral component.
    """
    ydata = ydata - np.mean(ydata)
    fourier = np.fft.rfft(ydata)  # rfft: no redundant negative freqs
    fft_freqs = np.fft.rfftfreq(len(ydata), d=xdata[1] - xdata[0])

    # Skip DC bin (index 0)
    mag = np.abs(fourier[1:])
    phase = np.angle(fourier[1:])
    freqs = fft_freqs[1:]

    max_ind = np.argmax(mag)
    max_freq = freqs[max_ind]
    max_phase = phase[max_ind]

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 6))
        ax[0].plot(freqs, mag, ".")
        ax[0].set_ylabel("Amplitude")
        ax[1].plot(freqs, phase * 180 / np.pi, ".")
        ax[1].plot(max_freq, max_phase * 180 / np.pi, "ro")
        ax[1].set_xlabel("Frequency (MHz)")
        ax[1].set_ylabel("Phase (deg)")
        print(f"Max freq={max_freq:.4f}  Max phase={max_phase:.4f} rad")
        plt.tight_layout()
        plt.show()

    return max_freq, max_phase


def validate_bounds(
    fitparams: List[float],
    bounds: Tuple[List[float], List[float]],
) -> List[float]:
    """
    Validate that initial fit parameters lie within specified bounds.

    Any parameter outside its bounds is reset to the midpoint of the
    corresponding interval, and a warning is printed.

    Parameters
    ----------
    fitparams : list of float
        Initial parameter estimates to validate.
    bounds : tuple of (list of float, list of float)
        ``(lower_bounds, upper_bounds)`` with one entry per parameter.

    Returns
    -------
    fitparams : list of float
        Validated parameter list with out-of-bounds values replaced.
    """
    fitparams = list(fitparams)
    for i, (param, lo, hi) in enumerate(zip(fitparams, bounds[0], bounds[1])):
        below = np.isfinite(lo) and param <= lo
        above = np.isfinite(hi) and param >= hi
        if below or above:
            lo_safe = lo if np.isfinite(lo) else param - 1
            hi_safe = hi if np.isfinite(hi) else param + 1
            fitparams[i] = (lo_safe + hi_safe) / 2
            print(
                f"fitparam[{i}]={param:.4g} out of bounds [{lo:.4g}, {hi:.4g}]"
                f" → reset to {fitparams[i]:.4g}"
            )
    return fitparams


def generic_fit(
    fitfunc: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: List[float],
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    error_message: str = "Warning: fit failed!",
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Generic curve-fitting wrapper using scipy Trust Region Reflective.

    Validates initial parameters against bounds (if provided), then calls
    ``scipy.optimize.curve_fit`` with the TRF method.

    Parameters
    ----------
    fitfunc : Callable
        Model function with signature f(x, *params).
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float
        Initial parameter guesses.
    bounds : tuple of (list of float, list of float), optional
        ``(lower_bounds, upper_bounds)`` passed to curve_fit.
    error_message : str, optional
        Message to print when the fit fails.

    Returns
    -------
    pOpt : list of float
        Optimized parameters, or NaN-filled list on failure.
    pCov : np.ndarray
        Covariance matrix of the fit, or inf-filled matrix on failure.
    fitparams : list of float
        Validated initial parameters used for the fit attempt.
    """
    if bounds:
        fitparams = validate_bounds(fitparams, bounds)

    pCov = np.full((len(fitparams), len(fitparams)), np.inf)
    pOpt = list(fitparams)

    try:
        kwargs = dict(p0=fitparams, method="trf", max_nfev=10_000)
        if bounds:
            kwargs["bounds"] = bounds
        pOpt, pCov = sp.optimize.curve_fit(fitfunc, xdata, ydata, **kwargs)
    except (RuntimeError, ValueError):
        print(error_message)
        pOpt = [np.nan] * len(fitparams)

    return pOpt, pCov, fitparams


# ====================================================== #
# Data Selection Functions
# ====================================================== #


def _fit_snr(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fit: List[float],
    fit_err: np.ndarray,
    fitfunc: Callable,
) -> float:
    """
    Score a fit by SNR = (peak-to-peak of fit curve) / (residual RMS).

    This is robust to large DC offsets — unlike R², which has a small
    denominator (ss_tot) when the signal swing is tiny relative to the
    offset, making it hypersensitive to noise.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Measured data.
    fit : list of float
        Fitted model parameters.
    fit_err : np.ndarray
        Covariance matrix from the fit.
    fitfunc : Callable
        Model function with signature f(x, *params).

    Returns
    -------
    snr : float
        Signal-to-noise ratio of the fit; -inf for invalid fits.
    """
    if np.any(np.isnan(fit)) or np.any(np.diag(fit_err) == np.inf):
        return -np.inf

    y_fit = fitfunc(xdata, *fit)
    fit_amplitude = np.max(y_fit) - np.min(y_fit)
    residual_rms = np.sqrt(np.mean((ydata - y_fit) ** 2))

    if residual_rms == 0:
        return np.inf
    if fit_amplitude == 0:
        return -np.inf  # flat fit — useless

    return fit_amplitude / residual_rms


def _calculate_normalized_errors(
    fits: List[Any],
    fit_errors: List[np.ndarray],
) -> np.ndarray:
    """
    Fallback scorer: mean(σ_i / |p_i|) across parameters.

    Lower values indicate better fits. Returns inf for invalid fits.

    Parameters
    ----------
    fits : list
        List of fitted parameter arrays, one per channel.
    fit_errors : list of np.ndarray
        List of covariance matrices, one per channel.

    Returns
    -------
    norm_errors : np.ndarray
        Normalized error score for each fit; inf for degenerate cases.
    """
    norm_errors = []
    for fit, err_matrix in zip(fits, fit_errors):
        param_errors = np.sqrt(np.abs(np.diag(err_matrix)))
        param_values = np.abs(fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(param_values > 0, param_errors / param_values, np.inf)
        norm_err = np.nanmean(ratios)
        norm_errors.append(np.inf if np.isnan(norm_err) else norm_err)
    return np.array(norm_errors)


def _find_best_fit_with_snr(
    data: Dict[str, Any],
    fits: List[Any],
    fit_errors: List[np.ndarray],
    check_measures: Tuple[str, ...],
    fitfunc: Callable,
) -> int:
    """
    Select the best measurement channel by SNR of the fit.

    SNR = (fit peak-to-peak) / (residual RMS) is insensitive to DC offset,
    directly reflecting how well the oscillation stands above the noise.
    Falls back to normalised-error ranking if every fit is invalid.

    Parameters
    ----------
    data : dict
        Experiment data dict containing ``'xpts'`` and each measure array.
    fits : list
        Fitted parameter arrays, indexed by check_measures order.
    fit_errors : list of np.ndarray
        Covariance matrices, indexed by check_measures order.
    check_measures : tuple of str
        Channel names to compare (e.g. ``("amps", "avgi", "avgq")``).
    fitfunc : Callable
        Model function f(x, *params) for computing the fit curve.

    Returns
    -------
    best_idx : int
        Index into check_measures of the best channel.
    """
    xdata = data["xpts"]
    scores = [
        _fit_snr(
            xdata,
            data[measure],
            fits[i],
            fit_errors[i],
            fitfunc,
        )
        for i, measure in enumerate(check_measures)
    ]

    best_idx = int(np.argmax(scores))
    if scores[best_idx] == -np.inf:
        # All fits failed — fall back to covariance-based ranking
        return _find_best_fit_simple(fits, fit_errors)

    return best_idx


def _find_best_fit_simple(
    fits: List[Any],
    fit_errors: List[np.ndarray],
) -> int:
    """
    Select the best fit by minimum normalised covariance error.

    Parameters
    ----------
    fits : list
        Fitted parameter arrays, one per channel.
    fit_errors : list of np.ndarray
        Covariance matrices, one per channel.

    Returns
    -------
    best_idx : int
        Index of the fit with the lowest normalised error.
    """
    return int(np.argmin(_calculate_normalized_errors(fits, fit_errors)))


def get_best_fit(
    data: Dict[str, Any],
    fitfunc: Optional[Callable] = None,
    prefixes: List[str] = ["fit"],
    check_measures: Tuple[str, ...] = ("amps", "avgi", "avgq"),
    get_best_data_params: Tuple[str, ...] = (),
    override: Optional[str] = None,
) -> List[Any]:
    """
    Compare fits across measurement channels and return the best one.

    Selection priority
    ------------------
    1. ``override``  — use this channel unconditionally if given.
    2. ``fitfunc``   — SNR-based selection (fit amplitude / residual RMS).
    3. fallback    — lowest normalised covariance error.

    Parameters
    ----------
    data : dict
        Experiment data dict; must contain keys like
        ``"{prefix}_{measure}"``, ``"{prefix}_err_{measure}"``,
        ``"xpts"``, and each measure name.
    fitfunc : Callable, optional
        Fitted model f(x, *params); required for SNR scoring.
    prefixes : list of str, optional
        Key prefixes in ``data`` (default ``["fit"]``).
    check_measures : tuple of str, optional
        Channels to compare (default ``("amps", "avgi", "avgq")``).
    get_best_data_params : tuple of str, optional
        Extra keys to return for the winning channel.
    override : str, optional
        If set, skip scoring and use this channel directly.

    Returns
    -------
    result : list
        ``[best_fit, best_fit_err, *extra_params, best_measure_name]``
    """
    # ── collect fits and covariance matrices ──────────────────────
    fits, fit_errors = [], []
    for measure in check_measures:
        for prefix in prefixes:
            fits.append(data[f"{prefix}_{measure}"])
            fit_errors.append(data[f"{prefix}_err_{measure}"])

    # Replace zero diagonal entries with inf (zero cov = degenerate fit)
    for err_matrix in fit_errors:
        diag = np.diag(err_matrix)
        zero_mask = diag == 0
        err_matrix[np.diag_indices_from(err_matrix)] = np.where(zero_mask, np.inf, diag)

    # ── select best channel ────────────────────────────────────────
    if override is not None and override in check_measures:
        best_index = list(check_measures).index(override)
    elif fitfunc is not None:
        best_index = _find_best_fit_with_snr(
            data, fits, fit_errors, check_measures, fitfunc
        )
    else:
        best_index = _find_best_fit_simple(fits, fit_errors)

    best_measure = check_measures[best_index % len(check_measures)]

    # ── assemble return value ──────────────────────────────────────
    result = [fits[best_index], fit_errors[best_index]]
    for param in get_best_data_params:
        result.append(data[f"{param}_{best_measure}"])
    result.append(best_measure)

    return result


# ====================================================== #
# Exponential Fit Functions
# ====================================================== #


def expfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(y0, yscale, decay)``.

    Returns
    -------
    y : np.ndarray
        ``y0 + yscale * exp(-x / decay)``
    """
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)


def expfunc2(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function with x offset.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(y0, yscale, x0, decay)``.

    Returns
    -------
    y : np.ndarray
        ``y0 + yscale * exp(-(x - x0) / decay)``
    """
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to an exponential decay model.

    Auto-initializes parameters from data statistics when not provided.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[y0, yscale, decay]``. Use None for any element
        to trigger auto-initialization.

    Returns
    -------
    pOpt : list of float
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 3

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = ydata[-1]  # y0
    if fitparams[1] is None:
        fitparams[1] = ydata[0] - ydata[-1]  # yscale
    if fitparams[2] is None:
        fitparams[2] = (xdata[-1] - xdata[0]) / 4  # decay

    return generic_fit(
        expfunc,
        xdata,
        ydata,
        fitparams,
        error_message="Warning: Fit exponential failed!",
    )


# ====================================================== #
# Lorentzian Fit Functions
# ====================================================== #


def lorfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Lorentzian function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(y0, yscale, x0, xscale)``.

    Returns
    -------
    y : np.ndarray
        ``y0 + yscale / (1 + (x - x0)^2 / xscale^2)``
    """
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a Lorentzian function.

    Auto-initializes parameters from data statistics when not provided.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[y0, yscale, x0, xscale]``. Use None for any
        element to trigger auto-initialization.

    Returns
    -------
    pOpt : list of float
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 4

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2  # y0
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)  # yscale
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]  # x0
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10  # xscale

    return generic_fit(
        lorfunc,
        xdata,
        ydata,
        fitparams,
        error_message="Warning: Fit Lorentzian failed!",
    )


# ====================================================== #
# asymlor Fit Functions
# ====================================================== #


def asym_lorfunc(x, *p):
    """
    Asymmetric Lorentzian function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(y0, A, x0, gamma, alpha)``.

    Returns
    -------
    y : np.ndarray
        ``y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0))))^2)``
    """
    y0, A, x0, gamma, alpha = p
    return y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)


def fit_asym_lor(xdata, ydata, fitparams=None):
    """
    Fit data to an asymmetric Lorentzian function.

    Auto-initializes parameters from data statistics when not provided.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[y0, A, x0, gamma, alpha]``. Use None for any
        element to trigger auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None:
        fitparams[4] = 0

    return generic_fit(
        asym_lorfunc,
        xdata,
        ydata,
        fitparams,
        error_message="Warning: Fit asymmetric Lorentzian failed!",
    )


# ====================================================== #
# Sinusoidal Fit Functions
# ====================================================== #


def sinfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Sinusoidal function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(yscale, freq, phase_deg, y0)``.

    Returns
    -------
    y : np.ndarray
        ``yscale * sin(2*pi*freq*x + phase_deg*pi/180) + y0``
    """
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0


def fitsin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a sinusoidal function.

    Uses ``fourier_init`` to auto-initialize frequency and phase from the FFT.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[yscale, freq, phase_deg, y0]``. Use None for
        any element to trigger auto-initialization.
    debug : bool, optional
        Passed to ``fourier_init`` for FFT debug plotting. Default is False.

    Returns
    -------
    pOpt : list of float
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 4

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = 1 / 2 * (max(ydata) - min(ydata))  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = np.mean(ydata)  # y0

    bounds = (
        [0.5 * fitparams[0], 1e-3, -360, np.min(ydata)],
        [2 * fitparams[0], 1000, 360, np.max(ydata)],
    )

    return generic_fit(
        sinfunc,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit sinusoidal failed!",
    )


def decaysin(x: np.ndarray, *p) -> np.ndarray:
    """
    Decaying sinusoidal function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(yscale, freq, phase_deg, decay, y0)``.

    Returns
    -------
    y : np.ndarray
        ``yscale * sin(2*pi*freq*x + phase_deg*pi/180) * exp(-x/decay) + y0``
    """
    yscale, freq, phase_deg, decay, y0 = p
    return (
        yscale
        * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
        * np.exp(-x / decay)
        + y0
    )


def fitdecaysin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a decaying sinusoidal function.

    Uses ``fourier_init`` to auto-initialize frequency and phase. If the
    first attempt fails, retries with an inverted phase.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[yscale, freq, phase_deg, decay, y0]``. Use
        None for any element to trigger auto-initialization.
    debug : bool, optional
        Passed to ``fourier_init`` for FFT debug plotting. Default is False.

    Returns
    -------
    pOpt : list of float
        Optimized parameters, or NaN-filled on failure.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 5

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = (max(ydata) - min(ydata)) / 2  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi + 90  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # decay
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y0

    bounds = (
        [
            0.75 * fitparams[0],
            0.1 * max_freq,
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
        ],
        [1.25 * fitparams[0], 1.5 * max_freq, 360, np.inf, np.max(ydata)],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            decaysin, xdata, ydata, p0=fitparams, bounds=bounds
        )

    except RuntimeError:
        try:
            # Try with inverted phase
            fitparams[2] = -fitparams[2]
            pOpt, pCov = sp.optimize.curve_fit(
                decaysin, xdata, ydata, p0=fitparams, bounds=bounds
            )
        except:
            print("Warning: Fit decaying sine failed!")
            pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


def decayslopesin(x: np.ndarray, *p) -> np.ndarray:
    """
    Decaying sinusoidal function with a linear slope envelope.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(yscale, freq, phase_deg, decay, y0, slope)``.

    Returns
    -------
    y : np.ndarray
        ``yscale * (sin(2*pi*freq*x + phase_deg*pi/180) + slope) * exp(-x/decay) + y0``
    """
    yscale, freq, phase_deg, decay, y0, slope = p
    return (
        yscale
        * (np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + slope)
        * np.exp(-x / decay)
        + y0
    )


def fitdecayslopesin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a decaying sinusoidal function with a linear slope envelope.

    Uses ``fourier_init`` to auto-initialize frequency and phase. Retries
    with phase adjustments on failure.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters ``[yscale, freq, phase_deg, decay, y0, slope]``.
        Use None for any element to trigger auto-initialization.
    debug : bool, optional
        Passed to ``fourier_init`` for FFT debug plotting. Default is False.

    Returns
    -------
    pOpt : list of float
        Optimized parameters, or NaN-filled on failure.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 6

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi + 90  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 4  # decay
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y0
    if fitparams[5] is None:
        fitparams[5] = 0  # slope

    bounds = (
        [0.6 * fitparams[0], 1e-3, -360, 0.1, np.min(ydata), -np.inf],
        [1.5 * fitparams[0], 1e3, 360, np.inf, np.max(ydata), np.inf],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams)
    except RuntimeError:
        try:
            # Try with phase shifted by -90 degrees
            fitparams[2] = fitparams[2] - 90
            pOpt, pCov = sp.optimize.curve_fit(
                decayslopesin, xdata, ydata, p0=fitparams
            )
        except:
            try:
                # Try with phase shifted by +180 degrees
                fitparams[2] = fitparams[2] + 180
                pOpt, pCov = sp.optimize.curve_fit(
                    decayslopesin, xdata, ydata, p0=fitparams
                )
            except:
                print("Warning: Fit decaying slope sine failed!")
                pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


def twofreq_decaysin(x: np.ndarray, *p) -> np.ndarray:
    """
    Two-frequency decaying sinusoidal function.

    Models a signal as a product of a decaying sinusoid and a slow sinusoidal
    modulation envelope.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    *p : float
        Parameters ``(yscale0, freq0, phase_deg0, decay0, y00, x00,
        yscale1, freq1, phase_deg1, y01)``.

    Returns
    -------
    y : np.ndarray
        ``y00 + decaysin(x, yscale0, freq0, phase_deg0, decay0, 0)
        * sinfunc(x, yscale1, freq1, phase_deg1, y01)``
    """
    yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01 = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1)


def fittwofreq_decaysin(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray]:
    """
    Fit data to a two-frequency decaying sinusoidal function.

    Auto-initializes parameters using the FFT of the data.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    fitparams : list of float, optional
        Initial parameters (10 elements). Use None for any element to
        trigger auto-initialization.

    Returns
    -------
    pOpt : list of float
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix.
    """
    if fitparams is None:
        fitparams = [None] * 10

    # Initialize using Fourier transform
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]

    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]

    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale0
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg0
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # decay0
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y00
    if fitparams[5] is None:
        fitparams[5] = xdata[0]  # x00
    if fitparams[6] is None:
        fitparams[6] = 1  # yscale1
    if fitparams[7] is None:
        fitparams[7] = 1 / 10  # freq1
    if fitparams[8] is None:
        fitparams[8] = 0  # phase_deg1
    if fitparams[9] is None:
        fitparams[9] = 0  # y01

    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
            xdata[0] - (xdata[-1] - xdata[0]),
            0.9,
            0.01,
            -360,
            -0.1,
        ],
        [
            1.25 * fitparams[0],
            15 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.max(ydata),
            xdata[-1] + (xdata[-1] - xdata[0]),
            1.1,
            10,
            360,
            0.1,
        ],
    )

    return generic_fit(
        twofreq_decaysin,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit two-frequency decaying sine failed!",
    )  # Return only pOpt and pCov


# ====================================================== #
# Gaussian Fit Functions
# ====================================================== #
def gaussian(x, a, x0, sigma, y0):
    """
    Single Gaussian function with baseline offset.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    a : float
        Amplitude of the Gaussian peak.
    x0 : float
        Center of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.
    y0 : float
        Baseline offset.

    Returns
    -------
    y : np.ndarray
        ``a * exp(-(x - x0)^2 / (2 * sigma^2)) + y0``
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + y0


def fit_gauss(xdata, ydata, fitparams=None):
    """
    Fit data to a single Gaussian function.

    Auto-initializes parameters from data statistics when not provided.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data (bin centers).
    ydata : np.ndarray
        Dependent variable data (counts or histogram values).
    fitparams : list of float, optional
        Initial parameters ``[a, x0, sigma, y0]``. Use None for any element
        to trigger auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    # xmed, xstd should be gotten from the single shot data prior to fitting the histogram
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = np.max(ydata)
    if fitparams[1] is None:
        fitparams[1] = xdata[np.argmax(ydata)]
    if fitparams[2] is None:
        fitparams[2] = (np.max(xdata) - np.min(xdata)) / 8
    if fitparams[3] is None:
        fitparams[3] = np.min(ydata)

    bounds = (
        [fitparams[0] * 0.5, np.min(xdata), fitparams[2] * 0.1, 0],
        [fitparams[0] * 1.5, np.max(xdata), fitparams[2] * 10, np.max(ydata)],
    )

    return generic_fit(
        gaussian,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit Gaussian failed!",
    )


# ====================================================== #
# Double Gaussian Fit Functions
# ====================================================== #


def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    """
    Standard double Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    a1 : float
        Amplitude of the first Gaussian.
    b1 : float
        Mean of the first Gaussian.
    c1 : float
        Standard deviation of the first Gaussian.
    a2 : float
        Amplitude of the second Gaussian.
    b2 : float
        Mean of the second Gaussian.
    c2 : float
        Standard deviation of the second Gaussian.

    Returns
    -------
    y : np.ndarray
        Sum of the two Gaussians.
    """
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1**2)) + a2 * np.exp(
        -((x - b2) ** 2) / (2 * c2**2)
    )


def fit_doublegauss(xdata, ydata, fitparams):
    """
    Robust fitting function for double Gaussian distributions.

    Constrains each Gaussian mean to within ±2σ of the initial guess to
    prevent the solver from merging both peaks into a single visible one.

    Parameters
    ----------
    xdata : np.ndarray
        Bin centers.
    ydata : np.ndarray
        Counts or histogram values.
    fitparams : list of float
        Initial guesses ``[amp_g, mu_g, sigma_g, amp_e, mu_e, sigma_e]``.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters, or initial guesses on failure.
    pCov : np.ndarray
        Covariance matrix, or zero matrix on failure.
    """
    # Unpack parameters to set dynamic bounds
    ag, mug, sig, ae, mue, sie = fitparams

    # --- Constraint Logic ---
    # Prevent the solver from drifting too far from the expected state centers (mu).
    # If the signal for one state is weak, the solver might try to merge both Gaussians
    # into the single visible peak. We constrain the mean to +/- 2 sigma (or a fixed buffer).

    delta_mu_g = abs(sig) * 2 if sig > 0 else 100
    delta_mu_e = abs(sie) * 2 if sie > 0 else 100

    # Lower Bounds: Amp > 0, Sigma > 0, Mean constrained
    lb = [0, mug - delta_mu_g, 0, 0, mue - delta_mu_e, 0]
    # Upper Bounds: Amp infinite, Sigma infinite, Mean constrained
    ub = [np.inf, mug + delta_mu_g, np.inf, np.inf, mue + delta_mu_e, np.inf]

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            double_gaussian,
            xdata,
            ydata,
            p0=fitparams,
            bounds=(lb, ub),
            maxfev=10000,  # Increase max iterations for convergence
        )
    except Exception:
        # print(f"Fit failed: {e}")
        # If fit fails, return the initial guesses so the code doesn't crash
        pOpt = fitparams
        pCov = np.zeros((6, 6))

    return pOpt, pCov


# ====================================================== #
# Hanger Resonator Fit Functions
# ====================================================== #


def hangerfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Complex Hanger (notch) function for resonator S21 fitting.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    *p : float
        Parameters ``(f0, Qi, Qe, phi, scale)``.

    Returns
    -------
    S21 : np.ndarray
        Complex S21 transmission response.
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))


def hangerS21func(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of the Hanger function for resonator S21 fitting.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    *p : float
        Parameters ``(f0, Qi, Qe, phi, scale)``.

    Returns
    -------
    mag : np.ndarray
        Magnitude of the S21 transmission response.
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return np.abs(hangerfunc(x, *p))


def hangerS21func_sloped(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of the Hanger function with a linear background slope.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    *p : float
        Parameters ``(f0, Qi, Qe, phi, scale, slope)``.

    Returns
    -------
    mag : np.ndarray
        Magnitude of S21 with a linear slope added.
    """
    f0, Qi, Qe, phi, scale, slope = p
    return hangerS21func(x, f0, 1e4 * Qi, 1e4 * Qe, phi, scale) + slope * (x - f0)


def hangerphasefunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Phase of the Hanger function for resonator fitting.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    *p : float
        Parameters ``(f0, Qi, Qe, phi, scale)``.

    Returns
    -------
    phase : np.ndarray
        Phase of the S21 transmission response (radians).
    """
    return np.angle(hangerfunc(x, *p))


def fithanger(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit resonator transmission data to a sloped Hanger function.

    Performs two sequential curve_fit calls for improved convergence.
    Auto-initializes parameters when not provided.

    Parameters
    ----------
    xdata : np.ndarray
        Frequency data points.
    ydata : np.ndarray
        Magnitude of S21 data points.
    fitparams : list of float, optional
        Initial parameters ``[f0, Qi, Qe, phi, scale, slope]``. Use None
        for any element to trigger auto-initialization.

    Returns
    -------
    pOpt : list of float
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    fitparams : list of float
        Initial parameters used.
    """
    if fitparams is None:
        fitparams = [None] * 6

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = xdata[np.argmin(np.abs(ydata))]  # f0
    if fitparams[1] is None:
        fitparams[1] = 8  # Qi
    if fitparams[2] is None:
        fitparams[2] = 3  # Qe
    if fitparams[3] is None:
        fitparams[3] = 0  # phi
    if fitparams[4] is None:
        fitparams[4] = max(ydata)  # scale
    if fitparams[5] is None:
        fitparams[5] = 0  # slope

    bounds = (
        [np.min(xdata), 0, 0, -np.inf, 0, -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, np.inf],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds
        )
        pOpt, pCov = sp.optimize.curve_fit(
            hangerS21func_sloped, xdata, ydata, p0=pOpt, bounds=bounds
        )
    except RuntimeError:
        print("Warning: Fit hanger failed!")
        traceback.print_exc()

    return pOpt, pCov, fitparams


# ====================================================== #
# Randomized Benchmarking Fit Functions
# ====================================================== #


def rb_func(depth: np.ndarray, p: float, a: float, b: float) -> np.ndarray:
    """
    Randomized benchmarking decay function.

    Parameters
    ----------
    depth : np.ndarray
        Sequence depth (number of Clifford gates).
    p : float
        Depolarizing parameter (decay base).
    a : float
        Amplitude of the decay.
    b : float
        Asymptotic offset (floor) at large depth.

    Returns
    -------
    fidelity : np.ndarray
        ``a * p**depth + b``
    """
    return a * p**depth + b


def rb_error(p: float, d: int) -> float:
    """
    Calculate the average error rate per Clifford from the RB decay parameter.

    Parameters
    ----------
    p : float
        Depolarizing parameter from the RB fit.
    d : int
        Hilbert space dimension (2 for a single qubit).

    Returns
    -------
    epc : float
        Error per Clifford (average gate error rate).
    """
    return 1 - (p + (1 - p) / d)


def error_fit_err(cov_p: float, d: int) -> float:
    """
    Propagate the covariance of p to the uncertainty in EPC.

    Parameters
    ----------
    cov_p : float
        Variance (diagonal covariance element) of the depolarizing parameter p.
    d : int
        Hilbert space dimension (2 for a single qubit).

    Returns
    -------
    var_epc : float
        Variance of the error per Clifford.
    """
    return cov_p * (1 / d - 1) ** 2


def rb_gate_fidelity(p_rb: float, p_irb: float, d: int) -> float:
    """
    Calculate gate fidelity from standard and interleaved RB parameters.

    Parameters
    ----------
    p_rb : float
        Depolarizing parameter from standard (reference) RB.
    p_irb : float
        Depolarizing parameter from interleaved RB.
    d : int
        Hilbert space dimension (2 for a single qubit).

    Returns
    -------
    fidelity : float
        Gate fidelity of the interleaved gate.
    """
    return 1 - (d - 1) * (1 - p_irb / p_rb) / d


# def fitrb(
#     xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
# ) -> Tuple[List[float], np.ndarray]:
#     """
#     Fit data to a randomized benchmarking function.

#     Args:
#         xdata: X-axis data points (sequence depth)
#         ydata: Y-axis data points (fidelity)
#         fitparams: Optional initial parameters [p, a, b]

#     Returns:
#         Tuple of (optimized_parameters, covariance_matrix)
#     """
#     if fitparams is None:
#         fitparams = [None] * 3

#     # Initialize parameters if not provided
#     if fitparams[0] is None:
#         fitparams[0] = 0.9  # p
#     if fitparams[1] is None:
#         fitparams[1] = np.max(ydata) - np.min(ydata)  # a
#     if fitparams[2] is None:
#         fitparams[2] = np.min(ydata)  # b

#     bounds = ([0, 0, 0], [1, 10 * np.max(ydata) - np.min(ydata), np.max(ydata)])

#     fitparams = validate_bounds(fitparams, bounds)
#     pOpt = fitparams
#     pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

#     try:
#         pOpt, pCov = sp.optimize.curve_fit(
#             rb_func, xdata, ydata, p0=fitparams, bounds=bounds
#         )
#         print(pOpt)
#         print(pCov[0][0], pCov[1][1], pCov[2][2])
#     except RuntimeError:
#         print("Warning: Fit randomized benchmarking failed!")
#         traceback.print_exc()
#         pOpt = [np.nan] * len(pOpt)

#     return pOpt, pCov


def fitrb(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    p_bounds: Tuple[float, float] = (0.0, 1.0),
    maxfev: int = 10000,
) -> Tuple[List[float], np.ndarray]:
    """
    Fit data to the RB decay model ``a * p**depth + b``.

    Uses a log-linear initial estimate for p and unconstrained bounds for
    a and b to handle inverted IQ signals. Prints fitted values and
    standard deviations on success.

    Parameters
    ----------
    xdata : np.ndarray
        Sequence depth values.
    ydata : np.ndarray
        Measured signal (fidelity or IQ amplitude) at each depth.
    fitparams : list of float, optional
        Initial parameters ``[p, a, b]``. Use None for any element to
        trigger smart auto-initialization.
    p_bounds : tuple of (float, float), optional
        ``(p_min, p_max)`` bounds for the depolarizing parameter.
        Tighten to constrain the fit, e.g. ``(0.95, 1.0)`` for a
        high-fidelity qubit. Default is ``(0.0, 1.0)``.
    maxfev : int, optional
        Maximum number of function evaluations. Default is 10000.

    Returns
    -------
    pOpt : list of float
        Optimized parameters ``[p, a, b]``, or NaN-filled on failure.
    pCov : np.ndarray
        3×3 covariance matrix, or inf-filled on failure.
    """
    y = np.asarray(ydata, dtype=float)
    x = np.asarray(xdata, dtype=float)

    # ── smart initial guess ──────────────────────────────────────────────────
    y_max, y_min = y.max(), y.min()
    a0 = y_max - y_min  # could be negative if signal is inverted
    b0 = y_min  # floor at large depth

    # Estimate p from the data: if the curve decays, fit log-linear on the
    # (y - b0) values that are positive; fall back to 0.99 if that fails.
    try:
        delta = y - b0
        mask = delta > 0
        if mask.sum() >= 2:
            slope, _ = np.polyfit(x[mask], np.log(delta[mask] + 1e-12), 1)
            p0 = float(np.clip(np.exp(slope), *p_bounds))
        else:
            p0 = float(np.clip(0.99, *p_bounds))
    except Exception:
        p0 = float(np.clip(0.99, *p_bounds))

    if fitparams is None:
        fitparams = [p0, a0, b0]
    else:
        fitparams = list(fitparams)
        if fitparams[0] is None:
            fitparams[0] = p0
        if fitparams[1] is None:
            fitparams[1] = a0
        if fitparams[2] is None:
            fitparams[2] = b0

    # ── bounds ───────────────────────────────────────────────────────────────
    # a and b are now unconstrained in sign (handles inverted IQ signals).
    # p is bounded by p_bounds.
    a_mag = max(abs(a0) * 10, 1.0)
    b_mag = max(abs(b0) * 10 + abs(a0) * 10, 1.0)

    bounds = (
        [p_bounds[0], -a_mag, -b_mag],
        [p_bounds[1], a_mag, b_mag],
    )

    # Clip initial guess inside bounds
    for i, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
        fitparams[i] = float(np.clip(fitparams[i], lo + 1e-9, hi - 1e-9))

    pOpt = fitparams
    pCov = np.full((3, 3), np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            rb_func,
            x,
            y,
            p0=fitparams,
            bounds=bounds,
            maxfev=maxfev,
            method="trf",  # Trust Region Reflective — better near bounds
        )
        print(f"[fitrb] p={pOpt[0]:.6f}  a={pOpt[1]:.4f}  b={pOpt[2]:.4f}")
        print(
            f"[fitrb] σ(p)={np.sqrt(pCov[0, 0]):.2e}  "
            f"σ(a)={np.sqrt(pCov[1, 1]):.2e}  "
            f"σ(b)={np.sqrt(pCov[2, 2]):.2e}"
        )
    except RuntimeError:
        print("Warning: RB fit failed!")
        traceback.print_exc()
        pOpt = [np.nan] * 3

    return pOpt, pCov


# ====================================================== #
# Adiabatic Pi Pulse Functions
# ====================================================== #


def adiabatic_amp(
    t: np.ndarray, amp_max: float, beta: float, period: float
) -> np.ndarray:
    """
    Amplitude envelope for an adiabatic pi pulse (sech shape).

    Parameters
    ----------
    t : np.ndarray
        Time points.
    amp_max : float
        Peak amplitude of the pulse.
    beta : float
        Controls the steepness of the sech envelope.
    period : float
        Total pulse duration.

    Returns
    -------
    amp : np.ndarray
        ``amp_max / cosh(beta * (2*t/period - 1))``
    """
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t: np.ndarray, mu: float, beta: float, period: float) -> np.ndarray:
    """
    Phase function for an adiabatic pi pulse.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    mu : float
        Width of the frequency sweep (related to bandwidth).
    beta : float
        Steepness parameter of the sech envelope.
    period : float
        Total pulse duration.

    Returns
    -------
    phase : np.ndarray
        ``mu * log(adiabatic_amp(t, 1, beta, period))``
    """
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(
    t: np.ndarray, amp_max: float, mu: float, beta: float, period: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate I and Q amplitudes for an adiabatic pi pulse.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    amp_max : float
        Peak amplitude.
    mu : float
        Width of the frequency sweep.
    beta : float
        Steepness of the sech envelope.
    period : float
        Total pulse duration.

    Returns
    -------
    iamp : np.ndarray
        In-phase (I) amplitude component.
    qamp : np.ndarray
        Quadrature (Q) amplitude component.
    """
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j * np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j * np.cos(phase))
    return np.real(iamp), np.real(qamp)


# ====================================================== #
# Correcting for over/under rotation
# delta: angle error in degrees
# See reference: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.012301


def probg_Xhalf(n, *p):
    """
    Ground-state probability for a repeated X/2 pulse sequence.

    Parameters
    ----------
    n : np.ndarray
        Number of repetitions.
    *p : float
        Parameters ``(a, delta)`` where a is an offset and delta is the
        rotation angle error in degrees.

    Returns
    -------
    prob : np.ndarray
        ``a + 0.5 * (-1)^n * cos(pi/2 + 2*n*delta_rad)``
    """
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta))


def probg_X(n, *p):
    """
    Ground-state probability for a repeated X pulse sequence.

    Parameters
    ----------
    n : np.ndarray
        Number of repetitions.
    *p : float
        Parameters ``(a, delta)`` where a is an offset and delta is the
        rotation angle error in degrees.

    Returns
    -------
    prob : np.ndarray
        ``a + 0.5 * cos(pi/2 + 2*n*delta_rad)``
    """
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * np.cos(np.pi / 2 + 2 * n * delta))


def probg_Xhalf_decay(n, *p):
    """
    Decaying ground-state probability for a repeated X/2 pulse sequence.

    Parameters
    ----------
    n : np.ndarray
        Number of repetitions.
    *p : float
        Parameters ``(a, delta, decay)`` where a is an offset, delta is
        the rotation angle error in degrees, and decay is the exponential
        decay constant.

    Returns
    -------
    prob : np.ndarray
        ``a + 0.5 * (-1)^n * cos(pi/2 + 2*n*delta_rad) * exp(-n/decay)``
    """
    a, delta, decay = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta)) * np.exp(
        -n / decay
    )


def fit_probg_Xhalf(xdata, ydata, fitparams=None):
    """
    Fit data to the X/2 repeated pulse rotation error model.

    Parameters
    ----------
    xdata : np.ndarray
        Number of repetitions.
    ydata : np.ndarray
        Measured ground-state probability.
    fitparams : list of float, optional
        Initial parameters ``[a, delta]``. Use None for auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0

    bounds = ([min(ydata), -20.0], [max(ydata), 20.0])

    return generic_fit(
        probg_Xhalf,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit X/2 rotation error failed!",
    )


def fit_probg_X(xdata, ydata, fitparams=None):
    """
    Fit data to the X repeated pulse rotation error model.

    Parameters
    ----------
    xdata : np.ndarray
        Number of repetitions.
    ydata : np.ndarray
        Measured ground-state probability.
    fitparams : list of float, optional
        Initial parameters ``[a, delta]``. Use None for auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0

    bounds = ([min(ydata), -20.0], [max(ydata), 20.0])

    return generic_fit(
        probg_X,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit X rotation error failed!",
    )


def fit_probg_Xhalf_decay(xdata, ydata, fitparams=None):
    """
    Fit data to the decaying X/2 repeated pulse rotation error model.

    Parameters
    ----------
    xdata : np.ndarray
        Number of repetitions.
    ydata : np.ndarray
        Measured ground-state probability.
    fitparams : list of float, optional
        Initial parameters ``[a, delta, decay]``. Use None for
        auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    if fitparams[2] is None:
        fitparams[2] = 10

    bounds = ([min(ydata), -20.0, 1], [max(ydata), 20.0, 100])

    return generic_fit(
        probg_Xhalf_decay,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit decaying X/2 rotation error failed!",
    )


# ====================================================== #
def poisson(n, *p):
    """
    Poisson distribution probability mass function.

    Parameters
    ----------
    n : np.ndarray
        Non-negative integer values (photon numbers).
    *p : float
        Parameters ``(nbar,)`` where nbar is the mean photon number.

    Returns
    -------
    prob : np.ndarray
        ``exp(-nbar) * nbar^n / n!``
    """
    nbar = p[0]
    return np.exp(-nbar) * (nbar**n) / sp.special.factorial(n)


def fit_poisson(xdata, ydata, fitparams=None):
    """
    Fit data to a Poisson distribution.

    Parameters
    ----------
    xdata : np.ndarray
        Non-negative integer values (photon numbers).
    ydata : np.ndarray
        Measured probability or count data.
    fitparams : list of float, optional
        Initial parameters ``[nbar]``. Use None for auto-initialization.

    Returns
    -------
    pOpt : np.ndarray
        Optimized parameters.
    pCov : np.ndarray
        Covariance matrix, or inf-filled on failure.
    """
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = list(fitparams)

    if fitparams[0] is None:
        fitparams[0] = ydata[0]

    bounds = ([0], [10])

    return generic_fit(
        poisson,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit Poisson failed!",
    )
