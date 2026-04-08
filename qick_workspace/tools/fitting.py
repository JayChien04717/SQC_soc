"""
Fitting module for quantum experiments.

This module provides functions for fitting various types of data from quantum experiments,
including exponential decays, sinusoids, Lorentzians, and more specialized functions like
Hanger resonator fits and randomized benchmarking.
"""

import numpy as np
import scipy as sp
import traceback
from typing import Tuple, List, Optional, Callable, Dict, Any, Union

# ====================================================== #
# Utility Functions
# ====================================================== #


def get_r2(
    xdata: np.ndarray, ydata: np.ndarray, fitfunc: Callable, fit_params: List[float]
) -> float:
    """
    Calculate the R-squared value for a fit.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitfunc: The fitting function
        `fit_params`: Parameters for the fitting function

    Returns:
        R-squared value of the fit
    """
    # Residual sum of squares
    ss_res = np.sum((fitfunc(xdata, *fit_params) - ydata) ** 2)
    # Total sum of squares
    ss_tot = np.sum((np.mean(ydata) - ydata) ** 2)
    # R^2 value
    r2 = 1 - ss_res / ss_tot
    return r2


def fix_phase(p: List[float]) -> float:
    """
    Normalize phase and calculate pi gain.

    Args:
        p: Parameters list containing phase information

    Returns:
        Pi gain value
    """
    if p[2] > 180:
        p[2] = p[2] - 360
    elif p[2] < -180:
        p[2] = p[2] + 360

    if p[2] < 0:
        pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
        pi2_gain = (0 - p[2] / 180) / 2 / p[1]
    else:
        pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
        pi2_gain = (1 - p[2] / 180) / 2 / p[1]
    return pi_gain, pi2_gain


def fourier_init(
    xdata: np.ndarray, ydata: np.ndarray, debug: bool = False
) -> Tuple[float, float]:
    """
    Initialize frequency and phase using Fourier transform.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        debug: If True, plots the Fourier transform for debugging

    Returns:
        Tuple of (max_frequency, max_phase)
    """
    ydata = ydata - np.mean(ydata)
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)

    half_N = len(ydata) // 2
    mag = np.abs(fourier[1:half_N])
    phase = fft_phases[1:half_N]
    freqs = fft_freqs[1:half_N]

    max_ind = np.argmax(mag)
    max_freq = freqs[max_ind]
    max_phase = phase[max_ind]

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 6))

        ax[0].plot(freqs, mag, ".")
        ax[1].set_xlabel("Frequency (MHz)")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(freqs, phase * 180 / np.pi, ".")
        ax[1].plot(max_freq, max_phase * 180 / np.pi, "ro")
        ax[1].set_ylabel("Phase (deg)")

        print(f"Max phase is {max_phase}")
        print(f"Max freq is {max_freq}")
        plt.show()

    return max_freq, max_phase


def validate_bounds(
    fitparams: List[float], bounds: Tuple[List[float], List[float]]
) -> List[float]:
    """
    Validate that parameters are within bounds and adjust if necessary.

    Args:
        fitparams: List of fit parameters
        bounds: Tuple of (lower_bounds, upper_bounds)

    Returns:
        Validated fit parameters
    """
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
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
    Generic fitting function that handles common fitting patterns.

    Args:
        fitfunc: The function to fit
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Initial parameters for fitting
        bounds: Optional bounds for parameters
        error_message: Message to display if fitting fails

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if bounds:
        fitparams = validate_bounds(fitparams, bounds)

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        if bounds:
            pOpt, pCov = sp.optimize.curve_fit(
                fitfunc, xdata, ydata, p0=fitparams, bounds=bounds
            )
        else:
            pOpt, pCov = sp.optimize.curve_fit(fitfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        print(error_message)
        pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


# ====================================================== #
# Data Selection Functions
# ====================================================== #


def get_best_fit(
    data: Dict[str, Any],
    fitfunc: Optional[Callable] = None,
    prefixes: List[str] = ["fit"],
    check_measures: Tuple[str,] = ("amps", "avgi", "avgq"),
    get_best_data_params: Tuple[str,] = (),
    override: Optional[str] = None,
) -> List[Any]:
    """
    Compare fits between different measurements and select the best one.

    Args:
        data: Dictionary containing fit data
        fitfunc: Optional function to use for R² calculation
        prefixes: List of prefixes to check in data keys
        check_measures: Tuple of measurement types to check
        get_best_data_params: Additional parameters to retrieve for best measurement
        override: Optional measurement to use regardless of fit quality

    Returns:
        List containing [best_fit, best_fit_err, *additional_params, best_measurement_type]
    """
    # Collect fit parameters and error matrices
    fits = []
    fit_errors = []

    for measure in check_measures:
        for prefix in prefixes:
            fits.append(data[f"{prefix}_{measure}"])
            fit_errors.append(data[f"{prefix}_err_{measure}"])

    # Fix error matrices: replace zeros with infinity (indicating bad fit)
    for error_matrix in fit_errors:
        diagonal = np.diag(error_matrix)
        zero_indices = np.where(diagonal == 0)[0]
        for idx in zero_indices:
            error_matrix[idx, idx] = np.inf

    # Use override if specified
    if override is not None and override in check_measures:
        best_index = np.argwhere(np.array(check_measures) == override)[0][0]
    else:
        # Calculate fit quality metrics
        if fitfunc is not None:
            # Method 1: Use both R² and normalized parameter errors
            best_index = _find_best_fit_with_r2(
                data, fits, fit_errors, check_measures, fitfunc
            )
        else:
            # Method 2: Use only normalized parameter errors
            best_index = _find_best_fit_simple(fits, fit_errors)

    # Get the best measurement type (accounting for prefixes)
    best_measure = check_measures[best_index % len(check_measures)]

    # Collect results
    result = [fits[best_index], fit_errors[best_index]]

    # Add any additional requested parameters
    for param in get_best_data_params:
        result.append(data[f"{param}_{best_measure}"])

    # Add the measurement type
    result.append(best_measure)

    return result


def _find_best_fit_with_r2(
    data: Dict[str, Any],
    fits: List[Any],
    fit_errors: List[np.ndarray],
    check_measures: Tuple[str,],
    fitfunc: Callable,
) -> int:
    """
    Find the best fit using R² values to maximize SNR.
    If multiple fits are valid, the one with the highest R² is selected.

    Args:
        data: Dictionary containing fit data
        fits: List of fit parameters
        fit_errors: List of error matrices
        check_measures: Tuple of measurement types
        fitfunc: Function to use for R² calculation

    Returns:
        Index of the best fit
    """
    # Get x and y data for R² calculation
    xdata = data["xpts"]
    ydata = [data[measure] for measure in check_measures]

    # Calculate R² values
    r2_values = []
    for i, (fit, y) in enumerate(zip(fits[: len(check_measures)], ydata)):
        residual_sum_sq = np.sum((fitfunc(xdata, *fit) - y) ** 2)
        total_sum_sq = np.sum((np.mean(y) - y) ** 2)
        
        if total_sum_sq == 0:
            r2 = -np.inf
        else:
            r2 = 1 - residual_sum_sq / total_sum_sq

        # Invalidate fits with infinite errors or NaNs
        if np.any(np.diag(fit_errors[i]) == np.inf) or np.any(np.isnan(fits[i])):
            r2 = -np.inf

        r2_values.append(r2)

    best_idx = np.argmax(r2_values)

    # If all fits failed (all R² are -inf), fallback to normalized errors
    if r2_values[best_idx] == -np.inf:
        return _find_best_fit_simple(fits, fit_errors)

    return int(best_idx)


def _find_best_fit_simple(fits: List[Any], fit_errors: List[np.ndarray]) -> int:
    """
    Find the best fit using only normalized parameter errors.

    Args:
        fits: List of fit parameters
        fit_errors: List of error matrices

    Returns:
        Index of the best fit
    """
    # Calculate normalized parameter errors
    norm_errors = _calculate_normalized_errors(fits, fit_errors)

    # Return index of fit with lowest normalized error
    return np.argmin(norm_errors)


def _calculate_normalized_errors(
    fits: List[Any], fit_errors: List[np.ndarray]
) -> np.ndarray:
    """
    Calculate normalized parameter errors for each fit.

    Args:
        fits: List of fit parameters
        fit_errors: List of error matrices

    Returns:
        Array of normalized errors
    """
    norm_errors = []

    for fit, error_matrix in zip(fits, fit_errors):
        # Calculate average of sqrt(|error|/|parameter|)
        param_errors = np.sqrt(np.abs(np.diag(error_matrix)))
        param_values = np.abs(fit)
        norm_error = np.mean(param_errors / param_values)

        # Handle NaN errors
        if np.isnan(norm_error):
            norm_error = np.inf

        norm_errors.append(norm_error)

    return np.array(norm_errors)


# ====================================================== #
# Exponential Fit Functions
# ====================================================== #


def expfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function.

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, decay]

    Returns:
        y = y0 + yscale*exp(-x/decay)
    """
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)


def expfunc2(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function with x offset.

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, x0, decay]

    Returns:
        y = y0 + yscale*exp(-(x-x0)/decay)
    """
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to an exponential decay.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [y0, yscale, decay]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, x0, xscale]

    Returns:
        y = y0 + yscale/(1+(x-x0)²/xscale²)
    """
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a Lorentzian function.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [y0, yscale, x0, xscale]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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
    y0, A, x0, gamma, alpha = p
    return y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)


def fit_asym_lor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = np.copy(fitparams)

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

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(asym_lorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        print("Warning: fit failed!")
    return pOpt, pCov


# ====================================================== #
# Sinusoidal Fit Functions
# ====================================================== #


def sinfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Sinusoidal function.

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, y0]

    Returns:
        y = yscale*sin(2π*freq*x + phase_deg*π/180) + y0
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

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, y0]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, decay, y0]

    Returns:
        y = yscale*sin(2π*freq*x + phase_deg*π/180)*exp(-x/decay) + y0
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

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, decay, y0]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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
    Decaying sinusoidal function with slope.

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, decay, y0, slope]

    Returns:
        y = yscale*(sin(2π*freq*x + phase_deg*π/180) + slope)*exp(-x/decay) + y0
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
    Fit data to a decaying sinusoidal function with slope.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, decay, y0, slope]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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

    Args:
        x: X-axis data points
        p: Parameters [yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01]

    Returns:
        y = y00 + decaysin(x, *p0) * sinfunc(x, *p1)
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

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters

    Returns:
        Tuple of (optimized_parameters, covariance_matrix)
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
    )[:2]  # Return only pOpt and pCov


# ====================================================== #
# Gaussian Fit Functions
# ====================================================== #
def gaussian(x, a, x0, sigma, y0):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + y0


def fit_gauss(xdata, ydata, fitparams=None):
    # xmed, xstd should be gotten from the single shot data prior to fitting the histogram
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.max(ydata)
    if fitparams[1] is None:
        fitparams[1] = xdata[np.argmax(ydata)]
    if fitparams[2] is None:
        fitparams[2] = (np.max(xdata) - np.min(xdata)) / 8
    if fitparams[3] is None:
        fitparams[3] = np.min(ydata)
    pOpt = fitparams
    # print('fitparams guess:', fitparams)
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    bounds = (
        [fitparams[0] * 0.5, np.min(xdata), fitparams[2] * 0.1, 0],
        [fitparams[0] * 1.5, np.max(xdata), fitparams[2] * 10, np.max(ydata)],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            gaussian,
            xdata,
            ydata,
            p0=np.array(fitparams, dtype="float64"),
            bounds=bounds,
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
# Double Gaussian Fit Functions
# ====================================================== #


def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    """
    Standard double Gaussian function.
    Params: a=amplitude, b=mean, c=sigma
    """
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1**2)) + a2 * np.exp(
        -((x - b2) ** 2) / (2 * c2**2)
    )


def fit_doublegauss(xdata, ydata, fitparams):
    """
    Robust fitting function for double Gaussian distributions.

    Args:
        xdata: Bin centers.
        ydata: Counts/Histogram values.
        fitparams: Initial guesses [amp_g, mu_g, sigma_g, amp_e, mu_e, sigma_e].

    Returns:
        pOpt: Optimized parameters.
        pCov: Covariance matrix.
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
    except Exception as e:
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
    Complex Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Complex S21 response
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))


def hangerS21func(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Magnitude of S21 response
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return np.abs(hangerfunc(x, *p))


def hangerS21func_sloped(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of Hanger function with slope for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale, slope]

    Returns:
        Magnitude of S21 response with slope
    """
    f0, Qi, Qe, phi, scale, slope = p
    return hangerS21func(x, f0, 1e4 * Qi, 1e4 * Qe, phi, scale) + slope * (x - f0)


def hangerphasefunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Phase of Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Phase of S21 response
    """
    return np.angle(hangerfunc(x, *p))


def fithanger(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a Hanger function.

    Args:
        xdata: X-axis data points (frequency)
        ydata: Y-axis data points (magnitude)
        fitparams: Optional initial parameters [f0, Qi, Qe, phi, scale, slope]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
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
    Randomized benchmarking function.

    Args:
        depth: Sequence depth
        p: Depolarizing parameter
        a: Amplitude
        b: Offset

    Returns:
        Fidelity as a function of sequence depth
    """
    return a * p**depth + b


def rb_error(p: float, d: int) -> float:
    """
    Calculate average error rate over all gates in sequence.

    Args:
        p: Depolarizing parameter
        d: Dimension of system (2^number of qubits)

    Returns:
        Average error rate
    """
    return 1 - (p + (1 - p) / d)


def error_fit_err(cov_p: float, d: int) -> float:
    """
    Return covariance of randomized benchmarking error.

    Args:
        cov_p: Covariance of depolarizing parameter
        d: Dimension of system (2^number of qubits)

    Returns:
        Covariance of error
    """
    return cov_p * (1 / d - 1) ** 2


def rb_gate_fidelity(p_rb: float, p_irb: float, d: int) -> float:
    """
    Calculate gate fidelity from regular and interleaved RB.

    Args:
        p_rb: Depolarizing parameter from regular RB
        p_irb: Depolarizing parameter from interleaved RB
        d: Dimension of system (2^number of qubits)

    Returns:
        Gate fidelity
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
    Fit data to a * p**depth + b.

    p_bounds : (p_min, p_max) — tighten this to constrain the depolarizing
               parameter, e.g. (0.95, 1.0) for a high-fidelity qubit.
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
    Amplitude function for adiabatic pi pulse.

    Args:
        t: Time points
        amp_max: Maximum amplitude
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Amplitude as a function of time
    """
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t: np.ndarray, mu: float, beta: float, period: float) -> np.ndarray:
    """
    Phase function for adiabatic pi pulse.

    Args:
        t: Time points
        mu: Width of frequency sweep
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Phase as a function of time
    """
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(
    t: np.ndarray, amp_max: float, mu: float, beta: float, period: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate I and Q amplitudes for adiabatic pi pulse.

    Args:
        t: Time points
        amp_max: Maximum amplitude
        mu: Width of frequency sweep
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Tuple of (I amplitude, Q amplitude)
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
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta))


def probg_X(n, *p):
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * np.cos(np.pi / 2 + 2 * n * delta))


def probg_Xhalf_decay(n, *p):
    a, delta, decay = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta)) * np.exp(
        -n / decay
    )


def fit_probg_Xhalf(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            probg_Xhalf, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def fit_probg_X(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    bounds = (
        [min(ydata), -20.0],
        [max(ydata), 20.0],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            probg_X, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


def fit_probg_Xhalf_decay(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = np.average(ydata)
    if fitparams[1] is None:
        fitparams[1] = 0.0
    if fitparams[2] is None:
        fitparams[2] = 10
    bounds = (
        [min(ydata), -20.0, 1],
        [max(ydata), 20.0, 100],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            probg_Xhalf_decay, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov


# ====================================================== #
def poisson(n, *p):
    nbar = p[0]
    return np.exp(-nbar) * (nbar**n) / sp.special.factorial(n)


def fit_poisson(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = np.copy(fitparams)
    if fitparams[0] is None:
        fitparams[0] = ydata[0]
    bounds = (
        [0],
        [10],
    )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            poisson, xdata, ydata, p0=fitparams, bounds=bounds
        )
        # return pOpt, pCov
    except RuntimeError:
        print("Warning: fit failed!")
        # return 0, 0
    return pOpt, pCov
