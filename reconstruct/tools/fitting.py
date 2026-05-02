"""
Fitting module for quantum experiments.

Provides functions for fitting exponential decays, sinusoids, Lorentzians,
Hanger resonator models, randomized benchmarking, and more.
"""

import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy as sp


def get_r2(xdata, ydata, fitfunc, fit_params):
    """Compute the coefficient of determination (R²) for a fit."""
    ss_res = np.sum((fitfunc(xdata, *fit_params) - ydata) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf


def fix_phase(p):
    """Normalize phase and calculate pi and pi/2 gains from sinusoidal fit parameters."""
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


def fourier_init(xdata, ydata, debug=False):
    """Estimate oscillation frequency and phase using a real-valued FFT."""
    ydata = ydata - np.mean(ydata)
    fourier = np.fft.rfft(ydata)
    fft_freqs = np.fft.rfftfreq(len(ydata), d=xdata[1] - xdata[0])
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
        plt.tight_layout()
        plt.show()
    return max_freq, max_phase


def validate_bounds(fitparams, bounds):
    """Validate that initial fit parameters lie within specified bounds."""
    fitparams = list(fitparams)
    for i, (param, lo, hi) in enumerate(zip(fitparams, bounds[0], bounds[1])):
        below = np.isfinite(lo) and param <= lo
        above = np.isfinite(hi) and param >= hi
        if below or above:
            lo_safe = lo if np.isfinite(lo) else param - 1
            hi_safe = hi if np.isfinite(hi) else param + 1
            fitparams[i] = (lo_safe + hi_safe) / 2
            print(f"fitparam[{i}]={param:.4g} out of bounds [{lo:.4g}, {hi:.4g}] → reset to {fitparams[i]:.4g}")
    return fitparams


def generic_fit(fitfunc, xdata, ydata, fitparams, bounds=None, error_message="Warning: fit failed!"):
    """Generic curve-fitting wrapper using scipy Trust Region Reflective."""
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


def _fit_snr(xdata, ydata, fit, fit_err, fitfunc):
    """Score a fit by SNR = (peak-to-peak of fit curve) / (residual RMS)."""
    if np.any(np.isnan(fit)) or np.any(np.diag(fit_err) == np.inf):
        return -np.inf
    y_fit = fitfunc(xdata, *fit)
    fit_amplitude = np.max(y_fit) - np.min(y_fit)
    residual_rms = np.sqrt(np.mean((ydata - y_fit) ** 2))
    if residual_rms == 0:
        return np.inf
    if fit_amplitude == 0:
        return -np.inf
    return fit_amplitude / residual_rms


def _calculate_normalized_errors(fits, fit_errors):
    """Fallback scorer: mean(σ_i / |p_i|) across parameters."""
    norm_errors = []
    for fit, err_matrix in zip(fits, fit_errors):
        param_errors = np.sqrt(np.abs(np.diag(err_matrix)))
        param_values = np.abs(fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(param_values > 0, param_errors / param_values, np.inf)
        norm_err = np.nanmean(ratios)
        norm_errors.append(np.inf if np.isnan(norm_err) else norm_err)
    return np.array(norm_errors)


def _find_best_fit_with_snr(data, fits, fit_errors, check_measures, fitfunc):
    """Select the best measurement channel by SNR of the fit."""
    xdata = data["xpts"]
    scores = [
        _fit_snr(xdata, data[measure], fits[i], fit_errors[i], fitfunc)
        for i, measure in enumerate(check_measures)
    ]
    best_idx = int(np.argmax(scores))
    if scores[best_idx] == -np.inf:
        return _find_best_fit_simple(fits, fit_errors)
    return best_idx


def _find_best_fit_simple(fits, fit_errors):
    """Select the best fit by minimum normalised covariance error."""
    return int(np.argmin(_calculate_normalized_errors(fits, fit_errors)))


def get_best_fit(data, fitfunc=None, prefixes=["fit"],
                 check_measures=("amps", "avgi", "avgq"),
                 get_best_data_params=(), override=None):
    """Compare fits across measurement channels and return the best one."""
    fits, fit_errors = [], []
    for measure in check_measures:
        for prefix in prefixes:
            fits.append(data[f"{prefix}_{measure}"])
            fit_errors.append(data[f"{prefix}_err_{measure}"])
    for err_matrix in fit_errors:
        diag = np.diag(err_matrix)
        zero_mask = diag == 0
        err_matrix[np.diag_indices_from(err_matrix)] = np.where(zero_mask, np.inf, diag)
    if override is not None and override in check_measures:
        best_index = list(check_measures).index(override)
    elif fitfunc is not None:
        best_index = _find_best_fit_with_snr(data, fits, fit_errors, check_measures, fitfunc)
    else:
        best_index = _find_best_fit_simple(fits, fit_errors)
    best_measure = check_measures[best_index % len(check_measures)]
    result = [fits[best_index], fit_errors[best_index]]
    for param in get_best_data_params:
        result.append(data[f"{param}_{best_measure}"])
    result.append(best_measure)
    return result


# ── Exponential ────────────────────────────────────────────────────────────────

def expfunc(x, *p):
    """Exponential decay: y0 + yscale * exp(-x / decay)."""
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)


def expfunc2(x, *p):
    """Exponential decay with x offset: y0 + yscale * exp(-(x-x0) / decay)."""
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(xdata, ydata, fitparams=None):
    """Fit data to an exponential decay model."""
    if fitparams is None:
        fitparams = [None] * 3
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0] - ydata[-1]
    if fitparams[2] is None: fitparams[2] = (xdata[-1] - xdata[0]) / 4
    return generic_fit(expfunc, xdata, ydata, fitparams, error_message="Warning: Fit exponential failed!")


def fitexp2(xdata, ydata, fitparams=None):
    """Fit data to an exponential decay with x offset."""
    if fitparams is None:
        fitparams = [None] * 4
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0] - ydata[-1]
    if fitparams[2] is None: fitparams[2] = xdata[0]
    if fitparams[3] is None: fitparams[3] = (xdata[-1] - xdata[0]) / 4
    return generic_fit(expfunc2, xdata, ydata, fitparams, error_message="Warning: Fit exponential2 failed!")


# ── Lorentzian ─────────────────────────────────────────────────────────────────

def lorfunc(x, *p):
    """Lorentzian: y0 + yscale / (1 + (x-x0)^2 / xscale^2)."""
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(xdata, ydata, fitparams=None):
    """Fit data to a Lorentzian function."""
    if fitparams is None:
        fitparams = [None] * 4
    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 10
    return generic_fit(lorfunc, xdata, ydata, fitparams, error_message="Warning: Fit Lorentzian failed!")


def asym_lorfunc(x, *p):
    """Asymmetric Lorentzian: y0 + A / (1 + ((x-x0)/(gamma*(1+alpha*(x-x0))))^2)."""
    y0, A, x0, gamma, alpha = p
    return y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)


def fit_asym_lor(xdata, ydata, fitparams=None):
    """Fit data to an asymmetric Lorentzian function."""
    if fitparams is None:
        fitparams = [None] * 5
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None: fitparams[4] = 0
    return generic_fit(asym_lorfunc, xdata, ydata, fitparams, error_message="Warning: Fit asymmetric Lorentzian failed!")


# ── Sinusoidal ─────────────────────────────────────────────────────────────────

def sinfunc(x, *p):
    """Sinusoidal: yscale * sin(2*pi*freq*x + phase_deg*pi/180) + y0."""
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0


def fitsin(xdata, ydata, fitparams=None, debug=False):
    """Fit data to a sinusoidal function."""
    if fitparams is None:
        fitparams = [None] * 4
    max_freq, max_phase = fourier_init(xdata, ydata, debug)
    if fitparams[0] is None: fitparams[0] = 1 / 2 * (max(ydata) - min(ydata))
    if fitparams[1] is None: fitparams[1] = max_freq
    if fitparams[2] is None: fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None: fitparams[3] = np.mean(ydata)
    bounds = (
        [0.5 * fitparams[0], 1e-3, -360, np.min(ydata)],
        [2 * fitparams[0], 1000, 360, np.max(ydata)],
    )
    return generic_fit(sinfunc, xdata, ydata, fitparams, bounds=bounds, error_message="Warning: Fit sinusoidal failed!")


def decaysin(x, *p):
    """Decaying sinusoid: yscale * sin(2*pi*freq*x + phase*pi/180) * exp(-x/decay) + y0."""
    yscale, freq, phase_deg, decay, y0 = p
    return (yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) * np.exp(-x / decay) + y0)


def fitdecaysin(xdata, ydata, fitparams=None, debug=False):
    """Fit data to a decaying sinusoidal function."""
    if fitparams is None:
        fitparams = [None] * 5
    max_freq, max_phase = fourier_init(xdata, ydata, debug)
    if fitparams[0] is None: fitparams[0] = (max(ydata) - min(ydata)) / 2
    if fitparams[1] is None: fitparams[1] = max_freq
    if fitparams[2] is None: fitparams[2] = max_phase * 180 / np.pi + 90
    if fitparams[3] is None: fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4] = np.mean(ydata)
    bounds = (
        [0.75 * fitparams[0], 0.1 * max_freq, -360, 0.3 * (max(xdata) - min(xdata)), np.min(ydata)],
        [1.25 * fitparams[0], 1.5 * max_freq, 360, np.inf, np.max(ydata)],
    )
    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
    except RuntimeError:
        try:
            fitparams[2] = -fitparams[2]
            pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        except:
            print("Warning: Fit decaying sine failed!")
            pOpt = [np.nan] * len(pOpt)
    return pOpt, pCov, fitparams


def decayslopesin(x, *p):
    """Decaying sinusoid with linear slope: yscale*(sin(...)+slope)*exp(-x/decay)+y0."""
    yscale, freq, phase_deg, decay, y0, slope = p
    return (yscale * (np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + slope) * np.exp(-x / decay) + y0)


def fitdecayslopesin(xdata, ydata, fitparams=None, debug=False):
    """Fit data to a decaying sinusoidal function with a linear slope envelope."""
    if fitparams is None:
        fitparams = [None] * 6
    max_freq, max_phase = fourier_init(xdata, ydata, debug)
    if fitparams[0] is None: fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None: fitparams[1] = max_freq
    if fitparams[2] is None: fitparams[2] = max_phase * 180 / np.pi + 90
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 4
    if fitparams[4] is None: fitparams[4] = np.mean(ydata)
    if fitparams[5] is None: fitparams[5] = 0
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
            fitparams[2] = fitparams[2] - 90
            pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams)
        except:
            try:
                fitparams[2] = fitparams[2] + 180
                pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams)
            except:
                print("Warning: Fit decaying slope sine failed!")
                pOpt = [np.nan] * len(pOpt)
    return pOpt, pCov, fitparams


def twofreq_decaysin(x, *p):
    """Two-frequency decaying sinusoidal function."""
    yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01 = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1)


def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    """Fit data to a two-frequency decaying sinusoidal function."""
    if fitparams is None:
        fitparams = [None] * 10
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0] = max(ydata) - min(ydata)
    if fitparams[1] is None: fitparams[1] = max_freq
    if fitparams[2] is None: fitparams[2] = max_phase * 180 / np.pi
    if fitparams[3] is None: fitparams[3] = max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4] = np.mean(ydata)
    if fitparams[5] is None: fitparams[5] = xdata[0]
    if fitparams[6] is None: fitparams[6] = 1
    if fitparams[7] is None: fitparams[7] = 1 / 10
    if fitparams[8] is None: fitparams[8] = 0
    if fitparams[9] is None: fitparams[9] = 0
    bounds = (
        [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)),
         np.min(ydata), xdata[0]-(xdata[-1]-xdata[0]), 0.9, 0.01, -360, -0.1],
        [1.25*fitparams[0], 15/(max(xdata)-min(xdata)), 360, np.inf,
         np.max(ydata), xdata[-1]+(xdata[-1]-xdata[0]), 1.1, 10, 360, 0.1],
    )
    return generic_fit(twofreq_decaysin, xdata, ydata, fitparams, bounds=bounds,
                       error_message="Warning: Fit two-frequency decaying sine failed!")


# ── Gaussian ───────────────────────────────────────────────────────────────────

def gaussian(x, a, x0, sigma, y0):
    """Single Gaussian: a * exp(-(x-x0)^2 / (2*sigma^2)) + y0."""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + y0


def fit_gauss(xdata, ydata, fitparams=None):
    """Fit data to a single Gaussian function."""
    if fitparams is None:
        fitparams = [None] * 4
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = np.max(ydata)
    if fitparams[1] is None: fitparams[1] = xdata[np.argmax(ydata)]
    if fitparams[2] is None: fitparams[2] = (np.max(xdata) - np.min(xdata)) / 8
    if fitparams[3] is None: fitparams[3] = np.min(ydata)
    bounds = (
        [fitparams[0] * 0.5, np.min(xdata), fitparams[2] * 0.1, 0],
        [fitparams[0] * 1.5, np.max(xdata), fitparams[2] * 10, np.max(ydata)],
    )
    return generic_fit(gaussian, xdata, ydata, fitparams, bounds=bounds, error_message="Warning: Fit Gaussian failed!")


def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    """Standard double Gaussian function."""
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1**2)) + a2 * np.exp(-((x - b2) ** 2) / (2 * c2**2))


def fit_doublegauss(xdata, ydata, fitparams):
    """Robust fitting for double Gaussian distributions with constrained means."""
    ag, mug, sig, ae, mue, sie = fitparams
    delta_mu_g = abs(sig) * 2 if sig > 0 else 100
    delta_mu_e = abs(sie) * 2 if sie > 0 else 100
    lb = [0, mug - delta_mu_g, 0, 0, mue - delta_mu_e, 0]
    ub = [np.inf, mug + delta_mu_g, np.inf, np.inf, mue + delta_mu_e, np.inf]
    try:
        pOpt, pCov = sp.optimize.curve_fit(double_gaussian, xdata, ydata, p0=fitparams,
                                           bounds=(lb, ub), maxfev=10000)
    except Exception:
        pOpt = fitparams
        pCov = np.zeros((6, 6))
    return pOpt, pCov


# ── Hanger resonator ───────────────────────────────────────────────────────────

def hangerfunc(x, *p):
    """Complex Hanger (notch) function for resonator S21 fitting."""
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))


def hangerS21func(x, *p):
    """Magnitude of the Hanger function."""
    return np.abs(hangerfunc(x, *p))


def hangerS21func_sloped(x, *p):
    """Magnitude of the Hanger function with a linear background slope."""
    f0, Qi, Qe, phi, scale, slope = p
    return hangerS21func(x, f0, 1e4 * Qi, 1e4 * Qe, phi, scale) + slope * (x - f0)


def hangerphasefunc(x, *p):
    """Phase of the Hanger function."""
    return np.angle(hangerfunc(x, *p))


def fithanger(xdata, ydata, fitparams=None):
    """Fit resonator transmission data to a sloped Hanger function."""
    if fitparams is None:
        fitparams = [None] * 6
    if fitparams[0] is None: fitparams[0] = xdata[np.argmin(np.abs(ydata))]
    if fitparams[1] is None: fitparams[1] = 8
    if fitparams[2] is None: fitparams[2] = 3
    if fitparams[3] is None: fitparams[3] = 0
    if fitparams[4] is None: fitparams[4] = max(ydata)
    if fitparams[5] is None: fitparams[5] = 0
    bounds = (
        [np.min(xdata), 0, 0, -np.inf, 0, -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, np.inf],
    )
    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds)
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=pOpt, bounds=bounds)
    except RuntimeError:
        print("Warning: Fit hanger failed!")
        traceback.print_exc()
    return pOpt, pCov, fitparams


# ── Randomized Benchmarking ────────────────────────────────────────────────────

def rb_func(depth, p, a, b):
    """RB decay function: a * p**depth + b."""
    return a * p**depth + b


def rb_error(p, d):
    """Average error rate per Clifford: 1 - (p + (1-p)/d)."""
    return 1 - (p + (1 - p) / d)


def error_fit_err(cov_p, d):
    """Propagate covariance of p to uncertainty in EPC."""
    return cov_p * (1 / d - 1) ** 2


def rb_gate_fidelity(p_rb, p_irb, d):
    """Gate fidelity from standard and interleaved RB: 1 - (d-1)*(1 - p_irb/p_rb)/d."""
    return 1 - (d - 1) * (1 - p_irb / p_rb) / d


def fitrb(xdata, ydata, fitparams=None, p_bounds=(0.0, 1.0), maxfev=10000):
    """Fit data to the RB decay model a * p**depth + b."""
    y = np.asarray(ydata, dtype=float)
    x = np.asarray(xdata, dtype=float)
    y_max, y_min = y.max(), y.min()
    a0 = y_max - y_min
    b0 = y_min
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
        if fitparams[0] is None: fitparams[0] = p0
        if fitparams[1] is None: fitparams[1] = a0
        if fitparams[2] is None: fitparams[2] = b0
    a_mag = max(abs(a0) * 10, 1.0)
    b_mag = max(abs(b0) * 10 + abs(a0) * 10, 1.0)
    bounds = (
        [p_bounds[0], -a_mag, -b_mag],
        [p_bounds[1], a_mag, b_mag],
    )
    for i, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
        fitparams[i] = float(np.clip(fitparams[i], lo + 1e-9, hi - 1e-9))
    pOpt = fitparams
    pCov = np.full((3, 3), np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(rb_func, x, y, p0=fitparams, bounds=bounds,
                                           maxfev=maxfev, method="trf")
        print(f"[fitrb] p={pOpt[0]:.6f}  a={pOpt[1]:.4f}  b={pOpt[2]:.4f}")
        print(f"[fitrb] σ(p)={np.sqrt(pCov[0, 0]):.2e}  σ(a)={np.sqrt(pCov[1, 1]):.2e}  σ(b)={np.sqrt(pCov[2, 2]):.2e}")
    except RuntimeError:
        print("Warning: RB fit failed!")
        traceback.print_exc()
        pOpt = [np.nan] * 3
    return pOpt, pCov


# ── Adiabatic pulse ────────────────────────────────────────────────────────────

def adiabatic_amp(t, amp_max, beta, period):
    """Amplitude envelope for an adiabatic pi pulse (sech shape)."""
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t, mu, beta, period):
    """Phase function for an adiabatic pi pulse."""
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(t, amp_max, mu, beta, period):
    """Calculate I and Q amplitudes for an adiabatic pi pulse."""
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j * np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j * np.cos(phase))
    return np.real(iamp), np.real(qamp)


# ── Rotation error models ──────────────────────────────────────────────────────

def probg_Xhalf(n, *p):
    """Ground-state probability for repeated X/2 pulse sequence."""
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta))


def probg_X(n, *p):
    """Ground-state probability for repeated X pulse sequence."""
    a, delta = p
    delta = delta * np.pi / 180
    return a + (0.5 * np.cos(np.pi / 2 + 2 * n * delta))


def probg_Xhalf_decay(n, *p):
    """Decaying ground-state probability for repeated X/2 pulse sequence."""
    a, delta, decay = p
    delta = delta * np.pi / 180
    return a + (0.5 * (-1) ** n * np.cos(np.pi / 2 + 2 * n * delta)) * np.exp(-n / decay)


def fit_probg_Xhalf(xdata, ydata, fitparams=None):
    """Fit data to the X/2 repeated pulse rotation error model."""
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = np.average(ydata)
    if fitparams[1] is None: fitparams[1] = 0.0
    bounds = ([min(ydata), -20.0], [max(ydata), 20.0])
    return generic_fit(probg_Xhalf, xdata, ydata, fitparams, bounds=bounds,
                       error_message="Warning: Fit X/2 rotation error failed!")


def fit_probg_X(xdata, ydata, fitparams=None):
    """Fit data to the X repeated pulse rotation error model."""
    if fitparams is None:
        fitparams = [None] * 2
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = np.average(ydata)
    if fitparams[1] is None: fitparams[1] = 0.0
    bounds = ([min(ydata), -20.0], [max(ydata), 20.0])
    return generic_fit(probg_X, xdata, ydata, fitparams, bounds=bounds,
                       error_message="Warning: Fit X rotation error failed!")


def fit_probg_Xhalf_decay(xdata, ydata, fitparams=None):
    """Fit data to the decaying X/2 repeated pulse rotation error model."""
    if fitparams is None:
        fitparams = [None] * 3
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = np.average(ydata)
    if fitparams[1] is None: fitparams[1] = 0.0
    if fitparams[2] is None: fitparams[2] = 10
    bounds = ([min(ydata), -20.0, 1], [max(ydata), 20.0, 100])
    return generic_fit(probg_Xhalf_decay, xdata, ydata, fitparams, bounds=bounds,
                       error_message="Warning: Fit decaying X/2 rotation error failed!")


# ── Poisson ────────────────────────────────────────────────────────────────────

def poisson(n, *p):
    """Poisson PMF: exp(-nbar) * nbar^n / n!"""
    nbar = p[0]
    return np.exp(-nbar) * (nbar**n) / sp.special.factorial(n)


def fit_poisson(xdata, ydata, fitparams=None):
    """Fit data to a Poisson distribution."""
    if fitparams is None:
        fitparams = [None] * 1
    else:
        fitparams = list(fitparams)
    if fitparams[0] is None: fitparams[0] = ydata[0]
    bounds = ([0], [10])
    return generic_fit(poisson, xdata, ydata, fitparams, bounds=bounds, error_message="Warning: Fit Poisson failed!")
