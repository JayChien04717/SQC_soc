"""Functions for evaluating the quality of TWPA gain vs frequency curves."""

import numpy as np
import xarray as xr
import logging
from typing import List

import matplotlib.pyplot as plt


def score_ai_twpa_c_gain_data(gain_data: xr.DataArray,
                              gain_min: float,
                              gain_median: float,
                              ripple_max: float,
                              ripple_window: float = 150e6,
                              gain_min_exp: float = 1.,
                              gain_median_weight: float = 20.,
                              ripple_weight: float = 80.,
                              f_min: float = 0,
                              f_max: float = np.inf,
                              freq_key: str = "frequency"):
    """
    Run individual score_*() functions and combine:

       total = (min. gain score)^gain_min_exp * [gain_median_weight*(median gain score) + ripple_weight*(ripple score)]
    """
    gain_data = gain_ensure_dB(gain_data)
    common_args = {"f_min": f_min, "f_max": f_max, "freq_key": freq_key}
    score  = gain_median_weight * score_median_gain(gain_data, gain_median=gain_median, **common_args)
    score += ripple_weight * score_gain_ripple(gain_data, ripple_max=ripple_max, ripple_window=ripple_window, **common_args)
    score *= score_min_gain(gain_data, gain_min=gain_min, **common_args)**gain_min_exp
    score = score.rename("total_score")
    return score


def gain_ensure_dB(gain: xr.DataArray):
    """Return gain in dB. Converts from linear amplitude if complex input."""
    if np.iscomplexobj(gain):
        gain = 20 * np.log10(np.abs(gain))
    if gain.max() < 2:
        logging.warning("Max gain appears to be < 2 dB. Is it on linear scale?")
    return gain


def score_min_gain(gain_data: xr.DataArray,
                   gain_min: float, f_min: float, f_max: float,
                   freq_key="frequency") -> xr.DataArray:
    """Fraction of frequency points in [f_min, f_max] that exceed gain_min."""
    assert freq_key in gain_data.coords
    f = gain_data[freq_key]
    f_region = (f > f_min) & (f < f_max)
    gain_data = gain_data.where(f_region, drop=True)
    ax = gain_data.dims.index(freq_key)
    score = (gain_data > gain_min).reduce(np.sum, axis=ax, keep_attrs=True) / np.sum(f_region)
    return score.rename("gain_min_score")


def score_median_gain(gain_data: xr.DataArray,
                      gain_median: float, f_min: float, f_max: float,
                      freq_key="frequency") -> xr.DataArray:
    """Ratio of median gain in [f_min, f_max] to target gain_median."""
    assert freq_key in gain_data.coords
    f = gain_data[freq_key]
    gain_data = gain_data.where((f > f_min) & (f < f_max), drop=True)
    ax = gain_data.dims.index(freq_key)
    score = gain_data.reduce(np.median, axis=ax, keep_attrs=True) / gain_median
    return score.rename("gain_median_score")


def score_gain_ripple(gain_data: xr.DataArray,
                      ripple_max: float, ripple_window: float,
                      f_min: float, f_max: float,
                      freq_key="frequency") -> xr.DataArray:
    """Fraction of frequency points where pk-to-pk ripple is below ripple_max."""
    assert freq_key in gain_data.coords
    ripple = compute_ripple(gain_data, freq_key=freq_key,
                            ripple_window=ripple_window, f_min=f_min, f_max=f_max)
    score = score_min_gain(-ripple, gain_min=-ripple_max,
                           f_min=f_min, f_max=f_max, freq_key=freq_key)
    return score.rename("ripple_score")


def compute_ripple(gain: xr.DataArray,
                   ripple_window: float,
                   f_min: float, f_max: float,
                   freq_key: str = "frequency") -> xr.DataArray:
    """Compute pk-to-pk ripple within a sliding window of width ripple_window Hz."""

    def max_minus_min(x): return x.max() - x.min()

    ripple = xr.zeros_like(gain).rename("ripple") + np.nan
    f = gain[freq_key]
    df = np.diff(f.to_numpy())
    assert df.max() - df.min() < 1, f"Frequency grid must be even and sorted: {np.unique(df.astype(int))}"

    int_period = int(np.round(ripple_window / df[0]))
    ripple_within_window = max_minus_min(
        gain.rolling({f.dims[0]: int_period}, min_periods=int_period // 2, center=True)
    )

    edge_exclusion = min(ripple_window / 2, (f_max - f_min) / 4)
    ripple = xr.where(
        (f >= f_min + edge_exclusion) & (f <= f_max - edge_exclusion),
        ripple_within_window, ripple
    )
    return ripple


def operation_point_distance(point0: xr.DataArray, point1: xr.DataArray,
                              characteristic_scale={"pump_power": 0.1, "pump_freq": 1e6, "ifbl": 2e-6}):
    """Distance metric between two operation points."""
    sum_of_squares = 0
    for dim in characteristic_scale.keys():
        if (dim in point0.coords) and (dim in point1.coords):
            sum_of_squares = sum_of_squares + ((point1[dim] - point0[dim]) / characteristic_scale[dim]) ** 2
        else:
            if dim != "pump_power":
                logging.warning(f"{dim} not present in point0 and point1.")
    return np.sqrt(sum_of_squares)


def find_best_operation_point(scored_data,
                               excluded_points=[], exclusion_radius=1,
                               distance_metric=operation_point_distance,
                               pump_power_key="pump_power", pump_freq_key="pump_freq", ifbl_key="ifbl"):
    """Find the best scoring point, excluding regions near already-found points."""
    for pt in excluded_points:
        scored_data = scored_data.where(distance_metric(scored_data, pt) > exclusion_radius, drop=True)

    argmax_dims = [pump_freq_key, ifbl_key]
    if pump_power_key is not None and pump_power_key in scored_data.coords:
        argmax_dims += [pump_power_key]

    try:
        best_point_indices = scored_data.argmax(dim=argmax_dims, skipna=True)
    except ValueError:
        logging.warning("Cannot determine argmax. No non-NaN scores after exclusions.")
        return None

    return scored_data.isel(best_point_indices)


def plot_one_gain_curve(gain_data: xr.DataArray, ifbl: float, pump_freq: float,
                        pump_power: float = None, ax: plt.Axes = None,
                        pump_power_key="pump_power", pump_freq_key="pump_freq", ifbl_key="ifbl"):
    """Plot gain vs frequency at the specified operation point coordinates."""
    if ifbl is not None: gain_data = gain_data.sel(**{ifbl_key: ifbl})
    if pump_freq is not None: gain_data = gain_data.sel(**{pump_freq_key: pump_freq})
    if pump_power is not None: gain_data = gain_data.sel(**{pump_power_key: pump_power})
    if ax is None: fig, ax = plt.subplots()
    gain = gain_ensure_dB(gain_data)
    gain.plot(ax=ax)
    ax.set_ylim(-5, 25)
    ax.set_ylabel("Gain (dB)")
    return gain


def plot_gain_at_operation_point(gain_data: xr.DataArray, op: xr.DataArray, ax: plt.Axes = None,
                                  pump_power_key="pump_power", pump_freq_key="pump_freq", ifbl_key="ifbl"):
    """Plot gain vs frequency at the specified operation point (op)."""
    coords = {k: float(op[k]) for k in op.coords}
    return plot_one_gain_curve(gain_data=gain_data, **coords, ax=ax,
                                pump_power_key=pump_power_key, pump_freq_key=pump_freq_key, ifbl_key=ifbl_key)


def plot_operation_point_parameters(scored_data: xr.DataArray, ops: List[xr.DataArray], ax: plt.Axes = None,
                                     pump_freq_key="pump_freq", ifbl_key="ifbl", heatmap_kwargs={}):
    """Plot heat map of scored data with operation points marked as red dots."""
    if ax is None: fig, ax = plt.subplots()
    scored_data.plot(x=ifbl_key, y=pump_freq_key, ax=ax, **heatmap_kwargs)
    for op in ops: ax.plot(op[ifbl_key], op[pump_freq_key], 'ro')
    return ax
