"""This module provides setup-independent functions for evaluating the quality of gain vs frequency curves."""

import numpy as np
import xarray as xr
import logging
from typing import List

import matplotlib.pyplot as plt

###################################
#
# Functions for scoring gain curves
#
###################################

def score_ai_twpa_c_gain_data(gain_data: xr.DataArray,
                              gain_min: float, # Target minimum gain at the bottom of gain ripples
                              gain_median: float, # Target median gain
                              ripple_max: float, # Target pk-to-pk gain ripple
                              ripple_window: float=150e6, # Window width in Hz for computing ripple
                              gain_min_exp: float=1., # Exponent for controlling how heavily deviations of gain below gain_min are punished
                              gain_median_weight: float=20., # Relative weight of median gain when computing total score
                              ripple_weight: float=80., # Relative weight of ripple violations when computing total score
                              f_min: float=0, # Frequency range to score
                              f_max: float=np.inf, # Frequency range to score
                              freq_key: str="frequency",
                              ):
    """This functions runs the individual score_*() functions and adds the scores together
       according to:

          total = (min. gain score)^gain_min_exp × [ gain_median_weight×(median gain score) + ripple_weight×(ripple score) ]
    """
    gain_data = gain_ensure_dB(gain_data)

    common_args = {"f_min": f_min, "f_max": f_max, "freq_key": freq_key}
    score  = gain_median_weight * score_median_gain(gain_data, gain_median=gain_median, **common_args)
    score += ripple_weight * score_gain_ripple(gain_data, ripple_max=ripple_max, ripple_window=ripple_window, **common_args)
    score *= score_min_gain(gain_data, gain_min=gain_min, **common_args)**gain_min_exp
    score = score.rename(f"total_score")
    return score

def gain_ensure_dB(gain: xr.DataArray):
    """Returns gain in dB. If the input data (gain) is complex, assume that it's on a linear amplitude scale.
       If the input is real, assume that it's already in dB."""
    if np.iscomplexobj(gain):
      gain = 20*np.log10(np.abs(gain)) # Convert from linear scale to dB

    if gain.max() < 2: logging.warning("Max gain in dataset appears to be less than 2 dB. Is it on linear scale? score_ai_twpa_c_gain_data() expects either complex data on linear scale, or magnitude in dB.")
    return gain

def score_min_gain(gain_data: xr.DataArray,
                   gain_min: float, f_min: float, f_max: float,
                   freq_key="frequency") -> xr.DataArray:
    '''
    Compute minimum gain score. Assumes that freq_key exists as a coordinate in gain_data.
    Arbitrary other coordinates are allowed, and are assumed to label distinct gain vs frequency curves.

    The min. gain score is the fraction of frequency points in the gain curve that exceed the gain_min.

    Returns xr.DataArray named gain_min_score with the same coordinates as gain_data, excluding freq_key.
    '''
    assert freq_key in gain_data.coords

    f = gain_data[freq_key]
    f_region = (f > f_min) & (f < f_max)
    gain_data = gain_data.where(f_region, drop=True)
    ax = gain_data.dims.index(freq_key) # axis index for frequency

    score = (gain_data > gain_min).reduce(
                np.sum, axis=ax, keep_attrs=True) / np.sum(f_region)
    score = score.rename(f"gain_min_score")

    return score

def score_median_gain(gain_data: xr.DataArray,
                      gain_median: float, f_min: float, f_max: float,
                      freq_key="frequency") -> xr.DataArray:
    '''
    Compute median gain score. Assumes that freq_key exists as a coordinate in gain_data.
    Arbitrary other coordinates are allowed, and are assumed to label distinct gain vs frequency curves.

    The median gain score for a gain curve is:
      (<median of the gain points> - gain_mean) / gain_mean

    Returns xr.DataArray named gain_median_score with the same coordinates as gain_data, excluding freq_key.
    '''
    assert freq_key in gain_data.coords

    f = gain_data[freq_key]
    gain_data = gain_data.where((f > f_min) & (f < f_max), drop=True)
    ax = gain_data.dims.index(freq_key) # axis index for frequency

    score = gain_data.reduce(
                np.median, axis=ax, keep_attrs=True) / gain_median
    score = score.rename(f"gain_median_score")

    return score

def score_gain_ripple(gain_data: xr.DataArray,
                      ripple_max: float, ripple_window: float,
                      f_min: float, f_max: float,
                      freq_key="frequency") -> xr.DataArray:
    '''
    Compute gain ripple score. Assumes that freq_key exists as a coordinate in gain_data.
    Arbitrary other coordinates are allowed, and are assumed to label distinct gain vs frequency curves.

    The gain ripple score for a gain curve is the fraction of the frequency points [fmin, fmax]
    where pk-to-pk ripple is below ripple_max. See compute_ripple() for definition of ripple.

    Returns xr.DataArray named ripple_score with the same coordinates as gain_data, excluding freq_key.
    '''
    assert freq_key in gain_data.coords

    ripple = compute_ripple(gain_data, freq_key=freq_key,
                            ripple_window=ripple_window, f_min=f_min, f_max=f_max)

    # We can reuse score_min_gain() if we invert the sign of the ripple
    score= score_min_gain(-ripple, gain_min=-ripple_max,
                          f_min=f_min, f_max=f_max, freq_key=freq_key)
    score = score.rename(f"ripple_score")
    return score

def compute_ripple(gain: xr.DataArray,
                   ripple_window: float,
                   f_min: float, f_max: float,
                   freq_key: str="frequency") -> xr.DataArray:
    '''
    Compute the ripple in gain DataArray.

    The ripple at frequency f is defined as max. minus min. gain within +/- ripple_window/2 Hz of f,
    *and* fully within [f_min, f_max].

    Returns DataArray with the same coordinates as gain. All values outside [f_min, f_max] are set to nan.
    '''

    def max_minus_min(x): return x.max() - x.min()

    ripple = xr.zeros_like(gain).rename("ripple") + np.nan # Initialize all ripple values to nan

    f = gain[freq_key]
    df = np.diff(f.to_numpy())
    assert df.max() - df.min() < 1, f"Frequency grid must be even and sorted: {np.unique(df.astype(int))}"

    int_period = int(np.round(ripple_window/df[0]))
    ripple_within_window = max_minus_min(gain.rolling({f.dims[0]: int_period}, min_periods=int_period//2, center=True))

    # Assign the result from this band to the ripple output buffer
    edge_exclusion = min(
        ripple_window/2, # The typical case where ripple_window << f_max-f_min
        (f_max - f_min)/4 # Very narrow [f_min, f_max] band comparable to ripple window --> Include points closer to band edge.
    )
    ripple = xr.where((f >= f_min + edge_exclusion) & (f <= f_max - edge_exclusion),
                      ripple_within_window, ripple)

    return ripple


#############################################################################
#
# Functions for evaluating scored data and choosing/plotting operation points
#
#############################################################################

def operation_point_distance(point0: xr.DataArray, point1: xr.DataArray,
                             characteristic_scale={ "pump_power": 0.1, "pump_freq": 1e6, "ifbl": 2e-6 }):
  """ A distance metric for evaluating how far two operation points are from each other.

      point0 can also be a collection of many points.
      
      Characteristic scales should reflect our expectation for the scale on which performance
      is sensitive to variations in the operation point parameters.
  """
  sum_of_squares = 0
  for dim in characteristic_scale.keys():
    if (dim in point0.coords) and (dim in point1.coords):
      sum_of_squares = sum_of_squares + ((point1[dim] - point0[dim])/characteristic_scale[dim])**2
    else:
      if dim!="pump_power": logging.warning(f"{dim} not present in point0 and point1. Assuming it's equal for both points.")

  return np.sqrt(sum_of_squares)

def find_best_operation_point(scored_data,
                              excluded_points=[], exclusion_radius=1,
                              distance_metric=operation_point_distance,
                              pump_power_key="pump_power", pump_freq_key="pump_freq", ifbl_key="ifbl"):
  """Find the most promising point in the scored data, excluding points near already found points ("excluded_points").

     Typically called in a loop to find the top few local maxima in a gain scan.

     A custom distance_metric can be used to override the default characteristic scales
     that determine how variations in the three different dimensions (flux bias, pump freq., pump power)
     are weighted when excluded volumes around excluded_points.
     You may wish to use functools.partial(operation_point_distance, ...) to implement your custom metric.
     """
  # Exclude operation points near excluded_points
  for pt in excluded_points:
    scored_data = scored_data.where(distance_metric(scored_data, pt) > exclusion_radius, drop=True)

  # Find the best point over flux bias and pump frequency, and potentially pump power
  argmax_dims = [pump_freq_key, ifbl_key]
  if pump_power_key is not None and pump_power_key in scored_data.coords: argmax_dims += [ pump_power_key ]

  try:
    best_point_indices = scored_data.argmax(dim=argmax_dims, skipna=True)
  except ValueError:
    logging.warning("Cannot determine argmax. This usually means that there are no non-NaN scores, after exclusions.")
    return None

  best_point = scored_data.isel( best_point_indices )

  return best_point

def plot_one_gain_curve(gain_data: xr.DataArray, ifbl: float, pump_freq: float, pump_power: float=None, ax: plt.Axes=None,
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

def plot_gain_at_operation_point(gain_data: xr.DataArray, op: xr.DataArray, ax: plt.Axes=None,
                                 pump_power_key="pump_power", pump_freq_key="pump_freq", ifbl_key="ifbl"):
  """Plot gain vs frequency at the specified operation point coordinates (op)."""
  coords = { k: float(op[k]) for k in op.coords }
  return plot_one_gain_curve(gain_data=gain_data, **coords, ax=ax,
                             pump_power_key=pump_power_key, pump_freq_key=pump_freq_key, ifbl_key=ifbl_key)

def plot_operation_point_parameters(scored_data: xr.DataArray, ops: List[xr.DataArray], ax: plt.Axes=None,
                                    pump_freq_key="pump_freq", ifbl_key="ifbl", heatmap_kwargs={}):
  """Plot a heat map of scored data vs flux bias and pump frequency,
     and plot the coordinates of the provided operation points (ops)
     as red dots over the heat map."""

  # Plot the score as background
  if ax is None: fig, ax = plt.subplots()
  scored_data.plot(x=ifbl_key, y=pump_freq_key, ax=ax, **heatmap_kwargs)

  # Mark the specified operation points with red dots
  for op in ops: ax.plot(op[ifbl_key], op[pump_freq_key], 'ro')

  return ax
