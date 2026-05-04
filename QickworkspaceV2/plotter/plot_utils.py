"""
plot_utils.py — professional fit-result figure for all Analysis classes.

Layout (2 × 3 GridSpec):

    ┌─────────────┬─────────────┬─────────────────────────────┐
    │  Amplitude  │      I      │                             │
    │   + fit     │    (raw)    │   Amplitude  ← main panel   │
    │             │             │   + fit overlay             │
    ├─────────────┼─────────────┤   + result text box         │
    │    Phase    │      Q      │   (title coloured by        │
    │    (raw)    │    (raw)    │    quality flag)            │
    └─────────────┴─────────────┴─────────────────────────────┘

The fit curve is drawn on the Amplitude panel (top-left) and the
large main panel (right).  Phase, I, Q are shown raw for reference.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Quality → colour mapping ──────────────────────────────────────────────────

_QUALITY_COLOR = {
    "good":           "#2ca02c",   # green
    "warning":        "#ff7f0e",   # orange
    "bad":            "#d62728",   # red
    "no_information": "#1f77b4",   # blue (neutral)
}

# ── Style constants ───────────────────────────────────────────────────────────

_DATA_KW  = dict(marker="o", ms=3.5, alpha=0.75, lw=0.9, color="#2171b5")
_FIT_KW   = dict(lw=2.0, color="#d62728", zorder=5)
_GRID_KW  = dict(alpha=0.25, lw=0.5, color="#888888")


# ── Public API ────────────────────────────────────────────────────────────────

def plot_fit_result(
    xpts: np.ndarray,
    iq_data: np.ndarray,
    simfunc,
    fit_params,
    *,
    x_label: str = "x",
    title: str = "",
    result_text: str = "",
    quality: str = "no_information",
    extra_lines=None,
) -> plt.Figure:
    """
    Render a professional 2×3 figure for a completed fit.

    Parameters
    ----------
    xpts        : sweep axis (x-values).
    iq_data     : complex IQ array (raw_iq from ExperimentData).
    simfunc     : fit function  ``f(x, *params) → y``.
    fit_params  : pre-computed fit parameters (``None`` → skip fit overlay).
    x_label     : shared x-axis label.
    title       : figure suptitle.
    result_text : newline-separated key results shown in the main panel.
    quality     : ``"good" | "warning" | "bad" | "no_information"``.
    extra_lines : list of dicts passed to ``ax.axvline()`` in every panel,
                  e.g. ``[{"x": 0.5, "color": "r", "ls": "--", "label": "π"}]``.
    """
    channels = {
        "Amplitude": np.abs(iq_data),
        "Phase":     np.unwrap(np.angle(iq_data)),
        "I":         iq_data.real,
        "Q":         iq_data.imag,
    }

    x_fit = np.linspace(xpts[0], xpts[-1], 600) if fit_params is not None else None
    fit_y = simfunc(x_fit, *fit_params) if fit_params is not None else None

    q_color = _QUALITY_COLOR.get(quality.lower(), "#333333")

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 1.9],
        hspace=0.48, wspace=0.32,
        left=0.06, right=0.97, top=0.88, bottom=0.12,
    )

    # ── 4 small panels ────────────────────────────────────────────────────────
    panel_order = ["Amplitude", "I", "Phase", "Q"]
    for idx, key in enumerate(panel_order):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(xpts, channels[key], **_DATA_KW)

        # Fit overlay on Amplitude only
        if key == "Amplitude" and fit_y is not None:
            ax.plot(x_fit, fit_y, **_FIT_KW)

        # Extra vertical markers
        if extra_lines:
            for kw in extra_lines:
                ax.axvline(**{k: v for k, v in kw.items() if k != "label"})

        ax.set_xlabel(x_label, fontsize=7.5)
        ax.set_ylabel("ADC", fontsize=7.5)
        ax.set_title(key, fontsize=8.5, fontweight="bold", pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, **_GRID_KW)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # ── Main panel ────────────────────────────────────────────────────────────
    ax_main = fig.add_subplot(gs[:, 2])
    ax_main.plot(xpts, channels["Amplitude"], label="data", **_DATA_KW)

    if fit_y is not None:
        ax_main.plot(x_fit, fit_y, label="fit", **_FIT_KW)

    if extra_lines:
        for kw in extra_lines:
            ax_main.axvline(**kw)
        ax_main.legend(fontsize=8.5, framealpha=0.85, loc="upper right")
    else:
        ax_main.legend(fontsize=8.5, framealpha=0.85)

    ax_main.set_xlabel(x_label, fontsize=10)
    ax_main.set_ylabel("Amplitude (ADC)", fontsize=10)
    ax_main.tick_params(labelsize=8.5)
    ax_main.grid(True, **_GRID_KW)
    for spine in ax_main.spines.values():
        spine.set_linewidth(0.8)

    # Result text box
    if result_text:
        ax_main.text(
            0.97, 0.97, result_text,
            transform=ax_main.transAxes,
            fontsize=9, va="top", ha="right", family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.45", facecolor="white",
                edgecolor="#aaaaaa", alpha=0.90, linewidth=0.8,
            ),
        )

    # Quality badge in top-right of main panel
    badge_label = f"● {quality.upper().replace('_', ' ')}"
    ax_main.text(
        0.03, 0.97, badge_label,
        transform=ax_main.transAxes,
        fontsize=8, va="top", ha="left", color=q_color, fontweight="bold",
    )

    # Suptitle
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=q_color, y=0.98)

    plt.show()
    return fig


# ── Legacy entry-point (kept for backward compatibility) ─────────────────────

def plot_final(xpts, data: np.ndarray, x_label: str, fitfunc, simfunc, return_ax=False):
    """
    Legacy: fit all four IQ channels independently and render a 2×3 figure.

    Prefer :func:`plot_fit_result` for new code (uses pre-computed params).
    """
    from ..tools import fitting as fitter

    d = {
        "xpts":  xpts,
        "amps":  np.abs(data),
        "phase": np.unwrap(np.angle(data)),
        "avgi":  data.real,
        "avgq":  data.imag,
    }

    for measure in ("amps", "phase", "avgi", "avgq"):
        popt, pcov, _ = fitfunc(d["xpts"], d[measure])
        d[f"fit_{measure}"]     = popt
        d[f"fit_err_{measure}"] = pcov

    fit_params, fit_err, best_measure = fitter.get_best_fit(d, fitfunc=simfunc)

    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

    marker_style = {"marker": "o", "markersize": 5, "alpha": 0.7, "linestyle": "-"}
    for i, measure in enumerate(("amps", "phase", "avgi", "avgq")):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.plot(d["xpts"], d[measure], **marker_style)
        ax.plot(d["xpts"], simfunc(d["xpts"], *d[f"fit_{measure}"]))
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{measure} (ADC unit)")
        ax.set_title(measure)

    ax_big = fig.add_subplot(gs[:, 2])
    ax_big.set_title(f"Best fit: {best_measure}")
    ax_big.plot(d["xpts"], d[best_measure], **marker_style)
    ax_big.plot(d["xpts"], simfunc(d["xpts"], *fit_params))
    ax_big.set_xlabel(x_label)
    ax_big.set_ylabel("ADC unit")

    error = np.sqrt(np.diag(fit_err))
    if return_ax:
        return fit_params, error, fig, ax_big
    return fit_params, error, fig


__all__ = ["plot_fit_result", "plot_final"]
