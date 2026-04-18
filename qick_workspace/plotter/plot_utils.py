import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..tools import fitting as fitter


def plot_final(xpts, data: np.ndarray, x_label: str, fitfunc, simfunc, return_ax=False):
    """
    Plot all four IQ quadratures, fit each, and highlight the best-fit channel.

    Fits *abs*, *phase*, *I*, and *Q* traces individually using *fitfunc*,
    selects the best-fitting channel via :func:`~tools.fitting.get_best_fit`,
    and renders a 2×3 figure layout: four small subplots on the left two
    columns and one large panel on the right showing the best channel.

    Parameters
    ----------
    xpts : np.ndarray
        Independent variable array (e.g. pulse length or frequency sweep).
    data : np.ndarray
        Complex-valued IQ data array (same length as *xpts*).
    x_label : str
        Label for the x-axis of all subplots.
    fitfunc : callable
        Fit routine with signature ``fitfunc(x, y) → (popt, pcov, info)``.
        Called once per quadrature.
    simfunc : callable
        Forward model function with signature ``simfunc(x, *params) → y``.
        Used to evaluate the fitted curve.
    return_ax : bool, optional
        When ``True``, also returns the ``Axes`` object for the large
        best-fit panel.  Default is ``False``.

    Returns
    -------
    fit_params : np.ndarray
        Best-fit parameter array for the channel selected by
        :func:`~tools.fitting.get_best_fit`.
    error : np.ndarray
        Standard errors (square root of the diagonal of *pcov*) for
        *fit_params*.
    fig : matplotlib.figure.Figure
        The rendered figure.
    ax_big : matplotlib.axes.Axes
        Large right-hand panel axes object.  Only returned when
        *return_ax* is ``True``.
    """
    marker_style = {"marker": "o", "markersize": 5, "alpha": 0.7, "linestyle": "-"}

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
