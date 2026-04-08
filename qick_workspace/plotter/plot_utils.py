import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..tools import fitting as fitter


def plot_final(xpts, data: np.ndarray, x_label: str, fitfunc, simfunc, return_ax=False):
    """
    Plot all four IQ quadratures (abs, phase, I, Q), fit each,
    and display the best-fit channel on a large right-hand panel.

    Returns
    -------
    fit_params, error, fig            (return_ax=False)
    fit_params, error, fig, ax_big   (return_ax=True)
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
