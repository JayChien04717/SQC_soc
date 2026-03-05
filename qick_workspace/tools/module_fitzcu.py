import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from .fitting import *
from typing import Optional, Dict
import pprint

try:
    from .abcd_rf_fit import *
except:
    print("No circle fit package")

figsize = (10, 4)
marker_style = {
    "marker": "o",
    "markersize": 5,
    "alpha": 0.7,
    "linestyle": "-",
}


def post_rotate(ydata):
    from scipy.optimize import minimize_scalar

    def rotate_complex(iq, angle):
        return (iq) * np.exp(1j * np.pi * angle / 180)

    def std_q(y, rot_agl_):
        iq = rotate_complex(y, rot_agl_)
        return np.std(iq.imag)

    res = minimize_scalar(lambda agl: std_q(ydata, agl), bounds=[0, 360])
    rotation_angle = res.x
    ydata = rotate_complex(ydata, rotation_angle)
    return ydata


def rsquare(y: float, ybar: float) -> float:
    """Calculate the coefficient of determination

    Parameters
    ----------
    y : float
        measure data
    ybar : float
        fitting data

    Returns
    -------
    float
        return the r square coefficient
    """
    ss_res = np.sum((y - ybar) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_square = 1 - ss_res / ss_tot
    return r_square


def resonator_circlefit(x: float, y: float, solve_type: str = "hm") -> Optional[Dict]:
    """Hanger geometry resonator circult fit
    This fitting tool is from https://github.com/sebastianprobst/resonator_tools
    Only for notch type hanger resonator

    Parameters
    ----------
    x : float
        frequency list/array
    y : float
        S21 data

    Returns
    -------
    Optional[Dict]
        fitting result, data contain Qc, Qi, Ql .ect.
    """
    fit = analyze(x * 1e6, y, solve_type, fit_edelay=True)
    fit.plot()
    param = fit.tolist()
    result_dict = {
        "Fres(GHz)": round(param[0] / 1e9, 4),
        "Qi": round((param[0] / (param[1] - param[2]))),
        "absQc": round(param[0] / param[2]),
        "Ql": round(param[0] / param[1]),
        "κ(MHz)": round(param[1] * 1e-6, 2),
    }
    pprint.pprint(result_dict)
    return param


def resonator_analyze(
    x: np.ndarray, y: np.ndarray, fit: bool = True
) -> Optional[float]:
    """
    Analyze the resonator spectrum by plotting amplitude and phase data.
    Optionally, fit the data using an asymmetric Lorentzian function.

    Parameters
    ----------
    x : np.ndarray
        Frequency list/array (in MHz).
    y : np.ndarray
        Complex S21 data.
    fit : bool, optional
        If True, fit the data using an asymmetric Lorentzian function.
        Otherwise, only plot the raw data. Default is True.

    Returns
    -------
    Optional[float]
        Resonance frequency (MHz), rounded to two decimal places, or None if fitting is not performed.
    """

    avg_abs = np.abs(y)
    plt.figure(figsize=(12, 4))
    plt.plot(x, avg_abs, label="meas", **marker_style)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Signal (a.u)")
    plt.title("Amplitude")

    res = None  # Initialize resonance frequency variable

    if fit:
        pOpt, _ = fit_asym_lor(x, avg_abs)  # Fit the data
        res = pOpt[2]  # Extract resonance frequency

        plt.plot(x, asym_lorfunc(x, *pOpt), label=f"Fit, $\kappa$={pOpt[3]:.2f} MHz")
        plt.axvline(res, color="r", ls="--", label=f"$f_{{res}}$ = {res:.2f} MHz")
    plt.legend()

    return round(res, 4) if res is not None else None


def spectrum_analyze(x: np.ndarray, y: np.ndarray, fit: bool = True) -> Optional[float]:
    """
    Analyze the spectrum by plotting amplitude and phase data.
    Optionally, fit the data using a Lorentzian function.

    Parameters
    ----------
    x : np.ndarray
        Frequency list/array (in MHz).
    y : np.ndarray
        Complex S21 data.
    fit : bool, optional
        If True, fit the data using a Lorentzian function.
        Otherwise, only plot the raw data. Default is True.

    Returns
    -------
    Optional[float]
        Resonance frequency (MHz), rounded to two decimal places, or None if fitting is not performed.
    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)  # Compute amplitude

    plt.title("Qubit Spectrum")
    plt.plot(x, data, label="meas", **marker_style)
    plt.ylabel("Signal (a.u)")
    plt.xlabel("Frequency (MHz)")
    plt.legend()

    res: Optional[float] = None  # Initialize resonance frequency variable

    if fit:
        pOpt, pCov, fitparams = fitlor(x, data)
        res = pOpt[2]  # Extract resonance frequency

        plt.plot(x, lorfunc(x, *pOpt), label="Fit")
        plt.axvline(res, color="r", ls="--", label=f"$f_{{res}}$ = {res:.2f} MHz")
        plt.legend()

    return round(res, 4) if res is not None else None


def dispersive_analyze(x: float, y1: float, y2: float, fit: bool = True):
    """Plot the dispersive shift and maximum shift frequency

    Parameters
    ----------
    x : float
        frequency
    y1 : float
        ground/excited state spectrum
    y2 : float
        ground/excited state spectrum
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    """
    y1 = np.abs(y1)
    y2 = np.abs(y2)
    pOpt1, pCov1 = fit_asym_lor(x, y1)
    res1 = pOpt1[2]
    pOpt2, pCov2 = fit_asym_lor(x, y2)
    res2 = pOpt2[2]

    plt.figure(figsize=figsize)
    plt.title(f"$\chi=${(res2 - res1):.3f}, unit = MHz", fontsize=15)
    plt.plot(x, y1, label="e", **marker_style)
    plt.plot(x, y2, label="g", **marker_style)
    if fit == True:
        plt.plot(
            x, asym_lorfunc(x, *pOpt1), label=f"fite, $\kappa$ = {pOpt1[3]:.2f}MHz"
        )
        plt.plot(
            x, asym_lorfunc(x, *pOpt2), label=f"fitg, $\kappa$ = {pOpt2[3]:.2f}MHz"
        )
    plt.axvline(res1, color="r", ls="--", label=f"$f_res$ = {res1:.2f}")
    plt.axvline(res2, color="g", ls="--", label=f"$f_res$ = {res2:.2f}")
    plt.legend()
    plt.figure(figsize=figsize)
    plt.plot(x, y1 - y2)
    plt.axvline(
        x[np.argmax(y1 - y2)],
        color="r",
        ls="--",
        label=f"max SNR1 = {x[np.argmax(y1 - y2)]:.2f}",
    )
    plt.axvline(
        x[np.argmin(y1 - y2)],
        color="g",
        ls="--",
        label=f"max SNR2 = {x[np.argmin(y1 - y2)]:.2f}",
    )
    plt.legend()


def pipulse_analyze(pOpt):
    if pOpt[2] > 180:
        pOpt[2] = pOpt[2] - 360
    elif pOpt[2] < -180:
        pOpt[2] = pOpt[2] + 360
    if pOpt[2] < 0:
        pi = (1 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2 = (0 - pOpt[2] / 180) / 2 / pOpt[1]
    else:
        pi = (3 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2 = (1 - pOpt[2] / 180) / 2 / pOpt[1]
    return pi, pi2


def amprabi_analyze(
    x: int,
    y: float,
    fit: bool = True,
    normalize: bool = False,
    protocal: str = None,
    debug=False,
):
    """Analyze and fit the amplitude Rabi

    Parameters
    ----------
    x : int
        gain/power
    y : float
        rabi data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    list
        return the pi pulse gain, pi/2 pulse gain and max value minus min value
    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)

    plt.title("Amplitude Rabi")
    plt.ylabel("Signal (a.u)")
    plt.xlabel("Dac Gain (a.u)")

    if fit:
        # Fit the data
        pOpt, pCov = fitdecaysin(x, data)
        pi_gain, pi2_gain = pipulse_analyze(pOpt)

        if protocal == "pi":
            plt.plot(x, data, label="pi pulse", **marker_style)
            plt.axvline(pi_gain, color="r", ls="--", label=f"pi gain = {pi_gain:.3f}")
            plt.legend(loc=1)
        elif protocal == "pipi":
            plt.plot(x, data, label="pipi pulse", **marker_style)
            plt.axvline(pi_gain, color="r", ls="--", label=f"pi/2 gain = {pi_gain:.3f}")
            plt.legend(loc=1)

        elif protocal == None:
            plt.plot(x, data, label="meas", **marker_style)
            plt.axvline(pi_gain, color="r", ls="--", label=f"pi gain = {pi_gain:.3f}")
            plt.axvline(
                pi2_gain, color="r", ls="--", label=f"pi2 gain = {pi2_gain:.3f}"
            )
            plt.legend(loc=1)
        plt.plot(x, decaysin(x, *pOpt), label="Fit")
    plt.tight_layout()
    if debug:
        return pOpt, pCov
    else:
        return round(pi_gain, 6), round(pi2_gain, 6)


def lengthrabi_analyze(x: float, y: float, fit: bool = True, normalize: bool = False):
    """Analyze and fit the length Rabi data

    Parameters
    ----------
    x : float
        Rabi pulse length
    y : float
        Rabi data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    List
        return the pi pulse legnth, pi/2 pulse length and max value minus min value
    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)  # Compute amplitude

    plt.plot(x, data, label="meas", **marker_style)
    plt.ylabel("Signal (a.u)")
    plt.xlabel("Pulse length (us)")
    plt.legend()

    if fit:
        # Fit the data
        pOpt, _ = fitdecaysin(x, data)
        pi_len, pi2_len = pipulse_analyze(pOpt)

        plt.plot(x, decaysin(x, *pOpt), label="Fit")
        plt.axvline(pi_len, color="r", ls="--", label=f"pi length = {pi_len:.3f} us")
        plt.axvline(pi2_len, color="r", ls="--", label=f"pi/2 length = {pi2_len:.3f}us")
        plt.legend()
    plt.title(f"Time Rabi, freq = {pOpt[1]:.2f} MHz")
    plt.tight_layout()
    return round(pi_len, 3), round(pi2_len, 3)


def rabichevron(x, y, data):
    """Analyze and fit the amplitude Rabi chevron

    Parameters
    ----------
    x : int
        gain/power
    y : float
        rabi chevron iteration.
    data : float
        rabi chevron 2D data

    Returns
    -------
    list
        return the rabi chevron fit pi pulse gain, pi/2 pulse gain and max value minus min value
    """
    data = np.abs(data)

    pi = 0
    pi2 = 0
    for i in range(len(data)):
        pi += np.abs(data[i])
        pi2 += np.abs(data[i]) * ((-1) ** i)
    pi_gain = (x[np.argmin((np.abs(pi)))] + x[np.argmax((np.abs(pi)))]) // 2
    peaks, _ = sp.signal.find_peaks(
        np.abs(pi2), height=(max(np.abs(pi2) + min(np.abs(pi2))) // 2)
    )
    pi2_gain = x[peaks[0]]

    plt.pcolormesh(x, y, np.abs(data))
    plt.title(f"Amplitude Rabi chevron", fontsize=15)
    plt.axvline(pi_gain, color="r", lw=2, ls="--", label=r"$\pi \quad gain$")
    plt.axvline(pi2_gain, color="r", lw=2, ls="--", label=r"$\pi /2 \quad gain$")
    plt.legend()
    plt.xlabel("$gain$", fontsize=15)
    plt.tight_layout()
    return round(pi_gain, 4), round(pi2_gain, 4)


def T1_analyze(
    x: float, y: float, fit: bool = True, normalize: bool = False, debug=False
):
    """T1 relaxation analyze

    Parameters
    ----------
    x : float
        T1 program relax time
    y : float
        T1 data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False
    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)

    plt.suptitle("T1")
    plt.plot(x, data, label="meas", **marker_style)
    plt.ylabel("Signal (a.u)")
    plt.xlabel("Time (us)")
    plt.legend()

    if fit:
        # Fit the data
        pOpt, pCov = fitexp(x, data)
        p_sigma = np.sqrt(np.diag(pCov))
        plt.plot(x, expfunc(x, *pOpt), label="Fit")
        plt.title(f"T1 = {pOpt[3]:.2f} +-{p_sigma[3]:.2f}$\mu s$", fontsize=15)
        plt.legend()
    plt.tight_layout()
    if debug:
        return pOpt, pCov
    else:
        return pOpt


def T2fring_analyze(
    x: float, y: float, fit: bool = True, normalize: bool = False, prefix=None
):
    """T2 ramsey analyze

    Parameters
    ----------
    x : float
        T2 ramsey program time
    y : float
        T2 data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    Returns
    -------
    float
        Detuning frequency
    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)

    pOpt, pCov = fitdecaysin(x, data)
    sim = decaysin(x, *pOpt)
    error = np.sqrt(np.diag(pCov))

    plt.figure(figsize=figsize)
    plt.plot(x, data, label="meas", **marker_style)
    if fit:
        plt.plot(x, sim, label="fit")
        plt.plot(x, pOpt[0] * np.exp(-x / pOpt[3]) + pOpt[4])
    plt.title(
        f"T2 {prefix} = {pOpt[3]:.2f}$\mu s, detune = {pOpt[1]:.5f}MHz \pm {(error[1]) * 1e3:.3f}kHz$",
        fontsize=15,
    )
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    if normalize:
        plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()

    return pOpt


def T2decay_analyze(
    x: float, y: float, fit: bool = True, normalize: bool = False, prefix=None
):
    """_summary_

    Parameters
    ----------
    x : float
        T2 echo program time
    y : float
        T2 echo data
    fit : bool, optional
        If fit is true, plot the fitting data, by default True
    normalize : bool, optional
        If normalize is true, normalize the data, by default False

    """
    y = post_rotate(y)
    data: np.ndarray = np.abs(y)

    pOpt, pCov = fitexp(x, y)
    p_sigma = np.sqrt(np.diag(pCov))
    sim = expfunc(x, *pOpt)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", **marker_style)
    if fit == True:
        plt.plot(x, sim, label="fit")
    plt.title(f"$T2 {prefix} = {pOpt[3]:.2f} \pm {p_sigma[3]:.2f}\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    if normalize == True:
        plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()


def hist(
    data,
    plot=True,
    span=None,
    verbose=True,
    title=None,
    fid_avg=False,
    plot_f=False,
    b_print=False,
    b_plot=False,
):
    """
    span: histogram limit is the mean +/- span
    fid_avg: if True, calculate fidelity F by the average mis-categorized e/g, ; otherwise count
        total number of miscategorized over total counts (gives F^2)
    """
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    plot_f = False
    if "If" in data.keys():
        plot_f = True
        If = data["If"]
        Qf = data["Qf"]

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print("Unrotated:")
        print(
            f"Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg + 1j * yg)} +/- {np.std(np.abs(Ig + 1j * Qg))}"
        )
        print(
            f"Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe + 1j * ye)} +/- {np.std(np.abs(Ig + 1j * Qe))}"
        )
        if plot_f:
            print(
                f"If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf + 1j * yf)}"
            )

    if plot:
        if b_plot:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
        else:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        if title is not None:
            plt.suptitle(title)
        fig.tight_layout()

        axs[0, 0].scatter(
            Ig, Qg, label="g", color="b", marker=".", edgecolor="None", alpha=0.2
        )
        axs[0, 0].scatter(
            Ie, Qe, label="e", color="r", marker=".", edgecolor="None", alpha=0.2
        )
        if plot_f:
            axs[0, 0].scatter(
                If, Qf, label="f", color="g", marker=".", edgecolor="None", alpha=0.2
            )
        axs[0, 0].plot(
            [xg],
            [yg],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="b",
            markersize=5,
        )
        axs[0, 0].plot(
            [xe],
            [ye],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="r",
            markersize=5,
        )
        if plot_f:
            axs[0, 0].plot(
                [xf],
                [yf],
                color="k",
                linestyle=":",
                marker="o",
                markerfacecolor="g",
                markersize=5,
            )

        axs[0, 0].set_xlabel("I [ADC levels]")
        axs[0, 0].set_ylabel("Q [ADC levels]")
        axs[0, 0].legend(loc="upper right")
        axs[0, 0].set_title("Unrotated", fontsize=14)
        axs[0, 0].axis("equal")

    """Compute the rotation angle"""
    theta = -np.arctan2((ye - yg), (xe - xg))
    if plot_f:
        theta = -np.arctan2((yf - yg), (xf - xg))

    """
    Adjust rotation angle
    """
    best_theta = theta
    I_tot = np.concatenate((Ie, Ig))
    span = (np.max(I_tot) - np.min(I_tot)) / 2
    midpoint = (np.max(I_tot) + np.min(I_tot)) / 2
    xlims = [midpoint - span, midpoint + span]
    ng, binsg = np.histogram(Ig, bins=numbins, range=xlims)
    ne, binse = np.histogram(Ie, bins=numbins, range=xlims)
    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    best_fid = np.max(contrast)
    for theta_i in np.linspace(theta - np.pi / 12, theta + np.pi / 12, 10):
        Ig_new = Ig * np.cos(theta_i) - Qg * np.sin(theta_i)
        Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)
        Ie_new = Ie * np.cos(theta_i) - Qe * np.sin(theta_i)
        Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)
        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        I_tot_new = np.concatenate((Ie_new, Ig_new))
        span = (np.max(I_tot_new) - np.min(I_tot_new)) / 2
        midpoint = (np.max(I_tot_new) + np.min(I_tot_new)) / 2
        xlims = [midpoint - span, midpoint + span]
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        contrast = np.abs(
            ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
        )
        fid = np.max(contrast)
        if fid > best_fid:
            best_theta = theta_i
            best_fid = fid
    theta = best_theta

    """Rotate the IQ data"""
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    if plot_f:
        If_new = If * np.cos(theta) - Qf * np.sin(theta)
        Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print("Rotated:")
        print(
            f"Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg + 1j * yg)} +/- {np.std(np.abs(Ig + 1j * Qg))}"
        )
        print(
            f"Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe + 1j * ye)} +/- {np.std(np.abs(Ig + 1j * Qe))}"
        )
        if plot_f:
            print(
                f"If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf + 1j * yf)}"
            )

    if span is None:
        span = (
            np.max(np.concatenate((Ie_new, Ig_new)))
            - np.min(np.concatenate((Ie_new, Ig_new)))
        ) / 2
    xlims = [(xg + xe) / 2 - span, (xg + xe) / 2 + span]

    if plot:
        axs[0, 1].scatter(
            Ig_new,
            Qg_new,
            label="g",
            color="b",
            marker=".",
            edgecolor="None",
            alpha=0.3,
        )
        axs[0, 1].scatter(
            Ie_new,
            Qe_new,
            label="e",
            color="r",
            marker=".",
            edgecolor="None",
            alpha=0.3,
        )
        if plot_f:
            axs[0, 1].scatter(
                If_new,
                Qf_new,
                label="f",
                color="g",
                marker=".",
                edgecolor="None",
                alpha=0.3,
            )
        axs[0, 1].plot(
            [xg],
            [yg],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="b",
            markersize=5,
        )
        axs[0, 1].plot(
            [xe],
            [ye],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="r",
            markersize=5,
        )
        if plot_f:
            axs[0, 1].plot(
                [xf],
                [yf],
                color="k",
                linestyle=":",
                marker="o",
                markerfacecolor="g",
                markersize=5,
            )

        axs[0, 1].set_xlabel("I [ADC levels]")
        axs[0, 1].legend(loc="upper right")
        axs[0, 1].set_title("Rotated", fontsize=14)
        axs[0, 1].axis("equal")

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1, 0].hist(
            Ig_new, bins=numbins, range=xlims, color="b", label="g", alpha=0.5
        )
        ne, binse, pe = axs[1, 0].hist(
            Ie_new, bins=numbins, range=xlims, color="r", label="e", alpha=0.5
        )
        if plot_f:
            nf, binsf, pf = axs[1, 0].hist(
                If_new, bins=numbins, range=xlims, color="g", label="f", alpha=0.5
            )
        axs[1, 0].set_ylabel("Counts", fontsize=14)
        axs[1, 0].set_xlabel("I [ADC levels]", fontsize=14)
        axs[1, 0].legend(loc="upper right")

    else:
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """fitting the shot gaussian"""
    # poptg, _ = fitdualgauss(binsg[:-1], ng)
    # popte, _ = fitdualgauss(binse[:-1], ne)
    # fitg = dualgauss(binsg[:-1], *poptg)
    # fite = dualgauss(binse[:-1], *popte)
    # axs[1, 0].plot(binsg[:-1], fitg)
    # axs[1, 0].plot(binsg[:-1], fite)
    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    # this method calculates fidelity as 1-2(Neg + Nge)/N
    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    tind = contrast.argmax()
    thresholds.append(binsg[tind])

    if not fid_avg:
        fids.append(contrast[tind])
    else:
        # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
        fids.append(
            0.5 * (1 - ng[tind:].sum() / ng.sum() + 1 - ne[:tind].sum() / ne.sum())
        )
    if verbose:
        print(f"g correctly categorized: {100 * (1 - ng[tind:].sum() / ng.sum())}%")
        print(f"e correctly categorized: {100 * (1 - ne[:tind].sum() / ne.sum())}%")

    if plot_f:
        contrast = np.abs(
            ((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg:
            fids.append(contrast[tind])
        else:
            fids.append(
                0.5 * (1 - ng[tind:].sum() / ng.sum() + 1 - nf[:tind].sum() / nf.sum())
            )

        contrast = np.abs(
            ((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg:
            fids.append(contrast[tind])
        else:
            fids.append(
                0.5 * (1 - ne[tind:].sum() / ne.sum() + 1 - nf[:tind].sum() / nf.sum())
            )

    if plot:
        title = "$\overline{F}_{ge}$" if fid_avg else "$F_{ge}$"
        axs[1, 0].set_title(f"Histogram ({title}: {100 * fids[0]:.3}%)", fontsize=14)
        axs[1, 0].axvline(thresholds[0], color="0.2", linestyle="--")
        if plot_f:
            axs[1, 0].axvline(thresholds[1], color="0.2", linestyle="--")
            axs[1, 0].axvline(thresholds[2], color="0.2", linestyle="--")

        axs[1, 1].set_title("Cumulative Counts", fontsize=14)
        axs[1, 1].plot(binsg[:-1], np.cumsum(ng), "b", label="g")
        axs[1, 1].plot(binse[:-1], np.cumsum(ne), "r", label="e")
        axs[1, 1].axvline(thresholds[0], color="0.2", linestyle="--")
        if plot_f:
            # axs[1, 1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1, 1].axvline(thresholds[1], color="0.2", linestyle="--")
            axs[1, 1].axvline(thresholds[2], color="0.2", linestyle="--")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("I [ADC levels]", fontsize=14)

    gg = 100 * (1 - ng[tind:].sum() / ng.sum())
    ge = 100 * (ng[tind:].sum() / ng.sum())
    eg = 100 * (1 - ne[tind:].sum() / ne.sum())
    ee = 100 * (ne[tind:].sum() / ne.sum())

    if b_print:
        print(
            f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f}% | {ge:.3f}% |
        ----------------
        | {eg:.3f}% | {ee:.3f}% |
        -----------------
        IQ plane rotated by: {180 / np.pi * theta:.1f}{chr(176)}
        Threshold: {thresholds[0]:.3e}
        Fidelity: {100 * fids[0]:.3f}%
        """
        )

    if b_plot:
        axs[0, 2].imshow(np.array([[gg, ge], [eg, ee]]))
        axs[0, 2].set_xticks([0, 1])
        axs[0, 2].set_yticks([0, 1])
        axs[0, 2].set_xticklabels(labels=["|g>", "|e>"])
        axs[0, 2].set_yticklabels(labels=["|g>", "|e>"])
        axs[0, 2].set_ylabel("Prepared", fontsize=14)
        axs[0, 2].set_xlabel("Measured", fontsize=14)
        axs[0, 2].text(0, 0, f"{gg:.1f}%", ha="center", va="center", color="k")
        axs[0, 2].text(1, 0, f"{ge:.1f}%", ha="center", va="center", color="w")
        axs[0, 2].text(0, 1, f"{eg:.1f}%", ha="center", va="center", color="w")
        axs[0, 2].text(1, 1, f"{ee:.1f}%", ha="center", va="center", color="k")
        axs[0, 2].set_title("Fidelities", fontsize=14)

        # lower right text setting
        text_kwargs = dict(ha="center", va="center", fontsize=12)
        axs[1, 2].text(
            0.45,
            0.5,
            f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f}% | {ge:.3f}% |
        ----------------
        | {eg:.3f}% | {ee:.3f}% |
        -----------------
        IQ plane rotated by: {180 / np.pi * theta:.1f}{chr(176)}
        Threshold: {thresholds[0]:.3e}
        Fidelity: {100 * fids[0]:.3f}%
        """,
            **text_kwargs,
        )
        axs[1, 2].spines["right"].set_color("none")
        axs[1, 2].spines["left"].set_color("none")
        axs[1, 2].spines["bottom"].set_color("none")
        axs[1, 2].spines["top"].set_color("none")
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.tight_layout()
    plt.show()
    return fids, thresholds, theta * 180 / np.pi  # fids: ge, gf, ef


if __name__ == "__main__":
    import resonator_tools
    # import sys
    # import os
    # os.chdir(r'D:\Jay PhD\Code\data_fitting\data')
    # sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
    # import Labber

    # np.bool = bool
    # np.float = np.float64
    # res = r'Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5'
    # pdr = r'r1_pdr.hdf5'
    # lenghrabi = r'q1_rabi.hdf5'
    # t1 = r"C:\Users\QEL\Desktop\New folder\q5 NDavg T1.hdf5"
    # t2 = r'C:\Users\QEL\Desktop\New folder\q5 NDavg T2 echo .hdf5'
    # spec = r'q1_twotone_4.hdf5'
    # # shot = r"C:\Users\QEL\Desktop\New folder\q5 single shot 5000 gain 0.9us -0.1mA.hdf5"
    # spec = Labber.LogFile(spec)
    # pdr = Labber.LogFile(pdr)
    # fr = Labber.LogFile(res)
    # f1 = Labber.LogFile(lenghrabi)
    # f2 = Labber.LogFile(t1)
    # f3 = Labber.LogFile(t2)
    # # shot = Labber.LogFile(shot)
    # (x, y) = fr.getTraceXY(entry=3)
    # (sx, sy) = spec.getTraceXY()
    # (rx, ry) = f1.getTraceXY()
    # (t1x, t1y) = f2.getTraceXY()
    # (t2x, t2y) = f3.getTraceXY()
    # T2decay_analyze(t2x*1e6, t2y, fit=True)
    # spectrum_analyze(sx, sy, plot=True)
    # lengthraig_analyze(rx, ry, plot=True, normalize=False)
    # amprabi_analyze(rx, ry, fit=True, normalize=True)
    # T1_analyze(t1x*1e6, t1y, fit=True, normalize=True)
    # T2r_analyze(t2x, t2y, fit=True, normalize=False)
    # resonator_analyze(x,y)
