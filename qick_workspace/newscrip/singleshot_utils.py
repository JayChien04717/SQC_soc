"""
plotter.singleshot_utils
=========================
Histogram and analysis utilities for SingleShot experiments.
Ported from scrip/s000_SingleShot_ge_prog_opt.py — full 4-panel plot
with IQ scatter, rotated scatter, histogram, and confusion matrix.
"""

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy.integrate import quad

from ..tools.fitting import fit_doublegauss, double_gaussian, fit_gauss, gaussian

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle = ["o", "*", "s", "^"]


def plot_hist(
    data,
    bins,
    ax=None,
    xlims=None,
    color=None,
    linestyle=None,
    label=None,
    alpha=None,
    normalize=True,
):
    if color is None:
        color = next(cycle(default_colors))
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        hist_sum = hist_data.sum()
        if hist_sum > 0:
            hist_data = hist_data / hist_sum

    for i in range(len(hist_data)):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [hist_data[i], hist_data[i]],
            color=color,
            linestyle=linestyle,
            label=label if i == 0 else None,
            alpha=alpha,
            linewidth=0.9,
        )
        if i < len(hist_data) - 1:
            ax.plot(
                [bin_edges[i + 1], bin_edges[i + 1]],
                [hist_data[i], hist_data[i + 1]],
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=0.9,
            )
    ax.relim()
    ax.set_ylim((0, None))
    return hist_data, bin_edges


def general_hist(
    iqshots,
    state_labels,
    g_states,
    e_states,
    e_label="e",
    check_qubit_label=None,
    numbins=200,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    gauss_overlap=False,
    plotoverlap=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    if numbins is None:
        numbins = 200

    has_f_state = len(iqshots) > 2

    # --- 1. Data Aggregation ---
    data_map = {"g": np.array([]), "e": np.array([]), "f": np.array([])}
    I_tot_all = np.array([])
    Q_tot_all = np.array([])

    for check_i, data_check in enumerate(iqshots):
        I, Q = data_check
        I_tot_all = np.concatenate((I_tot_all, I))
        Q_tot_all = np.concatenate((Q_tot_all, Q))
        if check_i in g_states:
            cat = "g"
        elif check_i in e_states:
            cat = "e"
        else:
            cat = "f"
        if data_map[cat].size == 0:
            data_map[cat] = I + 1j * Q
        else:
            data_map[cat] = np.concatenate((data_map[cat], I + 1j * Q))

    # --- 2. Rotation Calculation ---
    if not amplitude_mode:
        if theta is None:
            xg = np.mean(np.real(data_map["g"])) if data_map["g"].size > 0 else 0
            yg = np.mean(np.imag(data_map["g"])) if data_map["g"].size > 0 else 0
            xe = np.mean(np.real(data_map["e"])) if data_map["e"].size > 0 else 1
            ye = np.mean(np.imag(data_map["e"])) if data_map["e"].size > 0 else 1
            theta = -np.arctan2((ye - yg), (xe - xg))
        else:
            theta *= np.pi / 180

        def rotate_iq(c_data, ang):
            i_new = np.real(c_data) * np.cos(ang) - np.imag(c_data) * np.sin(ang)
            q_new = np.real(c_data) * np.sin(ang) + np.imag(c_data) * np.cos(ang)
            return i_new, q_new

        I_all_new, _ = rotate_iq(I_tot_all + 1j * Q_tot_all, theta)
        span = (np.max(I_all_new) - np.min(I_all_new)) / 2
        midpoint = (np.max(I_all_new) + np.min(I_all_new)) / 2
    else:
        theta = 0
        amp_all = np.abs(I_tot_all + 1j * Q_tot_all)
        span = (np.max(amp_all) - np.min(amp_all)) / 2
        midpoint = (np.max(amp_all) + np.min(amp_all)) / 2

        def rotate_iq(c_data, ang):
            return np.real(c_data), np.imag(c_data)

    xlims = [midpoint - span, midpoint + span]

    # --- 3. Plot Setup ---
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
        if title is None:
            title = "Readout Fidelity" + (
                f" on Q{check_qubit_label}" if check_qubit_label is not None else ""
            )
        fig.suptitle(title)
        fig.tight_layout()

        # Row 0: IQ Scatter
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 0].axis("equal")

        axs[0, 1].set_title(
            f"Rotated ($\\theta={theta * 180 / np.pi:.1f}^\\circ$)", fontsize=13
        )
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 1].axis("equal")

        # Row 1: Histogram & Confusion Matrix
        threshold_axis = "I" if not amplitude_mode else "Amplitude"
        axs[1, 0].set_ylabel("Counts", fontsize=12)
        axs[1, 0].set_xlabel(f"{threshold_axis} [ADC levels]", fontsize=11)

        plt.subplots_adjust(hspace=0.35, wspace=0.15)

    # Variables
    n_dist = {"g": None, "e": None, "f": None}
    bins_dist = None
    gauss_fit_fidelity = 0
    popts = []
    pcovs = []
    b_g_plot, c_g_plot = None, None
    b_e_plot, c_e_plot = None, None

    # --- 4. Process Each Input State ---
    for check_i, data_check in enumerate(iqshots):
        state_label = state_labels[check_i]
        I, Q = data_check
        complex_data = I + 1j * Q

        this_color = default_colors[check_i % len(default_colors)]
        this_marker = marker_cycle[check_i % len(marker_cycle)]
        this_linestyle = linestyle_cycle[0]

        # Rotation
        if not amplitude_mode:
            I_new, Q_new = rotate_iq(complex_data, theta)
            data_to_hist = I_new
        else:
            I_new, Q_new = I, Q
            data_to_hist = np.abs(complex_data)

        # Scatter Plot
        if plot:
            axs[0, 0].scatter(
                I,
                Q,
                label=state_label,
                color=this_color,
                marker=".",
                edgecolor="None",
                alpha=0.1,
            )
            axs[0, 0].plot(
                [np.mean(I)],
                [np.mean(Q)],
                color="k",
                marker=this_marker,
                markerfacecolor=this_color,
                markersize=6,
            )
            axs[0, 1].scatter(
                I_new,
                Q_new,
                label=state_label,
                color=this_color,
                marker=".",
                edgecolor="None",
                alpha=0.1,
            )
            axs[0, 1].plot(
                [np.mean(I_new)],
                [np.mean(Q_new)],
                color="k",
                marker=this_marker,
                markerfacecolor=this_color,
                markersize=6,
            )

        # Histogram Accumulation
        if plot:
            n, bins = plot_hist(
                data_to_hist,
                bins=numbins,
                ax=axs[1, 0],
                xlims=xlims,
                color=this_color,
                linestyle=this_linestyle,
                label=state_label,
                alpha=0.6,
                normalize=False,
            )
        else:
            n, bins = np.histogram(data_to_hist, bins=numbins, range=xlims)

        bins_dist = bins
        if check_i in g_states:
            cat = "g"
        elif check_i in e_states:
            cat = "e"
        else:
            cat = "f"
        if n_dist[cat] is None:
            n_dist[cat] = n
        else:
            n_dist[cat] += n

    # --- 5. Fitting ---
    def gaussian_norm(x, b, c):
        a = 1 / (np.sqrt(2 * np.pi) * c)
        return a * np.exp(-((x - b) ** 2) / (2 * c**2))

    def overlap_area_norm(b1, c1, b2, c2):
        def min_func(x):
            return np.minimum(gaussian_norm(x, b1, c1), gaussian_norm(x, b2, c2))

        x_min = min(b1 - 5 * c1, b2 - 5 * c2)
        x_max = max(b1 + 5 * c1, b2 + 5 * c2)
        area, _ = quad(min_func, x_min, x_max)
        return area

    def readout_fidelity_norm(b1, c1, b2, c2):
        return 1 - overlap_area_norm(b1, c1, b2, c2)

    do_fit = fit or gauss_overlap or plotoverlap

    if do_fit and n_dist["g"] is not None and n_dist["e"] is not None:
        bin_centers = (bins_dist[:-1] + bins_dist[1:]) / 2
        n_g = n_dist["g"]
        n_e = n_dist["e"]

        xmax_g_idx = np.argmax(n_g)
        xmax_e_idx = np.argmax(n_e)
        xmax_g_val = bin_centers[xmax_g_idx]
        xmax_e_val = bin_centers[xmax_e_idx]

        sigma_guess = abs(xmax_e_val - xmax_g_val) / 5.0
        if sigma_guess < 1e-3:
            sigma_guess = (bins_dist[-1] - bins_dist[0]) / 20.0

        if gauss_overlap:
            guess_g = [
                np.max(n_g),
                xmax_g_val,
                sigma_guess,
                np.max(n_g) * 0.1,
                xmax_e_val,
                sigma_guess,
            ]
            guess_e = [
                np.max(n_e),
                xmax_e_val,
                sigma_guess,
                np.max(n_e) * 0.2,
                xmax_g_val,
                sigma_guess,
            ]
            try:
                popt_g, pcov_g = fit_doublegauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_doublegauss(bin_centers, n_e, guess_e)
                if popt_g[0] > popt_g[3]:
                    b_g, c_g = popt_g[1], abs(popt_g[2])
                else:
                    b_g, c_g = popt_g[4], abs(popt_g[5])
                if popt_e[0] > popt_e[3]:
                    b_e, c_e = popt_e[1], abs(popt_e[2])
                else:
                    b_e, c_e = popt_e[4], abs(popt_e[5])
                gauss_fit_fidelity = readout_fidelity_norm(b_g, c_g, b_e, c_e)
                popts = [popt_g, popt_e]
                pcovs = [pcov_g, pcov_e]
                b_g_plot, c_g_plot = b_g, c_g
                b_e_plot, c_e_plot = b_e, c_e
            except Exception as e:
                print(f"Double Gaussian fit failed: {e}")
                gauss_fit_fidelity = 0
                popt_g, popt_e = None, None
            fit_func = double_gaussian
        else:
            guess_g = [np.max(n_g), xmax_g_val, sigma_guess, 0]
            guess_e = [np.max(n_e), xmax_e_val, sigma_guess, 0]
            try:
                popt_g, pcov_g = fit_gauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_gauss(bin_centers, n_e, guess_e)
                popts = [popt_g, popt_e]
                pcovs = [pcov_g, pcov_e]
                b_g_plot, c_g_plot = popt_g[1], abs(popt_g[2])
                b_e_plot, c_e_plot = popt_e[1], abs(popt_e[2])
            except Exception as e:
                print(f"Gaussian fit failed: {e}")
                popt_g, popt_e = None, None
            fit_func = gaussian

        if plot and popt_g is not None and popt_e is not None:
            x_dense = np.linspace(bins_dist[0], bins_dist[-1], 500)
            y_fit_g = fit_func(x_dense, *popt_g)
            y_fit_e = fit_func(x_dense, *popt_e)
            axs[1, 0].plot(
                x_dense,
                y_fit_g,
                color=default_colors[0],
                linestyle="-",
                linewidth=2,
                label="Fit G",
            )
            axs[1, 0].plot(
                x_dense,
                y_fit_e,
                color=default_colors[1],
                linestyle="-",
                linewidth=2,
                label="Fit E",
            )

            if plotoverlap and b_g_plot is not None and b_e_plot is not None:
                bin_width = bins_dist[1] - bins_dist[0]
                norm_g = gaussian_norm(x_dense, b_g_plot, c_g_plot)
                scaled_g = norm_g * np.sum(n_g) * bin_width
                norm_e = gaussian_norm(x_dense, b_e_plot, c_e_plot)
                scaled_e = norm_e * np.sum(n_e) * bin_width
                y_overlap = np.minimum(scaled_g, scaled_e)
                axs[1, 0].fill_between(
                    x_dense, 0, y_overlap, color="purple", alpha=0.3, label="Overlap"
                )

    # --- 6. Thresholds & Confusion Matrix ---
    fids = []
    thresholds = []

    contrast_ge = np.abs(
        (np.cumsum(n_dist["g"]) - np.cumsum(n_dist["e"]))
        / (np.sum(n_dist["g"]) + np.sum(n_dist["e"]))
    )
    tind_ge = contrast_ge.argmax()
    threshold_ge = bins_dist[tind_ge]
    thresholds.append(threshold_ge)

    if not fid_avg:
        fids.append(contrast_ge[tind_ge])
    else:
        fids.append(
            0.5
            * (
                1
                - n_dist["g"][tind_ge:].sum() / n_dist["g"].sum()
                + 1
                - n_dist["e"][:tind_ge].sum() / n_dist["e"].sum()
            )
        )

    # --- Confusion Matrix Calculation ---
    if not has_f_state:
        matrix_size = 2
        labels = ["|g⟩", f"|{e_label}⟩"]
        n00 = n_dist["g"][:tind_ge].sum()
        n01 = n_dist["g"][tind_ge:].sum()
        n10 = n_dist["e"][:tind_ge].sum()
        n11 = n_dist["e"][tind_ge:].sum()
        raw_matrix = np.array([[n00, n01], [n10, n11]])
    else:
        matrix_size = 3
        labels = ["|g⟩", f"|{e_label}⟩", "|f⟩"]
        if n_dist["f"] is not None:
            contrast_ef = np.abs(
                (np.cumsum(n_dist["e"]) - np.cumsum(n_dist["f"]))
                / (np.sum(n_dist["e"]) + np.sum(n_dist["f"]))
            )
            tind_ef = contrast_ef.argmax()
            threshold_ef = bins_dist[tind_ef]
            thresholds.append(threshold_ef)
            sorted_t_indices = sorted([tind_ge, tind_ef])
            t1, t2 = sorted_t_indices[0], sorted_t_indices[1]

            def classify_counts(n_arr):
                c0 = n_arr[:t1].sum()
                c1 = n_arr[t1:t2].sum()
                c2 = n_arr[t2:].sum()
                return [c0, c1, c2]

            row_g = classify_counts(n_dist["g"])
            row_e = classify_counts(n_dist["e"])
            row_f = classify_counts(n_dist["f"])
            raw_matrix = np.array([row_g, row_e, row_f])
        else:
            raw_matrix = np.zeros((3, 3))

    row_sums = raw_matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    conf_matrix = 100 * raw_matrix / row_sums

    # --- 7. Finalize Plots ---
    if plot:
        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--", label="Threshold")

        fid_title = "$\\overline{F}_{ge}$" if fid_avg else "$F_{ge}$"
        if gauss_overlap:
            axs[1, 0].set_title(
                f"{fid_title} (Gauss): {100 * gauss_fit_fidelity:.2f}%", fontsize=13
            )
        else:
            axs[1, 0].set_title(
                f"{fid_title} (Thresh): {100 * fids[0]:.2f}%", fontsize=13
            )

        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="gray", linestyle="-.")

        axs[1, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)

        # --- Confusion Matrix Plot ---
        ax_cm = axs[1, 1]
        ax_cm.clear()
        im = ax_cm.imshow(conf_matrix, cmap="Reds", vmin=0, vmax=100)
        ax_cm.set_xticks(np.arange(matrix_size))
        ax_cm.set_yticks(np.arange(matrix_size))
        ax_cm.set_xticklabels(labels)
        ax_cm.set_yticklabels(labels)
        ax_cm.set_xlabel("Declared output", fontsize=11)
        ax_cm.set_ylabel("Input state", fontsize=11)
        ax_cm.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

        for i in range(matrix_size):
            for j in range(matrix_size):
                val = conf_matrix[i, j]
                text_color = "white" if val > 50 else "black"
                ax_cm.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )

        ax_cm.set_title("Confusion Matrix (%)")

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    # --- 8. Returns ---
    if gauss_overlap:
        return_data = [gauss_fit_fidelity, thresholds, theta * 180 / np.pi]
    else:
        return_data = [fids, thresholds, theta * 180 / np.pi]

    if fit or gauss_overlap or plotoverlap:
        return_data += [popts, pcovs]

    if check_qnd:
        n_diff = np.abs((n_g_0 - n_g_1)) if "n_g_0" in locals() else 0
        n_diff_qnd = (
            np.sum(n_diff) / 2 / np.sum(n_dist["g"]) if np.sum(n_dist["g"]) > 0 else 0
        )
        return_data += [n_diff_qnd]

    if verbose:
        print(f"Theta: {theta * 180 / np.pi:.2f} deg")
        print(f"Threshold Fidelity: {100 * fids[0]:.3f}%")
        print("Confusion Matrix (%):\n", conf_matrix)
        if gauss_overlap:
            print(f"Gaussian Fit Fidelity: {100 * gauss_fit_fidelity:.3f}%")

    return return_data


def hist(
    data,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    gauss_overlap=False,
    plotoverlap=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
):
    Ig, Qg = data["Ig"], data["Qg"]
    Ie, Qe = data["Ie"], data["Qe"]
    iqshots = [(Ig, Qg), (Ie, Qe)]
    state_labels = ["g", "e"]
    g_states = [0]
    e_states = [1]

    if "If" in data:
        iqshots.append((data["If"], data["Qf"]))
        state_labels.append("f")
        e_states = [2]

    return general_hist(
        iqshots=iqshots,
        state_labels=state_labels,
        g_states=g_states,
        e_states=e_states,
        amplitude_mode=amplitude_mode,
        ps_threshold=ps_threshold,
        theta=theta,
        plot=plot,
        verbose=verbose,
        fid_avg=fid_avg,
        fit=fit,
        gauss_overlap=gauss_overlap,
        plotoverlap=plotoverlap,
        fitparams=fitparams,
        normalize=normalize,
        title=title,
        export=export,
    )
