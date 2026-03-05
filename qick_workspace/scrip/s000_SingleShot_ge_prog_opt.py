# ----- Qick package ----- #
from tabnanny import verbose
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# ----- User Library ----- #
from ..tools.system_cfg import *
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import get_next_filename_labber, hdf5_generator
from ..tools.yamltool import yml_comment

# from .singleshotplot import hist
from ..tools.fitting import fit_doublegauss, double_gaussian, fit_gauss, gaussian
from scipy.integrate import quad

##################
# plot hist
##################


# Use np.hist and plt.plot to accomplish plt.hist with less memory usage
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
        color_cycle = cycle(default_colors)
        color = next(color_cycle)
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        # Avoid division by zero error
        hist_sum = hist_data.sum()
        if hist_sum > 0:
            hist_data = hist_data / hist_sum

    for i in range(len(hist_data)):
        if i > 0:
            label = None
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [hist_data[i], hist_data[i]],
            color=color,
            linestyle=linestyle,
            label=label,
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


# ===================================================================== #


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
    plotoverlap=False,  # 新增選項
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    if numbins is None:
        numbins = 200

    # Detect states
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

    xlims = [midpoint - span, midpoint + span]

    # --- 3. Plot Setup ---
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
        if title is None:
            title = f"Readout Fidelity" + (
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

    # Store extracted params for overlap plotting
    b_g_plot, c_g_plot = None, None
    b_e_plot, c_e_plot = None, None

    # --- 4. Process Each Input State ---
    for check_i, data_check in enumerate(iqshots):
        state_label = state_labels[check_i]
        I, Q = data_check
        complex_data = I + 1j * Q

        # Apply styles based on cycle
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

        # Accumulate
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

    # --- 5. Fitting (Modified) ---
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
            # --- DOUBLE GAUSSIAN FIT ---
            fit_func = double_gaussian
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

                # Extract main peak
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

                # Save for plotting
                b_g_plot, c_g_plot = b_g, c_g
                b_e_plot, c_e_plot = b_e, c_e

            except Exception as e:
                print(f"Double Gaussian fit failed: {e}")
                gauss_fit_fidelity = 0
                popt_g, popt_e = None, None

        else:
            # --- SINGLE GAUSSIAN FIT ---
            fit_func = gaussian
            guess_g = [np.max(n_g), xmax_g_val, sigma_guess, 0]
            guess_e = [np.max(n_e), xmax_e_val, sigma_guess, 0]

            try:
                popt_g, pcov_g = fit_gauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_gauss(bin_centers, n_e, guess_e)
                popts = [popt_g, popt_e]
                pcovs = [pcov_g, pcov_e]

                # Extract simple gaussian params for overlap plot if requested
                b_g_plot, c_g_plot = popt_g[1], abs(popt_g[2])
                b_e_plot, c_e_plot = popt_e[1], abs(popt_e[2])

            except Exception as e:
                print(f"Gaussian fit failed: {e}")
                popt_g, popt_e = None, None

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

            # --- Plot Overlap Area ---
            if plotoverlap and b_g_plot is not None and b_e_plot is not None:
                # Reconstruct the single gaussians used for fidelity calculation
                # Scale them to counts: N_counts = PDF * Total_Counts * Bin_Width
                bin_width = bins_dist[1] - bins_dist[0]

                # Scale G
                norm_g = gaussian_norm(x_dense, b_g_plot, c_g_plot)
                scaled_g = norm_g * np.sum(n_g) * bin_width

                # Scale E
                norm_e = gaussian_norm(x_dense, b_e_plot, c_e_plot)
                scaled_e = norm_e * np.sum(n_e) * bin_width

                # Calculate overlap of these scaled distributions
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

    # --- Matrix Calculation ---
    if not has_f_state:
        matrix_size = 2
        labels = ["|g>", f"|{e_label}>"]
        n00 = n_dist["g"][:tind_ge].sum()
        n01 = n_dist["g"][tind_ge:].sum()
        n10 = n_dist["e"][:tind_ge].sum()
        n11 = n_dist["e"][tind_ge:].sum()
        raw_matrix = np.array([[n00, n01], [n10, n11]])
    else:
        matrix_size = 3
        labels = ["|g>", f"|{e_label}>", "|f>"]
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

        fid_title = "$\overline{F}_{ge}$" if fid_avg else "$F_{ge}$"
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

        # Confusion Matrix
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
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )

        if title is not None:
            ax_cm.set_title("Readout Fidelity Matrix (%)")

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
        print("Fidelity Matrix (%):\n", conf_matrix)
        if gauss_overlap:
            print(f"Gaussian Fit Fidelity: {100 * gauss_fit_fidelity:.3f}%")

    return return_data


# ===================================================================== #


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
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    iqshots = [(Ig, Qg), (Ie, Qe)]
    state_labels = ["g", "e"]
    g_states = [0]
    e_states = [1]

    if "If" in data.keys():
        If = data["If"]
        Qf = data["Qf"]
        iqshots.append((If, Qf))
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


# ===================================================================== #


def multihist(
    data,
    check_qubit,
    check_states,
    play_pulses_list,
    g_states,
    e_states,
    numbins=200,
    amplitude_mode=False,
    ps_threshold=None,
    theta=None,
    plot=True,
    verbose=True,
    fid_avg=False,
    fit=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
    check_qnd=False,
):
    """
    Assumes data is passed in via data["iqshots"] = [(idata, qdata)]*len(check_states), idata=[... *num_shots]*num_qubits_sample

    These are mostly for labeling purposes:
    check_states: an array of strs of the init_state specifying each configuration to plot a histogram for
    play_pulses_list: list of play_pulses corresponding to check_states, see code for play_pulses
    """
    state_labels = []
    assert len(play_pulses_list) == len(check_states)
    for i in range(len(check_states)):
        check_state = check_states[i]
        play_pulses = play_pulses_list[i]
        label = f"{check_state}"
        # print('check state', check_state)
        if len(play_pulses) > 1 or play_pulses[0] != 0:
            label += f" play {play_pulses}"
        state_labels.append(label)
    all_q_iqshots = data["iqshots"]
    iqshots = []
    for check_i, data_check in enumerate(all_q_iqshots):
        I, Q = data_check
        I = I[check_qubit]
        Q = Q[check_qubit]
        iqshots.append((I, Q))
    check_qubit_label = check_qubit
    return_data = general_hist(
        iqshots=iqshots,
        check_qubit_label=check_qubit_label,
        state_labels=state_labels,
        g_states=g_states,
        e_states=e_states,
        numbins=numbins,
        amplitude_mode=amplitude_mode,
        ps_threshold=ps_threshold,
        theta=theta,
        plot=plot,
        verbose=verbose,
        fid_avg=fid_avg,
        fit=fit,
        fitparams=fitparams,
        normalize=normalize,
        title=title,
        export=export,
        check_qnd=check_qnd,
    )
    if check_qnd:
        data["n_diff_qnd"] = return_data[-1]
    return return_data


##################
# Define Program #
##################

# Separate g and e per each experiment defined.


class SingleShotProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("shotloop", cfg["shots"])

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("shotloop", cfg["shots"])

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        self.add_gauss(
            ch=qb_ch,
            name="ramp",
            sigma=cfg["sigma_ge"],
            length=cfg["sigma_ge"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse",
                ro_ch=ro_ch,
                style="arb",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
            )
        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch,
                name="qb_pulse",
                ro_ch=ro_ch,
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShotProgram_f(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])

        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("shotloop", cfg["shots"])

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        self.add_gauss(
            ch=qb_ch,
            name="ramp_ge",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_ge_pulse",
            style="arb",
            envelope="ramp_ge",
            freq=cfg["qb_freq_ge"],
            phase=cfg["qb_phase"],
            gain=cfg["pi_gain_ge"],
        )

        self.add_gauss(
            ch=qb_ch,
            name="ramp_ef",
            sigma=cfg["sigma"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qb_ch,
            name="qb_ef_pulse",
            style="arb",
            envelope="ramp_ef",
            freq=cfg["qb_freq_ef"],
            phase=cfg["qb_phase"],
            gain=cfg["pi_gain_ef"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait1")
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShot_ge_opt:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, SHOTS, sweep_para: dict):
        self.cfg["shots"] = SHOTS

        raw_length = sweep_para.get("length")
        self.length_sweep = (
            raw_length
            if isinstance(raw_length, (list, tuple, np.ndarray))
            else [raw_length]
        )

        raw_gain = sweep_para.get("gain")
        self.gain_sweep = (
            raw_gain if isinstance(raw_gain, (list, tuple, np.ndarray)) else [raw_gain]
        )

        raw_freq = sweep_para.get("freq")
        self.freq_sweep = (
            raw_freq if isinstance(raw_freq, (list, tuple, np.ndarray)) else [raw_freq]
        )

        final_shape = (
            len(self.length_sweep),
            len(self.gain_sweep),
            len(self.freq_sweep),
            SHOTS,
        )

        self.I_g_array = np.full(final_shape, np.nan)
        self.Q_g_array = np.full(final_shape, np.nan)
        self.I_e_array = np.full(final_shape, np.nan)
        self.Q_e_array = np.full(final_shape, np.nan)

        # --- TQDM 動態設定 ---
        is_l_sweep = len(self.length_sweep) > 1
        is_g_sweep = len(self.gain_sweep) > 1
        is_f_sweep = len(self.freq_sweep) > 1

        outermost_real_sweep = None
        if is_l_sweep:
            outermost_real_sweep = "l"
        elif is_g_sweep:
            outermost_real_sweep = "g"
        elif is_f_sweep:
            outermost_real_sweep = "f"

        # 1. 建立 L (最外層) 迭代器
        l_iter = self.length_sweep
        if "l" == outermost_real_sweep:
            l_iter = tqdm(self.length_sweep, desc="Length loop")

        for l_idx, l_val in enumerate(l_iter):
            # --- 修正點：G 迭代器在 L 迴圈內建立 ---
            g_iter = self.gain_sweep
            if "g" == outermost_real_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop")
            elif is_g_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop", leave=False)

            for g_idx, g_val in enumerate(g_iter):
                # --- 修正點：F 迭代器在 G 迴圈內建立 ---
                f_iter = self.freq_sweep
                if "f" == outermost_real_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop")
                elif is_f_sweep:
                    f_iter = tqdm(self.freq_sweep, desc="Freq loop", leave=False)

                for f_idx, f_val in enumerate(f_iter):
                    cfg_update = {"steps": SHOTS}
                    if l_val is not None:
                        cfg_update["ro_length"] = l_val
                    if g_val is not None:
                        cfg_update["res_gain_ge"] = g_val
                    if f_val is not None:
                        cfg_update["res_freq_ge"] = f_val

                    self.cfg.update(cfg_update)

                    ssp_g = SingleShotProgram_g(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list_g = ssp_g.acquire(self.soc, rounds=1, progress=False)

                    ssp_e = SingleShotProgram_e(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list_e = ssp_e.acquire(self.soc, rounds=1, progress=False)

                    I_g = iq_list_g[0][0].T[0]
                    Q_g = iq_list_g[0][0].T[1]
                    I_e = iq_list_e[0][0].T[0]
                    Q_e = iq_list_e[0][0].T[1]

                    self.I_g_array[l_idx, g_idx, f_idx, :] = I_g
                    self.Q_g_array[l_idx, g_idx, f_idx, :] = Q_g
                    self.I_e_array[l_idx, g_idx, f_idx, :] = I_e
                    self.Q_e_array[l_idx, g_idx, f_idx, :] = Q_e

        self.data = {
            "Ig": self.I_g_array,
            "Qg": self.Q_g_array,
            "Ie": self.I_e_array,
            "Qe": self.Q_e_array,
        }

    def analyze(self):
        try:
            from scipy.interpolate import RegularGridInterpolator
            from scipy.optimize import minimize

            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False

        try:
            len_L = len(self.length_sweep)
            len_G = len(self.gain_sweep)
            len_F = len(self.freq_sweep)
        except AttributeError:
            print("Error: 'run' method must be called first to define sweep axes.")
            return

        fid_Array = np.zeros((len_L, len_G, len_F))

        I_g_data = self.data["Ig"]
        Q_g_data = self.data["Qg"]
        I_e_data = self.data["Ie"]
        Q_e_data = self.data["Qe"]

        for l_idx in tqdm(range(len_L), desc="Analyze max fidelity"):
            for g_idx in range(len_G):
                for f_idx in range(len_F):
                    I_g = I_g_data[l_idx, g_idx, f_idx]
                    Q_g = Q_g_data[l_idx, g_idx, f_idx]
                    I_e = I_e_data[l_idx, g_idx, f_idx]
                    Q_e = Q_e_data[l_idx, g_idx, f_idx]

                    vec_I = np.mean(I_e) - np.mean(I_g)
                    vec_Q = np.mean(Q_e) - np.mean(Q_g)
                    denom = np.abs(vec_I + 1j * vec_Q) ** 2

                    if denom == 0:
                        fid_Array[l_idx, g_idx, f_idx] = 0.0
                        continue

                    gstate = (
                        (I_g - np.mean(I_g)) * vec_I + (Q_g - np.mean(Q_g)) * vec_Q
                    ) / denom
                    estate = (
                        (I_e - np.mean(I_g)) * vec_I + (Q_e - np.mean(Q_g)) * vec_Q
                    ) / denom

                    if gstate.size == 0 or estate.size == 0:
                        fid_Array[l_idx, g_idx, f_idx] = 0.0
                        continue

                    min_g = np.min(gstate)
                    max_e = np.max(estate)
                    if min_g >= max_e:
                        min_g, max_e = min(min_g, max_e), max(min_g, max_e)
                        if min_g == max_e:
                            max_e += 1e-9

                    th_list = np.linspace(min_g, max_e, 1000).reshape(-1, 1)
                    g_proj = np.array(gstate).reshape(1, -1)
                    e_proj = np.array(estate).reshape(1, -1)

                    g_cdf = np.sum(g_proj < th_list, axis=1) / g_proj.shape[1]
                    e_cdf = np.sum(e_proj > th_list, axis=1) / e_proj.shape[1]
                    fidelity = (g_cdf + e_cdf) / 2

                    fid_Array[l_idx, g_idx, f_idx] = np.max(fidelity)

        max_idx = np.unravel_index(np.argmax(fid_Array), fid_Array.shape)
        max_l_idx, max_g_idx, max_f_idx = max_idx

        max_fid_grid = fid_Array[max_idx]
        max_length_grid = self.length_sweep[max_l_idx]
        max_gain_grid = self.gain_sweep[max_g_idx]
        max_freq_grid = self.freq_sweep[max_f_idx]

        print(f"\n--- Grid Search Result ---")
        print(f"Max fidelity (on grid): {max_fid_grid:.4f}")

        length_str_grid = (
            f"{max_length_grid:.3f} us" if max_length_grid is not None else "default"
        )
        gain_str_grid = (
            f"{max_gain_grid:.5f} DAC" if max_gain_grid is not None else "default"
        )
        freq_str_grid = (
            f"{max_freq_grid:.5f} MHz" if max_freq_grid is not None else "default"
        )

        print(
            f"At length={length_str_grid}, gain={gain_str_grid}, freq={freq_str_grid}"
        )

        max_length, max_gain, max_freq = max_length_grid, max_gain_grid, max_freq_grid

        if not SCIPY_AVAILABLE:
            print("\nWarning: `scipy` not found. Skipping interpolation.")
            print("To enable, run: pip install scipy")
        else:
            try:
                all_axes_data = [
                    (self.length_sweep, len_L, max_length_grid),
                    (self.gain_sweep, len_G, max_gain_grid),
                    (self.freq_sweep, len_F, max_freq_grid),
                ]

                opt_param_indices = []
                opt_axes = []
                opt_initial_guess = []
                opt_bounds = []
                fixed_params = {}

                for i, (axis, length, initial_val) in enumerate(all_axes_data):
                    if length > 1 and axis[0] is not None:
                        opt_param_indices.append(i)
                        opt_axes.append(np.array(axis))
                        opt_initial_guess.append(initial_val)
                        opt_bounds.append((np.min(axis), np.max(axis)))
                    else:
                        fixed_params[i] = initial_val

                if not opt_param_indices:
                    print(
                        "\nNo parameters to optimize (all sweeps have length 1 or are None)."
                    )
                    print("Returning grid search result.")
                else:
                    squeezed_fid_Array = np.squeeze(fid_Array)

                    if squeezed_fid_Array.ndim == 0:
                        squeezed_fid_Array = squeezed_fid_Array.reshape(
                            (1,) * len(opt_param_indices)
                        )

                    try:
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="cubic"
                        )
                    except ValueError:
                        print(
                            "Warning: Not enough data for cubic interpolation. Falling back to linear."
                        )
                        interpolator = RegularGridInterpolator(
                            tuple(opt_axes), squeezed_fid_Array, method="linear"
                        )

                    def objective_func(opt_params):
                        return -interpolator(opt_params)[0]

                    result = minimize(
                        objective_func,
                        opt_initial_guess,
                        method="L-BFGS-B",
                        bounds=opt_bounds,
                    )

                    if not result.success:
                        print(
                            f"\nWarning: Interpolation optimization failed. {result.message}"
                        )
                        print("Returning grid search result.")
                    else:
                        max_fid_interp = -result.fun

                        final_params = [None, None, None]

                        for i, param_val in enumerate(result.x):
                            final_params[opt_param_indices[i]] = param_val

                        for i, param_val in fixed_params.items():
                            final_params[i] = param_val

                        max_length, max_gain, max_freq = final_params

                        print(f"\n--- Interpolated Result ---")
                        print(f"Max fidelity (interpolated): {max_fid_interp:.4f}")

                        length_str = (
                            f"{max_length:.3f} us"
                            if max_length is not None
                            else "default"
                        )
                        gain_str = (
                            f"{max_gain:.5f} DAC" if max_gain is not None else "default"
                        )
                        freq_str = (
                            f"{max_freq:.5f} MHz" if max_freq is not None else "default"
                        )

                        print(
                            f"At length={length_str}, gain={gain_str}, freq={freq_str}"
                        )

            except Exception as e:
                print(f"\nAn error occurred during interpolation: {e}")
                print("Returning grid search result.")

        self.fid_Array = fid_Array  # 將 fid_Array 存儲在 self 中 供 plot 函數使用

        return_L = round(max_length, 3) if max_length is not None else None
        return_G = round(max_gain, 6) if max_gain is not None else None
        return_F = round(max_freq, 6) if max_freq is not None else None

        return return_L, return_G, return_F

    def plot_top_fidelity_histograms(self, top_n=9):
        if not hasattr(self, "fid_Array"):
            print("Running analyze() to generate fidelity data...")
            self.analyze()
            if not hasattr(self, "fid_Array"):
                print(
                    "Error: Fidelity data not available after running analyze(). Cannot plot."
                )
                return

        fid_Array = self.fid_Array

        if fid_Array.ndim != 3:
            print("Error: fid_Array must be a 3D array (length, gain, freq).")
            return

        flat_fid_Array = fid_Array.flatten()

        top_n_flat_indices = np.argsort(flat_fid_Array)[-top_n:][::-1]

        top_n_indices = np.unravel_index(top_n_flat_indices, fid_Array.shape)

        top_n_I_g = [self.data["Ig"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_g = [self.data["Qg"][idx] for idx in zip(*top_n_indices)]
        top_n_I_e = [self.data["Ie"][idx] for idx in zip(*top_n_indices)]
        top_n_Q_e = [self.data["Qe"][idx] for idx in zip(*top_n_indices)]

        all_I_data = np.concatenate(top_n_I_g + top_n_I_e)
        all_Q_data = np.concatenate(top_n_Q_g + top_n_Q_e)

        overall_min = min(np.min(all_I_data), np.min(all_Q_data))
        overall_max = max(np.max(all_I_data), np.max(all_Q_data))
        range_span = overall_max - overall_min
        padding_val = range_span * 0.05

        plot_min = overall_min - padding_val
        plot_max = overall_max + padding_val

        plot_extent = [plot_min, plot_max, plot_min, plot_max]

        hexbin_gridsize = 50

        grid_size = int(np.ceil(np.sqrt(top_n)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size)
        )

        axes = axes.flatten()

        print(f"\nPlotting top {top_n} fidelity points using hexbin...")
        for i in range(min(top_n, len(axes))):
            l_idx, g_idx, f_idx = (
                top_n_indices[0][i],
                top_n_indices[1][i],
                top_n_indices[2][i],
            )

            I_g = self.data["Ig"][l_idx, g_idx, f_idx]
            Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
            I_e = self.data["Ie"][l_idx, g_idx, f_idx]
            Q_e = self.data["Qe"][l_idx, g_idx, f_idx]

            current_fid = fid_Array[l_idx, g_idx, f_idx]

            length = self.length_sweep[l_idx]
            gain = self.gain_sweep[g_idx]
            freq = self.freq_sweep[f_idx]

            ax = axes[i]

            ax.hexbin(
                I_e,
                Q_e,
                gridsize=hexbin_gridsize,
                cmap="Reds",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )

            ax.hexbin(
                I_g,
                Q_g,
                gridsize=hexbin_gridsize,
                cmap="Blues",
                alpha=0.6,
                extent=plot_extent,
                mincnt=1,
            )

            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

            ax.set_aspect("equal", adjustable="box")

            title_str = (
                f"Fidelity: {current_fid:.4f}\n"
                f"L={length:.3f} us, G={gain:.5f} DAC, F={freq:.5f} MHz"
            )
            ax.set_title(title_str, fontsize=10)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

        for j in range(top_n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
