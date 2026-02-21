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
from ..tools.system_tool import get_next_filename_labber, hdf5_generator, config_to_yaml

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
    plotoverlap=False,
    fitparams=None,
    normalize=True,
    title=None,
    export=False,
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
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 0].axis("equal")
        axs[0, 1].set_title(
            f"Rotated ($\\theta={theta * 180 / np.pi:.1f}^\\circ$)", fontsize=13
        )
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 1].axis("equal")
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

        if not amplitude_mode:
            I_new, Q_new = rotate_iq(complex_data, theta)
            data_to_hist = I_new
        else:
            I_new, Q_new = I, Q
            data_to_hist = np.abs(complex_data)

        if plot:
            axs[0, 0].scatter(
                I,
                Q,
                label=state_label,
                color=this_color,
                marker=this_marker,
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
                marker=this_marker,
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

    # --- 5. Fitting Logic (Skipped definition for brevity, assuming standard Gaussian funcs exist) ---
    def gaussian_norm(x, b, c):
        return 1 / (np.sqrt(2 * np.pi) * c) * np.exp(-((x - b) ** 2) / (2 * c**2))

    def overlap_area_norm(b1, c1, b2, c2):
        def min_func(x):
            return np.minimum(gaussian_norm(x, b1, c1), gaussian_norm(x, b2, c2))

        x_min, x_max = min(b1 - 5 * c1, b2 - 5 * c2), max(b1 + 5 * c1, b2 + 5 * c2)
        area, _ = quad(min_func, x_min, x_max)
        return area

    def readout_fidelity_norm(b1, c1, b2, c2):
        return 1 - overlap_area_norm(b1, c1, b2, c2)

    do_fit = fit or gauss_overlap or plotoverlap
    if do_fit and n_dist["g"] is not None and n_dist["e"] is not None:
        bin_centers = (bins_dist[:-1] + bins_dist[1:]) / 2
        n_g, n_e = n_dist["g"], n_dist["e"]
        xmax_g_idx, xmax_e_idx = np.argmax(n_g), np.argmax(n_e)
        xmax_g_val, xmax_e_val = bin_centers[xmax_g_idx], bin_centers[xmax_e_idx]
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
                b_g, c_g = (
                    (popt_g[1], abs(popt_g[2]))
                    if popt_g[0] > popt_g[3]
                    else (popt_g[4], abs(popt_g[5]))
                )
                b_e, c_e = (
                    (popt_e[1], abs(popt_e[2]))
                    if popt_e[0] > popt_e[3]
                    else (popt_e[4], abs(popt_e[5]))
                )
                gauss_fit_fidelity = readout_fidelity_norm(b_g, c_g, b_e, c_e)
                popts, pcovs = [popt_g, popt_e], [pcov_g, pcov_e]
                b_g_plot, c_g_plot, b_e_plot, c_e_plot = b_g, c_g, b_e, c_e
            except Exception as e:
                print(f"Fit failed: {e}")
                gauss_fit_fidelity, popts = 0, [None, None]
        else:
            guess_g = [np.max(n_g), xmax_g_val, sigma_guess, 0]
            guess_e = [np.max(n_e), xmax_e_val, sigma_guess, 0]
            try:
                popt_g, pcov_g = fit_gauss(bin_centers, n_g, guess_g)
                popt_e, pcov_e = fit_gauss(bin_centers, n_e, guess_e)
                popts, pcovs = [popt_g, popt_e], [pcov_g, pcov_e]
                b_g_plot, c_g_plot = popt_g[1], abs(popt_g[2])
                b_e_plot, c_e_plot = popt_e[1], abs(popt_e[2])
            except:
                popts = [None, None]

        if plot and popts[0] is not None:
            x_dense = np.linspace(bins_dist[0], bins_dist[-1], 500)
            fit_f = double_gaussian if gauss_overlap else gaussian
            axs[1, 0].plot(
                x_dense,
                fit_f(x_dense, *popts[0]),
                color=default_colors[0],
                linestyle="-",
                linewidth=2,
                label="Fit G",
            )
            axs[1, 0].plot(
                x_dense,
                fit_f(x_dense, *popts[1]),
                color=default_colors[1],
                linestyle="-",
                linewidth=2,
                label="Fit E",
            )
            if plotoverlap and b_g_plot is not None:
                bin_width = bins_dist[1] - bins_dist[0]
                y_overlap = np.minimum(
                    gaussian_norm(x_dense, b_g_plot, c_g_plot)
                    * np.sum(n_g)
                    * bin_width,
                    gaussian_norm(x_dense, b_e_plot, c_e_plot)
                    * np.sum(n_e)
                    * bin_width,
                )
                axs[1, 0].fill_between(
                    x_dense, 0, y_overlap, color="purple", alpha=0.3, label="Overlap"
                )

    # --- 6. Thresholds & Matrix ---
    fids, thresholds = [], []
    contrast_ge = np.abs(
        (np.cumsum(n_dist["g"]) - np.cumsum(n_dist["e"]))
        / (np.sum(n_dist["g"]) + np.sum(n_dist["e"]))
    )
    tind_ge = contrast_ge.argmax()
    thresholds.append(bins_dist[tind_ge])
    fids.append(
        contrast_ge[tind_ge]
        if not fid_avg
        else 0.5
        * (
            1
            - n_dist["g"][tind_ge:].sum() / n_dist["g"].sum()
            + 1
            - n_dist["e"][:tind_ge].sum() / n_dist["e"].sum()
        )
    )

    if not has_f_state:
        matrix_size, labels = 2, ["|g>", f"|{e_label}>"]
        raw_matrix = np.array(
            [
                [n_dist["g"][:tind_ge].sum(), n_dist["g"][tind_ge:].sum()],
                [n_dist["e"][:tind_ge].sum(), n_dist["e"][tind_ge:].sum()],
            ]
        )
    else:
        matrix_size, labels = 3, ["|g>", f"|{e_label}>", "|f>"]
        if n_dist["f"] is not None:
            contrast_ef = np.abs(
                (np.cumsum(n_dist["e"]) - np.cumsum(n_dist["f"]))
                / (np.sum(n_dist["e"]) + np.sum(n_dist["f"]))
            )
            tind_ef = contrast_ef.argmax()
            thresholds.append(bins_dist[tind_ef])
            ts = sorted([tind_ge, tind_ef])

            def cls(n):
                return [n[: ts[0]].sum(), n[ts[0] : ts[1]].sum(), n[ts[1] :].sum()]

            raw_matrix = np.array(
                [cls(n_dist["g"]), cls(n_dist["e"]), cls(n_dist["f"])]
            )
        else:
            raw_matrix = np.zeros((3, 3))

    row_sums = raw_matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    conf_matrix = 100 * raw_matrix / row_sums

    # --- 7. Finalize Plots ---

    if plot:
        # 繪製 Threshold 線
        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--", label="Threshold")

        # 設定標題
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

        # --- 修正圖例顯示問題 (Fix Legend Visibility) ---

        # 1. Histogram Legend
        axs[1, 0].legend(fontsize=8, loc="upper right")

        # 2. Scatter Legend (Unrotated) - 強制讓圖例點變為不透明
        leg0 = axs[0, 0].legend(fontsize=8, loc="upper right")
        if leg0:
            # 修改點： legendHandles -> legend_handles
            for lh in leg0.legend_handles:
                lh.set_alpha(1)  # 設定圖例中的點為不透明

        # 3. Scatter Legend (Rotated) - 強制讓圖例點變為不透明
        leg1 = axs[0, 1].legend(fontsize=8, loc="upper right")
        if leg1:
            # 修改點： legendHandles -> legend_handles
            for lh in leg1.legend_handles:
                lh.set_alpha(1)

        # ----------------------------------------------

        # Confusion Matrix 繪圖
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
            plt.close()
        else:
            plt.show()

    # --- 8. Returns (Dictionary Format) ---

    # 8.1 Create Raw Data Packet (For external analysis like fit_single_shot)
    formatted_data = {
        "Ig": np.real(data_map["g"]),
        "Qg": np.imag(data_map["g"]),
        "Ie": np.real(data_map["e"]),
        "Qe": np.imag(data_map["e"]),
    }
    if has_f_state:
        formatted_data["If"] = np.real(data_map["f"])
        formatted_data["Qf"] = np.imag(data_map["f"])

    # 8.2 Construct Result Dictionary
    result_dict = {
        "fidelity": gauss_fit_fidelity if gauss_overlap else fids[0],
        "thresholds": thresholds,
        "angle": theta * 180 / np.pi,
        "confusion_matrix": conf_matrix,
        "data": formatted_data,
        "fids_list": fids,
    }

    # 8.3 Add Fit Params if they exist
    if fit or gauss_overlap or plotoverlap:
        result_dict["fit_params"] = popts
        result_dict["fit_cov"] = pcovs

    if verbose:
        print(f"Theta: {result_dict['angle']:.2f} deg")
        print(f"Fidelity: {100 * result_dict['fidelity']:.3f}%")
        print("Fidelity Matrix (%):\n", conf_matrix)

    return result_dict


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


##################
# Define Program #
##################

# Separate g and e per each experiment defined.


class SingleShotProgram_gef(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qb_ch = cfg["qb_ch"]
        qb_ch_ef = cfg["qb_ch_ef"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"], mixer_freq=cfg["qb_mixer"])
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg["nqz_qb"])
        if self.soccfg["gens"][qb_ch_ef]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qb_ch_ef, nqz=cfg["nqz_qb_ef"], mixer_freq=cfg["qb_mixer_ef"]
            )
        else:
            self.declare_gen(ch=qb_ch_ef, nqz=cfg["nqz_qb_ef"])

        self.add_loop("shotloop", cfg["shots"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )
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
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch,
                name="qb_ge_pulse",
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
                name="qb_ge_pulse",
                ro_ch=ro_ch,
                style="flat_top",
                envelope="ramp",
                freq=cfg["qb_freq_ge"],
                phase=cfg["qb_phase"],
                gain=cfg["pi_gain_ge"],
                length=cfg["qb_flat_top_length_ge"],
            )
        self.add_gauss(
            ch=qb_ch_ef,
            name="ramp_ef",
            sigma=cfg["sigma_ef"],
            length=cfg["sigma_ef"] * 5,
            even_length=True,
        )
        if cfg["pulse_type"] == "arb":
            self.add_pulse(
                ch=qb_ch_ef,
                name="qb_ef_pulse",
                style="arb",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["pi_gain_ef"],
            )

        elif cfg["pulse_type"] == "flat_top":
            self.add_pulse(
                ch=qb_ch_ef,
                name="qb_ef_pulse",
                style="flat_top",
                envelope="ramp_ef",
                freq=cfg["qb_freq_ef"],
                phase=cfg["qb_phase_ef"],
                gain=cfg["pi_gain_ef"],
                length=cfg["qb_flat_top_length_ef"],
            )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.delay_auto(cfg["relax_delay"], tag="relax_wait")

        self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        if cfg["shot_f"] is True:
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=self.cfg["qb_ch"], name="qb_ge_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=self.cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleShot_gef:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, SHOTS, shot_f=False):
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f

        shot_gef = SingleShotProgram_gef(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        iq_list_gef = shot_gef.acquire(self.soc, rounds=1, progress=True)

        # 5. 根據結果拆分數據
        Ig = iq_list_gef[0][0, :, 0]
        Qg = iq_list_gef[0][0, :, 1]
        Ie = iq_list_gef[0][1, :, 0]
        Qe = iq_list_gef[0][1, :, 1]

        if shot_f:
            If = iq_list_gef[0][2, :, 0]
            Qf = iq_list_gef[0][2, :, 1]
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe, "If": If, "Qf": Qf}
        else:
            self.data = {"Ig": Ig, "Qg": Qg, "Ie": Ie, "Qe": Qe}

        return self.data

    def plot(
        self, fid_avg=False, fit=False, normalize=False, verbose=True, overlap=False
    ):
        return hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=True,
            verbose=verbose,
            gauss_overlap=overlap,
            fid_avg=fid_avg,
            fit=fit,
            fitparams=[None, None, 5, None, None, 5],
            normalize=normalize,
            title=None,
            export=False,
        )

    def autorun(self, SHOTS, shot_f=False):
        self.cfg["shots"] = SHOTS
        shot_g = SingleShotProgram_g(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        shot_e = SingleShotProgram_e(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        iq_list_g = shot_g.acquire(self.soc, rounds=1, progress=False)
        iq_list_e = shot_e.acquire(self.soc, rounds=1, progress=False)

        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
        if shot_f:
            shot_f = SingleShotProgram_f(
                self.soccfg,
                reps=1,
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list_f = shot_f.acquire(self.soc, rounds=1, progress=True)
            I_f = iq_list_f[0][0].T[0]
            Q_f = iq_list_f[0][0].T[1]

        if shot_f:
            self.data = {
                "Ig": I_g,
                "Qg": Q_g,
                "Ie": I_e,
                "Qe": Q_e,
                "If": I_f,
                "Qf": Q_f,
            }
        else:
            self.data = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}

        data = hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=False,
            verbose=False,
            fid_avg=False,
            fit=False,
            fitparams=[None, None, 20, None, None, 20],
            normalize=False,
            title=None,
            export=False,
        )
        return data

    def saveLabber(self, qb_idx, yoko_value=None):
        if "If" in self.data.keys():
            expt_name = "s000_singleshot_gef" + f"_{qb_idx}"
        else:
            expt_name = "s000_singleshot_ge" + f"_{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        print("Current data file: " + file_path)

        dict_val = config_to_yaml(self.cfg)
        if "If" in self.data.keys():
            shotdata = np.array(
                (
                    self.data["Ig"],
                    self.data["Qg"],
                    self.data["Ie"],
                    self.data["Qe"],
                    self.data["If"],
                    self.data["Qf"],
                )
            )
            hdf5_generator(
                filepath=file_path,
                x_info={
                    "name": "# shot",
                    "unit": "#",
                    "values": np.arange(self.cfg["shots"]),
                },
                y_info={"name": "State", "unit": "", "values": [0, 1, 2]},
                z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
                comment=(f"{dict_val}"),
                tag="SingleSht",
            )
        else:
            shotdata = np.array(
                (
                    self.data["Ig"],
                    self.data["Qg"],
                    self.data["Ie"],
                    self.data["Qe"],
                )
            )
            hdf5_generator(
                filepath=file_path,
                x_info={
                    "name": "# shot",
                    "unit": "#",
                    "values": np.arange(self.cfg["shots"]),
                },
                y_info={"name": "State", "unit": "", "values": [0, 1]},
                z_info={"name": "Signal", "unit": "ADC unit", "values": shotdata},
                comment=(f"{dict_val}"),
                tag="SingleShot",
            )
