"""
Histogram and analysis utilities for SingleShot experiments.

Fitting:  Gaussian Mixture Model (GMM) via scikit-learn.
Fidelity: mean of the confusion-matrix diagonal.
"""

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy.stats import norm as _norm

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle    = ["o", "*", "s", "^"]


def plot_hist(data, bins, ax=None, xlims=None, color=None, linestyle=None,
              label=None, alpha=None, normalize=True):
    if color is None:
        color = next(cycle(default_colors))
    hist_data, bin_edges = np.histogram(data, bins=bins, range=xlims)
    if normalize:
        s = hist_data.sum()
        if s > 0:
            hist_data = hist_data / s

    for i in range(len(hist_data)):
        ax.plot([bin_edges[i], bin_edges[i + 1]], [hist_data[i], hist_data[i]],
                color=color, linestyle=linestyle,
                label=label if i == 0 else None,
                alpha=alpha, linewidth=0.9)
        if i < len(hist_data) - 1:
            ax.plot([bin_edges[i + 1], bin_edges[i + 1]],
                    [hist_data[i], hist_data[i + 1]],
                    color=color, linestyle=linestyle, alpha=alpha, linewidth=0.9)
    ax.relim()
    ax.set_ylim((0, None))
    return hist_data, bin_edges


def _bic_gmm(X, max_components=2, n_init=5):
    best_gmm, best_bic = None, np.inf
    for k in range(1, max_components + 1):
        try:
            g = GaussianMixture(
                n_components=k, covariance_type="full",
                n_init=n_init, random_state=0,
            )
            g.fit(X)
            b = g.bic(X)
            if b < best_bic:
                best_bic, best_gmm = b, g
        except Exception:
            pass
    return best_gmm


def _fit_gmm(I_projs, xlims, n_init=5, max_components=2):
    n_states = len(I_projs)
    _MAX_G_SECONDARY = 0.30
    state_gmms = []
    for i, proj in enumerate(I_projs):
        gmm = _bic_gmm(proj.reshape(-1, 1), max_components, n_init)
        if i == 0 and gmm.n_components > 1:
            dominant_w = float(np.max(gmm.weights_))
            secondary_w = 1.0 - dominant_w
            if secondary_w > _MAX_G_SECONDARY:
                gmm = _bic_gmm(proj.reshape(-1, 1), 1, n_init)
        state_gmms.append(gmm)

    primary_means   = np.zeros(n_states)
    primary_stds    = np.zeros(n_states)
    primary_weights = np.zeros(n_states)
    for i, gmm in enumerate(state_gmms):
        idx = int(np.argmax(gmm.weights_))
        primary_means[i]   = float(gmm.means_[idx, 0])
        primary_stds[i]    = float(np.sqrt(gmm.covariances_[idx, 0, 0]))
        primary_weights[i] = float(gmm.weights_[idx])

    if n_states >= 2:
        pmin, pmax = primary_means.min(), primary_means.max()
        for i, gmm in enumerate(state_gmms):
            if gmm.n_components <= 1:
                continue
            dom_idx = int(np.argmax(gmm.weights_))
            out_of_range = any(
                not (pmin - 1e-9 <= float(gmm.means_[j, 0]) <= pmax + 1e-9)
                for j in range(gmm.n_components)
                if j != dom_idx
            )
            if out_of_range:
                state_gmms[i] = _bic_gmm(I_projs[i].reshape(-1, 1), 1, n_init)
                idx2 = int(np.argmax(state_gmms[i].weights_))
                primary_means[i]   = float(state_gmms[i].means_[idx2, 0])
                primary_stds[i]    = float(np.sqrt(state_gmms[i].covariances_[idx2, 0, 0]))
                primary_weights[i] = float(state_gmms[i].weights_[idx2])

    state_order = np.argsort(primary_means)

    x_dense = np.linspace(xlims[0], xlims[1], 2000).reshape(-1, 1)
    log_liks_dense = np.array([gmm.score_samples(x_dense) for gmm in state_gmms])

    conf_matrix = np.zeros((n_states, n_states))
    for i, proj in enumerate(I_projs):
        X = proj.reshape(-1, 1)
        ll = np.array([gmm.score_samples(X) for gmm in state_gmms])
        preds = np.argmax(ll, axis=0)
        for j in range(n_states):
            conf_matrix[i, j] = np.mean(preds == j)

    thresholds = []
    for k in range(n_states - 1):
        s1, s2 = state_order[k], state_order[k + 1]
        midpoint = (primary_means[s1] + primary_means[s2]) / 2
        diff  = log_liks_dense[s1] - log_liks_dense[s2]
        cross = np.where(np.diff(np.sign(diff)))[0]
        if cross.size:
            cross_vals = x_dense[cross, 0]
            best = cross_vals[np.argmin(np.abs(cross_vals - midpoint))]
            t = float(best)
        else:
            t = float(midpoint)
        thresholds.append(t)

    return (state_gmms, state_order, conf_matrix, thresholds,
            primary_means, primary_stds, primary_weights)


def general_hist(iqshots, state_labels, g_states, e_states, e_label="e",
                 check_qubit_label=None, numbins=200, amplitude_mode=False,
                 ps_threshold=None, theta=None, plot=True, verbose=True,
                 fid_avg=False, normalize=True, title=None, export=False):
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for GMM fitting. "
            "Install with:  pip install scikit-learn"
        )

    if numbins is None:
        numbins = 200

    n_states = len(iqshots)

    if not amplitude_mode:
        if theta is None:
            g_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in g_states])
            e_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in e_states])
            theta_rad = -np.arctan2(np.mean(e_c.imag) - np.mean(g_c.imag),
                                    np.mean(e_c.real) - np.mean(g_c.real))

            def _make_rot(tr):
                def _rot_I(c):
                    return c.real * np.cos(tr) - c.imag * np.sin(tr)
                def _rot_IQ(c):
                    I = c.real * np.cos(tr) - c.imag * np.sin(tr)
                    Q = c.real * np.sin(tr) + c.imag * np.cos(tr)
                    return I, Q
                return _rot_I, _rot_IQ

            _rot_I_coarse, _ = _make_rot(theta_rad)

            if _HAS_SKLEARN:
                try:
                    g_proj_coarse = _rot_I_coarse(g_c)
                    e_proj_coarse = _rot_I_coarse(e_c)
                    gmm_g = _bic_gmm(g_proj_coarse.reshape(-1, 1), 2, 5)
                    gmm_e = _bic_gmm(e_proj_coarse.reshape(-1, 1), 2, 5)
                    g_primary_I = float(gmm_g.means_[np.argmax(gmm_g.weights_), 0])
                    e_primary_I = float(gmm_e.means_[np.argmax(gmm_e.weights_), 0])
                    Q_g_mean = float(np.mean(g_c.real * np.sin(theta_rad) + g_c.imag * np.cos(theta_rad)))
                    Q_e_mean = float(np.mean(e_c.real * np.sin(theta_rad) + e_c.imag * np.cos(theta_rad)))
                    dI = e_primary_I - g_primary_I
                    dQ = Q_e_mean - Q_g_mean
                    delta = np.arctan2(dQ, dI)
                    theta_rad = theta_rad + delta
                except Exception:
                    pass
        else:
            theta_rad = float(theta) * np.pi / 180.0

        def _rot_I(c):
            return c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)

        def _rot_IQ(c):
            I = c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)
            Q = c.real * np.sin(theta_rad) + c.imag * np.cos(theta_rad)
            return I, Q
    else:
        theta_rad = 0.0
        _rot_I  = lambda c: np.abs(c)
        _rot_IQ = lambda c: (c.real, c.imag)

    all_c    = np.concatenate([I + 1j * Q for I, Q in iqshots])
    proj_all = _rot_I(all_c)
    span     = (proj_all.max() - proj_all.min()) / 2
    mid      = (proj_all.max() + proj_all.min()) / 2
    xlims    = [mid - span, mid + span]

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
        _title = title or (
            "Readout Fidelity"
            + (f" on Q{check_qubit_label}" if check_qubit_label is not None else "")
        )
        fig.suptitle(_title)
        axs[0, 0].set_title("Unrotated", fontsize=13)
        axs[0, 0].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 0].set_ylabel("Q [ADC levels]", fontsize=11)
        axs[0, 0].axis("equal")
        axs[0, 1].set_title(
            f"Rotated ($\\theta = {theta_rad * 180 / np.pi:.1f}^\\circ$)", fontsize=13)
        axs[0, 1].set_xlabel("I [ADC levels]", fontsize=11)
        axs[0, 1].axis("equal")
        x_axis_lbl = "Amplitude" if amplitude_mode else "I"
        axs[1, 0].set_xlabel(f"{x_axis_lbl} [ADC levels]", fontsize=11)
        axs[1, 0].set_ylabel("Counts", fontsize=12)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)

    I_projs   = []
    bins_dist = None

    for idx, (I, Q) in enumerate(iqshots):
        cmplx        = I + 1j * Q
        I_new, Q_new = _rot_IQ(cmplx)
        proj         = _rot_I(cmplx)
        I_projs.append(proj)
        color     = default_colors[idx % len(default_colors)]
        marker    = marker_cycle[idx % len(marker_cycle)]
        lbl       = state_labels[idx]

        if plot:
            axs[0, 0].scatter(I, Q, label=lbl, color=color, marker=".", edgecolor="None", alpha=0.1)
            axs[0, 0].plot(np.mean(I), np.mean(Q), color="k", marker=marker,
                           markerfacecolor=color, markersize=6)
            axs[0, 1].scatter(I_new, Q_new, label=lbl, color=color, marker=".", edgecolor="None", alpha=0.1)
            axs[0, 1].plot(np.mean(I_new), np.mean(Q_new), color="k", marker=marker,
                           markerfacecolor=color, markersize=6)
            _, bins_dist = plot_hist(
                proj, bins=numbins, ax=axs[1, 0], xlims=xlims,
                color=color, linestyle=linestyle_cycle[0],
                label=lbl, alpha=0.6, normalize=False,
            )
        else:
            _, bins_dist = np.histogram(proj, bins=numbins, range=xlims)

    state_gmms, state_order, conf_matrix, thresholds, gmm_means, gmm_stds, gmm_weights = \
        _fit_gmm(I_projs, xlims)

    fid  = float(np.mean(np.diag(conf_matrix)))
    fids = [fid]
    conf_matrix_pct = conf_matrix * 100.0

    if plot:
        x_plot    = np.linspace(xlims[0], xlims[1], 500)
        x_col     = x_plot.reshape(-1, 1)
        bin_width = bins_dist[1] - bins_dist[0]

        for idx, gmm in enumerate(state_gmms):
            n_shots   = len(I_projs[idx])
            scale_s   = n_shots * bin_width
            state_pdf = np.exp(gmm.score_samples(x_col))
            c = default_colors[idx % len(default_colors)]

            for comp_i in range(gmm.n_components):
                w   = float(gmm.weights_[comp_i])
                mu  = float(gmm.means_[comp_i, 0])
                sig = float(np.sqrt(gmm.covariances_[comp_i, 0, 0]))
                comp_pdf = w * _norm.pdf(x_plot, mu, sig)
                axs[1, 0].fill_between(x_plot, 0, comp_pdf * scale_s, alpha=0.15, color=c)
                axs[1, 0].plot(x_plot, comp_pdf * scale_s, color=c, linewidth=0.9, linestyle="--")

            axs[1, 0].plot(x_plot, state_pdf * scale_s, color=c, linewidth=1.8, alpha=0.85,
                           label=f"GMM {state_labels[idx]}")

        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--", linewidth=1.2, label="Threshold")

        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="gray", linestyle="-.")

        fid_title = "$F_{\\overline{ge}}$" if fid_avg else "$F_{ge}$"
        axs[1, 0].set_title(f"{fid_title} (GMM): {100 * fid:.2f}%", fontsize=13)
        axs[1, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)

        cm_labels = [f"|{lbl}⟩" for lbl in state_labels]
        ax_cm = axs[1, 1]
        ax_cm.clear()
        im = ax_cm.imshow(conf_matrix_pct, cmap="Reds", vmin=0, vmax=100)
        ax_cm.set_xticks(np.arange(n_states))
        ax_cm.set_yticks(np.arange(n_states))
        ax_cm.set_xticklabels(cm_labels)
        ax_cm.set_yticklabels(cm_labels)
        ax_cm.set_xlabel("Declared output", fontsize=11)
        ax_cm.set_ylabel("Input state", fontsize=11)
        ax_cm.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        for i in range(n_states):
            for j in range(n_states):
                val = conf_matrix_pct[i, j]
                ax_cm.text(j, i, f"{val:.1f}%", ha="center", va="center",
                           color="white" if val > 50 else "black", fontsize=12)
        ax_cm.set_title("Confusion Matrix (%)")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("Exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    if verbose:
        print(f"Rotation angle : {theta_rad * 180 / np.pi:.2f} deg")
        print(f"GMM Fidelity   : {100 * fid:.3f}%")
        for idx, (lbl, gmm) in enumerate(zip(state_labels, state_gmms)):
            sec_weight = 1.0 - float(gmm_weights[idx]) if gmm.n_components > 1 else 0.0
            quality_flag = "  ⚠ secondary component large" if sec_weight > 0.25 else ""
            print(f"  |{lbl}⟩  components={gmm.n_components}  "
                  f"primary_mean={gmm_means[idx]:.3f}  primary_std={gmm_stds[idx]:.3f}  "
                  f"secondary_weight={sec_weight:.3f}{quality_flag}")
        print(f"Thresholds     : {[f'{t:.3f}' for t in thresholds]}")
        print("Confusion Matrix (%):\n", np.round(conf_matrix_pct, 1))

    return [fids, thresholds, theta_rad * 180 / np.pi, conf_matrix_pct]


def hist(data, amplitude_mode=False, ps_threshold=None, theta=None,
         plot=True, verbose=True, fid_avg=False,
         normalize=True, title=None, export=False):
    iqshots      = [(data["Ig"], data["Qg"]), (data["Ie"], data["Qe"])]
    state_labels = ["g", "e"]
    g_states     = [0]
    e_states     = [1]

    if "If" in data:
        iqshots.append((data["If"], data["Qf"]))
        state_labels.append("f")
        e_states = [2]

    return general_hist(
        iqshots=iqshots, state_labels=state_labels,
        g_states=g_states, e_states=e_states,
        amplitude_mode=amplitude_mode, ps_threshold=ps_threshold,
        theta=theta, plot=plot, verbose=verbose,
        fid_avg=fid_avg, normalize=normalize,
        title=title, export=export,
    )


__all__ = ["plot_hist", "general_hist", "hist", "_fit_gmm", "_bic_gmm"]
