"""
newscrip.singleshot_utils
=========================
Histogram and analysis utilities for SingleShot experiments.

Fitting:  Gaussian Mixture Model (GMM) via scikit-learn.
Fidelity: mean of the confusion-matrix diagonal — generalises correctly to any
          number of states (ge, gef, ...).
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


# ── Step-histogram renderer ────────────────────────────────────────────────────

def plot_hist(data, bins, ax=None, xlims=None, color=None, linestyle=None,
              label=None, alpha=None, normalize=True):
    """Draw a step histogram onto *ax*."""
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


# ── GMM core ──────────────────────────────────────────────────────────────────

def _fit_gmm(I_projs, xlims, n_init=5):
    """
    Fit a Gaussian Mixture Model on the combined 1-D projections of all states
    and derive the confusion matrix and classification thresholds.

    Parameters
    ----------
    I_projs : list of ndarray
        Rotated I-axis projections, one array per prepared state (g, e, [f, ...]).
    xlims : [float, float]
        Search window for threshold finding.
    n_init : int
        Number of GMM random restarts.

    Returns
    -------
    gmm : GaussianMixture
        Fitted model.
    order : ndarray
        Indices that sort GMM components by ascending mean  (g → e → f → ...).
    conf_matrix : ndarray, shape (n_states, n_states)
        Fraction of state-i shots declared as state-j
        (rows = prepared state, cols = declared state).
    thresholds : list[float]
        Decision boundaries between adjacent sorted components.
    means : ndarray   – sorted component means
    stds  : ndarray   – sorted component standard deviations
    weights : ndarray – sorted component mixture weights
    """
    n_states = len(I_projs)
    all_I = np.concatenate(I_projs).reshape(-1, 1)

    # "tied" forces all components to share one covariance (= equal sigma).
    # Physical motivation: both g and e states see the same amplifier noise
    # floor, so their intrinsic readout sigma should be equal.  "full" lets
    # each component fit its own sigma, which causes the T1-decay-broadened
    # tail of the |e⟩ distribution to inflate sigma_e and pull the threshold
    # toward |g⟩, artificially degrading the |e⟩ assignment.
    gmm = GaussianMixture(
        n_components=n_states, covariance_type="tied",
        n_init=n_init, random_state=0,
    )
    gmm.fit(all_I)

    # Sort components left → right (ascending mean)
    order     = np.argsort(gmm.means_.ravel())
    inv_order = np.argsort(order)   # original component index → sorted label

    means   = gmm.means_.ravel()[order]
    # "tied": single shared covariance matrix → same std for all components
    shared_std = float(np.sqrt(gmm.covariances_[0, 0]))
    stds    = np.full(n_states, shared_std)
    weights = gmm.weights_[order]

    # ── Confusion matrix ───────────────────────────────────────────────────
    # conf[i, j] = fraction of prepared-state-i shots assigned to sorted class j
    conf_matrix = np.zeros((n_states, n_states))
    for i, I_proj in enumerate(I_projs):
        raw_preds    = gmm.predict(I_proj.reshape(-1, 1))
        sorted_preds = inv_order[raw_preds]
        for j in range(n_states):
            conf_matrix[i, j] = (sorted_preds == j).sum() / len(sorted_preds)

    # ── Thresholds: posterior crossings between adjacent sorted components ─
    x_dense    = np.linspace(xlims[0], xlims[1], 2000)
    posteriors = gmm.predict_proba(x_dense.reshape(-1, 1))  # (2000, n_components)
    thresholds = []
    for k in range(n_states - 1):
        diff  = posteriors[:, order[k]] - posteriors[:, order[k + 1]]
        cross = np.where(np.diff(np.sign(diff)))[0]
        t = float(x_dense[cross[0]]) if cross.size else float((means[k] + means[k + 1]) / 2)
        thresholds.append(t)

    return gmm, order, conf_matrix, thresholds, means, stds, weights


# ── Main analysis ─────────────────────────────────────────────────────────────

def general_hist(iqshots, state_labels, g_states, e_states, e_label="e",
                 check_qubit_label=None, numbins=200, amplitude_mode=False,
                 ps_threshold=None, theta=None, plot=True, verbose=True,
                 fid_avg=False, normalize=True, title=None, export=False):
    """
    Analyse multi-state single-shot readout data with GMM fitting.

    Fits a GMM to the combined (rotated) I projections, then builds a
    confusion matrix by classifying each state's shots.  Fidelity is the
    mean of the confusion-matrix diagonal — correct for any number of states.

    Parameters
    ----------
    iqshots : list of (I, Q) array pairs
        One pair per prepared state, ordered [g, e] or [g, e, f, ...].
    state_labels : list[str]
        Display labels corresponding to *iqshots*.
    g_states : list[int]
        Indices in *iqshots* representing the ground state.
    e_states : list[int]
        Indices in *iqshots* representing the first excited state.
    e_label : str
        Label for the excited state (default "e").
    check_qubit_label : int or None
        Appended to the figure title as "on Q<n>".
    numbins : int
        Number of histogram bins.
    amplitude_mode : bool
        Use |IQ| amplitude instead of rotated I projection.
    ps_threshold : float or None
        Extra vertical marker on the histogram (e.g. post-selection cut).
    theta : float or None
        IQ rotation angle in degrees.  None → auto-computed from g/e means.
    plot : bool
        Produce the 4-panel figure.
    verbose : bool
        Print numerical results.
    fid_avg : bool
        API-compatibility flag; no effect (confusion-matrix fidelity is always
        the mean diagonal).
    normalize : bool
        Normalise histogram bins (passed through to plot_hist).
    title : str or None
        Override figure suptitle.
    export : bool
        Save figure to 'multihist.jpg' and close instead of plt.show().

    Returns
    -------
    list : [fids, thresholds, angle_deg, conf_matrix_pct]
        *fids*            – [F] where F is the mean confusion-matrix diagonal (0–1)
        *thresholds*      – decision boundaries between GMM components
        *angle_deg*       – IQ rotation angle used (degrees)
        *conf_matrix_pct* – (n_states × n_states) confusion matrix in %
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for GMM fitting. "
            "Install with:  pip install scikit-learn"
        )

    if numbins is None:
        numbins = 200

    n_states = len(iqshots)

    # ── 1. Rotation angle ──────────────────────────────────────────────────
    if not amplitude_mode:
        if theta is None:
            g_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in g_states])
            e_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in e_states])
            theta_rad = -np.arctan2(np.mean(e_c.imag) - np.mean(g_c.imag),
                                    np.mean(e_c.real) - np.mean(g_c.real))
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
        _rot_IQ = lambda c: (c.real, c.imag)   # scatter shows raw IQ

    # ── 2. Compute xlims from all shots ────────────────────────────────────
    all_c    = np.concatenate([I + 1j * Q for I, Q in iqshots])
    proj_all = _rot_I(all_c)
    span     = (proj_all.max() - proj_all.min()) / 2
    mid      = (proj_all.max() + proj_all.min()) / 2
    xlims    = [mid - span, mid + span]

    # ── 3. Plot setup ──────────────────────────────────────────────────────
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

    # ── 4. Scatter + histogram per state ──────────────────────────────────
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
            # Unrotated scatter
            axs[0, 0].scatter(I, Q, label=lbl, color=color,
                              marker=".", edgecolor="None", alpha=0.1)
            axs[0, 0].plot(np.mean(I), np.mean(Q), color="k",
                           marker=marker, markerfacecolor=color, markersize=6)
            # Rotated scatter
            axs[0, 1].scatter(I_new, Q_new, label=lbl, color=color,
                              marker=".", edgecolor="None", alpha=0.1)
            axs[0, 1].plot(np.mean(I_new), np.mean(Q_new), color="k",
                           marker=marker, markerfacecolor=color, markersize=6)
            # Histogram
            _, bins_dist = plot_hist(
                proj, bins=numbins, ax=axs[1, 0], xlims=xlims,
                color=color, linestyle=linestyle_cycle[0],
                label=lbl, alpha=0.6, normalize=False,
            )
        else:
            _, bins_dist = np.histogram(proj, bins=numbins, range=xlims)

    # ── 5. GMM fitting & confusion matrix ─────────────────────────────────
    gmm, order, conf_matrix, thresholds, gmm_means, gmm_stds, gmm_weights = \
        _fit_gmm(I_projs, xlims)

    # Fidelity = mean of confusion-matrix diagonal (fraction, 0–1)
    fid  = float(np.mean(np.diag(conf_matrix)))
    fids = [fid]

    # Confusion matrix in percent
    conf_matrix_pct = conf_matrix * 100.0

    # ── 6. GMM overlay on histogram ───────────────────────────────────────
    if plot:
        x_plot    = np.linspace(xlims[0], xlims[1], 500)
        bin_width = bins_dist[1] - bins_dist[0]
        n_total   = sum(len(p) for p in I_projs)
        scale     = n_total * bin_width   # converts PDF → expected counts

        # Individual GMM components (shaded + dashed)
        for k in range(n_states):
            comp_pdf = gmm_weights[k] * _norm.pdf(x_plot, gmm_means[k], gmm_stds[k])
            c = default_colors[k % len(default_colors)]
            axs[1, 0].fill_between(x_plot, 0, comp_pdf * scale,
                                   alpha=0.20, color=c)
            axs[1, 0].plot(x_plot, comp_pdf * scale,
                           color=c, linewidth=1.1, linestyle="--")

        # Total GMM mixture
        total_pdf = np.exp(gmm.score_samples(x_plot.reshape(-1, 1)))
        axs[1, 0].plot(x_plot, total_pdf * scale,
                       color="black", linewidth=1.5, alpha=0.7, label="GMM total")

        # Decision thresholds
        for th in thresholds:
            axs[1, 0].axvline(th, color="k", linestyle="--",
                              linewidth=1.2, label="Threshold")

        if ps_threshold is not None:
            axs[1, 0].axvline(ps_threshold, color="gray", linestyle="-.")

        fid_title = "$F_{\\overline{ge}}$" if fid_avg else "$F_{ge}$"
        axs[1, 0].set_title(f"{fid_title} (GMM): {100 * fid:.2f}%", fontsize=13)
        axs[1, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)

        # ── Confusion matrix heatmap ───────────────────────────────────────
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
                ax_cm.text(j, i, f"{val:.1f}%",
                           ha="center", va="center",
                           color="white" if val > 50 else "black", fontsize=12)
        ax_cm.set_title("Confusion Matrix (%)")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if export:
            plt.savefig("multihist.jpg", dpi=1000)
            print("Exported multihist.jpg")
            plt.close()
        else:
            plt.show()

    # ── 7. Verbose ────────────────────────────────────────────────────────
    if verbose:
        print(f"Rotation angle : {theta_rad * 180 / np.pi:.2f} deg")
        print(f"GMM Fidelity   : {100 * fid:.3f}%")
        print(f"GMM means      : {[f'{m:.3f}' for m in gmm_means]}")
        print(f"Thresholds     : {[f'{t:.3f}' for t in thresholds]}")
        print("Confusion Matrix (%):\n", np.round(conf_matrix_pct, 1))

    return [fids, thresholds, theta_rad * 180 / np.pi, conf_matrix_pct]


# ── Convenience wrapper ────────────────────────────────────────────────────────

def hist(data, amplitude_mode=False, ps_threshold=None, theta=None,
         plot=True, verbose=True, fid_avg=False,
         normalize=True, title=None, export=False):
    """
    Wrapper around general_hist for the standard IQ-dict format.

    Parameters
    ----------
    data : dict
        Keys: 'Ig', 'Qg', 'Ie', 'Qe'.  Optionally 'If', 'Qf' for f-state.
    """
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
