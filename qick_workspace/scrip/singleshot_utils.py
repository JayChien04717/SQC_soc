"""
scrip.singleshot_utils
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

def _bic_gmm(X, max_components=2, n_init=5):
    """
    Fit 1..max_components GMMs on 1-D data X and return the one with
    lowest BIC (Bayesian Information Criterion).

    BIC penalises model complexity, so it selects 2 components only when
    the data is genuinely bimodal (e.g. T1-decay tail on |e⟩).
    """
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
    """
    Fit per-state GMMs with BIC-selected number of components (1 or 2),
    then classify using a Bayes (max log-likelihood) rule.

    Why per-state instead of combined:
    - T1 decay during readout makes |e⟩ bimodal in I: one peak at the
      true |e⟩ position, one near |g⟩ from decayed shots.
    - A combined 2-component GMM on g+e shots cannot distinguish a
      bimodal |e⟩ from two well-separated states — it picks arbitrary
      boundaries and reports inflated fidelity.
    - Fitting each state separately with BIC lets the algorithm
      automatically use 2 Gaussians for a T1-broadened |e⟩ and 1 for a
      clean |g⟩, then place the Bayes-optimal boundary between the
      per-state likelihoods.

    Component budget per state
    --------------------------
    All states allow up to max_components via BIC selection.
    |g⟩ can legitimately have a small secondary mode from thermal population
    (typically < 15%).  However, if BIC selects 2 components for |g⟩ and the
    secondary weight exceeds 0.30, it is treated as overfitting of the |e⟩
    overlap tail and |g⟩ is refit with a single component.
    |e⟩ and higher states always allow 2 components for T1-decay tails.

    Parameters
    ----------
    I_projs : list of ndarray
        Rotated I-axis projections, one array per prepared state.
    xlims : [float, float]
        Search window for threshold finding.
    n_init : int
        GMM random restarts per candidate k.
    max_components : int
        Maximum number of Gaussian components for non-ground states (default 2).

    Returns
    -------
    state_gmms  : list[GaussianMixture]  – one fitted GMM per state
    state_order : ndarray                – state indices sorted by primary mean
    conf_matrix : ndarray (n_states, n_states)
    thresholds  : list[float]
    primary_means : ndarray              – dominant component mean per state
    primary_stds  : ndarray              – dominant component std  per state
    primary_weights : ndarray            – dominant component weight per state
    """
    n_states = len(I_projs)

    # ── Fit each state independently ──────────────────────────────────────
    # All states allow up to max_components so that:
    # - |g⟩ can model a small thermal-population secondary mode
    # - |e⟩/|f⟩ can model T1-decay tails
    #
    # Guard for |g⟩ (index 0): if the secondary Gaussian weight is large
    # (> 0.30), it is almost certainly overfitting the overlap tail from |e⟩
    # rather than a real thermal population (which is typically < 15%).
    # In that case, refit |g⟩ with a single component.
    _MAX_G_SECONDARY = 0.30
    state_gmms = []
    for i, proj in enumerate(I_projs):
        gmm = _bic_gmm(proj.reshape(-1, 1), max_components, n_init)
        if i == 0 and gmm.n_components > 1:
            # Check secondary weight
            dominant_w = float(np.max(gmm.weights_))
            secondary_w = 1.0 - dominant_w
            if secondary_w > _MAX_G_SECONDARY:
                # Refit with 1 component — overlap tail, not thermal population
                gmm = _bic_gmm(proj.reshape(-1, 1), 1, n_init)
        state_gmms.append(gmm)

    # ── Primary component per state (highest-weight Gaussian) ─────────────
    # Used for rotation-axis refinement and display.
    primary_means   = np.zeros(n_states)
    primary_stds    = np.zeros(n_states)
    primary_weights = np.zeros(n_states)
    for i, gmm in enumerate(state_gmms):
        idx = int(np.argmax(gmm.weights_))
        primary_means[i]   = float(gmm.means_[idx, 0])
        primary_stds[i]    = float(np.sqrt(gmm.covariances_[idx, 0, 0]))
        primary_weights[i] = float(gmm.weights_[idx])

    # ── Guard: secondary components must lie within the primary-mean range ──
    # Physical constraint: T1 decay brings |e⟩ toward |g⟩, thermal excitation
    # brings |g⟩ toward |e⟩.  A secondary Gaussian whose mean lies *outside*
    # the envelope spanned by all primary means is modelling noise or a
    # systematic artifact (e.g. RITS, readout-induced transitions), not a
    # physical decay channel.  Refit any such state with a single Gaussian so
    # the fidelity estimate is not inflated by a spurious component.
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

    # Sort states by primary mean (left → right in histogram)
    state_order = np.argsort(primary_means)

    # ── Bayes confusion matrix ─────────────────────────────────────────────
    # For each shot, pick the state whose per-state GMM gives the highest
    # log-likelihood.  This is the optimal classifier under these models.
    x_dense = np.linspace(xlims[0], xlims[1], 2000).reshape(-1, 1)
    log_liks_dense = np.array(
        [gmm.score_samples(x_dense) for gmm in state_gmms]
    )  # (n_states, 2000)

    conf_matrix = np.zeros((n_states, n_states))
    for i, proj in enumerate(I_projs):
        X = proj.reshape(-1, 1)
        ll = np.array([gmm.score_samples(X) for gmm in state_gmms])  # (n_states, n_shots)
        preds = np.argmax(ll, axis=0)                                  # (n_shots,)
        for j in range(n_states):
            conf_matrix[i, j] = np.mean(preds == j)

    # ── Thresholds: Bayes log-likelihood crossings ─────────────────────────
    # Take the crossing closest to the midpoint of the two adjacent primary
    # means, not cross[0].  Multi-component GMMs can produce spurious early
    # crossings far from the true decision boundary.
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
    # Two-pass refinement when theta is auto-computed:
    #   Pass 1 — raw IQ mean → coarse theta (fast, no model)
    #   Pass 2 — fit 1D GMM on coarse-rotated I projections, use primary
    #             Gaussian means as reference → refined theta
    # This prevents T1-decay tails and thermal population secondary modes
    # from biasing the rotation axis.
    if not amplitude_mode:
        if theta is None:
            g_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in g_states])
            e_c = np.concatenate([iqshots[i][0] + 1j * iqshots[i][1] for i in e_states])

            # Pass 1: coarse rotation from raw means
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

            # Pass 2: fit GMM on coarse I-projections, use primary means
            if _HAS_SKLEARN:
                try:
                    g_proj_coarse = _rot_I_coarse(g_c)
                    e_proj_coarse = _rot_I_coarse(e_c)
                    gmm_g = _bic_gmm(g_proj_coarse.reshape(-1, 1), 2, 5)
                    gmm_e = _bic_gmm(e_proj_coarse.reshape(-1, 1), 2, 5)
                    # Primary mean = component with highest weight
                    g_primary_I = float(gmm_g.means_[np.argmax(gmm_g.weights_), 0])
                    e_primary_I = float(gmm_e.means_[np.argmax(gmm_e.weights_), 0])
                    # Back-project primary means to original 2D space to get refined angle
                    # The coarse rotation already put g/e mostly on the I axis;
                    # refine by computing the angle correction from primary I positions.
                    # In the coarse-rotated frame the primary g/e centroids are
                    # (g_primary_I, Q_g_mean) and (e_primary_I, Q_e_mean).
                    Q_g_mean = float(np.mean(g_c.real * np.sin(theta_rad) + g_c.imag * np.cos(theta_rad)))
                    Q_e_mean = float(np.mean(e_c.real * np.sin(theta_rad) + e_c.imag * np.cos(theta_rad)))
                    dI = e_primary_I - g_primary_I
                    dQ = Q_e_mean - Q_g_mean
                    # Additional rotation to align g→e direction to I axis
                    delta = np.arctan2(dQ, dI)
                    theta_rad = theta_rad + delta
                except Exception:
                    pass  # keep coarse theta on any failure
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

    # ── 5. Per-state GMM fitting & confusion matrix ───────────────────────
    state_gmms, state_order, conf_matrix, thresholds, gmm_means, gmm_stds, gmm_weights = \
        _fit_gmm(I_projs, xlims)

    # Fidelity = mean of confusion-matrix diagonal (fraction, 0–1)
    fid  = float(np.mean(np.diag(conf_matrix)))
    fids = [fid]

    # Confusion matrix in percent
    conf_matrix_pct = conf_matrix * 100.0

    # ── 6. Per-state GMM overlay on histogram ─────────────────────────────
    if plot:
        x_plot    = np.linspace(xlims[0], xlims[1], 500)
        x_col     = x_plot.reshape(-1, 1)
        bin_width = bins_dist[1] - bins_dist[0]

        for idx, gmm in enumerate(state_gmms):
            n_shots   = len(I_projs[idx])
            scale_s   = n_shots * bin_width          # per-state scale
            state_pdf = np.exp(gmm.score_samples(x_col))
            c = default_colors[idx % len(default_colors)]

            # Each Gaussian component of this state (dashed + light fill)
            for comp_i in range(gmm.n_components):
                w   = float(gmm.weights_[comp_i])
                mu  = float(gmm.means_[comp_i, 0])
                sig = float(np.sqrt(gmm.covariances_[comp_i, 0, 0]))
                comp_pdf = w * _norm.pdf(x_plot, mu, sig)
                axs[1, 0].fill_between(x_plot, 0, comp_pdf * scale_s,
                                       alpha=0.15, color=c)
                axs[1, 0].plot(x_plot, comp_pdf * scale_s,
                               color=c, linewidth=0.9, linestyle="--")

            # Full per-state mixture envelope
            axs[1, 0].plot(x_plot, state_pdf * scale_s,
                           color=c, linewidth=1.8, alpha=0.85,
                           label=f"GMM {state_labels[idx]}")

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
        for idx, (lbl, gmm) in enumerate(zip(state_labels, state_gmms)):
            sec_weight = 1.0 - float(gmm_weights[idx]) if gmm.n_components > 1 else 0.0
            quality_flag = "  ⚠ secondary component large — possible state leakage" if sec_weight > 0.25 else ""
            print(f"  |{lbl}⟩  components={gmm.n_components}  "
                  f"primary_mean={gmm_means[idx]:.3f}  primary_std={gmm_stds[idx]:.3f}  "
                  f"secondary_weight={sec_weight:.3f}{quality_flag}")
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
