import matplotlib.pyplot as plt
import numpy as np

# ----- Qick package ----- #
from qick import *
from tqdm.auto import tqdm

# ----- User Library ----- #
from .base_program import BaseProgram
from .singleshot_utils import _fit_gmm, hist

##################
# Define Program #
##################


class SingleShotOptProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("shotloop", cfg["shots"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")
        if cfg.get("shot_f", False):
            self.setup_qubit_gen(cfg, "ef")
            self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse", gain_key="pi_gain_ef")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

        # 1. Ground state readout
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        self.delay_auto(cfg["relax_delay"], tag="relax_wait")

        # 2. Excited state readout
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(0.01, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

        # 3. Optional f-state readout (ge pi + ef pi -> |f>)
        if cfg.get("shot_f", False):
            self.delay_auto(cfg["relax_delay"], tag="relax_wait2")
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(0.01, tag="wait1")
            self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
            self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


#####################
# Define Experiment #
#####################


class SingleShot_ge_opt:
    """
    Grid search + Gaussian Process optimization for single-shot readout
    parameters (readout length, drive gain, readout frequency).

    Workflow
    --------
    1. run()     : Sweep over a user-defined grid; store raw IQ shots.
    2. analyze() : Compute per-point metrics, apply physical constraints,
                   fit a GP surrogate, optionally run Bayesian optimization
                   (online, requires hardware), and report the Pareto front.
    3. plot_*()  : Visualize results.

    Metrics computed per grid point
    --------------------------------
    fid_Array      : GMM hard-classification fidelity (mean conf-matrix diag)
    snr_array      : SNR = ||delta_mu||^2 / (sigma_g^2 + sigma_e^2)
    sep_array      : LDA separation ||delta_mu||
    soft_fid_array : Posterior-weighted soft fidelity — more robust than hard
                     fidelity when distributions overlap.
    leakage_array  : Secondary GMM component weight of |e> (proxy for T1-decay
                     leakage or readout-induced transitions to higher states).
    thermal_array  : Secondary GMM component weight of |g> when BIC selects 2
                     components (proxy for thermal population of |e>).

    Physical interpretation of leakage and thermal
    -----------------------------------------------
    leakage > 0.15  -> readout length is likely comparable to T1/2, or
                       readout-induced state transitions (RITS) are active.
    thermal > 0.05  -> qubit temperature is non-negligible, or relax_delay
                       is too short for the qubit to fully reset to |g>.

    Key improvements over the original analyze()
    ---------------------------------------------
    - Physical constraint filtering: points with leakage or thermal above
      user-specified thresholds are excluded from the optimum search.
      This prevents selecting a high-fidelity-but-physically-unsound config.
    - GP surrogate (Matern-2.5 + WhiteKernel): replaces RegularGridInterpolator
      cubic spline.  More robust on sparse, noisy grids; avoids Runge
      overshoot; automatically estimates measurement noise level.
    - Bug fix: the original code had `max_fid_interp = min(-result.fun,
      max_fid_grid)`, which forced the interpolated result to never exceed
      the grid maximum, completely defeating the purpose of interpolation.
      Removed; GP predictions can legitimately exceed grid points.
    - Bayesian optimization (optional, online): uses Expected Improvement (EI)
      as the acquisition function to propose the next hardware measurement
      point.  Balances exploration vs exploitation.  Physical constraints are
      enforced as a penalty on infeasible points.
    - Pareto front: exposes the fidelity-vs-leakage trade-off so users can
      choose a config based on their T1 budget.
    """

    def __init__(self, config):
        from .base_experiment import BaseExperiment

        if BaseExperiment._soc is None:
            raise RuntimeError(
                "Call BaseExperiment.setup(soc, soccfg, data_path) first."
            )
        self.soc = BaseExperiment._soc
        self.soccfg = BaseExperiment._soccfg
        self.cfg = config

    # =========================================================================
    # run
    # =========================================================================

    def run(self, SHOTS, sweep_para: dict, shot_f=False):
        """
        Acquire raw IQ shots over the Cartesian grid defined by sweep_para.

        Parameters
        ----------
        SHOTS      : int   — number of single shots per state per grid point.
        sweep_para : dict  — keys are 'length', 'gain', 'freq'; values are
                             scalars or array-likes.  Scalar -> single point.
        shot_f     : bool  — if True, also prepare and measure the |f> state
                             (requires pi_ge + pi_ef pulses).
        """
        self.cfg["shots"] = SHOTS
        self.cfg["shot_f"] = shot_f
        self._shot_f = shot_f

        # Normalize each sweep parameter to a list
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

        # Pre-allocate storage; NaN makes missing data visible
        self.I_g_array = np.full(final_shape, np.nan)
        self.Q_g_array = np.full(final_shape, np.nan)
        self.I_e_array = np.full(final_shape, np.nan)
        self.Q_e_array = np.full(final_shape, np.nan)
        if shot_f:
            self.I_f_array = np.full(final_shape, np.nan)
            self.Q_f_array = np.full(final_shape, np.nan)

        # Determine which axes are actually being swept for tqdm placement
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

        # Build iterators with tqdm only at the outermost active sweep level
        l_iter = self.length_sweep
        if "l" == outermost_real_sweep:
            l_iter = tqdm(self.length_sweep, desc="Length loop")

        for l_idx, l_val in enumerate(l_iter):
            g_iter = self.gain_sweep
            if "g" == outermost_real_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop")
            elif is_g_sweep:
                g_iter = tqdm(self.gain_sweep, desc="Gain loop", leave=False)

            for g_idx, g_val in enumerate(g_iter):
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

                    ssp = SingleShotOptProgram(
                        self.soccfg,
                        reps=1,
                        final_delay=self.cfg["relax_delay"],
                        cfg=self.cfg,
                    )
                    iq_list = ssp.acquire(self.soc, rounds=1, progress=False)

                    I_g = iq_list[0][0, :, 0]
                    Q_g = iq_list[0][0, :, 1]
                    I_e = iq_list[0][1, :, 0]
                    Q_e = iq_list[0][1, :, 1]

                    self.I_g_array[l_idx, g_idx, f_idx, :] = I_g
                    self.Q_g_array[l_idx, g_idx, f_idx, :] = Q_g
                    self.I_e_array[l_idx, g_idx, f_idx, :] = I_e
                    self.Q_e_array[l_idx, g_idx, f_idx, :] = Q_e

                    if shot_f:
                        self.I_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 0]
                        self.Q_f_array[l_idx, g_idx, f_idx, :] = iq_list[0][2, :, 1]

        self.data = {
            "Ig": self.I_g_array,
            "Qg": self.Q_g_array,
            "Ie": self.I_e_array,
            "Qe": self.Q_e_array,
        }
        if shot_f:
            self.data["If"] = self.I_f_array
            self.data["Qf"] = self.Q_f_array

    # =========================================================================
    # _compute_metrics  (static helper)
    # =========================================================================

    @staticmethod
    def _compute_metrics(I_g, Q_g, I_e, Q_e):
        """
        Compute all per-grid-point readout quality metrics from raw IQ shots.

        Parameters
        ----------
        I_g, Q_g : ndarray  — I and Q quadratures for |g> shots.
        I_e, Q_e : ndarray  — I and Q quadratures for |e> shots.

        Returns
        -------
        dict with keys:
          fid      : GMM hard-classification fidelity (mean diag conf matrix).
          soft_fid : Posterior-weighted soft fidelity (see note below).
          snr      : ||delta_mu||^2 / (sigma_g^2 + sigma_e^2).
          sep      : LDA separation ||delta_mu||.
          leakage  : Secondary GMM weight of |e> (T1-decay / RITS proxy).
          thermal  : Secondary GMM weight of |g> (thermal population proxy).

        Soft fidelity
        -------------
        Instead of hard 0/1 classification, each shot x is scored by the
        per-state posterior P(state_i | x) under the GMM:
            soft_fid = mean_i [ mean_{x in state_i} P(state_i | x) ]
        This equals the expected accuracy under the model and is strictly >=
        hard fidelity when the threshold is sub-optimal.  Particularly useful
        for overlapping distributions.

        Note: QICK acquire() returns integrated IQ only, so a matched-filter
        / optimal-weighting kernel cannot be applied here.  Soft fidelity is
        the best post-hoc improvement achievable without time-trace data.
        """
        # ------------------------------------------------------------------
        # LDA projection onto the |g>-|e> axis
        # ------------------------------------------------------------------
        mg = np.array([I_g.mean(), Q_g.mean()])
        me = np.array([I_e.mean(), Q_e.mean()])
        v = me - mg
        n = float(np.linalg.norm(v))

        sep = 0.0
        snr = 0.0
        if n > 1e-12:
            pg = ((I_g - mg[0]) * v[0] + (Q_g - mg[1]) * v[1]) / n
            pe = ((I_e - mg[0]) * v[0] + (Q_e - mg[1]) * v[1]) / n
            sep = n
            snr = n**2 / (pg.var() + pe.var() + 1e-30)

        # ------------------------------------------------------------------
        # GMM fit in the LDA-rotated 1-D projection
        # ------------------------------------------------------------------
        all_c = np.concatenate([I_g + 1j * Q_g, I_e + 1j * Q_e])
        theta_rad = -np.arctan2(me[1] - mg[1], me[0] - mg[0])

        def _rot_I(c):
            return c.real * np.cos(theta_rad) - c.imag * np.sin(theta_rad)

        proj_g = _rot_I(I_g + 1j * Q_g)
        proj_e = _rot_I(I_e + 1j * Q_e)
        proj_all = _rot_I(all_c)
        span = (proj_all.max() - proj_all.min()) / 2
        mid = (proj_all.max() + proj_all.min()) / 2
        xlims = [mid - span, mid + span]

        (
            state_gmms,
            state_order,
            conf_matrix,
            thresholds,
            primary_means,
            primary_stds,
            primary_weights,
        ) = _fit_gmm([proj_g, proj_e], xlims)

        # Hard fidelity: mean of the confusion-matrix diagonal
        fid = float(np.mean(np.diag(conf_matrix)))

        # ------------------------------------------------------------------
        # Soft fidelity via per-shot Bayesian posterior under the GMM
        # ------------------------------------------------------------------
        # For each state i:
        #   P(i | x) = p_i(x) / sum_k p_k(x)   [uniform prior]
        # soft accuracy for state i = mean_j P(i | x_j) over shots of state i.
        soft_accs = []
        for i, proj in enumerate([proj_g, proj_e]):
            X = proj.reshape(-1, 1)
            # Log-likelihood from each state's GMM; shape = (n_states, n_shots)
            ll = np.array([gmm.score_samples(X) for gmm in state_gmms])
            # Numerically stable softmax -> posterior
            ll_shifted = ll - ll.max(axis=0)
            posteriors = np.exp(ll_shifted)
            posteriors /= posteriors.sum(axis=0)
            soft_accs.append(float(posteriors[i].mean()))
        soft_fid = float(np.mean(soft_accs))

        # ------------------------------------------------------------------
        # Leakage: secondary GMM weight of |e> (index 1)
        # Physical meaning: fraction of |e> shots that decayed toward |g>
        # during readout (T1) or were excited to |f> (RITS).
        # BIC selects 2 components only when bimodality is genuine.
        # ------------------------------------------------------------------
        gmm_e = state_gmms[1]
        leakage = 0.0
        if gmm_e.n_components > 1:
            leakage = float(1.0 - np.max(gmm_e.weights_))

        # ------------------------------------------------------------------
        # Thermal population: secondary GMM weight of |g> (index 0)
        # Physical meaning: fraction of nominally-ground-state shots that
        # are actually in |e> due to finite qubit temperature.
        # _fit_gmm enforces secondary_w < 0.30, so values are meaningful.
        # ------------------------------------------------------------------
        gmm_g = state_gmms[0]
        thermal = 0.0
        if gmm_g.n_components > 1:
            thermal = float(1.0 - np.max(gmm_g.weights_))

        return dict(
            fid=fid,
            soft_fid=soft_fid,
            snr=snr,
            sep=sep,
            leakage=leakage,
            thermal=thermal,
        )

    # =========================================================================
    # _is_pareto_efficient  (static helper)
    # =========================================================================

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Identify Pareto-efficient points under cost minimization.

        Parameters
        ----------
        costs : ndarray, shape (N, M)
            Each row is a point; each column is a cost to minimize.
            To maximize a metric, pass its negation as the corresponding column.

        Returns
        -------
        is_eff : bool ndarray, shape (N,)
            True for Pareto-efficient points.
        """
        is_eff = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                # A point dominates c if it is <= c on ALL objectives and
                # strictly < c on at least one.
                dominated = np.all(costs[is_eff] <= c, axis=1) & np.any(
                    costs[is_eff] < c, axis=1
                )
                is_eff[is_eff] = ~dominated
                is_eff[i] = True  # Never let a point disqualify itself
        return is_eff

    # =========================================================================
    # _expected_improvement  (static helper)
    # =========================================================================

    @staticmethod
    def _expected_improvement(
        gp, X_candidates: np.ndarray, y_best: float, xi: float = 0.01
    ) -> np.ndarray:
        """
        Expected Improvement (EI) acquisition function for Bayesian optimization.

        EI(x) = E[max(f(x) - y_best - xi, 0)]
               = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
        where Z = (mu - y_best - xi) / sigma, Phi = normal CDF, phi = normal PDF.

        Parameters
        ----------
        gp           : fitted GaussianProcessRegressor.
        X_candidates : ndarray, shape (N, D) — candidate points (scaled).
        y_best       : float — best observed fidelity so far.
        xi           : float — exploration-exploitation trade-off.
                       Larger xi (0.05) -> more exploration.
                       Smaller xi (0.01) -> more exploitation.

        Returns
        -------
        ei : ndarray, shape (N,) — EI value at each candidate point.
        """
        from scipy.stats import norm as sp_norm

        mu, sigma = gp.predict(X_candidates, return_std=True)
        sigma = sigma.reshape(-1)
        imp = mu - y_best - xi
        # Avoid division by near-zero sigma (flat GP region)
        Z = np.where(sigma > 1e-9, imp / sigma, 0.0)
        ei = imp * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei

    # =========================================================================
    # _acquire_single_point  (helper: run hardware at one parameter config)
    # =========================================================================

    def _acquire_single_point(self, length, gain, freq, SHOTS):
        """
        Run a single hardware acquisition at the given (length, gain, freq).
        Returns (I_g, Q_g, I_e, Q_e) arrays and the full metrics dict.
        """
        cfg_update = {"steps": SHOTS}
        if length is not None:
            cfg_update["ro_length"] = length
        if gain is not None:
            cfg_update["res_gain_ge"] = gain
        if freq is not None:
            cfg_update["res_freq_ge"] = freq
        self.cfg.update(cfg_update)

        ssp = SingleShotOptProgram(
            self.soccfg,
            reps=1,
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        iq_list = ssp.acquire(self.soc, rounds=1, progress=False)

        I_g = iq_list[0][0, :, 0]
        Q_g = iq_list[0][0, :, 1]
        I_e = iq_list[0][1, :, 0]
        Q_e = iq_list[0][1, :, 1]

        data_slice = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}
        fid = hist(data_slice, plot=False, verbose=False)[0][0]
        metrics = self._compute_metrics(I_g, Q_g, I_e, Q_e)
        metrics["fid"] = fid  # Overwrite with hist()-consistent value

        return I_g, Q_g, I_e, Q_e, metrics

    # =========================================================================
    # analyze
    # =========================================================================

    def analyze(
        self,
        leakage_threshold: float = 0.20,
        thermal_threshold: float = 0.10,
        bo_n_iter: int = 0,
        bo_xi: float = 0.01,
        pareto: bool = True,
    ):
        """
        Compute metrics for every grid point, then find the optimal readout
        configuration using physical constraints + GP surrogate interpolation.
        Optionally refine online with Bayesian optimization.

        Parameters
        ----------
        leakage_threshold : float
            Maximum allowed |e> leakage.  Points above this are excluded from
            the optimum search (but still shown in heatmaps).
            Rule of thumb: 0.20 is conservative; relax to 0.30 if the qubit
            T1 is short relative to the readout duration you need.
        thermal_threshold : float
            Maximum allowed thermal population.  Values above ~0.05 suggest
            the qubit is not fully resetting between shots; increase relax_delay.
        bo_n_iter : int
            Number of Bayesian optimization iterations.  Each iteration runs
            one hardware acquisition at the EI-optimal candidate point.
            Set to 0 (default) to use GP interpolation only (offline; no
            additional hardware time required).
        bo_xi : float
            Exploration factor for the EI acquisition function.
            0.01 -> exploitation-focused; 0.05 -> more exploratory.
        pareto : bool
            If True, compute and print the Pareto front (fidelity vs leakage).

        Returns
        -------
        (best_length, best_gain, best_freq) : rounded floats or None.
        """
        # ------------------------------------------------------------------
        # Optional imports: GP requires scikit-learn
        # ------------------------------------------------------------------
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler

            GP_AVAILABLE = True
        except ImportError:
            GP_AVAILABLE = False
            print(
                "Warning: scikit-learn not found.  GP interpolation disabled.\n"
                "Install with:  pip install scikit-learn"
            )

        try:
            len_L = len(self.length_sweep)
            len_G = len(self.gain_sweep)
            len_F = len(self.freq_sweep)
        except AttributeError:
            print("Error: call run() first to define sweep axes.")
            return

        shape3 = (len_L, len_G, len_F)
        fid_Array = np.zeros(shape3)
        soft_fid_array = np.zeros(shape3)
        snr_array = np.zeros(shape3)
        sep_array = np.zeros(shape3)
        leakage_array = np.zeros(shape3)
        thermal_array = np.zeros(shape3)

        shot_f = getattr(self, "_shot_f", False)
        metric_label = "GMM fidelity (gef)" if shot_f else "GMM fidelity (ge)"

        # ==================================================================
        # Step 1: compute all metrics over the grid
        # ==================================================================
        for l_idx in tqdm(range(len_L), desc=f"Analyze [{metric_label}]"):
            for g_idx in range(len_G):
                for f_idx in range(len_F):
                    I_g = self.data["Ig"][l_idx, g_idx, f_idx]
                    Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
                    I_e = self.data["Ie"][l_idx, g_idx, f_idx]
                    Q_e = self.data["Qe"][l_idx, g_idx, f_idx]

                    # Use hist() for GMM fidelity (supports gef mode)
                    data_slice = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}
                    if shot_f:
                        data_slice["If"] = self.data["If"][l_idx, g_idx, f_idx]
                        data_slice["Qf"] = self.data["Qf"][l_idx, g_idx, f_idx]
                    result = hist(data_slice, plot=False, verbose=False)
                    fid_Array[l_idx, g_idx, f_idx] = result[0][0]

                    # Extended metrics (ge only; f-state extension adds complexity)
                    m = self._compute_metrics(I_g, Q_g, I_e, Q_e)
                    soft_fid_array[l_idx, g_idx, f_idx] = m["soft_fid"]
                    snr_array[l_idx, g_idx, f_idx] = m["snr"]
                    sep_array[l_idx, g_idx, f_idx] = m["sep"]
                    leakage_array[l_idx, g_idx, f_idx] = m["leakage"]
                    thermal_array[l_idx, g_idx, f_idx] = m["thermal"]

        # ==================================================================
        # Step 2: physical constraint filtering
        # Exclude grid points where leakage or thermal exceeds the thresholds.
        # Leakage reflects T1 decay during readout; thermal reflects incomplete
        # qubit reset.  Both degrade the usable fidelity in practice.
        # ==================================================================
        feasible_mask = (leakage_array <= leakage_threshold) & (
            thermal_array <= thermal_threshold
        )
        n_feasible = feasible_mask.sum()
        print(
            f"\n{n_feasible}/{feasible_mask.size} grid points pass physical "
            f"constraints (leakage <= {leakage_threshold}, "
            f"thermal <= {thermal_threshold})"
        )

        if n_feasible == 0:
            # Fallback: relax the constraints to find the least-bad point
            print(
                "Warning: no feasible points found.  Relaxing to leakage <= min + 0.10."
            )
            feasible_mask = leakage_array <= (leakage_array.min() + 0.10)

        # Mask infeasible points with -inf so argmax ignores them
        fid_feasible = np.where(feasible_mask, fid_Array, -np.inf)

        # ==================================================================
        # Step 3: best feasible point on the grid
        # ==================================================================
        max_idx = np.unravel_index(np.argmax(fid_feasible), fid_feasible.shape)
        max_l_idx, max_g_idx, max_f_idx = max_idx

        best_fid_grid = float(fid_Array[max_idx])
        best_length_grid = self.length_sweep[max_l_idx]
        best_gain_grid = self.gain_sweep[max_g_idx]
        best_freq_grid = self.freq_sweep[max_f_idx]

        print("\n--- Grid best (feasible) ---")
        print(
            f"  fid={best_fid_grid:.4f}"
            f"  soft={soft_fid_array[max_idx]:.4f}"
            f"  SNR={snr_array[max_idx]:.3f}"
            f"  leakage={leakage_array[max_idx]:.3f}"
            f"  thermal={thermal_array[max_idx]:.3f}"
        )
        print(
            f"  length={best_length_grid}  gain={best_gain_grid}  freq={best_freq_grid}"
        )

        # Start with grid best; may be refined by GP or BO below
        max_length, max_gain, max_freq = (
            best_length_grid,
            best_gain_grid,
            best_freq_grid,
        )

        # ==================================================================
        # Step 4: GP surrogate fit + offline interpolation
        #
        # Why GP instead of RegularGridInterpolator cubic?
        # ------------------------------------------------
        # 1. Cubic splines on sparse grids (N ~ 5-20 per axis) suffer from
        #    Runge overshoot — wildly oscillating predictions between nodes.
        # 2. RegularGridInterpolator requires equal spacing; our sweeps often
        #    are not uniform.
        # 3. GP with Matern(nu=2.5) assumes C^2 smoothness (one continuous
        #    derivative), which matches the typical fidelity-vs-parameter
        #    landscape for dispersive readout.
        # 4. WhiteKernel automatically estimates shot-noise variance, making
        #    the surrogate robust to finite-shot statistical fluctuations.
        # 5. GP gives a posterior uncertainty (sigma), which is exploited by
        #    the EI acquisition function in Step 5.
        # ==================================================================
        if GP_AVAILABLE:
            # Build coordinate arrays for each axis (replace None with index)
            l_vals = np.array(
                [
                    v if v is not None else float(i)
                    for i, v in enumerate(self.length_sweep)
                ]
            )
            g_vals = np.array(
                [
                    v if v is not None else float(i)
                    for i, v in enumerate(self.gain_sweep)
                ]
            )
            f_vals = np.array(
                [
                    v if v is not None else float(i)
                    for i, v in enumerate(self.freq_sweep)
                ]
            )

            # Identify which axes are actually swept (more than one point)
            swept_axes = []  # axis index in (L, G, F)
            swept_vals = []  # corresponding coordinate arrays
            fixed_vals = {}  # axis index -> fixed value for unsewpt axes
            for ax_i, (vals, length) in enumerate(
                [(l_vals, len_L), (g_vals, len_G), (f_vals, len_F)]
            ):
                if length > 1:
                    swept_axes.append(ax_i)
                    swept_vals.append(vals)
                else:
                    fixed_vals[ax_i] = vals[0]

            if len(swept_axes) == 0:
                print("\nOnly one grid point — no interpolation needed.")
            else:
                # Build (N_total, D) training matrix where D = # swept axes
                idx_arrays = np.indices(shape3)
                flat_idx = [idx_arrays[ax].ravel() for ax in range(3)]

                coord_cols = []
                for ax_i in range(3):
                    if ax_i in [swept_axes[k] for k in range(len(swept_axes))]:
                        vals_for_ax = [l_vals, g_vals, f_vals][ax_i]
                        coord_cols.append(vals_for_ax[flat_idx[ax_i]])
                X_train = np.column_stack(coord_cols)  # shape (N_total, D)
                y_train = fid_Array.ravel()  # shape (N_total,)

                # StandardScaler normalises each parameter axis to zero mean /
                # unit variance, making GP length-scale estimation stable when
                # length, gain, and freq have very different physical scales.
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)

                # Matern(nu=2.5): C^2 smooth landscape (appropriate for readout
                # fidelity vs params).  WhiteKernel: explicit noise floor for
                # finite-shot GMM fidelity estimates.
                kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,  # Multi-start to avoid local optima in hyperparameter MLE
                    normalize_y=True,
                    random_state=42,
                )
                gp.fit(X_scaled, y_train)

                # Evaluate the GP on a fine grid (50 pts per swept axis)
                n_interp = 50
                fine_swept = [
                    np.linspace(swept_vals[k].min(), swept_vals[k].max(), n_interp)
                    for k in range(len(swept_axes))
                ]
                fine_grids = np.meshgrid(*fine_swept, indexing="ij")
                X_fine_sweep = np.column_stack([g.ravel() for g in fine_grids])

                # Reconstruct full (L, G, F) coordinates for the fine grid
                full_cols = []
                swept_col_cursor = 0
                for ax_i in range(3):
                    if ax_i in fixed_vals:
                        full_cols.append(np.full(len(X_fine_sweep), fixed_vals[ax_i]))
                    else:
                        full_cols.append(X_fine_sweep[:, swept_col_cursor])
                        swept_col_cursor += 1
                X_fine_full = np.column_stack(full_cols)  # shape (N_fine, 3)

                X_fine_scaled = scaler.transform(X_fine_sweep)
                mu_fine, sigma_fine = gp.predict(X_fine_scaled, return_std=True)

                best_interp_idx = int(np.argmax(mu_fine))
                best_interp_fid = float(mu_fine[best_interp_idx])
                best_interp_params = X_fine_full[best_interp_idx]  # (L, G, F)

                # NOTE: We intentionally do NOT cap the GP prediction at the
                # grid maximum.  The original code had:
                #   max_fid_interp = min(-result.fun, max_fid_grid)
                # which prevented interpolation from ever improving on the grid,
                # defeating its purpose.  If the GP predicts a significantly
                # higher value (> 5%), it is flagged for hardware verification.
                if best_interp_fid > best_fid_grid * 1.05:
                    print(
                        f"\nNote: GP predicts fid={best_interp_fid:.4f}, which is "
                        f">5% above the grid best ({best_fid_grid:.4f}).  "
                        f"Verify with hardware before committing to this config."
                    )
                else:
                    print("\n--- GP interpolation result ---")
                    print(f"  predicted fid = {best_interp_fid:.4f}")

                print(
                    f"  length={best_interp_params[0]:.4f}  "
                    f"gain={best_interp_params[1]:.6f}  "
                    f"freq={best_interp_params[2]:.6f}"
                )

                # Update the returned optimum from GP prediction
                max_length = best_interp_params[0] if len_L > 1 else best_length_grid
                max_gain = best_interp_params[1] if len_G > 1 else best_gain_grid
                max_freq = best_interp_params[2] if len_F > 1 else best_freq_grid

                # Store for use in BO and for external access
                self._gp = gp
                self._scaler = scaler
                self._X_fine_full = X_fine_full
                self._X_fine_scaled = X_fine_scaled
                self._gp_best_fid = best_interp_fid

                # ==============================================================
                # Step 5: Bayesian optimization (online; requires hardware)
                #
                # Each BO iteration:
                #   a) Re-fit the GP on all observed points.
                #   b) Evaluate EI on the fine candidate grid.
                #   c) Acquire hardware data at the EI-maximizing candidate.
                #   d) Apply a fidelity penalty if the point is infeasible
                #      (leakage or thermal exceeds threshold) so the GP learns
                #      to avoid those regions.
                #   e) Update the observed dataset and check for improvement.
                # ==============================================================
                if bo_n_iter > 0:
                    print(f"\n--- Bayesian optimization ({bo_n_iter} iterations) ---")

                    X_observed = X_scaled.copy()
                    y_observed = y_train.copy()
                    SHOTS = self.cfg["shots"]

                    for bo_it in range(bo_n_iter):
                        # Re-fit GP on all observations so far
                        gp.fit(X_observed, y_observed)
                        y_best_so_far = float(y_observed.max())

                        # Select next point by maximising EI
                        ei = self._expected_improvement(
                            gp, X_fine_scaled, y_best_so_far, xi=bo_xi
                        )
                        next_idx = int(np.argmax(ei))
                        next_params = X_fine_full[next_idx]  # (L, G, F)

                        print(
                            f"  BO iter {bo_it + 1}/{bo_n_iter}: "
                            f"L={next_params[0]:.4f}  "
                            f"G={next_params[1]:.6f}  "
                            f"F={next_params[2]:.6f}  "
                            f"EI={ei[next_idx]:.5f}"
                        )

                        # Hardware acquisition at the proposed point
                        _, _, _, _, metrics_new = self._acquire_single_point(
                            length=next_params[0] if len_L > 1 else None,
                            gain=next_params[1] if len_G > 1 else None,
                            freq=next_params[2] if len_F > 1 else None,
                            SHOTS=SHOTS,
                        )
                        fid_new = metrics_new["fid"]

                        print(
                            f"         measured fid={fid_new:.4f}  "
                            f"leakage={metrics_new['leakage']:.3f}  "
                            f"thermal={metrics_new['thermal']:.3f}"
                        )

                        # Penalty for physically infeasible points:
                        # Halve the fidelity so the GP learns to avoid that region.
                        is_feasible_new = (
                            metrics_new["leakage"] <= leakage_threshold
                            and metrics_new["thermal"] <= thermal_threshold
                        )
                        y_new = fid_new if is_feasible_new else fid_new * 0.5

                        # Append new observation to the training dataset
                        next_scaled = scaler.transform(next_params[swept_axes].reshape(1, -1))
                        X_observed = np.vstack([X_observed, next_scaled])
                        y_observed = np.append(y_observed, y_new)

                        # Update best if this point improves on the current optimum
                        if fid_new > best_fid_grid and is_feasible_new:
                            best_fid_grid = fid_new
                            max_length = (
                                next_params[0] if len_L > 1 else best_length_grid
                            )
                            max_gain = next_params[1] if len_G > 1 else best_gain_grid
                            max_freq = next_params[2] if len_F > 1 else best_freq_grid
                            print("         *** New best feasible point! ***")

        # ==================================================================
        # Step 6: Pareto front (fidelity vs leakage)
        #
        # The Pareto front exposes the trade-off between maximising fidelity
        # and minimising leakage (which is bounded by T1 and readout length).
        # Users can pick a point based on their specific T1 budget:
        #   - Short T1 qubit: prefer a Pareto point with lower leakage even
        #     if it sacrifices a few tenths of a percent in fidelity.
        #   - Long T1 qubit: the highest-fidelity Pareto point is likely safe.
        # ==================================================================
        if pareto:
            flat_fid = fid_Array.ravel()
            flat_leak = leakage_array.ravel()
            flat_therm = thermal_array.ravel()
            flat_l_idx = np.repeat(np.arange(len_L), len_G * len_F)
            flat_g_idx = np.tile(np.repeat(np.arange(len_G), len_F), len_L)
            flat_f_idx = np.tile(np.arange(len_F), len_L * len_G)

            # Costs: maximise fidelity -> minimise (-fid); minimise leakage
            costs = np.column_stack([-flat_fid, flat_leak])
            pareto_mask = self._is_pareto_efficient(costs)

            print(
                f"\n=== Pareto front (fidelity vs leakage): "
                f"{pareto_mask.sum()} points ==="
            )
            pareto_pts = sorted(
                zip(
                    flat_fid[pareto_mask],
                    flat_leak[pareto_mask],
                    flat_therm[pareto_mask],
                    flat_l_idx[pareto_mask],
                    flat_g_idx[pareto_mask],
                    flat_f_idx[pareto_mask],
                ),
                reverse=True,
            )
            print(
                f"  {'fid':>7}  {'leak':>6}  {'therm':>6}"
                f"  {'length':>8}  {'gain':>10}  {'freq':>10}"
            )
            for fid_p, leak_p, therm_p, li, gi, fi in pareto_pts:
                print(
                    f"  {fid_p:.4f}  {leak_p:.4f}  {therm_p:.4f}"
                    f"  {self.length_sweep[li]!s:>8}"
                    f"  {self.gain_sweep[gi]!s:>10}"
                    f"  {self.freq_sweep[fi]!s:>10}"
                )
            self._pareto_pts = pareto_pts

        # ==================================================================
        # Store metric arrays for plotting
        # ==================================================================
        self.fid_Array = fid_Array
        self.soft_fid_array = soft_fid_array
        self.snr_array = snr_array
        self.sep_array = sep_array
        self.leakage_array = leakage_array
        self.thermal_array = thermal_array
        self._feasible_mask = feasible_mask

        return_L = round(float(max_length), 3) if max_length is not None else None
        return_G = round(float(max_gain), 6) if max_gain is not None else None
        return_F = round(float(max_freq), 6) if max_freq is not None else None

        return return_L, return_G, return_F

    # =========================================================================
    # plot_grid_analysis
    # =========================================================================

    def plot_grid_analysis(self):
        """
        Six-panel heatmap overview of all grid metrics, followed by a full
        hist() IQ plot for the best feasible grid point.

        Panels
        ------
        Row 1:  GMM Fidelity  |  Soft Fidelity  |  SNR
        Row 2:  IQ Separation |  Leakage (|e>)  |  Thermal pop. (|g>)

        Each heatmap collapses the freq dimension by taking the best freq
        (by fidelity) at each (length, gain) pair.

        Infeasible grid points (leakage or thermal above threshold) are
        marked with a grey 'X' overlay to make them visually distinct.

        Soft Fidelity
        -------------
        Posterior-weighted accuracy (see _compute_metrics).  Higher than hard
        fidelity when the Bayes boundary is sub-optimal.  Use it to spot grid
        points where more shots would help vs where the physics limits you.

        Leakage heatmap
        ---------------
        Secondary GMM weight of |e>.  Large values (> 0.15) indicate T1 decay
        during readout or readout-induced transitions (RITS).  If leakage is
        high everywhere, the readout length is too long relative to T1.

        Thermal population heatmap
        --------------------------
        Secondary GMM weight of |g> (only when BIC selects 2 components and
        secondary weight < 0.30).  Reflects residual |e> population before the
        pi-pulse, caused by finite qubit temperature or incomplete reset.
        """
        if not hasattr(self, "fid_Array"):
            print("Running analyze() first ...")
            self.analyze()

        fid_arr = self.fid_Array
        soft_arr = self.soft_fid_array
        snr_arr = self.snr_array
        sep_arr = self.sep_array
        leak_arr = self.leakage_array
        therm_arr = self.thermal_array
        len_L, len_G, len_F = fid_arr.shape

        # Collapse the freq dimension: pick the best freq per (L, G) by fidelity
        best_f = np.argmax(fid_arr, axis=2)

        def _take(arr):
            return np.take_along_axis(arr, best_f[:, :, None], axis=2)[:, :, 0]

        fid_2d = _take(fid_arr)
        soft_2d = _take(soft_arr)
        snr_2d = _take(snr_arr)
        sep_2d = _take(sep_arr)
        leak_2d = _take(leak_arr)
        therm_2d = _take(therm_arr)

        # Feasibility mask collapsed to (L, G) using the same best_f index
        if hasattr(self, "_feasible_mask"):
            feasible_2d = _take(self._feasible_mask.astype(float)) > 0.5
        else:
            feasible_2d = np.ones((len_L, len_G), dtype=bool)

        def _labels(sweep):
            if sweep[0] is None:
                return [str(i) for i in range(len(sweep))]
            return [f"{v:.3g}" for v in sweep]

        l_labels = _labels(self.length_sweep)
        g_labels = _labels(self.gain_sweep)
        font_sz = max(4, 7 - max(len_L, len_G) // 5)

        def _imshow(
            ax,
            data,
            title,
            cmap,
            vmin=None,
            vmax=None,
            fmt=".3f",
            cbar_label="",
            mark_infeasible=False,
        ):
            im = ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="auto"
            )
            plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Gain", fontsize=9)
            ax.set_ylabel("Length", fontsize=9)
            ax.set_xticks(range(len_G))
            ax.set_xticklabels(g_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len_L))
            ax.set_yticklabels(l_labels, fontsize=7)
            for i in range(len_L):
                for j in range(len_G):
                    v = data[i, j]
                    if np.isnan(v):
                        continue
                    bg = im.cmap(im.norm(v))
                    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    tc = "white" if lum < 0.45 else "black"
                    ax.text(
                        j,
                        i,
                        format(v, fmt),
                        ha="center",
                        va="center",
                        fontsize=font_sz,
                        color=tc,
                    )
                    # Grey 'X' overlay for infeasible points
                    if mark_infeasible and not feasible_2d[i, j]:
                        ax.text(
                            j,
                            i,
                            "X",
                            ha="center",
                            va="center",
                            fontsize=font_sz + 2,
                            color="grey",
                            alpha=0.6,
                            fontweight="bold",
                        )

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("SingleShot Optimization — Grid Analysis", fontsize=13)

        _imshow(
            axs[0, 0],
            fid_2d,
            "GMM Fidelity",
            "RdYlGn",
            0.5,
            1.0,
            ".3f",
            "Fidelity",
            mark_infeasible=True,
        )
        _imshow(
            axs[0, 1],
            soft_2d,
            "Soft Fidelity",
            "RdYlGn",
            0.5,
            1.0,
            ".3f",
            "Soft Fidelity",
            mark_infeasible=True,
        )
        _imshow(
            axs[0, 2],
            snr_2d,
            "SNR  (||delta_mu||^2/sigma^2)",
            "plasma",
            0,
            None,
            ".2f",
            "SNR",
        )
        _imshow(
            axs[1, 0],
            sep_2d,
            "IQ Separation (||delta_mu||)",
            "Blues",
            0,
            None,
            ".3f",
            "Separation [ADC]",
        )
        _imshow(
            axs[1, 1],
            leak_2d,
            "|e> Leakage  (T1/RITS)",
            "Reds",
            0,
            0.5,
            ".3f",
            "Secondary weight",
        )
        _imshow(
            axs[1, 2],
            therm_2d,
            "Thermal Pop.  (|g>)",
            "Oranges",
            0,
            0.3,
            ".3f",
            "Secondary weight",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # ------------------------------------------------------------------
        # Console: top 5 feasible points
        # ------------------------------------------------------------------
        def _val_str(sweep, idx):
            v = sweep[idx]
            return f"{v:.4g}" if v is not None else str(idx)

        all_pts = [
            (
                l,
                g,
                int(best_f[l, g]),
                fid_2d[l, g],
                soft_2d[l, g],
                snr_2d[l, g],
                leak_2d[l, g],
                therm_2d[l, g],
                bool(feasible_2d[l, g]),
            )
            for l in range(len_L)
            for g in range(len_G)
        ]
        # Sort by fidelity; feasible points first
        all_pts.sort(key=lambda x: (x[8], x[3]), reverse=True)

        print("\n=== Top 5 points (feasible first, then by fidelity) ===")
        print(
            f"  {'L':>8}  {'G':>8}  {'F':>8}"
            f"  {'fid':>6}  {'soft':>6}  {'snr':>6}"
            f"  {'leak':>6}  {'therm':>6}  {'ok?':>4}"
        )
        for l, g, f, fid, soft, snr, leak, therm, ok in all_pts[:5]:
            print(
                f"  {_val_str(self.length_sweep, l):>8}"
                f"  {_val_str(self.gain_sweep, g):>8}"
                f"  {_val_str(self.freq_sweep, f):>8}"
                f"  {fid:.4f}  {soft:.4f}  {snr:6.3f}"
                f"  {leak:.3f}  {therm:.3f}  {'yes' if ok else 'NO':>4}"
            )

        # ------------------------------------------------------------------
        # Diagnostics for the best feasible point
        # ------------------------------------------------------------------
        l, g, f, fid, soft, snr, leak, therm, _ = all_pts[0]
        print("\n=== Diagnostics for best point ===")
        if leak > 0.15:
            print(
                f"  [!] High leakage ({leak:.3f}) — readout length may exceed T1/2, "
                f"or RITS is active.  Consider shorter readout."
            )
        else:
            print(f"  [ok] Leakage OK ({leak:.3f})")

        if therm > 0.05:
            print(
                f"  [!] Thermal population ({therm:.3f}) — qubit temperature is "
                f"non-negligible or relax_delay is too short."
            )
        else:
            print(f"  [ok] Thermal population OK ({therm:.3f})")

        delta_fid = soft - fid
        if delta_fid > 0.01:
            print(
                f"  [i] soft_fid - fid = {delta_fid:.4f}: the Bayesian soft "
                f"boundary outperforms the hard threshold — more shots or a "
                f"better threshold placement may improve fidelity."
            )
        else:
            print(f"  [ok] Hard and soft fidelities agree (delta = {delta_fid:.4f}).")

        # ------------------------------------------------------------------
        # Full IQ hist for the best point
        # ------------------------------------------------------------------
        l_v = _val_str(self.length_sweep, l)
        g_v = _val_str(self.gain_sweep, g)
        f_v = _val_str(self.freq_sweep, f)
        print(
            f"\n=== Full hist for best point: L={l_v}, G={g_v}, F={f_v}"
            f"  (fid={fid:.4f}, soft={soft:.4f}) ==="
        )
        data_slice = {
            "Ig": self.data["Ig"][l, g, f],
            "Qg": self.data["Qg"][l, g, f],
            "Ie": self.data["Ie"][l, g, f],
            "Qe": self.data["Qe"][l, g, f],
        }
        if getattr(self, "_shot_f", False):
            data_slice["If"] = self.data["If"][l, g, f]
            data_slice["Qf"] = self.data["Qf"][l, g, f]

        hist(
            data_slice,
            plot=True,
            verbose=True,
            title=(
                f"Best point  L={l_v}, G={g_v}, F={f_v}"
                f"  —  fid={fid:.4f}  soft={soft:.4f}"
                f"  leak={leak:.3f}  therm={therm:.3f}"
            ),
        )
        plt.show()

    # =========================================================================
    # plot_pareto
    # =========================================================================

    def plot_pareto(self):
        """
        Scatter plot of all grid points in (leakage, fidelity) space with
        the Pareto front highlighted.

        Requires analyze(pareto=True) to have been called first.
        """
        if not hasattr(self, "_pareto_pts"):
            print("Run analyze(pareto=True) first.")
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        leak_all = self.leakage_array.ravel()
        fid_all = self.fid_Array.ravel()

        # All grid points (grey)
        ax.scatter(leak_all, fid_all, c="grey", s=20, alpha=0.4, label="Grid points")

        # Pareto front points (coloured)
        pareto_fid = [p[0] for p in self._pareto_pts]
        pareto_leak = [p[1] for p in self._pareto_pts]
        ax.scatter(
            pareto_leak,
            pareto_fid,
            c="tab:blue",
            s=60,
            zorder=3,
            label="Pareto front",
        )
        # Connect Pareto points with a step line (sorted by leakage)
        sorted_pairs = sorted(zip(pareto_leak, pareto_fid))
        ax.step(
            [p[0] for p in sorted_pairs],
            [p[1] for p in sorted_pairs],
            where="post",
            color="tab:blue",
            linewidth=1.2,
            alpha=0.6,
        )

        # Mark the overall best feasible point
        if hasattr(self, "_feasible_mask"):
            fid_feasible = np.where(self._feasible_mask, self.fid_Array, -np.inf)
            best_idx = np.unravel_index(np.argmax(fid_feasible), fid_feasible.shape)
            ax.scatter(
                self.leakage_array[best_idx],
                self.fid_Array[best_idx],
                c="red",
                s=100,
                zorder=4,
                marker="*",
                label="Best feasible",
            )

        ax.set_xlabel("Leakage (secondary |e> weight)", fontsize=11)
        ax.set_ylabel("GMM Fidelity", fontsize=11)
        ax.set_title("Pareto Front: Fidelity vs Leakage", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # plot_top_fidelity_histograms
    # =========================================================================

    def plot_top_fidelity_histograms(self, top_n=9, feasible_only=True):
        """
        IQ hexbin scatter plots for the top-N grid points by GMM fidelity.

        Parameters
        ----------
        top_n          : int  — number of top points to plot.
        feasible_only  : bool — if True (default), restrict the ranking to
                                points that passed the physical constraints.
                                Set to False to see all points including those
                                with high leakage / thermal.
        """
        if not hasattr(self, "fid_Array"):
            print("Running analyze() to generate fidelity data...")
            self.analyze()
            if not hasattr(self, "fid_Array"):
                print("Error: fidelity data not available.")
                return

        fid_Array = self.fid_Array
        if fid_Array.ndim != 3:
            print("Error: fid_Array must be 3-D (length, gain, freq).")
            return

        # Filter to feasible points if requested
        if feasible_only and hasattr(self, "_feasible_mask"):
            scored = np.where(self._feasible_mask, fid_Array, -np.inf)
        else:
            scored = fid_Array

        flat_scored = scored.flatten()
        top_n_flat_indices = np.argsort(flat_scored)[-top_n:][::-1]
        # Drop any -inf (infeasible) entries that sneak in
        top_n_flat_indices = [
            idx for idx in top_n_flat_indices if flat_scored[idx] > -np.inf
        ][:top_n]
        top_n_indices = np.unravel_index(top_n_flat_indices, fid_Array.shape)

        all_I = np.concatenate(
            [self.data["Ig"][idx] for idx in zip(*top_n_indices)]
            + [self.data["Ie"][idx] for idx in zip(*top_n_indices)]
        )
        all_Q = np.concatenate(
            [self.data["Qg"][idx] for idx in zip(*top_n_indices)]
            + [self.data["Qe"][idx] for idx in zip(*top_n_indices)]
        )

        overall_min = min(all_I.min(), all_Q.min())
        overall_max = max(all_I.max(), all_Q.max())
        span = (overall_max - overall_min) * 0.05
        plot_min = overall_min - span
        plot_max = overall_max + span
        plot_extent = [plot_min, plot_max, plot_min, plot_max]

        hexbin_gridsize = 50
        grid_size = int(np.ceil(np.sqrt(len(top_n_flat_indices))))
        fig, axes = plt.subplots(
            grid_size,
            grid_size,
            figsize=(5 * grid_size, 5 * grid_size),
        )
        axes = axes.flatten()

        feasible_suffix = "(feasible only)" if feasible_only else "(all points)"
        print(
            f"\nPlotting top {len(top_n_flat_indices)} fidelity points {feasible_suffix}..."
        )

        for i, (l_idx, g_idx, f_idx) in enumerate(zip(*top_n_indices)):
            I_g = self.data["Ig"][l_idx, g_idx, f_idx]
            Q_g = self.data["Qg"][l_idx, g_idx, f_idx]
            I_e = self.data["Ie"][l_idx, g_idx, f_idx]
            Q_e = self.data["Qe"][l_idx, g_idx, f_idx]

            current_fid = fid_Array[l_idx, g_idx, f_idx]
            soft_fid_val = self.soft_fid_array[l_idx, g_idx, f_idx]
            leak_val = self.leakage_array[l_idx, g_idx, f_idx]
            therm_val = self.thermal_array[l_idx, g_idx, f_idx]
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

            # Indicate infeasible points with a red border
            if hasattr(self, "_feasible_mask"):
                ok = bool(self._feasible_mask[l_idx, g_idx, f_idx])
                if not ok:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)

            title_str = (
                f"fid={current_fid:.4f}  soft={soft_fid_val:.4f}\n"
                f"L={length:.3f}us  G={gain:.5f}  F={freq:.5f}MHz\n"
                f"leak={leak_val:.3f}  therm={therm_val:.3f}"
            )
            ax.set_title(title_str, fontsize=9)
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

        for j in range(len(top_n_flat_indices), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
