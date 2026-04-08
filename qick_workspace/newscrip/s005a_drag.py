"""
s005a — DRAG Calibration (Alpha Sweep)
=====================================
Sweeps the DRAG parameter alpha over a 2D grid of (alpha, iter) using the
two-for-loop liveplotfun (scan_x_axis + scan_y_axis).
For each iteration row, finds the alpha that maximises the signal amplitude.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .base_program import BaseProgram
from .base_experiment import BaseExperiment
from ..tools.system_cfg import DATA_PATH
from ..tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ..plotter.liveplot import liveplotfun


# ── Program ──────────────────────────────────────────────────────────────────


class DragProgram(BaseProgram):
    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.declare_gen_auto(cfg["qb_ch"], cfg["nqz_qb"], "qb_mixer", cfg)

        # x180 / mx180: always use DRAG-shaped envelope (shape="drag"),
        # but pulse_type (arb / flat_top / ...) follows cfg["pulse_type"].
        self.setup_qb_pulse(
            cfg,
            prefix="ge",
            shape="drag",
            name="x180_ge",
            phase=0,
            gain_key="pi_gain_ge",
        )
        self.setup_qb_pulse(
            cfg,
            prefix="ge",
            shape="drag",
            name="mx180_ge",
            phase=180,
            gain_key="pi_gain_ge",
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        iterations = int(cfg.get("iter", 10))
        for _ in range(iterations):
            self.pulse(ch=cfg["qb_ch"], name="x180_ge", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=cfg["qb_ch"], name="mx180_ge", t=0)
            self.delay_auto(0.01)

        self.measure(cfg)


# ── Experiment ────────────────────────────────────────────────────────────────


class DragCalibration(BaseExperiment):
    """DRAG Calibration: 2D sweep over (alpha × iteration) using liveplotfun 2-for-loop scan."""

    EXPT_NAME = "s005a_drag_ge"
    TAG = "DRAGCalibration"
    X_LABEL = "DRAG Parameter (α)"
    Y_LABEL = "Iterations (N)"
    TITLE_PREFIX = "DRAG Calibration"
    SWEEP_KEYS_TO_REMOVE = []

    X_SAVE_NAME = "Alpha"
    X_SAVE_UNIT = "a.u."
    X_SAVE_SCALE = 1.0

    Y_SAVE_NAME = "Iterations"
    Y_SAVE_UNIT = "N"
    Y_SAVE_SCALE = 1.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_scan_axes(self):
        cfg = self.cfg
        if (
            "alpha_start" not in cfg
            or "alpha_stop" not in cfg
            or "alpha_steps" not in cfg
        ):
            raise ValueError(
                "cfg must contain 'alpha_start', 'alpha_stop', 'alpha_steps'."
            )

        alphas = np.linspace(cfg["alpha_start"], cfg["alpha_stop"], cfg["alpha_steps"])

        if "iter_start" in cfg and "iter_stop" in cfg:
            iters = np.arange(
                cfg["iter_start"],
                cfg["iter_stop"] + 1,
                cfg.get("iter_step", 1),
                dtype=float,
            )
        else:
            iters = np.array([float(cfg.get("iter", 10))])

        return alphas, iters

    # ── override run() ───────────────────────────────────────────────────────

    def run(self, py_avg, show_final_plot=False, **kwargs):
        """
        2-D parameter scan: outer loop = iter (N), inner loop = alpha.
        Delegates to liveplotfun's _liveplot_2d_scan via scan_x_axis + scan_y_axis.
        """
        alphas, iters = self._build_scan_axes()
        self._sweep_vals_x = alphas
        self._sweep_vals_y = iters

        # ── hardware: 2-for-loop scan ─────────────────────────────────────────
        def _make_prog(alpha_val, iter_val):
            self.cfg["drag_alpha"] = float(alpha_val)
            self.cfg["iter"] = int(iter_val)
            return DragProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

        self.iqdata, interrupted, avg_count = liveplotfun(
            soc=self.soc,
            py_avg=py_avg,
            scan_x_axis=alphas,  # inner loop → alpha
            scan_y_axis=iters,  # outer loop → N
            get_prog_callback=_make_prog,
            x_label=self.X_LABEL,
            y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX,
            show_final_plot=show_final_plot,
        )

        if self.iqdata is None:
            print("No data acquired.")
            return None
        if interrupted:
            print(f"Interrupted at avg {avg_count}.")

        return self._post_fit(alphas)

    # ── _create_program / _extract_sweep_axis (required by BaseExperiment) ───

    def _create_program(self):
        """Not used directly (run() is overridden), but required by ABC."""
        self.cfg.setdefault("drag_alpha", self.cfg.get("alpha_start", 0.5))
        self.cfg.setdefault("iter", self.cfg.get("iter_start", 1))
        return DragProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Not used directly (run() is overridden)."""
        return self._sweep_vals_x

    def _extract_sweep_axis_y(self, prog):
        """Not used directly (run() is overridden)."""
        return self._sweep_vals_y

    # ── analysis ─────────────────────────────────────────────────────────────

    def analyze_and_plot(self):
        """Backward-compatible alias for _post_fit()."""
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        """
        Sum all iteration traces (abs), then find the alpha at the
        max (or min) of the summed trace — mimics error-amplification logic.
        Returns: optimal_alpha (float).
        """
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        alphas = self._sweep_vals_x
        iters = self._sweep_vals_y

        # ── Sum every iteration row ───────────────────────────────────────────
        sum_trace = np.sum(np.abs(self.iqdata), axis=0)  # shape: (n_alpha,)

        # Peak: index of maximum in the summed trace
        idx_max = int(np.argmax(sum_trace))
        idx_min = int(np.argmin(sum_trace))
        optimal_alpha_max = alphas[idx_max]
        optimal_alpha_min = alphas[idx_min]

        # Parabola sub-pixel refinement around the max peak
        try:

            def parabola(x, a, b, c):
                return a * (x - b) ** 2 + c

            for idx_pk, label in [(idx_max, "max"), (idx_min, "min")]:
                i0 = max(idx_pk - 3, 0)
                i1 = min(idx_pk + 4, len(alphas))
                popt, _ = curve_fit(
                    parabola,
                    alphas[i0:i1],
                    sum_trace[i0:i1],
                    p0=[1.0, alphas[idx_pk], sum_trace[idx_pk]],
                )
                if alphas.min() <= popt[1] <= alphas.max():
                    if label == "max":
                        optimal_alpha_max = popt[1]
                    else:
                        optimal_alpha_min = popt[1]
        except Exception:
            pass

        # ── Pick the larger peak relative to baseline ────────────────────────────
        baseline = np.median(sum_trace)
        dev_max = abs(sum_trace[idx_max] - baseline)
        dev_min = abs(sum_trace[idx_min] - baseline)

        if dev_max >= dev_min:
            optimal_alpha = optimal_alpha_max
        else:
            optimal_alpha = optimal_alpha_min
        print(f"\n[DRAG Sum] Optimal α = {optimal_alpha:.6f})")

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: 2D heatmap
        ax0 = axes[0]
        im = ax0.pcolormesh(
            alphas, iters, np.abs(self.iqdata), shading="auto", cmap="viridis"
        )
        fig.colorbar(im, ax=ax0, label="ADC Units (Abs)")
        ax0.axvline(
            optimal_alpha,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"α={optimal_alpha:.4f}",
        )
        ax0.set_xlabel(self.X_LABEL)
        ax0.set_ylabel(self.Y_LABEL)
        ax0.set_title("Drag calibration")
        ax0.legend(fontsize=9)

        # Right: summed trace
        ax1 = axes[1]
        ax1.plot(
            alphas,
            sum_trace,
            "o-",
            color="steelblue",
            linewidth=2,
            markersize=5,
            label="Σ iterations",
        )
        ax1.axvline(
            optimal_alpha,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"α={optimal_alpha:.4f}",
        )
        ax1.set_xlabel(self.X_LABEL)
        ax1.set_ylabel("ADC Units")
        ax1.set_title("Summed Trace")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        fig.suptitle(self.TITLE_PREFIX, fontsize=13)
        fig.tight_layout()
        plt.show()

        self.fit_params = {
            "optimal_alpha": round(float(optimal_alpha), 6),
        }
        return self.fit_params

    def _save_comment(self, dict_val):
        if self.fit_params:
            a = self.fit_params.get("optimal_alpha", "N/A")
            return f"DRAG Calibration\nOptimal alpha = {a}\n{dict_val}"
        return f"{dict_val}"
