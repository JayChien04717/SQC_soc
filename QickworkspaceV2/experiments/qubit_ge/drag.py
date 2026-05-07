"""
QubitGE/drag — s005a: DRAG calibration (alpha sweep, 2D scan).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...core.experiment_data import ExperimentData, QualityFlag
from ...tools.system_tool import hdf5_generator, get_next_filename_labber, config_to_yaml
from ...plotter.liveplot import liveplotfun


class DragProgram(BaseProgram):
    """QICK program for DRAG calibration using an ASMv2 hardware iteration loop."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.declare_gen_auto(cfg["qb_ch"], cfg["nqz_qb"], "qb_mixer", cfg)
        self.setup_qb_pulse(cfg, prefix="ge", shape="drag", name="x180_ge",
                            phase=0, gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, prefix="ge", shape="drag", name="mx180_ge",
                            phase=180, gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)

        # ASMv2 hardware loop. open_loop() allocates and initializes the
        # tProc counter; close_loop() increments it and jumps back as needed.
        self.open_loop(int(cfg["iteration"]), name="iter_loop")

        self.pulse(ch=cfg["qb_ch"], name="x180_ge", t=0)
        self.delay_auto(t=0.01)
        self.pulse(ch=cfg["qb_ch"], name="mx180_ge", t=0)
        self.delay_auto(t=0.01)

        self.close_loop()
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class DragCalibration(BaseExperiment):
    """
    DRAG Calibration: 2D sweep over (alpha x iteration) using liveplotfun 2-for-loop scan.
    """

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

    def _build_scan_axes(self):
        cfg = self.cfg
        if "alpha_start" not in cfg or "alpha_stop" not in cfg or "alpha_steps" not in cfg:
            raise ValueError("cfg must contain 'alpha_start', 'alpha_stop', 'alpha_steps'.")
        alphas = np.linspace(cfg["alpha_start"], cfg["alpha_stop"], cfg["alpha_steps"])
        if "iteration_start" in cfg and "iteration_stop" in cfg:
            iters = np.arange(
                cfg["iteration_start"], cfg["iteration_stop"] + 1,
                cfg.get("iteration_step", 1), dtype=float,
            )
        else:
            iters = np.array([float(cfg["iteration"])])
        return alphas, iters

    def run(self, py_avg, show_final_plot=False, **kwargs):
        alphas, iters = self._build_scan_axes()
        self._sweep_vals_x = alphas
        self._sweep_vals_y = iters

        def _make_prog(alpha_val, iter_val):
            self.cfg["drag_alpha"] = float(alpha_val)
            self.cfg["iteration"] = int(iter_val)
            return DragProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

        self.iqdata, interrupted, avg_count = liveplotfun(
            soc=self.soc,
            py_avg=py_avg,
            scan_x_axis=alphas,
            scan_y_axis=iters,
            get_prog_callback=_make_prog,
            x_label=self.X_LABEL,
            y_label=self.Y_LABEL,
            title_prefix=self.TITLE_PREFIX,
            show_final_plot=show_final_plot,
        )

        if self.iqdata is None:
            print("No data acquired.")
            result = ExperimentData(
                experiment_type=self.EXPT_NAME,
                quality=QualityFlag.BAD,
                quality_message="No data acquired",
                interrupted=True,
                avg_count=0,
            )
            self.result = result
            return result
        if interrupted:
            print(f"Interrupted at avg {avg_count}.")

        fit_result = self._post_fit(alphas) or {}
        optimal_alpha = fit_result.get("optimal_alpha")
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=alphas,
            y_axis=iters,
            fit_params=self.fit_params,
            fit_errors=self.fit_errors,
            fit_result={k: (v, None) for k, v in fit_result.items()},
            scalar_result=float(optimal_alpha) if optimal_alpha is not None else None,
            quality=QualityFlag.NO_INFORMATION,
            config=dict(self.cfg) if hasattr(self.cfg, "__iter__") else {},
            interrupted=interrupted,
            avg_count=avg_count,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
        )
        self.result = result
        return result

    def _create_program(self):
        self.cfg.setdefault("drag_alpha", self.cfg.get("alpha_start", 0.5))
        self.cfg.setdefault("iteration", self.cfg.get("iteration_start", 1))
        return DragProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return self._sweep_vals_x

    def _extract_sweep_axis_y(self, prog):
        return self._sweep_vals_y

    def analyze_and_plot(self):
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        alphas = self._sweep_vals_x
        iters = self._sweep_vals_y
        sum_trace = np.sum(np.abs(self.iqdata), axis=0)

        idx_max = int(np.argmax(sum_trace))
        idx_min = int(np.argmin(sum_trace))
        optimal_alpha_max = alphas[idx_max]
        optimal_alpha_min = alphas[idx_min]

        try:
            def parabola(x, a, b, c):
                return a * (x - b) ** 2 + c

            for idx_pk, label in [(idx_max, "max"), (idx_min, "min")]:
                i0 = max(idx_pk - 3, 0)
                i1 = min(idx_pk + 4, len(alphas))
                popt, _ = curve_fit(
                    parabola, alphas[i0:i1], sum_trace[i0:i1],
                    p0=[1.0, alphas[idx_pk], sum_trace[idx_pk]],
                )
                if alphas.min() <= popt[1] <= alphas.max():
                    if label == "max":
                        optimal_alpha_max = popt[1]
                    else:
                        optimal_alpha_min = popt[1]
        except Exception:
            pass

        baseline = np.median(sum_trace)
        dev_max = abs(sum_trace[idx_max] - baseline)
        dev_min = abs(sum_trace[idx_min] - baseline)
        optimal_alpha = optimal_alpha_max if dev_max >= dev_min else optimal_alpha_min
        print(f"\n[DRAG Sum] Optimal α = {optimal_alpha:.6f})")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax0 = axes[0]
        im = ax0.pcolormesh(alphas, iters, np.abs(self.iqdata), shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax0, label="ADC Units (Abs)")
        ax0.axvline(optimal_alpha, color="red", linestyle="--", linewidth=2,
                    label=f"α={optimal_alpha:.4f}")
        ax0.set_xlabel(self.X_LABEL)
        ax0.set_ylabel(self.Y_LABEL)
        ax0.set_title("Drag calibration")
        ax0.legend(fontsize=9)

        ax1 = axes[1]
        ax1.plot(alphas, sum_trace, "o-", color="steelblue", linewidth=2,
                 markersize=5, label="Σ iterations")
        ax1.axvline(optimal_alpha, color="red", linestyle="--", linewidth=2,
                    label=f"α={optimal_alpha:.4f}")
        ax1.set_xlabel(self.X_LABEL)
        ax1.set_ylabel("ADC Units")
        ax1.set_title("Summed Trace")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        fig.suptitle(self.TITLE_PREFIX, fontsize=13)
        fig.tight_layout()
        plt.show()

        optimal_alpha = round(float(optimal_alpha), 6)
        self.fit_params = np.array([optimal_alpha])
        self.fit_errors = None
        self._drag_fit_result = {"optimal_alpha": optimal_alpha}
        return self._drag_fit_result

    def _save_comment(self, dict_val):
        fit_result = getattr(self, "_drag_fit_result", None)
        if fit_result:
            a = fit_result.get("optimal_alpha", "N/A")
            return f"DRAG Calibration\nOptimal alpha = {a}\n{dict_val}"
        if self.result is not None:
            a = self.result.get_param("optimal_alpha", "N/A")
            return f"DRAG Calibration\nOptimal alpha = {a}\n{dict_val}"
        return f"{dict_val}"


__all__ = ["DragProgram", "DragCalibration"]
