from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QCheckBox, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib import cm, colors as mcolors

from gui.theme import BG0, BG1, BG3, ACCENT, TEXT, TEXT_DIM


ALLXY_GATE_LABELS = [
    "I,I", "X,X", "Y,Y", "X,Y", "Y,X",
    "X/2,I", "Y/2,I", "X/2,Y/2", "Y/2,X/2", "X/2,Y",
    "Y/2,X", "X,Y/2", "Y,X/2", "X/2,X", "X,X/2",
    "Y/2,Y", "Y,Y/2", "X,I", "Y,I", "X/2,X/2", "Y/2,Y/2",
]


class PlotPanel(QWidget):
    """Live matplotlib canvas with IQ-mode selector and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x      = None
        self._iq     = None
        self._y_axis = None
        self._xlabel = ""
        self._title  = ""
        self._experiment_type = ""
        self._line   = None   # reused Line2D for liveplot
        self._fit_line = None
        self._fit_text = None
        self._mesh   = None   # reused QuadMesh for 2D liveplot
        self._cbar   = None   # colorbar reference for 2D
        self._single_shot_axes = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)

        # Controls bar
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Channel:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["mag", "phase", "avgi", "avgq"])
        self.mode_combo.setToolTip("IQ channel to display")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        ctrl.addWidget(self.mode_combo)

        self.autoscale_cb = QCheckBox("Autoscale")
        self.autoscale_cb.setChecked(True)
        ctrl.addWidget(self.autoscale_cb)

        ctrl.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setToolTip("Clear the plot")
        clear_btn.clicked.connect(self.clear)
        ctrl.addWidget(clear_btn)

        export_btn = QPushButton("Export…")
        export_btn.setFixedWidth(70)
        export_btn.setToolTip("Save figure as PNG / PDF / SVG")
        export_btn.clicked.connect(self.export_figure)
        ctrl.addWidget(export_btn)

        root.addLayout(ctrl)

        # Matplotlib canvas + toolbar
        self.fig = Figure(tight_layout=True)
        self.fig.patch.set_facecolor(BG0)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax(self.ax)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {BG0};")

        toolbar = NavToolbar(self.canvas, self)
        root.addWidget(toolbar)
        root.addWidget(self.canvas)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_data(self, x, iq, xlabel: str = "x", title: str = ""):
        """Refresh plot with new IQ data."""
        self._x      = np.asarray(x)
        self._iq     = np.asarray(iq)
        self._y_axis = None
        self._xlabel = xlabel
        self._title  = title
        self._experiment_type = title
        self._replot()

    def set_result(self, result):
        """Add a fitted curve and result summary after acquisition finishes."""
        if result is None:
            return
        if result.x_axis is not None:
            self._x = np.asarray(result.x_axis)
        if result.raw_iq is not None:
            self._iq = np.asarray(result.raw_iq)
        self._y_axis = np.asarray(result.y_axis) if getattr(result, "y_axis", None) is not None else None
        self._experiment_type = str(getattr(result, "experiment_type", "") or self._title or "")
        self._replot()
        self._draw_fit(result)

    def clear_fit(self):
        """Remove the final fit overlay while keeping live data."""
        if self._fit_line is not None:
            try:
                self._fit_line.remove()
            except ValueError:
                pass
            self._fit_line = None
        if self._fit_text is not None:
            try:
                self._fit_text.remove()
            except ValueError:
                pass
            self._fit_text = None
        self.canvas.draw_idle()

    def clear(self):
        self._x = self._iq = self._y_axis = None
        self._experiment_type = ""
        self._line = self._fit_line = self._mesh = None
        self._fit_text = None
        self._reset_single_axis()
        self.canvas.draw_idle()

    def _draw_fit(self, result):
        self.clear_fit()
        if self._x is None or self._iq is None:
            self._draw_fit_text(result)
            self.canvas.draw_idle()
            return
        if np.asarray(self._iq).ndim > 1:
            self._draw_fit_text(result)
            self.canvas.draw_idle()
            return

        y_fit = self._fit_curve(result)
        if y_fit is not None:
            x = np.asarray(self._x, dtype=float)
            self._fit_line, = self.ax.plot(
                x,
                y_fit,
                "-",
                color="#f2cc60",
                linewidth=2.0,
                alpha=0.95,
                label="fit",
            )
            self.ax.legend(loc="best", facecolor=BG1, edgecolor=BG3, labelcolor=TEXT_DIM)
        self._draw_fit_text(result)
        if self.autoscale_cb.isChecked():
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _fit_curve(self, result):
        x = np.asarray(self._x, dtype=float)
        fit_result = getattr(result, "fit_result", {}) or {}
        try:
            from QickworkspaceV2.tools.fitting import decaysin, expfunc, fitlor, lorfunc, rb_func
            fit_params = getattr(result, "fit_params", None)
            if fit_params is None and any(k in fit_result for k in ("f_res[MHz]", "kappa_MHz")):
                p, _, _ = fitlor(x, np.abs(np.asarray(self._iq)))
                return lorfunc(x, *p)
            if fit_params is None:
                return None
            p = np.asarray(fit_params, dtype=float)
            if len(p) == 4 and (
                "f0_MHz" in fit_result
                or "f_res[MHz]" in fit_result
                or "linewidth_MHz" in fit_result
                or "kappa_MHz" in fit_result
            ):
                return lorfunc(x, *p)
            if len(p) == 3 and any(k in fit_result for k in ("T1_us", "T2r_us", "T2e_us")):
                return expfunc(x, *p)
            if len(p) == 5 and any(k in fit_result for k in ("pi_gain", "pi_length_us", "T2r_us", "T2e_us")):
                return decaysin(x, *p)
            if len(p) == 3 and any(k in fit_result for k in ("p", "EPC", "epc")):
                return rb_func(x, *p)
        except Exception:
            return None
        return None

    def _draw_fit_text(self, result):
        fit_result = getattr(result, "fit_result", {}) or {}
        if not fit_result:
            return
        lines = []
        for key, raw in fit_result.items():
            value = raw[0] if isinstance(raw, (tuple, list)) and raw else raw
            if isinstance(value, (int, float, np.number)):
                lines.append(f"{key}: {float(value):.6g}")
            else:
                lines.append(f"{key}: {value}")
        if not lines:
            return
        if getattr(result, "interrupted", False):
            lines.insert(0, f"partial avg: {getattr(result, 'avg_count', 0)}")
        text_kwargs = {
            "transform": self.ax.transAxes,
            "va": "top",
            "ha": "left",
            "fontsize": 9,
            "color": TEXT,
            "bbox": {"boxstyle": "round,pad=0.35", "facecolor": BG0, "edgecolor": BG3, "alpha": 0.88},
        }
        if getattr(self.ax, "name", "") == "3d":
            self._fit_text = self.ax.text2D(0.02, 0.98, "\n".join(lines[:8]), **text_kwargs)
        else:
            self._fit_text = self.ax.text(0.02, 0.98, "\n".join(lines[:8]), **text_kwargs)

    def export_figure(self):
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Figure", "figure",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)")
        if path:
            self.fig.savefig(path, dpi=150, facecolor=BG0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        """Force full redraw when IQ channel selection changes."""
        self._line = self._mesh = None
        self._replot()

    def _reset_single_axis(self):
        self.fig.clear()
        self.fig.patch.set_facecolor(BG0)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)
        self._cbar = None
        self._single_shot_axes = None

    def _ensure_single_axis(self):
        if self._single_shot_axes is not None or self.ax not in self.fig.axes:
            self._reset_single_axis()

    @staticmethod
    def _looks_like_singleshot_iq(iq, experiment_type=""):
        arr = np.asarray(iq)
        context = str(experiment_type or "").lower()
        return (
            "single" in context and
            np.iscomplexobj(arr)
            and arr.ndim == 2
            and 1 < arr.shape[0] <= 4
            and arr.shape[1] >= 10
            and arr.shape[0] <= arr.shape[1]
        )

    @staticmethod
    def _looks_like_tomography(iq, experiment_type=""):
        arr = np.asarray(iq)
        context = str(experiment_type or "").lower()
        return "tomo" in context and arr.ndim == 1 and arr.shape[0] == 3

    @staticmethod
    def _looks_like_allxy(iq, experiment_type=""):
        arr = np.asarray(iq)
        context = str(experiment_type or "").lower()
        return "allxy" in context and arr.ndim == 1 and arr.shape[0] == len(ALLXY_GATE_LABELS)

    def _plot_allxy(self):
        iq = np.asarray(self._iq)
        y = np.real(iq) if self.mode_combo.currentText() == "avgi" else np.abs(iq)
        x = np.arange(len(y), dtype=float)
        if y[0] < y[-1]:
            ref = [np.nanmin(y)] * 5 + [(np.nanmax(y) + np.nanmin(y)) / 2] * 12 + [np.nanmax(y)] * 4
        else:
            ref = [np.nanmax(y)] * 5 + [(np.nanmax(y) + np.nanmin(y)) / 2] * 12 + [np.nanmin(y)] * 4
        if len(ref) != len(y):
            ref = ref[:len(y)] if len(ref) > len(y) else ref + [ref[-1]] * (len(y) - len(ref))

        self._reset_single_axis()
        self._line, = self.ax.plot(x, y, "o", color=ACCENT, markersize=4, label="data")
        self.ax.plot(x, ref, "-", color="#f2cc60", linewidth=1.8, label="ideal")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(ALLXY_GATE_LABELS, rotation=55, ha="right", fontsize=8)
        self.ax.set_xlabel("Gate set")
        self.ax.set_ylabel("I (ADC)" if self.mode_combo.currentText() == "avgi" else "|IQ| (ADC)")
        self.ax.set_title(self._title or "AllXY", fontsize=9)
        self.ax.grid(True, color=BG3, alpha=0.35, linewidth=0.6)
        self.ax.legend(loc="best", facecolor=BG1, edgecolor=BG3, labelcolor=TEXT_DIM)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_tomography_expectations(self):
        if self._y_axis is not None and np.asarray(self._y_axis).shape == (2, 2):
            self._plot_tomography_density_matrix()
            return
        vals = np.asarray(self._iq, dtype=float)
        self._reset_single_axis()
        labels = ["<X>", "<Y>", "<Z>"]
        colors = [ACCENT, "#f2cc60", "#ff7b72"]
        self.ax.bar(np.arange(3), vals, color=colors, alpha=0.78, edgecolor=BG0, linewidth=0.8)
        self.ax.axhline(0.0, color=BG3, linewidth=1.0)
        self.ax.set_xticks(np.arange(3), labels)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_ylabel("Expectation")
        self.ax.set_title(self._title or "State Tomography", fontsize=9)
        for idx, val in enumerate(vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.04 if val >= 0 else -0.04
            self.ax.text(idx, val + offset, f"{val:.3f}", ha="center", va=va, color=TEXT, fontsize=9)
        self.canvas.draw_idle()

    def _plot_tomography_density_matrix(self):
        rho = np.asarray(self._y_axis, dtype=np.complex128)
        magnitude = np.abs(rho).ravel()
        phase = np.angle(rho).ravel()
        xpos, ypos = np.meshgrid(np.arange(2), np.arange(2), indexing="xy")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos, dtype=float)
        dx = dy = np.full_like(xpos, 0.58, dtype=float)
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        facecolors = cm.twilight(norm(phase))

        self.fig.clear()
        self.fig.patch.set_facecolor(BG0)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._single_shot_axes = None
        self._line = self._mesh = self._fit_line = self._fit_text = None
        self._cbar = None
        self.ax.set_facecolor(BG1)
        self.ax.bar3d(xpos, ypos, zpos, dx, dy, magnitude, color=facecolors, shade=True, alpha=0.92)
        self.ax.set_zlim(-1.0, 1.0)
        self.ax.set_xticks([0.29, 1.29])
        self.ax.set_xticklabels(["|0>", "|1>"])
        self.ax.set_yticks([0.29, 1.29])
        self.ax.set_yticklabels(["<0|", "<1|"])
        self.ax.set_zlabel("|rho|")
        self.ax.set_title(self._title or "State Tomography", fontsize=9, color=TEXT)
        self.ax.tick_params(colors=TEXT_DIM, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_DIM)
        self.ax.yaxis.label.set_color(TEXT_DIM)
        self.ax.zaxis.label.set_color(TEXT_DIM)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.set_facecolor(BG1)
            axis.pane.set_edgecolor(BG3)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.twilight)
        mapper.set_array([])
        self._cbar = self.fig.colorbar(mapper, ax=self.ax, pad=0.08, shrink=0.72)
        self._cbar.set_label("phase (rad)", color=TEXT_DIM)
        self._cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
        self._cbar.outline.set_edgecolor(BG3)
        for x, y, z, mag, ph in zip(xpos, ypos, magnitude, magnitude, phase):
            self.ax.text(x + 0.29, y + 0.29, z + 0.04, f"{mag:.2f}\n{ph:.2f}", color=TEXT, ha="center", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_singleshot_histograms(self):
        iq = np.asarray(self._iq, dtype=np.complex128)
        n_states = iq.shape[0]
        labels = ["g", "e", "f", "h"][:n_states]
        colors = [ACCENT, "#f2cc60", "#ff7b72", "#a371f7"][:n_states]

        centers = np.nanmean(iq, axis=1)
        delta = centers[1] - centers[0] if n_states >= 2 else 0.0
        angle = np.angle(delta) if np.isfinite(delta) and abs(delta) > 0 else 0.0
        rotated = iq * np.exp(-1j * angle)

        self.fig.clear()
        self.fig.patch.set_facecolor(BG0)
        ax_iq = self.fig.add_subplot(2, 1, 1)
        ax_i = self.fig.add_subplot(2, 1, 2)
        self.ax = ax_iq
        self._single_shot_axes = (ax_iq, ax_i)
        self._line = self._mesh = self._fit_line = self._fit_text = None
        self._cbar = None
        self._style_ax(ax_iq)
        self._style_ax(ax_i)

        flat_iq = iq.ravel()
        flat_iq = flat_iq[np.isfinite(flat_iq.real) & np.isfinite(flat_iq.imag)]
        if flat_iq.size:
            bins_2d = min(140, max(40, int(np.sqrt(flat_iq.size) * 1.5)))
            _, _, _, image = ax_iq.hist2d(
                flat_iq.real,
                flat_iq.imag,
                bins=bins_2d,
                cmap="inferno",
            )
            self._cbar = self.fig.colorbar(image, ax=ax_iq, pad=0.02)
            self._cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
            self._cbar.outline.set_edgecolor(BG3)

        for idx, center in enumerate(centers):
            if not (np.isfinite(center.real) and np.isfinite(center.imag)):
                continue
            ax_iq.plot(
                center.real,
                center.imag,
                "o",
                color=colors[idx],
                markersize=5,
                markeredgecolor=BG0,
                markeredgewidth=0.8,
            )
            ax_iq.text(
                center.real,
                center.imag,
                f" {labels[idx]}",
                color=colors[idx],
                fontsize=9,
                va="center",
                ha="left",
            )

        projection = rotated.real
        finite_projection = projection[np.isfinite(projection)]
        if finite_projection.size:
            lo, hi = np.nanpercentile(finite_projection, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                center = float(np.nanmean(finite_projection))
                span = max(abs(center) * 0.1, 1.0)
                lo, hi = center - span, center + span
            bins_1d = np.linspace(lo, hi, min(90, max(35, int(np.sqrt(finite_projection.size)))))
            for idx in range(n_states):
                vals = projection[idx]
                vals = vals[np.isfinite(vals)]
                ax_i.hist(
                    vals,
                    bins=bins_1d,
                    alpha=0.45,
                    color=colors[idx],
                    label=labels[idx],
                    histtype="stepfilled",
                    edgecolor=colors[idx],
                    linewidth=1.1,
                )
                if vals.size:
                    ax_i.axvline(np.nanmean(vals), color=colors[idx], linewidth=1.2, alpha=0.95)
            ax_i.legend(loc="best", facecolor=BG1, edgecolor=BG3, labelcolor=TEXT_DIM)

        ax_iq.set_xlabel("I (ADC)")
        ax_iq.set_ylabel("Q (ADC)")
        ax_iq.set_title(f"{self._title or 'SingleShot'} - IQ 2D histogram", fontsize=9)
        ax_i.set_xlabel("Rotated I (ADC)")
        ax_i.set_ylabel("Counts")
        ax_i.set_title(f"I-axis projection after rotation ({np.degrees(angle):.1f} deg)", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _replot(self):
        if self._x is None or self._iq is None:
            return
        if self._looks_like_allxy(self._iq, self._experiment_type or self._title):
            self._plot_allxy()
            return
        if self._looks_like_tomography(self._iq, self._experiment_type or self._title):
            self._plot_tomography_expectations()
            return
        if self._looks_like_singleshot_iq(self._iq, self._experiment_type or self._title):
            self._plot_singleshot_histograms()
            return
        self._ensure_single_axis()

        mode = self.mode_combo.currentText()
        ylabel_map = {
            "mag":   "|IQ| (ADC)",
            "phase": "Phase (deg)",
            "avgi":  "I (ADC)",
            "avgq":  "Q (ADC)",
        }
        extractors = {
            "mag":   lambda q: np.abs(q),
            "phase": lambda q: np.degrees(np.angle(q)),
            "avgi":  lambda q: np.real(q),
            "avgq":  lambda q: np.imag(q),
        }
        y = extractors[mode](self._iq)
        is_2d = y.ndim > 1
        x_plot = self._x
        if not is_2d and x_plot.shape[0] != y.shape[0]:
            x_plot = np.arange(y.shape[0], dtype=float)
        elif is_2d and x_plot.shape[0] != y.shape[-1]:
            x_plot = np.arange(y.shape[-1], dtype=float)

        # Determine if we can update in-place (same dimensionality and shape)
        can_update = (
            (not is_2d and self._line is not None and
             self._line.get_xdata().shape == x_plot.shape)
            or
            (is_2d and self._mesh is not None and
             self._mesh.get_array().shape == y.ravel().shape)
        )

        if can_update:
            if is_2d:
                self._mesh.set_array(y.ravel())
                self._mesh.set_clim(y.min(), y.max())
            else:
                self._line.set_ydata(y)
                if self.autoscale_cb.isChecked():
                    self.ax.relim()
                    self.ax.autoscale_view()
            self.ax.set_title(self._title or "", fontsize=9)
        else:
            # Full redraw on first call or shape change
            self._line = self._mesh = None
            if self._cbar is not None:
                self._cbar.remove()
                self._cbar = None
            self.ax.cla()
            self._style_ax(self.ax)

            if is_2d:
                self._mesh = self.ax.pcolormesh(
                    x_plot, np.arange(y.shape[0]), y,
                    cmap="RdBu_r", shading="auto",
                )
                self._cbar = self.fig.colorbar(self._mesh, ax=self.ax, pad=0.02)
            else:
                (self._line,) = self.ax.plot(
                    x_plot, y, ".-", color=ACCENT, markersize=3, linewidth=1.2,
                )
                if self.autoscale_cb.isChecked():
                    self.ax.relim()
                    self.ax.autoscale_view()

            self.ax.set_xlabel(self._xlabel)
            self.ax.set_ylabel(ylabel_map[mode])
            self.ax.set_title(self._title or "", fontsize=9)

        self.canvas.draw_idle()

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG3)
