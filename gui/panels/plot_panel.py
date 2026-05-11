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

from gui.theme import BG0, BG1, BG3, ACCENT, TEXT, TEXT_DIM


class PlotPanel(QWidget):
    """Live matplotlib canvas with IQ-mode selector and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x      = None
        self._iq     = None
        self._xlabel = ""
        self._title  = ""
        self._line   = None   # reused Line2D for liveplot
        self._fit_line = None
        self._fit_text = None
        self._mesh   = None   # reused QuadMesh for 2D liveplot
        self._cbar   = None   # colorbar reference for 2D
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
        self._xlabel = xlabel
        self._title  = title
        self._replot()

    def set_result(self, result):
        """Add a fitted curve and result summary after acquisition finishes."""
        if result is None:
            return
        if result.x_axis is not None:
            self._x = np.asarray(result.x_axis)
        if result.raw_iq is not None:
            self._iq = np.asarray(result.raw_iq)
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
        self._x = self._iq = None
        self._line = self._fit_line = self._mesh = None
        self._fit_text = None
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self.ax.cla()
        self._style_ax(self.ax)
        self.canvas.draw_idle()

    def _draw_fit(self, result):
        self.clear_fit()
        if self._x is None or self._iq is None:
            self._draw_fit_text(result)
            return
        if np.asarray(self._iq).ndim > 1:
            self._draw_fit_text(result)
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
        self._fit_text = self.ax.text(
            0.02,
            0.98,
            "\n".join(lines[:8]),
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": BG0, "edgecolor": BG3, "alpha": 0.88},
        )

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

    def _replot(self):
        if self._x is None or self._iq is None:
            return

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
