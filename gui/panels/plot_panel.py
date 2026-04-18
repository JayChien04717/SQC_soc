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
        self.mode_combo.currentTextChanged.connect(self._replot)
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

    def clear(self):
        self._x = self._iq = None
        self.ax.cla()
        self._style_ax(self.ax)
        self.canvas.draw_idle()

    def export_figure(self):
        path, filt = QFileDialog.getSaveFileName(
            self, "Export Figure", "figure",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)")
        if path:
            self.fig.savefig(path, dpi=150, facecolor=BG0)

    # ── Internal ──────────────────────────────────────────────────────────────

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

        self.ax.cla()
        self._style_ax(self.ax)

        if y.ndim == 1:
            self.ax.plot(self._x, y, ".-", color=ACCENT, markersize=3, linewidth=1.2)
        else:
            im = self.ax.imshow(
                y, aspect="auto",
                extent=[self._x[0], self._x[-1], 0, y.shape[0]],
                origin="lower", cmap="RdBu_r",
            )
            self.fig.colorbar(im, ax=self.ax, pad=0.02)

        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(ylabel_map[mode])
        if self._title:
            self.ax.set_title(self._title, fontsize=9)

        if self.autoscale_cb.isChecked():
            self.ax.relim()
            self.ax.autoscale_view()

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
