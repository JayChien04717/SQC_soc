"""
Standalone Data Browser window.

    conda activate scqenv
    python gui/data_browser_app.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QButtonGroup, QRadioButton, QGroupBox,
    QTextEdit, QTabWidget, QFileDialog, QSizePolicy, QMessageBox,
)
from PySide6.QtCore import Qt, QSize, QSettings
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

try:
    from qick_workspace.tools.data_manager import load_data, list_data_files
    _HAS_DM = True
except ImportError:
    _HAS_DM = False

from gui import theme
from gui.theme import BG0, BG1, BG3, ACCENT, TEXT, TEXT_DIM


class DataBrowserWindow(QMainWindow):
    """Standalone full-screen data browser."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Browser  ·  qick_workspace")
        self.resize(1400, 900)
        self._current_data = None
        self._current_path = None
        self._all_files    = []
        self._build_ui()
        self._setup_shortcuts()
        self._restore_geometry()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 1080])
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Open a folder to browse data files.")

    # ── Left panel: file tree ─────────────────────────────────────────────────

    def _build_left(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        # Folder row
        folder_row = QHBoxLayout()
        self.folder_label = QLineEdit()
        self.folder_label.setReadOnly(True)
        self.folder_label.setPlaceholderText("No folder selected")
        folder_row.addWidget(self.folder_label)
        open_btn = QPushButton("Open…")
        open_btn.setFixedWidth(60)
        open_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(open_btn)
        lay.addLayout(folder_row)

        # Filter row
        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter name…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit)
        self.qubit_combo = QComboBox()
        self.qubit_combo.addItems(["All", "Q0", "Q1", "Q2", "Q3"])
        self.qubit_combo.currentTextChanged.connect(self._apply_filter)
        self.qubit_combo.setFixedWidth(60)
        filter_row.addWidget(self.qubit_combo)
        lay.addLayout(filter_row)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Time"])
        self.tree.setColumnWidth(0, 190)
        self.tree.setSortingEnabled(False)
        self.tree.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.tree)

        # Refresh + Delete row
        action_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-scan the folder  (F5)")
        refresh_btn.clicked.connect(self._refresh)
        action_row.addWidget(refresh_btn)
        self._del_btn = QPushButton("Delete")
        self._del_btn.setToolTip("Delete the selected file  (Del)")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._delete_current)
        action_row.addWidget(self._del_btn)
        lay.addLayout(action_row)
        return w

    # ── Right panel: plot + info tabs ─────────────────────────────────────────

    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        # Controls row: channel + plot type
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self._ch_bg = QButtonGroup(self)
        for i, ch in enumerate(["mag", "phase", "avgi", "avgq"]):
            rb = QRadioButton(ch)
            rb.setFixedWidth(66)
            if i == 0:
                rb.setChecked(True)
            self._ch_bg.addButton(rb, i)
            ch_row.addWidget(rb)
        self._ch_bg.idToggled.connect(
            lambda _id, checked: self._replot() if checked else None
        )
        ch_row.addSpacing(16)
        ch_row.addWidget(QLabel("Plot:"))
        self._plot_type = QComboBox()
        self._plot_type.addItems(["Scatter", "Hist 1D", "Hist 2D"])
        self._plot_type.setFixedWidth(90)
        self._plot_type.currentTextChanged.connect(lambda _: self._replot())
        ch_row.addWidget(self._plot_type)
        export_btn = QPushButton("Export…")
        export_btn.setFixedWidth(72)
        export_btn.setToolTip("Save figure as PNG / PDF / SVG  (Ctrl+S)")
        export_btn.clicked.connect(self._export_figure)
        ch_row.addWidget(export_btn)
        lay.addLayout(ch_row)

        # Matplotlib canvas + toolbar
        self.fig = Figure(tight_layout=True)
        self.fig.patch.set_facecolor(BG0)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax(self.ax)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {BG0};")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(self.canvas, self)
        lay.addWidget(toolbar)
        lay.addWidget(self.canvas, stretch=3)

        # Info tabs (Metadata + Config JSON)
        self.info_tabs = QTabWidget()
        self.info_tabs.setFixedHeight(220)

        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setFont(QFont("Consolas", 9))
        self.info_tabs.addTab(self.meta_text, "Metadata")

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setFont(QFont("Consolas", 9))
        self.info_tabs.addTab(self.config_text, "Config JSON")

        lay.addWidget(self.info_tabs, stretch=1)
        return w

    # ── File list ─────────────────────────────────────────────────────────────

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        if path:
            self.folder_label.setText(path)
            self._load_folder(path)

    def _load_folder(self, path):
        if not _HAS_DM:
            self.statusBar().showMessage("data_manager not available.")
            return
        self._all_files = list_data_files(path)
        self._populate_tree(self._all_files)
        self.statusBar().showMessage(f"{len(self._all_files)} files found in {path}")

    def _refresh(self):
        path = self.folder_label.text()
        if path:
            self._load_folder(path)

    def _populate_tree(self, files):
        self.tree.clear()
        grouped = {}
        for f in files:
            date = f["timestamp"][:10] if f["timestamp"] else "unknown"
            grouped.setdefault(date, []).append(f)

        for date, items in sorted(grouped.items(), reverse=True):
            parent = QTreeWidgetItem(self.tree, [date, f"{len(items)} files"])
            parent.setExpanded(True)
            for f in items:
                child = QTreeWidgetItem(parent, [
                    f["filename"],
                    f["timestamp"][11:16] if len(f["timestamp"]) > 10 else "",
                ])
                child.setData(0, Qt.UserRole, f["path"])

    def _apply_filter(self):
        keyword = self.filter_edit.text().lower()
        qsel    = self.qubit_combo.currentText()
        filtered = [
            f for f in self._all_files
            if (not keyword or keyword in f["filename"].lower())
            and (qsel == "All" or f["qubit"] == int(qsel[1]))
        ]
        self._populate_tree(filtered)

    def _on_item_clicked(self, item, _col):
        path = item.data(0, Qt.UserRole)
        if not path or not os.path.isfile(path):
            self._del_btn.setEnabled(False)
            return
        self._current_path = path
        self._del_btn.setEnabled(True)
        self._load_file(path)

    def _delete_current(self):
        if not self._current_path or not os.path.isfile(self._current_path):
            return
        name = os.path.basename(self._current_path)
        reply = QMessageBox.question(
            self, "Delete file",
            f"Permanently delete:\n{name}?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            os.remove(self._current_path)
            self._current_path = None
            self._current_data = None
            self._del_btn.setEnabled(False)
            self._refresh()

    def _export_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "figure",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)")
        if path:
            self.fig.savefig(path, dpi=150, facecolor=BG0)
            self.statusBar().showMessage(f"Saved → {path}")

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F5"),     self, self._refresh)
        QShortcut(QKeySequence("Delete"), self, self._delete_current)
        QShortcut(QKeySequence("Ctrl+S"), self, self._export_figure)

    def _restore_geometry(self):
        s = QSettings("qick_workspace", "DataBrowser")
        geom = s.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event):
        s = QSettings("qick_workspace", "DataBrowser")
        s.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    # ── Data loading & display ────────────────────────────────────────────────

    def _load_file(self, path):
        if not _HAS_DM:
            return
        try:
            self._current_data = load_data(path)
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            return

        d = self._current_data
        self.statusBar().showMessage(os.path.basename(path))

        # Metadata tab
        is_2d = d["y"] is not None
        meta_lines = [
            f"Experiment  : {d['experiment']}",
            f"Qubit       : {d['qubit']}",
            f"Timestamp   : {d['timestamp']}",
            f"Tag         : {d['tag']}",
            f"Dimensions  : {'2D' if is_2d else '1D'}",
            f"X axis      : {d['x']['name']} ({d['x']['unit']})"
            f"  [{d['x']['values'][0]:.4g} … {d['x']['values'][-1]:.4g}]"
            f"  {len(d['x']['values'])} pts",
        ]
        if is_2d:
            meta_lines.append(
                f"Y axis      : {d['y']['name']} ({d['y']['unit']})"
                f"  [{d['y']['values'][0]:.4g} … {d['y']['values'][-1]:.4g}]"
                f"  {len(d['y']['values'])} pts"
            )
        for ch in ("avgi", "avgq", "mag", "phase"):
            arr = d[ch]
            meta_lines.append(
                f"{ch:<12}: min={arr.min():.4g}  max={arr.max():.4g}  "
                f"mean={arr.mean():.4g}"
            )
        self.meta_text.setPlainText("\n".join(meta_lines))

        # Config tab
        try:
            cfg = d["config"]
            pretty = json.dumps(cfg, indent=2, default=str)
        except Exception:
            pretty = str(d["config"])
        self.config_text.setPlainText(pretty)

        self._replot()

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG3)

    def _replot(self):
        if self._current_data is None:
            return
        d        = self._current_data
        cid      = self._ch_bg.checkedId()
        ch       = ["mag", "phase", "avgi", "avgq"][cid]
        y        = d[ch]
        x        = d["x"]["values"]
        ptype    = self._plot_type.currentText()
        ylabel_map = {
            "mag":   "|IQ| (ADC)",
            "phase": "Phase (deg)",
            "avgi":  "I (ADC)",
            "avgq":  "Q (ADC)",
        }

        # Rebuild figure cleanly every time to avoid colorbar / axis pollution
        self.fig.clear()
        self.fig.patch.set_facecolor(BG0)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)

        is_2d_data = d["y"] is not None and y.ndim == 2

        # ── Hist 2D (IQ plane density) ────────────────────────────────────────
        if ptype == "Hist 2D":
            avgi = d["avgi"].ravel()
            avgq = d["avgq"].ravel()
            bins = min(120, max(30, int(np.sqrt(avgi.size) * 1.5)))
            _, _, _, img = self.ax.hist2d(
                avgi, avgq, bins=bins, cmap="inferno",
                density=True,
            )
            cbar = self.fig.colorbar(img, ax=self.ax, pad=0.02)
            cbar.set_label("Density", color=TEXT_DIM)
            cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelcolor=TEXT_DIM)
            self.ax.set_xlabel("I (ADC)")
            self.ax.set_ylabel("Q (ADC)")
            self.ax.set_title(
                f"{d['experiment']}  Q{d['qubit']}  —  IQ histogram",
                fontsize=10,
            )

        # ── Hist 1D (selected channel) ────────────────────────────────────────
        elif ptype == "Hist 1D":
            vals = y.ravel()
            bins = min(120, max(30, int(np.sqrt(vals.size) * 2)))
            n, edges, patches = self.ax.hist(
                vals, bins=bins,
                color=ACCENT, alpha=0.85, edgecolor=BG0, linewidth=0.4,
            )
            # Overlay a subtle KDE
            try:
                from scipy.stats import gaussian_kde
                kde_x = np.linspace(edges[0], edges[-1], 300)
                kde = gaussian_kde(vals)
                scale = n.max() / kde(kde_x).max()
                self.ax.plot(kde_x, kde(kde_x) * scale,
                             color=TEXT, linewidth=1.2, zorder=5)
            except ImportError:
                pass
            self.ax.set_xlabel(ylabel_map[ch])
            self.ax.set_ylabel("Counts")
            self.ax.set_title(
                f"{d['experiment']}  Q{d['qubit']}  —  {ch} histogram",
                fontsize=10,
            )

        # ── Scatter / line (default) ──────────────────────────────────────────
        else:
            if is_2d_data:
                yv = d["y"]["values"]
                xc = np.concatenate([[x[0]  - (x[1] -x[0] )/2],
                                      (x[:-1]  + x[1:] ) / 2,
                                      [x[-1]  + (x[-1] -x[-2] )/2]])
                yc = np.concatenate([[yv[0] - (yv[1]-yv[0])/2],
                                      (yv[:-1] + yv[1:]) / 2,
                                      [yv[-1] + (yv[-1]-yv[-2])/2]])
                pcm = self.ax.pcolormesh(xc, yc, y, cmap="RdBu_r", shading="flat")
                cbar = self.fig.colorbar(pcm, ax=self.ax, pad=0.02)
                cbar.set_label(ylabel_map[ch], color=TEXT_DIM)
                cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelcolor=TEXT_DIM)
                self.ax.set_xlabel(f"{d['x']['name']} ({d['x']['unit']})")
                self.ax.set_ylabel(f"{d['y']['name']} ({d['y']['unit']})")
            else:
                self.ax.plot(x, y.ravel(), ".-", color=ACCENT,
                             markersize=4, linewidth=1.2)
                self.ax.set_xlabel(f"{d['x']['name']} ({d['x']['unit']})")
                self.ax.set_ylabel(ylabel_map[ch])
            self.ax.set_title(
                f"{d['experiment']}  Q{d['qubit']}  —  {ch}",
                fontsize=10,
            )

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Data Browser")
    theme.apply(app)
    win = DataBrowserWindow()
    win.show()

    # Auto-open example folder if run from repo root
    example_dir = os.path.join(os.path.dirname(__file__), "..", "example")
    if os.path.isdir(example_dir):
        win.folder_label.setText(os.path.abspath(example_dir))
        win._load_folder(os.path.abspath(example_dir))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
