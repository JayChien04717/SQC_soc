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
    QTextEdit, QTabWidget, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

try:
    from qick_workspace.tools.data_manager import load_data, list_data_files
    _HAS_DM = True
except ImportError:
    _HAS_DM = False


class DataBrowserWindow(QMainWindow):
    """Standalone full-screen data browser."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Browser")
        self.resize(1400, 900)
        self._current_data = None
        self._all_files    = []
        self._build_ui()

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

        # Refresh
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        lay.addWidget(refresh_btn)
        return w

    # ── Right panel: plot + info tabs ─────────────────────────────────────────

    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        # Channel selector
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self._ch_bg = QButtonGroup(self)
        for i, ch in enumerate(["mag", "phase", "avgi", "avgq"]):
            rb = QRadioButton(ch)
            rb.setFixedWidth(70)
            if i == 0:
                rb.setChecked(True)
            self._ch_bg.addButton(rb, i)
            ch_row.addWidget(rb)
        self._ch_bg.idToggled.connect(
            lambda _id, checked: self._replot() if checked else None
        )
        ch_row.addStretch()
        lay.addLayout(ch_row)

        # Matplotlib canvas + toolbar
        self.fig = Figure(tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
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
            parent = QTreeWidgetItem(self.tree, [date, ""])
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
            return
        self._load_file(path)

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

    def _replot(self):
        if self._current_data is None:
            return
        d   = self._current_data
        cid = self._ch_bg.checkedId()
        ch  = ["mag", "phase", "avgi", "avgq"][cid]
        y   = d[ch]
        x   = d["x"]["values"]
        ylabel_map = {
            "mag":   "|IQ| (ADC)",
            "phase": "Phase (deg)",
            "avgi":  "I (ADC)",
            "avgq":  "Q (ADC)",
        }

        # Always rebuild the figure cleanly to avoid colorbar layout pollution
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        is_2d = d["y"] is not None and y.ndim == 2

        if is_2d:
            yv = d["y"]["values"]
            # pcolormesh gives correct axis tick labels for non-uniform grids
            xc = np.concatenate([[x[0] - (x[1]-x[0])/2],
                                  (x[:-1] + x[1:]) / 2,
                                  [x[-1] + (x[-1]-x[-2])/2]])
            yc = np.concatenate([[yv[0] - (yv[1]-yv[0])/2],
                                  (yv[:-1] + yv[1:]) / 2,
                                  [yv[-1] + (yv[-1]-yv[-2])/2]])
            pcm = self.ax.pcolormesh(xc, yc, y, cmap="RdBu_r", shading="flat")
            self.fig.colorbar(pcm, ax=self.ax, pad=0.02, label=ylabel_map[ch])
            self.ax.set_xlabel(f"{d['x']['name']} ({d['x']['unit']})")
            self.ax.set_ylabel(f"{d['y']['name']} ({d['y']['unit']})")
        else:
            self.ax.plot(x, y.ravel(), ".-", markersize=4, linewidth=1.2)
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
