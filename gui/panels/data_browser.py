import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QLabel,
    QPushButton, QLineEdit, QComboBox, QButtonGroup,
    QRadioButton, QGroupBox, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Lazy import so GUI loads even without qick_workspace on path
try:
    from qick_workspace.tools.data_manager import list_data_files, load_data
    _HAS_DM = True
except ImportError:
    _HAS_DM = False


class DataBrowser(QWidget):
    """[6] Data browser: file list (left) + plot viewer (right)."""

    file_selected = Signal(str)   # emits file path on selection

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_dir = ""
        self._current_data = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_viewer_panel())
        splitter.setSizes([260, 600])
        root.addWidget(splitter)

    # ── File list (left) ──────────────────────────────────────────────────────

    def _build_file_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)

        # Filter bar
        frow = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        frow.addWidget(self.filter_edit)
        self.qubit_combo = QComboBox()
        self.qubit_combo.addItems(["All", "Q0", "Q1", "Q2", "Q3"])
        self.qubit_combo.currentTextChanged.connect(self._apply_filter)
        frow.addWidget(self.qubit_combo)
        lay.addLayout(frow)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Time"])
        self.tree.setColumnWidth(0, 160)
        self.tree.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.tree)

        # Buttons
        brow = QHBoxLayout()
        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self._browse_folder)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        brow.addWidget(open_btn)
        brow.addWidget(refresh_btn)
        lay.addLayout(brow)
        return w

    # ── Viewer (right) ────────────────────────────────────────────────────────

    def _build_viewer_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)

        # Channel toggle
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("View:"))
        self._ch_bg = QButtonGroup(self)
        for i, ch in enumerate(["mag", "phase", "avgi", "avgq"]):
            rb = QRadioButton(ch)
            if i == 0:
                rb.setChecked(True)
            self._ch_bg.addButton(rb, i)
            ch_row.addWidget(rb)
        self._ch_bg.idToggled.connect(
            lambda _id, checked: self._replot() if checked else None
        )
        ch_row.addStretch()
        lay.addLayout(ch_row)

        # Matplotlib canvas
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, stretch=3)

        # Metadata box
        meta_grp = QGroupBox("Metadata")
        meta_lay = QVBoxLayout(meta_grp)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(120)
        meta_lay.addWidget(self.meta_text)
        lay.addWidget(meta_grp, stretch=1)
        return w

    # ── Public API ────────────────────────────────────────────────────────────

    def set_folder(self, path):
        self._root_dir = path
        self.refresh()

    def refresh(self):
        self.tree.clear()
        if not self._root_dir or not _HAS_DM:
            return
        files = list_data_files(self._root_dir)
        self._all_files = files
        self._populate_tree(files)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _populate_tree(self, files):
        self.tree.clear()
        grouped = {}
        for f in files:
            date = f["timestamp"][:10] if f["timestamp"] else "unknown"
            grouped.setdefault(date, []).append(f)

        for date, items in grouped.items():
            parent = QTreeWidgetItem(self.tree, [date, ""])
            parent.setExpanded(True)
            for f in items:
                child = QTreeWidgetItem(parent, [
                    f["filename"],
                    f["timestamp"][11:16] if len(f["timestamp"]) > 10 else "",
                ])
                child.setData(0, Qt.UserRole, f["path"])

    def _apply_filter(self):
        if not hasattr(self, "_all_files"):
            return
        keyword = self.filter_edit.text().lower()
        qsel    = self.qubit_combo.currentText()

        filtered = []
        for f in self._all_files:
            if keyword and keyword not in f["filename"].lower():
                continue
            if qsel != "All":
                qidx = int(qsel[1])
                if f["qubit"] != qidx:
                    continue
            filtered.append(f)
        self._populate_tree(filtered)

    def _on_item_clicked(self, item, _col):
        path = item.data(0, Qt.UserRole)
        if not path or not os.path.isfile(path):
            return
        self.file_selected.emit(path)
        self._load_and_display(path)

    def _load_and_display(self, path):
        if not _HAS_DM:
            self.meta_text.setPlainText("data_manager not available")
            return
        try:
            self._current_data = load_data(path)
        except Exception as e:
            self.meta_text.setPlainText(f"Error: {e}")
            return

        d = self._current_data
        meta = (
            f"Experiment : {d['experiment']}\n"
            f"Qubit      : {d['qubit']}\n"
            f"Timestamp  : {d['timestamp']}\n"
            f"Tag        : {d['tag']}\n"
            f"X axis     : {d['x']['name']} ({d['x']['unit']}), "
            f"{len(d['x']['values'])} pts"
        )
        if d["y"] is not None:
            meta += f"\nY axis     : {d['y']['name']} ({d['y']['unit']})"
        self.meta_text.setPlainText(meta)
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

        self.ax.cla()
        if y.ndim == 1:
            self.ax.plot(x, y, ".-", markersize=3)
        else:
            self.ax.imshow(
                y, aspect="auto",
                extent=[x[0], x[-1], 0, y.shape[0]],
                origin="lower",
            )
        self.ax.set_xlabel(f"{d['x']['name']} ({d['x']['unit']})")
        self.ax.set_ylabel(ylabel_map[ch])
        self.canvas.draw_idle()

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        if path:
            self.set_folder(path)
