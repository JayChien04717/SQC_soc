"""
Dark futuristic theme for qick_workspace GUI.
Apply with:  app.setStyleSheet(DARK_QSS)
"""

# Colour palette
BG0      = "#0d1117"   # deepest background
BG1      = "#161b22"   # panel background
BG2      = "#21262d"   # input / widget background
BG3      = "#30363d"   # hover / border
ACCENT   = "#00d4ff"   # cyan accent
ACCENT2  = "#7ee787"   # green accent (success)
WARN     = "#f0883e"   # orange warning
TEXT     = "#e6edf3"   # primary text
TEXT_DIM = "#8b949e"   # secondary text
BORDER   = "#30363d"

DARK_QSS = f"""

/* ── Global ─────────────────────────────────────────────────────────── */
* {{
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
    color: {TEXT};
    background-color: {BG0};
    selection-background-color: {ACCENT};
    selection-color: {BG0};
}}

QMainWindow, QDialog, QWidget {{
    background-color: {BG0};
}}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BG3};
    width: 2px;
    height: 2px;
}}

/* ── Group Box ───────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {BG1};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
    font-size: 11px;
    color: {ACCENT};
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 0 4px;
    background-color: {BG1};
}}

/* ── Labels ─────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {TEXT};
}}

/* ── Line Edit ───────────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:read-only {{
    color: {TEXT_DIM};
}}
QLineEdit::placeholder {{
    color: {TEXT_DIM};
}}

/* ── Push Button ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    color: {TEXT};
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {BG3};
    border: 1px solid {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: {BG0};
}}
QPushButton:checked {{
    background-color: {ACCENT};
    color: {BG0};
    border: 1px solid {ACCENT};
    font-weight: 700;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: {BG3};
}}

/* ── Combo Box ───────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT};
}}
QComboBox:hover {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    width: 0;
    height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG2};
    border: 1px solid {ACCENT};
    selection-background-color: {ACCENT};
    selection-color: {BG0};
    outline: none;
}}

/* ── Spin Boxes ──────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    color: {TEXT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {BG3};
    border: none;
    width: 16px;
}}

/* ── Radio Button ────────────────────────────────────────────────────── */
QRadioButton {{
    background: transparent;
    spacing: 6px;
    color: {TEXT};
}}
QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border-radius: 7px;
    border: 2px solid {BORDER};
    background-color: {BG2};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border: 2px solid {ACCENT};
}}
QRadioButton:hover {{
    color: {ACCENT};
}}

/* ── Check Box ───────────────────────────────────────────────────────── */
QCheckBox {{
    background: transparent;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border-radius: 3px;
    border: 2px solid {BORDER};
    background-color: {BG2};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 2px solid {ACCENT};
}}

/* ── Tree Widget ─────────────────────────────────────────────────────── */
QTreeWidget {{
    background-color: {BG1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    alternate-background-color: {BG2};
    outline: none;
}}
QTreeWidget::item {{
    padding: 3px 4px;
    border: none;
}}
QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: {BG0};
}}
QTreeWidget::item:hover {{
    background-color: {BG3};
}}
QTreeWidget QHeaderView::section {{
    background-color: {BG2};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
    color: {TEXT_DIM};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Tab Widget ──────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 0 4px 4px 4px;
    background-color: {BG1};
}}
QTabBar::tab {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 5px 14px;
    color: {TEXT_DIM};
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {BG1};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}

/* ── Text Edit ───────────────────────────────────────────────────────── */
QTextEdit {{
    background-color: {BG1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
    color: {TEXT};
    font-family: "Consolas", "JetBrains Mono", monospace;
    font-size: 11px;
    line-height: 1.5;
}}

/* ── Scroll Bar ──────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {BG1};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background-color: {BG3};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background-color: {BG1};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background-color: {BG3};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Menu Bar ────────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {BG1};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {BG3};
    color: {ACCENT};
}}
QMenu {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 20px 5px 12px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {ACCENT};
    color: {BG0};
}}

/* ── Status Bar ──────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG1};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 11px;
}}

/* ── Dock Widget ─────────────────────────────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    font-weight: 600;
    font-size: 11px;
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QDockWidget::title {{
    background-color: {BG1};
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    text-align: left;
}}

/* ── Scroll Area ─────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* ── Tool Bar (matplotlib) ───────────────────────────────────────────── */
QToolBar {{
    background-color: {BG1};
    border-bottom: 1px solid {BORDER};
    spacing: 2px;
    padding: 2px;
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px;
    color: {TEXT};
}}
QToolButton:hover {{
    background-color: {BG3};
    border: 1px solid {ACCENT};
    color: {ACCENT};
}}
QToolButton:checked {{
    background-color: {ACCENT};
    color: {BG0};
}}

"""


def apply(app):
    """Apply the dark theme to a QApplication instance."""
    app.setStyleSheet(DARK_QSS)
    # Set dark palette for widgets not covered by QSS
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(BG0))
    pal.setColor(QPalette.WindowText,      QColor(TEXT))
    pal.setColor(QPalette.Base,            QColor(BG1))
    pal.setColor(QPalette.AlternateBase,   QColor(BG2))
    pal.setColor(QPalette.Text,            QColor(TEXT))
    pal.setColor(QPalette.Button,          QColor(BG2))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(BG0))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    pal.setColor(QPalette.ToolTipBase,     QColor(BG2))
    pal.setColor(QPalette.ToolTipText,     QColor(TEXT))
    app.setPalette(pal)
