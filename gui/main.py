"""
Entry point for QickworkspaceV2 GUI.

    pip install PySide6 matplotlib h5py
    python gui/main.py            # normal mode
    python gui/main.py --debug    # simulation mode (no hardware required)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from gui.main_window import MainWindow
from gui import theme


def main():
    """
    Application entry point.

    Parameters
    ----------
    --debug : flag
        Launch in simulation mode.  Hardware connection is replaced by
        mock objects and experiment runs return synthetic data.
    """
    parser = argparse.ArgumentParser(description="QickworkspaceV2 GUI")
    parser.add_argument(
        "--debug", action="store_true",
        help="Simulation mode — no hardware required",
    )
    args, qt_args = parser.parse_known_args()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName("QickworkspaceV2 GUI")
    theme.apply(app)

    win = MainWindow(debug=args.debug)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
