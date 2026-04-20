"""
Strip Labber's bundled qtpy from sys.path before any Qt import.
Runs automatically when any gui.* module is imported.
"""
import sys
import os

sys.path = [p for p in sys.path if "_include" not in p]
os.environ.setdefault("QT_API", "pyside6")
