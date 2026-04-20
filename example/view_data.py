"""
Quick standalone data viewer — no GUI framework needed.
Shows all example files as matplotlib plots.

    python example/view_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from qick_workspace.tools.data_manager import load_data, list_data_files

EXAMPLE_DIR = os.path.dirname(__file__)
files = list_data_files(EXAMPLE_DIR)

print(f"Found {len(files)} files:\n")
for i, f in enumerate(files):
    print(f"  [{i}] {f['filename']}  — {f['experiment']}  Q{f['qubit']}  {f['timestamp']}")

print()

for f in files:
    d = load_data(f["path"])
    x = d["x"]["values"]
    xlabel = f"{d['x']['name']} ({d['x']['unit']})"

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig.suptitle(f"{d['experiment']}  |  Q{d['qubit']}  |  {d['timestamp'][:16]}", fontsize=10)

    channels = [("mag", d["mag"]), ("phase", d["phase"]),
                ("avgi", d["avgi"]), ("avgq", d["avgq"])]

    for ax, (ch, y) in zip(axes, channels):
        if d["y"] is not None:
            # 2D
            yv = d["y"]["values"]
            im = ax.imshow(y, aspect="auto",
                           extent=[x[0], x[-1], yv[0], yv[-1]],
                           origin="lower")
            plt.colorbar(im, ax=ax, pad=0.02)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(f"{d['y']['name']} ({d['y']['unit']})", fontsize=7)
        else:
            # 1D
            ax.plot(x, y, ".-", markersize=3)
            ax.set_xlabel(xlabel, fontsize=7)

        ax.set_title(ch, fontsize=9)
        ax.tick_params(labelsize=7)

    plt.tight_layout()

plt.show()
