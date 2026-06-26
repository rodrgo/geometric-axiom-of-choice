"""Standalone version of the hero figure's third panel.

Renders the depth-stratified "three measurements, one gradient" view
(k-NN AUC, reconstruction loss, fraction outside the q=90 superlevel
set, each rescaled into a normalized boundary-strength axis) as its
own figure for paste-in after the hero figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hero_figure import panel_c_lines, FIG


def main():
    plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    panel_c_lines(ax)
    # Strip the (c) panel-label from the title since this is a
    # standalone figure now.
    ax.set_title("Three measurements, one gradient", fontsize=11)
    plt.tight_layout()
    out = FIG / "three_measurements_optB.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
