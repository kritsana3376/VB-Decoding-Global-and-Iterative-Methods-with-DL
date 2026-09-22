"""Shared figure styling for the paper's plots.

Both plot scripts used to carry an identical block of font sizes, grid
settings and axis tweaks, plus their own copy of the decoder colour table.
Edit here to change both figures at once.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Fonts. Tahoma was chosen because it also renders Thai; any installed font
# works, and matplotlib falls back with a warning if it is missing.
TARGET_FONT = "Tahoma"
FONT_LABEL = 24      # axis label size
FONT_TICKS = 20      # tick number size
FONT_LEGEND = 18     # legend size

GRID_THICKNESS = 0.6
GRID_STYLE = "--"
GRID_COLOR = "#757575"

AXIS_THICKNESS = 1.2  # spine width
AXIS_COLOR = "#333333"
TICK_LENGTH = 6
TICK_THICKNESS = 1.2

FIG_SIZE = (14, 10)
MARKER_SIZE_DATA = 12

# X axis. X_AUTO=True uses the SER values present in the data; False forces
# the fixed X_START/X_END/X_STEP range used in the paper's figures.
X_AUTO = False
X_START = 0.01
X_END = 0.16
X_STEP = 0.01

# One colour per decoder family: E-hMP red, M-hMP blue, VSD green, with the
# baseline lightest and the guided variants darker.
DECODER_STYLES = {
    "EhMP":           {"color": "red",            "label": "E-hMP"},
    "E_hMP_guid_CNN": {"color": "darkred",        "label": "E-hMP CNN"},
    "E_hMP_guid_GNN": {"color": "salmon",         "label": "E-hMP GNN"},

    "MhMP":           {"color": "blue",           "label": "M-hMP"},
    "M_hMP_guid_CNN": {"color": "darkblue",       "label": "M-hMP CNN"},
    "M_hMP_guid_GNN": {"color": "dodgerblue",     "label": "M-hMP GNN"},

    "VSD_AFV":        {"color": "limegreen",      "label": "VSD-pv"},
    "VSD_guid_CNN":   {"color": "darkgreen",      "label": "VSD CNN"},
    "VSD_guid_GNN":   {"color": "mediumseagreen", "label": "VSD GNN"},
}


def new_figure():
    """Open a themed figure and axis."""
    sns.set_theme(style="whitegrid", rc={"font.family": TARGET_FONT})
    return plt.subplots(figsize=FIG_SIZE)


def positive_only(y):
    """Replace non-positive values with NaN and report how many there were.

    A log scale cannot draw zero or negative values. Turning them into NaN
    breaks the line visibly rather than dropping points silently.
    """
    y = np.asarray(y, dtype=float)
    n_dropped = int(np.sum(~np.isnan(y) & (y <= 0)))
    return np.where(y > 0, y, np.nan), n_dropped


def finish_axes(ax, x, xlabel, ylabel):
    """Apply the shared log-scale, tick, grid and spine styling."""
    ax.set_yscale("log")

    ticks = (np.unique(x[~np.isnan(x)]) if X_AUTO
             else np.arange(X_START, X_END + 1e-9, X_STEP))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.2f}" for v in ticks], fontsize=FONT_TICKS)
    ax.tick_params(axis="y", labelsize=FONT_TICKS)

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, labelpad=12)

    ax.legend(fontsize=FONT_LEGEND, loc="best", frameon=True,
              facecolor="white", edgecolor="none")
    ax.grid(True, which="both", linestyle=GRID_STYLE,
            linewidth=GRID_THICKNESS, color=GRID_COLOR, alpha=0.7)

    # Solid spines with outward-facing major ticks.
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(AXIS_THICKNESS)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=TICK_LENGTH, width=TICK_THICKNESS, colors=AXIS_COLOR,
                   left=True, bottom=True)
    ax.tick_params(axis="y", which="minor", left=False)


def save_figure(fig, path, n_dropped=0):
    """Save at print resolution. bbox_inches keeps large labels from clipping."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to:\n   {path}")
    if n_dropped:
        print(f"Note: {n_dropped} point(s) were <= 0 and cannot be drawn on a "
              f"log scale, so gaps were left in those lines.")
