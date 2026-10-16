"""The one figure style shared by scripts/09_figures.py and the figures drawn in analyses/.

Only the settings that differ from matplotlib's defaults.
"""

import matplotlib.pyplot as plt

TEXT_WIDTH_IN = 6.3  # width of the thesis text block

STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
}


def apply_style():
    plt.rcParams.update(STYLE)
