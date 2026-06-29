from __future__ import annotations

from collections.abc import Sequence

import corner
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


TRUE_THETA_COLOR = "#000000"
TRUE_THETA_LINESTYLE = "--"


def overplot_true_values(fig: Figure, values: Sequence[float]) -> None:
    corner.overplot_lines(
        fig,
        values,
        color=TRUE_THETA_COLOR,
        linestyle=TRUE_THETA_LINESTYLE,
        linewidth=1.7,
        alpha=0.95,
        zorder=10,
    )


def true_theta_legend_handle(label: str = "True theta") -> Line2D:
    return Line2D(
        [0],
        [0],
        color=TRUE_THETA_COLOR,
        linestyle=TRUE_THETA_LINESTYLE,
        lw=2,
        label=label,
    )
