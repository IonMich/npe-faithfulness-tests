from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/"
    "results/mdn_vs_spline_fixed_p_summary_16m.json"
)
DEFAULT_OUTPUT = (
    ROOT / "runs/00_shared_assets/readme_scaling/decay_mdn_vs_spline_fixed_p_2x2_16m.png"
)

SERIES_STYLE = {
    "mdn_base": {
        "label": "Mixture density network (MDN), 44,722 parameters",
        "color": "#5b56b3",
        "marker": "o",
    },
    "spline_flow_small": {
        "label": "Conditional spline-flow NPE, 45,844 parameters",
        "color": "#c45a2d",
        "marker": "s",
    },
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series_arrays(series: dict, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = series["points"]
    x = np.asarray([point["train_simulations"] for point in points], dtype=float)
    median = np.asarray([point[metric]["median"] for point in points], dtype=float)
    q16 = np.asarray([point[metric]["q16"] for point in points], dtype=float)
    q84 = np.asarray([point[metric]["q84"] for point in points], dtype=float)
    return x, median, q16, q84


def draw_panel(
    *,
    ax: plt.Axes,
    summary: dict,
    metric: str,
    ylabel: str,
    log_y: bool,
) -> None:
    for series in summary["series"]:
        style = SERIES_STYLE[series["short"]]
        x, median, q16, q84 = series_arrays(series, metric)
        ax.plot(
            x,
            median,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
            label=style["label"],
        )
        ax.fill_between(x, q16, q84, color=style["color"], alpha=0.16)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("simulated training pairs")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.24, which="both")
    ax.legend(frameon=False, fontsize=8)


def plot(summary_path: Path = DEFAULT_SUMMARY, output_path: Path = DEFAULT_OUTPUT) -> Path:
    summary = load_summary(summary_path)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)
    panels = [
        (
            "panel_marginal_wasserstein_mean",
            "panel mean normalized Wasserstein distance",
            True,
        ),
        (
            "panel_marginal_target_ratio_mean",
            "distance / numerical evaluation floor",
            True,
        ),
        (
            "full_val_nll_z_units",
            "validation negative log likelihood, z units",
            False,
        ),
        ("training_seconds", "training seconds", True),
    ]
    for ax, (metric, ylabel, log_y) in zip(axes.ravel(), panels, strict=True):
        draw_panel(ax=ax, summary=summary, metric=metric, ylabel=ylabel, log_y=log_y)

    figure.suptitle("Fixed parameter-count single-decay NPE data scaling", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    print(plot())


if __name__ == "__main__":
    main()
