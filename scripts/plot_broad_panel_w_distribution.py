from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SPLINE_SUMMARY = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "34_mini_fixed_p_4m_diagnostic/spline/results/broad_scaling_summary.json"
)
DEFAULT_MDN_SUMMARY = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "09_ui_best_broad_mdn_512k_seed20260902/results/broad_scaling_summary.json"
)
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "37_panel_w_distribution_mdn512k_spline4m"
)


def load_first_row(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def per_signal_by_index(row: dict[str, object]) -> dict[int, dict[str, object]]:
    metrics = row.get("panel_marginal_metrics") or {}
    values = metrics.get("per_signal") or []
    return {int(item["index"]): item for item in values}


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q16": float(np.quantile(finite, 0.16)),
        "median": float(np.median(finite)),
        "q84": float(np.quantile(finite, 0.84)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    return finite, np.arange(1, finite.size + 1, dtype=np.float64) / finite.size


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "index",
        "label",
        "A",
        "k",
        "sigma",
        "target_wasserstein",
        "spline_wasserstein",
        "spline_target_ratio",
        "mdn_wasserstein",
        "mdn_target_ratio",
        "spline_minus_mdn",
        "spline_over_mdn",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def plot(rows: list[dict[str, object]], output_path: Path) -> None:
    spline_w = np.asarray([row["spline_wasserstein"] for row in rows], dtype=np.float64)
    mdn_w = np.asarray([row["mdn_wasserstein"] for row in rows], dtype=np.float64)
    spline_ratio = np.asarray([row["spline_target_ratio"] for row in rows], dtype=np.float64)
    mdn_ratio = np.asarray([row["mdn_target_ratio"] for row in rows], dtype=np.float64)
    sigma = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
    indices = np.asarray([row["index"] for row in rows], dtype=np.int64)

    colors = {
        "spline": "#c45a2d",
        "mdn": "#6d4aff",
        "target": "#172033",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    for values, label, color in [
        (spline_w, "Broad spline 4.096M", colors["spline"]),
        (mdn_w, "Broad MDN 512k", colors["mdn"]),
    ]:
        x, y = ecdf(values)
        ax.step(x, y, where="post", lw=2.4, color=color, label=label)
        ax.scatter(values, np.full_like(values, 0.04 if "spline" in label else 0.08), s=30, color=color, alpha=0.78)
        ax.axvline(np.median(values), color=color, lw=1.6, alpha=0.65)
    ax.set_xlabel("panel marginal mean normalized W")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Distribution across 16 cached prior-panel signals")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    positions = np.arange(len(rows))
    order = np.argsort(mdn_w)[::-1]
    for rank, row_index in enumerate(order):
        left = spline_w[row_index]
        right = mdn_w[row_index]
        ax.plot([rank - 0.18, rank + 0.18], [left, right], color="#9ca3af", lw=1.0, alpha=0.8)
        ax.scatter(rank - 0.18, left, color=colors["spline"], s=38, zorder=3)
        ax.scatter(rank + 0.18, right, color=colors["mdn"], s=38, zorder=3)
        if rank < 4 or left > right:
            ax.text(rank, max(left, right) * 1.035, str(indices[row_index]), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(indices[item])) for item in order], rotation=0)
    ax.set_xlabel("signal index, sorted by MDN W")
    ax.set_ylabel("panel marginal mean normalized W")
    ax.set_title("Per-signal paired comparison")
    ax.grid(axis="y", which="both", alpha=0.22)

    ax = axes[1, 0]
    bins = np.linspace(0, max(float(np.max(spline_ratio)), float(np.max(mdn_ratio))) * 1.05, 12)
    ax.hist(spline_ratio, bins=bins, alpha=0.42, color=colors["spline"], label="Broad spline 4.096M")
    ax.hist(mdn_ratio, bins=bins, alpha=0.36, color=colors["mdn"], label="Broad MDN 512k")
    ax.axvline(np.median(spline_ratio), color=colors["spline"], lw=2.0)
    ax.axvline(np.median(mdn_ratio), color=colors["mdn"], lw=2.0)
    ax.set_xlabel("W / panel target numerical floor")
    ax.set_ylabel("signal count")
    ax.set_title("Distance to evaluation floor")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(sigma, spline_w, color=colors["spline"], s=42, label="Broad spline 4.096M")
    ax.scatter(sigma, mdn_w, color=colors["mdn"], s=42, label="Broad MDN 512k")
    for index, x_value, y_spline, y_mdn in zip(indices, sigma, spline_w, mdn_w, strict=True):
        if max(y_spline, y_mdn) >= np.quantile(np.concatenate([spline_w, mdn_w]), 0.82):
            ax.text(x_value, max(y_spline, y_mdn) * 1.04, str(int(index)), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true sigma for panel signal")
    ax.set_ylabel("panel marginal mean normalized W")
    ax.set_title("Failure concentration by noise level")
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False)

    figure.suptitle("Broad spline 4.096M vs Broad MDN 512k: corrected panel marginal W", y=1.02)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot panel marginal W distributions for the best MDN and spline broad NPEs.")
    parser.add_argument("--spline-summary", type=Path, default=DEFAULT_SPLINE_SUMMARY)
    parser.add_argument("--mdn-summary", type=Path, default=DEFAULT_MDN_SUMMARY)
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    figures_dir = args.output_root / "figures"
    results_dir = args.output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    spline_row = load_first_row(args.spline_summary)
    mdn_row = load_first_row(args.mdn_summary)
    spline_by_index = per_signal_by_index(spline_row)
    mdn_by_index = per_signal_by_index(mdn_row)
    panel = np.load(args.panel_cache)
    theta_panel = np.asarray(panel["theta_panel"], dtype=np.float64)
    labels = [str(item) for item in panel["labels"]]

    common_indices = sorted(set(spline_by_index) & set(mdn_by_index))
    rows = []
    for index in common_indices:
        spline_item = spline_by_index[index]
        mdn_item = mdn_by_index[index]
        target = float(spline_item["target_wasserstein"])
        mdn_w = float(mdn_item["wasserstein"])
        spline_w = float(spline_item["wasserstein"])
        rows.append(
            {
                "index": int(index),
                "label": labels[index],
                "A": float(theta_panel[index, 0]),
                "k": float(theta_panel[index, 1]),
                "sigma": float(theta_panel[index, 2]),
                "target_wasserstein": target,
                "spline_wasserstein": spline_w,
                "spline_target_ratio": float(spline_item["target_ratio"]),
                "mdn_wasserstein": mdn_w,
                "mdn_target_ratio": float(mdn_item["target_ratio"]),
                "spline_minus_mdn": float(spline_w - mdn_w),
                "spline_over_mdn": float(spline_w / mdn_w) if mdn_w > 0 else float("nan"),
            }
        )

    csv_path = results_dir / "broad_panel_w_distribution_rows.csv"
    json_path = results_dir / "broad_panel_w_distribution_summary.json"
    figure_path = figures_dir / "broad_panel_w_distribution_mdn512k_vs_spline4m.png"
    write_csv(rows, csv_path)
    plot(rows, figure_path)

    spline_w = np.asarray([row["spline_wasserstein"] for row in rows], dtype=np.float64)
    mdn_w = np.asarray([row["mdn_wasserstein"] for row in rows], dtype=np.float64)
    summary = {
        "panel_cache": str(args.panel_cache),
        "signal_count": len(rows),
        "spline_summary": str(args.spline_summary),
        "mdn_summary": str(args.mdn_summary),
        "spline_wasserstein": quantile_summary(spline_w),
        "mdn_wasserstein": quantile_summary(mdn_w),
        "spline_better_count": int(np.sum(spline_w < mdn_w)),
        "mdn_better_count": int(np.sum(mdn_w < spline_w)),
        "mean_improvement_mdn_minus_spline": float(np.mean(mdn_w - spline_w)),
        "median_improvement_mdn_minus_spline": float(np.median(mdn_w - spline_w)),
        "outputs": {
            "figure": str(figure_path),
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
