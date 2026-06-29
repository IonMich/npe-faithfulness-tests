from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/01_exponential_decay/10_wasserstein_distributions/01_grid_eval")


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_model_list(value: str) -> list[str]:
    models = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model name is required.")
    return models


def collect_rows(panel: dict[str, object], models: list[str] | None) -> tuple[list[dict[str, object]], list[str]]:
    available = [str(model) for model in panel.get("models", [])]
    selected = available if models is None else models
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Requested models not in panel summary: {missing}. Available: {available}")

    rows: list[dict[str, object]] = []
    for observation in panel.get("observations", []):
        obs_index = int(observation["index"])
        distance_metrics = observation.get("distance_metrics", {}) or {}
        timing = observation.get("timing_seconds", {}) or {}
        model_timing = timing.get("model_sampling", {}) or {}
        grid_reference_seconds = timing.get("grid_reference")
        for model in selected:
            result = observation["models"][model]
            rows.append(
                {
                    "observation_index": obs_index,
                    "model": model,
                    "mean_normalized_wasserstein": float(result["mean_normalized_wasserstein"]),
                    "npe_sampling_seconds": model_timing.get(model),
                    "grid_reference_seconds": grid_reference_seconds,
                    "grid_tolerance_ratio": float(result.get("grid_tolerance_ratio", np.nan)),
                    "tolerance_ratio": float(result.get("tolerance_ratio", np.nan)),
                    "tau_grid": float(result.get("tau_grid", np.nan)),
                    "tau": float(result.get("tau", np.nan)),
                    "tolerance_valid": result.get("tolerance_valid"),
                    "ratio_pass": result.get("ratio_pass"),
                    "prior_z_radius": distance_metrics.get("prior_z_radius"),
                    "summary_prior_predictive_radius": distance_metrics.get("summary_prior_predictive_radius"),
                    "local_x0_distance_over_radius": distance_metrics.get("local_x0_distance_over_radius"),
                    "parameter_region_distance_over_radius": distance_metrics.get("parameter_region_distance_over_radius"),
                }
            )
    return rows, selected


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
        "q90": float(np.quantile(finite, 0.90)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    return finite, np.arange(1, finite.size + 1) / finite.size


def plot_wasserstein_distribution(
    *,
    rows: list[dict[str, object]],
    models: list[str],
    panel_kind: str,
    output_path: Path,
) -> None:
    colors = [
        "#2f6fbb",
        "#b85c38",
        "#2f855a",
        "#7a5cc2",
        "#374151",
    ]
    has_timing = np.isfinite(
        np.asarray(
            [
                row.get("npe_sampling_seconds", np.nan)
                for row in rows
            ]
            + [
                row.get("grid_reference_seconds", np.nan)
                for row in rows
            ],
            dtype=np.float64,
        )
    ).any()
    ncols = 3 if has_timing else 2
    figure, axes = plt.subplots(1, ncols, figsize=(5.9 * ncols, 5.2))
    all_values = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    finite_all = all_values[np.isfinite(all_values)]
    if finite_all.size == 0:
        raise RuntimeError("No finite Wasserstein values found in panel summary.")
    bins = min(max(8, int(np.sqrt(finite_all.size) * 2)), 28)
    use_log_x = finite_all.min() > 0 and finite_all.max() / finite_all.min() > 100
    hist_bins: int | np.ndarray
    if use_log_x:
        hist_bins = np.geomspace(finite_all.min(), finite_all.max(), bins + 1)
    else:
        hist_bins = bins
    for index, model in enumerate(models):
        values = np.asarray(
            [
                row["mean_normalized_wasserstein"]
                for row in rows
                if row["model"] == model
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        color = colors[index % len(colors)]
        label = model
        axes[0].hist(
            values,
            bins=hist_bins,
            density=True,
            alpha=0.28,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        axes[0].axvline(np.median(values), color=color, linewidth=2.0)
        x, y = ecdf(values)
        axes[1].step(x, y, where="post", color=color, linewidth=2.2, label=label)
        axes[1].scatter(values, np.full_like(values, 0.04 + 0.025 * index), color=color, s=26, alpha=0.78)
        axes[1].axvline(np.median(values), color=color, linewidth=1.8)

    axes[0].set_xlabel("mean normalized Wasserstein to grid posterior")
    axes[0].set_ylabel("density")
    axes[0].set_title("Distribution across random signals")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("mean normalized Wasserstein to grid posterior")
    axes[1].set_ylabel("empirical CDF")
    axes[1].set_title("ECDF across random signals")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)

    if has_timing:
        timing_ax = axes[2]
        timing_values = []
        timing_labels = []
        timing_colors = []
        for index, model in enumerate(models):
            values = np.asarray(
                [
                    row.get("npe_sampling_seconds", np.nan)
                    for row in rows
                    if row["model"] == model
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if values.size:
                timing_values.append(values)
                timing_labels.append(f"{model} NPE")
                timing_colors.append(colors[index % len(colors)])
        grid_by_observation: dict[int, float] = {}
        for row in rows:
            value = row.get("grid_reference_seconds")
            if value is not None and np.isfinite(float(value)):
                grid_by_observation[int(row["observation_index"])] = float(value)
        grid_values = np.asarray(list(grid_by_observation.values()), dtype=np.float64)
        if grid_values.size:
            timing_values.append(grid_values)
            timing_labels.append("grid reference")
            timing_colors.append("#172033")
        positions = np.arange(1, len(timing_values) + 1)
        box = timing_ax.boxplot(
            timing_values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], timing_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for position, values, color in zip(positions, timing_values, timing_colors, strict=True):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            timing_ax.scatter(
                np.full_like(values, position, dtype=np.float64) + jitter,
                values,
                color=color,
                s=26,
                alpha=0.8,
                zorder=3,
            )
            timing_ax.axhline(np.median(values), color=color, linewidth=1.4, alpha=0.5)
        timing_ax.set_xticks(positions)
        timing_ax.set_xticklabels(timing_labels, rotation=20, ha="right")
        timing_ax.set_ylabel("seconds per observation")
        timing_ax.set_title("Measured runtime")
        timing_ax.grid(axis="y", alpha=0.22)
        finite_times = np.concatenate([values for values in timing_values if values.size])
        if finite_times.max() / max(finite_times.min(), 1e-12) > 30:
            timing_ax.set_yscale("log")

    if use_log_x:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[0].tick_params(axis="x", labelrotation=20)
        axes[1].tick_params(axis="x", labelrotation=20)
    figure.suptitle(f"Decay NPE Wasserstein error distribution: {panel_kind}", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "observation_index",
        "model",
        "mean_normalized_wasserstein",
        "grid_tolerance_ratio",
        "tolerance_ratio",
        "tau_grid",
        "tau",
        "npe_sampling_seconds",
        "grid_reference_seconds",
        "tolerance_valid",
        "ratio_pass",
        "prior_z_radius",
        "summary_prior_predictive_radius",
        "local_x0_distance_over_radius",
        "parameter_region_distance_over_radius",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distribution of NPE-to-grid Wasserstein distances across decay panel signals.",
    )
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=parse_model_list, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    panel_kind = str(panel.get("panel_distribution", {}).get("kind", "unknown"))
    rows, models = collect_rows(panel, args.models)

    figure_path = figure_dir / "grid_eval_wasserstein_distribution.png"
    csv_path = results_dir / "grid_eval_wasserstein_distribution.csv"
    json_path = results_dir / "grid_eval_wasserstein_distribution_summary.json"
    plot_wasserstein_distribution(
        rows=rows,
        models=models,
        panel_kind=panel_kind,
        output_path=figure_path,
    )
    write_csv(rows, csv_path)

    summary = {}
    for model in models:
        values = np.asarray(
            [row["mean_normalized_wasserstein"] for row in rows if row["model"] == model],
            dtype=np.float64,
        )
        ratios = np.asarray(
            [row["tolerance_ratio"] for row in rows if row["model"] == model],
            dtype=np.float64,
        )
        npe_seconds = np.asarray(
            [row.get("npe_sampling_seconds", np.nan) for row in rows if row["model"] == model],
            dtype=np.float64,
        )
        grid_seconds = np.asarray(
            [row.get("grid_reference_seconds", np.nan) for row in rows if row["model"] == model],
            dtype=np.float64,
        )
        summary[model] = {
            "mean_normalized_wasserstein": quantile_summary(values),
            "tolerance_ratio": quantile_summary(ratios),
            "npe_sampling_seconds": quantile_summary(npe_seconds),
            "grid_reference_seconds": quantile_summary(grid_seconds),
        }

    output = {
        "panel_summary": str(args.panel_summary),
        "panel_kind": panel_kind,
        "models": models,
        "model_labels": panel.get("model_labels", {}),
        "summary": summary,
        "rows": rows,
        "outputs": {
            "figure": str(figure_path),
            "csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    print(f"csv: {csv_path}")
    for model, model_summary in summary.items():
        w = model_summary["mean_normalized_wasserstein"]
        print(
            f"{model}: W median={w['median']:.4f}, "
            f"q90={w['q90']:.4f}, max={w['max']:.4f}, n={w['n']}"
        )


if __name__ == "__main__":
    main()
