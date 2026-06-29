from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/10_wasserstein_distributions/09_grid_size_comparison"
)


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


def parse_panel(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Panels must be given as label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Panel label cannot be empty")
    return label, Path(path.strip())


def parse_panel_list(value: str) -> list[tuple[str, Path]]:
    panels = [parse_panel(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(panels) < 2:
        raise argparse.ArgumentTypeError("At least two panels are required")
    return panels


def parse_model_list(value: str) -> list[str]:
    models = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model name is required")
    return models


def label_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"\d+", label)
    if match is None:
        return 10**9, label
    return int(match.group(0)), label


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


def collect_rows(
    *,
    panels: list[tuple[str, Path]],
    models: list[str] | None,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    selected_models: list[str] | None = models
    for label, path in panels:
        panel = json.loads(path.read_text(encoding="utf-8"))
        available = [str(model) for model in panel.get("models", [])]
        if selected_models is None:
            selected_models = available
        missing = sorted(set(selected_models) - set(available))
        if missing:
            raise ValueError(f"{path} is missing requested models {missing}; available={available}")
        for observation in panel.get("observations", []):
            obs_index = int(observation["index"])
            grid_reference = observation.get("grid_reference", {}) or {}
            timing = observation.get("timing_seconds", {}) or {}
            model_timing = timing.get("model_sampling", {}) or {}
            distance_metrics = observation.get("distance_metrics", {}) or {}
            for model in selected_models:
                result = observation["models"][model]
                rows.append({
                    "grid_label": label,
                    "panel_summary": str(path),
                    "observation_index": obs_index,
                    "model": model,
                    "grid_size": int(grid_reference.get("grid_size", 0)),
                    "grid_points": int(grid_reference.get("grid_points", 0)),
                    "grid_expansions": int(grid_reference.get("grid_expansions", 0)),
                    "max_edge_mass": float(grid_reference.get("max_edge_mass", np.nan)),
                    "mean_normalized_wasserstein": float(result["mean_normalized_wasserstein"]),
                    "grid_tolerance_ratio": float(result.get("grid_tolerance_ratio", np.nan)),
                    "tau_grid": float(result.get("tau_grid", np.nan)),
                    "npe_sampling_seconds": float(model_timing.get(model, np.nan)),
                    "grid_reference_seconds": float(timing.get("grid_reference", np.nan)),
                    "prior_z_radius": float(distance_metrics.get("prior_z_radius", np.nan)),
                })
    return rows, selected_models or []


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_delta_rows(
    rows: list[dict[str, object]],
    *,
    reference_label: str,
) -> list[dict[str, object]]:
    baseline = {
        (int(row["observation_index"]), str(row["model"])): float(row["mean_normalized_wasserstein"])
        for row in rows
        if row["grid_label"] == reference_label
    }
    output = []
    for row in rows:
        key = (int(row["observation_index"]), str(row["model"]))
        if key not in baseline:
            continue
        delta = float(row["mean_normalized_wasserstein"]) - baseline[key]
        output.append({
            **row,
            "reference_grid_label": reference_label,
            "delta_w_to_reference_grid": delta,
            "abs_delta_w_to_reference_grid": abs(delta),
        })
    return output


def plot_comparison(
    *,
    rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    labels: list[str],
    models: list[str],
    reference_label: str,
    outfile: Path,
) -> None:
    if len(models) != 1:
        raise ValueError("The comparison plot currently expects one model")
    model = models[0]
    colors = ["#2f6fbb", "#b85c38", "#2f855a", "#7a5cc2", "#374151", "#bf4d5a"]
    all_values = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    finite = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    use_log_x = finite.size > 0 and finite.max() / max(finite.min(), 1e-12) > 80
    if use_log_x:
        bins: int | np.ndarray = np.geomspace(finite.min(), finite.max(), 44)
    else:
        bins = 32

    figure, axes = plt.subplots(2, 3, figsize=(17.2, 9.6))
    hist_ax, ecdf_ax, quant_ax = axes[0]
    delta_ax, runtime_ax, edge_ax = axes[1]

    for index, label in enumerate(labels):
        group = [row for row in rows if row["grid_label"] == label and row["model"] == model]
        values = np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
        color = colors[index % len(colors)]
        hist_ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=label,
        )
        x, y = ecdf(values)
        ecdf_ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)

    hist_ax.set_xlabel("mean normalized Wasserstein to grid posterior")
    hist_ax.set_ylabel("density")
    hist_ax.set_title("Distribution")
    hist_ax.grid(alpha=0.22)
    hist_ax.legend(frameon=False)
    ecdf_ax.set_xlabel("mean normalized Wasserstein to grid posterior")
    ecdf_ax.set_ylabel("empirical CDF")
    ecdf_ax.set_title("ECDF")
    ecdf_ax.grid(alpha=0.22)
    ecdf_ax.legend(frameon=False)
    if use_log_x:
        hist_ax.set_xscale("log")
        ecdf_ax.set_xscale("log")

    x_positions = np.arange(len(labels))
    medians = []
    q90s = []
    q95s = []
    for label in labels:
        group = [row for row in rows if row["grid_label"] == label and row["model"] == model]
        values = np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
        medians.append(float(np.median(values)))
        q90s.append(float(np.quantile(values, 0.90)))
        q95s.append(float(np.quantile(values, 0.95)))
    quant_ax.plot(x_positions, medians, marker="o", linewidth=2.0, label="median", color="#2f6fbb")
    quant_ax.plot(x_positions, q90s, marker="s", linewidth=2.0, label="q90", color="#b85c38")
    quant_ax.plot(x_positions, q95s, marker="^", linewidth=2.0, label="q95", color="#2f855a")
    quant_ax.set_xticks(x_positions)
    quant_ax.set_xticklabels(labels, rotation=20, ha="right")
    quant_ax.set_ylabel("mean normalized Wasserstein")
    quant_ax.set_title("Distribution quantiles")
    quant_ax.grid(alpha=0.22)
    quant_ax.legend(frameon=False)

    delta_values = []
    delta_labels = []
    for label in labels:
        if label == reference_label:
            continue
        group = [
            row
            for row in delta_rows
            if row["grid_label"] == label and row["model"] == model
        ]
        values = np.asarray([row["abs_delta_w_to_reference_grid"] for row in group], dtype=np.float64)
        delta_values.append(values)
        delta_labels.append(label)
    if delta_values:
        delta_ax.boxplot(delta_values, tick_labels=delta_labels, showfliers=False)
        for position, values in enumerate(delta_values, start=1):
            jitter = np.linspace(-0.08, 0.08, min(len(values), 120))
            subset = values[: len(jitter)]
            delta_ax.scatter(
                np.full_like(subset, position, dtype=np.float64) + jitter,
                subset,
                s=14,
                alpha=0.32,
                color="#374151",
            )
    delta_ax.set_ylabel(f"|W(grid) - W({reference_label})|")
    delta_ax.set_title("Same-signal shift vs finest grid")
    delta_ax.grid(axis="y", alpha=0.22)

    runtime_values = []
    for label in labels:
        group = [row for row in rows if row["grid_label"] == label and row["model"] == model]
        runtime_values.append(np.asarray([row["grid_reference_seconds"] for row in group], dtype=np.float64))
    runtime_ax.boxplot(runtime_values, tick_labels=labels, showfliers=False)
    runtime_ax.set_yscale("log")
    runtime_ax.set_ylabel("grid reference seconds")
    runtime_ax.set_title("Runtime")
    runtime_ax.grid(axis="y", alpha=0.22)

    edge_values = []
    for label in labels:
        group = [row for row in rows if row["grid_label"] == label and row["model"] == model]
        edge_values.append(np.asarray([row["max_edge_mass"] for row in group], dtype=np.float64))
    edge_ax.boxplot(edge_values, tick_labels=labels, showfliers=False)
    edge_ax.set_yscale("log")
    edge_ax.set_ylabel("max edge mass")
    edge_ax.set_title("Grid range check")
    edge_ax.grid(axis="y", alpha=0.22)

    figure.suptitle(f"Decay {model} W distribution sensitivity to grid size", y=1.02)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare decay NPE-to-grid Wasserstein distributions across grid sizes.",
    )
    parser.add_argument("--panels", type=parse_panel_list, required=True)
    parser.add_argument("--models", type=parse_model_list, default=None)
    parser.add_argument("--reference-label", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panels = sorted(args.panels, key=lambda item: label_sort_key(item[0]))
    rows, models = collect_rows(panels=panels, models=args.models)
    labels = [label for label, _ in panels]
    reference_label = args.reference_label or labels[-1]
    if reference_label not in labels:
        raise ValueError(f"--reference-label {reference_label!r} is not among labels {labels}")
    delta_rows = build_delta_rows(rows, reference_label=reference_label)

    summary: dict[str, object] = {}
    for label in labels:
        for model in models:
            group = [row for row in rows if row["grid_label"] == label and row["model"] == model]
            key = f"{label}:{model}"
            summary[key] = {
                "grid_size": int(group[0]["grid_size"]) if group else None,
                "grid_points": int(group[0]["grid_points"]) if group else None,
                "mean_normalized_wasserstein": quantile_summary(
                    np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
                ),
                "grid_reference_seconds": quantile_summary(
                    np.asarray([row["grid_reference_seconds"] for row in group], dtype=np.float64)
                ),
                "max_edge_mass": quantile_summary(
                    np.asarray([row["max_edge_mass"] for row in group], dtype=np.float64)
                ),
                "grid_expansions": {
                    str(expansion): int(
                        sum(1 for row in group if int(row["grid_expansions"]) == expansion)
                    )
                    for expansion in sorted({int(row["grid_expansions"]) for row in group})
                },
            }
            if label != reference_label:
                delta_group = [
                    row
                    for row in delta_rows
                    if row["grid_label"] == label and row["model"] == model
                ]
                summary[key]["abs_delta_w_to_reference_grid"] = quantile_summary(
                    np.asarray(
                        [row["abs_delta_w_to_reference_grid"] for row in delta_group],
                        dtype=np.float64,
                    )
                )

    figure_path = figure_dir / "grid_size_wasserstein_comparison.png"
    rows_csv = results_dir / "grid_size_wasserstein_rows.csv"
    delta_csv = results_dir / "grid_size_wasserstein_deltas.csv"
    json_path = results_dir / "grid_size_wasserstein_comparison_summary.json"
    plot_comparison(
        rows=rows,
        delta_rows=delta_rows,
        labels=labels,
        models=models,
        reference_label=reference_label,
        outfile=figure_path,
    )
    write_csv(rows, rows_csv)
    write_csv(delta_rows, delta_csv)
    output = {
        "panels": [
            {"label": label, "panel_summary": str(path)}
            for label, path in panels
        ],
        "models": models,
        "reference_label": reference_label,
        "summary": summary,
        "outputs": {
            "figure": str(figure_path),
            "rows_csv": str(rows_csv),
            "delta_csv": str(delta_csv),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    for key, values in summary.items():
        w = values["mean_normalized_wasserstein"]
        print(
            f"{key}: median={w['median']:.4f}, q90={w['q90']:.4f}, "
            f"q95={w['q95']:.4f}, max={w['max']:.4f}"
        )


if __name__ == "__main__":
    main()
