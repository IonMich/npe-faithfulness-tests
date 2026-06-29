from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/01_exponential_decay/10_wasserstein_distributions/05_w_vs_prior_radius")


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
        prior_z_radius = distance_metrics.get("prior_z_radius")
        if prior_z_radius is None:
            continue
        for model in selected:
            result = observation["models"][model]
            rows.append({
                "observation_index": obs_index,
                "model": model,
                "prior_z_radius": float(prior_z_radius),
                "mean_normalized_wasserstein": float(result["mean_normalized_wasserstein"]),
                "grid_tolerance_ratio": float(result.get("grid_tolerance_ratio", np.nan)),
                "tau_grid": float(result.get("tau_grid", np.nan)),
            })
    return rows, selected


def summarize_bins(
    *,
    rows: list[dict[str, object]],
    model: str,
    bin_count: int,
) -> list[dict[str, float | int | str]]:
    model_rows = [row for row in rows if row["model"] == model]
    radius = np.asarray([row["prior_z_radius"] for row in model_rows], dtype=np.float64)
    w = np.asarray([row["mean_normalized_wasserstein"] for row in model_rows], dtype=np.float64)
    if radius.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, min(bin_count, radius.size) + 1)
    edges = np.quantile(radius, quantiles)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    output: list[dict[str, float | int | str]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (radius > left) & (radius <= right)
        if not np.any(mask):
            continue
        values = w[mask]
        radii = radius[mask]
        output.append({
            "model": model,
            "bin_index": index,
            "radius_min": float(np.min(radii)),
            "radius_max": float(np.max(radii)),
            "radius_median": float(np.median(radii)),
            "n": int(values.size),
            "w_q25": float(np.quantile(values, 0.25)),
            "w_median": float(np.median(values)),
            "w_q75": float(np.quantile(values, 0.75)),
            "w_mean": float(np.mean(values)),
        })
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_w_vs_radius(
    *,
    rows: list[dict[str, object]],
    bin_rows: list[dict[str, object]],
    models: list[str],
    output_path: Path,
) -> None:
    colors = ["#2f6fbb", "#b85c38", "#2f855a", "#7a5cc2", "#374151"]
    figure, ax = plt.subplots(figsize=(10.5, 6.3))
    all_w = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    for index, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        radius = np.asarray([row["prior_z_radius"] for row in model_rows], dtype=np.float64)
        w = np.asarray([row["mean_normalized_wasserstein"] for row in model_rows], dtype=np.float64)
        color = colors[index % len(colors)]
        ax.scatter(radius, w, s=20, alpha=0.42, color=color, label=f"{model} signals")
        model_bins = [row for row in bin_rows if row["model"] == model]
        if model_bins:
            x = np.asarray([row["radius_median"] for row in model_bins], dtype=np.float64)
            median = np.asarray([row["w_median"] for row in model_bins], dtype=np.float64)
            q25 = np.asarray([row["w_q25"] for row in model_bins], dtype=np.float64)
            q75 = np.asarray([row["w_q75"] for row in model_bins], dtype=np.float64)
            ax.plot(x, median, color=color, linewidth=2.2, marker="o", label=f"{model} binned median")
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, label=f"{model} binned IQR")
    if all_w.size and all_w.max() / max(all_w.min(), 1e-12) > 25:
        ax.set_yscale("log")
    ax.set_xlabel("prior-space radius of generating theta")
    ax.set_ylabel("mean normalized Wasserstein to grid posterior")
    ax.set_title("Decay NPE error versus generating-parameter prior radius")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot decay NPE-to-grid Wasserstein against prior-space radius of true parameters.",
    )
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=parse_model_list, default=None)
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    rows, models = collect_rows(panel, args.models)
    bin_rows: list[dict[str, object]] = []
    for model in models:
        bin_rows.extend(summarize_bins(rows=rows, model=model, bin_count=args.bins))

    figure_path = figure_dir / "npe_wasserstein_vs_prior_radius.png"
    rows_csv = results_dir / "npe_wasserstein_vs_prior_radius.csv"
    bins_csv = results_dir / "npe_wasserstein_vs_prior_radius_bins.csv"
    json_path = results_dir / "npe_wasserstein_vs_prior_radius_summary.json"
    plot_w_vs_radius(
        rows=rows,
        bin_rows=bin_rows,
        models=models,
        output_path=figure_path,
    )
    write_csv(rows, rows_csv)
    write_csv(bin_rows, bins_csv)

    output = {
        "panel_summary": str(args.panel_summary),
        "models": models,
        "rows": rows,
        "bin_summary": bin_rows,
        "outputs": {
            "figure": str(figure_path),
            "rows_csv": str(rows_csv),
            "bins_csv": str(bins_csv),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    print(f"rows_csv: {rows_csv}")


if __name__ == "__main__":
    main()
