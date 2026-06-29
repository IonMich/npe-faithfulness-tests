from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from abc_faithfulness_decay import make_k_grid
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD
from npe_flow_decay import make_context_summaries, sample_prior_z

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/01_exponential_decay/09_distance_distributions/01_grid_eval")


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


def parse_metrics(value: str) -> list[str]:
    metrics = [piece.strip() for piece in value.split(",") if piece.strip()]
    valid = {
        "prior_z_radius",
        "summary_prior_predictive_radius",
        "local_x0_distance_over_radius",
        "parameter_region_distance_over_radius",
    }
    invalid = sorted(set(metrics) - valid)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown distance metrics: {invalid}")
    return metrics


def simulate_x_from_z(z: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.exp(z)
    mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
    return mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)


def prior_z_radius(z: np.ndarray) -> np.ndarray:
    mean = PRIOR_LOG_MEAN.numpy()
    std = PRIOR_LOG_STD.numpy()
    return np.sqrt(np.sum(((z - mean[None, :]) / std[None, :]) ** 2, axis=1))


def theta_true_to_z(observation: dict[str, object]) -> np.ndarray:
    if observation.get("z_true") is not None:
        return np.asarray(observation["z_true"], dtype=np.float64)
    theta_true = observation["theta_true"]
    if isinstance(theta_true, dict):
        theta = np.asarray([theta_true[name] for name in PARAMETER_NAMES], dtype=np.float64)
    else:
        theta = np.asarray(theta_true, dtype=np.float64)
    return np.log(theta)


def fit_summary_metric(
    *,
    t: np.ndarray,
    k_grid: np.ndarray,
    context_kind: str,
    pilot_simulations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = sample_prior_z(pilot_simulations, rng)
    x = simulate_x_from_z(z, t, rng)
    summary = make_context_summaries(
        x,
        t,
        k_grid,
        kind=context_kind,
        chunk_size=chunk_size,
    )
    mean = summary.mean(axis=0)
    scale = np.maximum(summary.std(axis=0, ddof=1), 1e-8)
    distances = np.sqrt(np.sum(((summary - mean[None, :]) / scale[None, :]) ** 2, axis=1))
    return {
        "mean": mean,
        "scale": scale,
        "pilot_distances": distances,
        "pilot_simulations": pilot_simulations,
        "seed": seed,
        "context_kind": context_kind,
        "metric": "diagonal-whitened Euclidean radius in decay summary space",
    }


def summary_radius_for_x(
    *,
    x: np.ndarray,
    t: np.ndarray,
    k_grid: np.ndarray,
    context_kind: str,
    metric: dict[str, object],
) -> float:
    summary = make_context_summaries(
        x[None, :],
        t,
        k_grid,
        kind=context_kind,
        chunk_size=1,
    )[0]
    mean = np.asarray(metric["mean"], dtype=np.float64)
    scale = np.asarray(metric["scale"], dtype=np.float64)
    return float(np.sqrt(np.sum(((summary - mean) / scale) ** 2)))


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
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
    y = np.arange(1, finite.size + 1) / finite.size
    return finite, y


def plot_distributions(
    *,
    rows: list[dict[str, object]],
    pilot: dict[str, np.ndarray],
    metrics: list[str],
    outfile: Path,
    panel_kind: str,
) -> None:
    active_metrics = [
        metric
        for metric in metrics
        if np.isfinite(np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)).any()
    ]
    if not active_metrics:
        raise RuntimeError("No requested distance metrics were available to plot.")
    figure, axes = plt.subplots(
        1,
        len(active_metrics),
        figsize=(5.2 * len(active_metrics), 4.6),
        squeeze=False,
    )
    labels = {
        "prior_z_radius": "prior z-score radius",
        "summary_prior_predictive_radius": "prior-predictive summary radius",
        "local_x0_distance_over_radius": "local x_0 distance / radius",
        "parameter_region_distance_over_radius": "parameter distance / radius",
    }
    for ax, metric in zip(axes[0], active_metrics, strict=True):
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        pilot_values = pilot.get(metric)
        if pilot_values is not None and np.isfinite(pilot_values).any():
            px, py = ecdf(pilot_values)
            ax.plot(px, py, color="#6b7280", linewidth=1.8, label="prior-predictive pilot")
        x, y = ecdf(values)
        ax.step(x, y, where="post", color="#2f6fbb", linewidth=2.2, label="grid-evaluated observations")
        ax.scatter(values, np.full_like(values, 0.04), color="#2f6fbb", s=28, alpha=0.75)
        median = float(np.median(values))
        q90 = float(np.quantile(values, 0.90))
        ax.axvline(median, color="#b85c38", linewidth=1.7, label="panel median")
        ax.axvline(q90, color="#b85c38", linestyle="--", linewidth=1.3, label="panel q90")
        if metric.endswith("_over_radius"):
            ax.axvline(1.0, color="#172033", linestyle=":", linewidth=1.6, label="declared boundary")
        ax.set_xlabel(labels[metric])
        ax.set_ylabel("empirical CDF")
        ax.set_title(labels[metric])
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    figure.suptitle(f"Decay grid-evaluation distance coverage: {panel_kind}", y=1.02)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "index",
        "prior_z_radius",
        "summary_prior_predictive_radius",
        "local_x0_distance",
        "local_x0_distance_over_radius",
        "parameter_region_distance",
        "parameter_region_distance_over_radius",
    ]
    model_names = sorted({
        model_name
        for row in rows
        for model_name in row.get("model_discrepancies", {})
    })
    fields.extend(f"{model}_mean_normalized_wasserstein" for model in model_names)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for model_name in model_names:
                output[f"{model_name}_mean_normalized_wasserstein"] = row.get("model_discrepancies", {}).get(model_name)
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot decay grid-evaluation observation distance distributions.",
    )
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metrics",
        type=parse_metrics,
        default=parse_metrics("prior_z_radius,summary_prior_predictive_radius"),
    )
    parser.add_argument("--pilot-simulations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--context-kind", choices=["indirect", "enhanced"], default=None)
    parser.add_argument("--k-grid-points", type=int, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--summary-chunk-size", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    config = panel.get("config", {})
    panel_kind = str(panel.get("panel_distribution", {}).get("kind", "unknown"))
    n_observations = int(config.get("n_observations_per_curve", 40))
    t = np.linspace(0.0, 6.0, n_observations)
    context_kind = args.context_kind or str(config.get("context_kind", "indirect"))
    k_grid = make_k_grid(
        int(args.k_grid_points or config.get("k_grid_points", 260)),
        float(args.k_min or config.get("k_min", 0.04)),
        float(args.k_max or config.get("k_max", 3.0)),
    )

    need_summary_radius = "summary_prior_predictive_radius" in args.metrics
    summary_metric = None
    if need_summary_radius:
        summary_metric = fit_summary_metric(
            t=t,
            k_grid=k_grid,
            context_kind=context_kind,
            pilot_simulations=args.pilot_simulations,
            seed=args.seed,
            chunk_size=args.summary_chunk_size,
        )

    rng = np.random.default_rng(args.seed + 1)
    pilot_z = sample_prior_z(args.pilot_simulations, rng)
    pilot: dict[str, np.ndarray] = {
        "prior_z_radius": prior_z_radius(pilot_z),
    }
    if summary_metric is not None:
        pilot["summary_prior_predictive_radius"] = np.asarray(
            summary_metric["pilot_distances"],
            dtype=np.float64,
        )

    rows = []
    missing_summary_x = 0
    for observation in panel.get("observations", []):
        z = theta_true_to_z(observation)
        distance_metrics = observation.get("distance_metrics", {}) or {}
        row: dict[str, object] = {
            "index": int(observation["index"]),
            "prior_z_radius": float(distance_metrics.get("prior_z_radius") or prior_z_radius(z[None, :])[0]),
            "summary_prior_predictive_radius": None,
            "local_x0_distance": distance_metrics.get("local_x0_distance"),
            "local_x0_distance_over_radius": distance_metrics.get("local_x0_distance_over_radius"),
            "parameter_region_distance": distance_metrics.get("parameter_region_distance"),
            "parameter_region_distance_over_radius": distance_metrics.get("parameter_region_distance_over_radius"),
            "model_discrepancies": {},
        }
        if summary_metric is not None:
            if observation.get("x") is None:
                missing_summary_x += 1
            else:
                row["summary_prior_predictive_radius"] = summary_radius_for_x(
                    x=np.asarray(observation["x"], dtype=np.float64),
                    t=t,
                    k_grid=k_grid,
                    context_kind=context_kind,
                    metric=summary_metric,
                )
        for model_name, model_result in observation.get("models", {}).items():
            row["model_discrepancies"][model_name] = model_result.get("mean_normalized_wasserstein")
        rows.append(row)

    figure_path = figure_dir / "grid_eval_distance_distribution.png"
    csv_path = results_dir / "grid_eval_distance_distribution.csv"
    json_path = results_dir / "grid_eval_distance_distribution_summary.json"
    plot_distributions(
        rows=rows,
        pilot=pilot,
        metrics=args.metrics,
        outfile=figure_path,
        panel_kind=panel_kind,
    )
    write_csv(rows, csv_path)

    metric_summary = {}
    for metric in args.metrics:
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        metric_summary[metric] = {
            "panel": quantile_summary(values),
            "pilot": quantile_summary(pilot[metric]) if metric in pilot else {"n": 0},
        }
    output = {
        "panel_summary": str(args.panel_summary),
        "panel_kind": panel_kind,
        "config": json_ready(vars(args)),
        "context": {
            "context_kind": context_kind,
            "k_grid_points": int(len(k_grid)),
            "n_observations_per_curve": n_observations,
        },
        "missing": {
            "summary_radius_missing_x_count": missing_summary_x,
        },
        "metric_summary": metric_summary,
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
    for metric, summary in metric_summary.items():
        panel_summary = summary["panel"]
        if panel_summary.get("n", 0):
            print(
                f"{metric}: median={panel_summary['median']:.3f}, "
                f"q90={panel_summary['q90']:.3f}, n={panel_summary['n']}"
            )
        else:
            print(f"{metric}: unavailable")


if __name__ == "__main__":
    main()
