from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from scipy.stats import wasserstein_distance

import npe_flow_decay as decay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_RUN_SUMMARY = Path(
    "runs/01_exponential_decay/03_npe_flow_search/"
    "11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/"
    "npe_flow_decay_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe")


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


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


def load_flow_checkpoint(path: Path, device: torch.device) -> tuple[decay.ConditionalSplineFlow, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = decay.ConditionalSplineFlow(
        z_dim=3,
        context_dim=len(np.asarray(checkpoint["context_mean"])),
        transforms=int(config["transforms"]),
        hidden_features=tuple(int(v) for v in config["hidden_features"]),
        bins=int(config["bins"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "config": config,
        "context_mean": np.asarray(checkpoint["context_mean"], dtype=np.float64),
        "context_std": np.asarray(checkpoint["context_std"], dtype=np.float64),
        "z_mean": np.asarray(checkpoint["z_mean"], dtype=np.float64),
        "z_std": np.asarray(checkpoint["z_std"], dtype=np.float64),
        "linear_adjustment": checkpoint.get("linear_adjustment"),
    }


def mean_normalized_reference_distance(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    values = []
    left_theta = np.asarray(left["theta_grid"])
    right_theta = np.asarray(right["theta_grid"])
    left_weights = np.asarray(left["weights"])
    right_weights = np.asarray(right["weights"])
    right_summary = right["summary"]
    for index, name in enumerate(decay.PARAMETER_NAMES):
        w = wasserstein_distance(
            left_theta[:, index],
            right_theta[:, index],
            u_weights=left_weights,
            v_weights=right_weights,
        )
        normalized = float(w / max(right_summary[name]["sd"], 1e-12))
        metrics[name] = {
            "wasserstein": float(w),
            "wasserstein_in_right_ref_sd": normalized,
        }
        values.append(normalized)
    metrics["mean_normalized_wasserstein"] = {"value": float(np.mean(values))}
    return metrics


def sorted_weighted_cache(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.concatenate(([0.0], np.cumsum(sorted_weights, dtype=np.float64)))
    cdf /= max(cdf[-1], 1e-300)
    return sorted_values, cdf


def wasserstein_uniform_to_cached(sample_values: np.ndarray, ref_values: np.ndarray, ref_cdf: np.ndarray) -> float:
    sample_sorted = np.sort(sample_values.astype(np.float64, copy=False))
    n = sample_sorted.size
    all_values = np.concatenate((sample_sorted, ref_values))
    all_values.sort()
    deltas = np.diff(all_values)
    if deltas.size == 0:
        return 0.0
    points = all_values[:-1]
    sample_cdf = np.searchsorted(sample_sorted, points, side="right") / max(n, 1)
    ref_cdf_values = ref_cdf[np.searchsorted(ref_values, points, side="right")]
    return float(np.sum(np.abs(sample_cdf - ref_cdf_values) * deltas))


def wasserstein_uniform_to_uniform(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_sorted = np.sort(left_values.astype(np.float64, copy=False))
    right_sorted = np.sort(right_values.astype(np.float64, copy=False))
    all_values = np.concatenate((left_sorted, right_sorted))
    all_values.sort()
    deltas = np.diff(all_values)
    if deltas.size == 0:
        return 0.0
    points = all_values[:-1]
    left_cdf = np.searchsorted(left_sorted, points, side="right") / max(left_sorted.size, 1)
    right_cdf = np.searchsorted(right_sorted, points, side="right") / max(right_sorted.size, 1)
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def compare_samples_to_reference_fast(
    samples: np.ndarray,
    reference: dict[str, object],
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    values = []
    ref_summary = reference["summary"]
    for index, name in enumerate(decay.PARAMETER_NAMES):
        ref_values, ref_cdf = reference_cache[name]
        w = wasserstein_uniform_to_cached(samples[:, index], ref_values, ref_cdf)
        normalized = float(w / max(ref_summary[name]["sd"], 1e-12))
        metrics[name] = {
            "wasserstein_to_grid": float(w),
            "wasserstein_to_grid_in_ref_sd": normalized,
        }
        values.append(normalized)
    metrics["mean_normalized_wasserstein"] = {"value": float(np.mean(values))}
    return metrics


def compare_samples_to_samples(
    left: np.ndarray,
    right: np.ndarray,
    reference: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    values = []
    ref_summary = reference["summary"]
    for index, name in enumerate(decay.PARAMETER_NAMES):
        w = wasserstein_uniform_to_uniform(left[:, index], right[:, index])
        normalized = float(w / max(ref_summary[name]["sd"], 1e-12))
        metrics[name] = {
            "wasserstein": float(w),
            "wasserstein_in_ref_sd": normalized,
        }
        values.append(normalized)
    metrics["mean_normalized_wasserstein"] = {"value": float(np.mean(values))}
    return metrics


def max_edge_mass(reference: dict[str, object]) -> float:
    values = []
    for item in reference["edge_mass"].values():
        values.extend([float(item["lower"]), float(item["upper"])])
    return max(values) if values else float("nan")


def build_reference_cache(reference: dict[str, object]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    theta_grid = np.asarray(reference["theta_grid"])
    weights = np.asarray(reference["weights"])
    return {
        name: sorted_weighted_cache(theta_grid[:, index], weights)
        for index, name in enumerate(decay.PARAMETER_NAMES)
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_key(rows: list[dict[str, object]], key: str, value: str) -> list[dict[str, object]]:
    output = []
    for key_value in sorted({row[key] for row in rows}):
        group = [row for row in rows if row[key] == key_value]
        data = np.asarray([row[value] for row in group], dtype=np.float64)
        output.append({
            key: key_value,
            "n": int(data.size),
            f"{value}_mean": float(np.mean(data)),
            f"{value}_sd": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
            f"{value}_min": float(np.min(data)),
            f"{value}_q16": float(np.quantile(data, 0.16)),
            f"{value}_median": float(np.median(data)),
            f"{value}_q84": float(np.quantile(data, 0.84)),
            f"{value}_max": float(np.max(data)),
        })
    return output


def plot_probe(
    *,
    grid_rows: list[dict[str, object]],
    sample_summary: list[dict[str, object]],
    self_summary: list[dict[str, object]],
    fixed_sample_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax = axes[0, 0]
    ax.plot(
        [row["grid_size"] for row in grid_rows],
        [row["grid_to_max_mean_normalized_wasserstein"] for row in grid_rows],
        marker="o",
        color="#2f6fbb",
        label="grid vs max grid",
    )
    ax.plot(
        [row["grid_size"] for row in grid_rows],
        [row["mcmc_to_grid_mean_normalized_wasserstein"] for row in grid_rows],
        marker="o",
        color="#2f855a",
        label="MCMC vs grid",
    )
    ax.plot(
        [row["grid_size"] for row in grid_rows],
        [row["hmc_to_grid_mean_normalized_wasserstein"] for row in grid_rows],
        marker="o",
        color="#b85c38",
        label="HMC vs grid",
    )
    ax.set_xlabel("reference grid size per dimension")
    ax.set_ylabel("mean normalized Wasserstein")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(
        [row["grid_size"] for row in fixed_sample_rows],
        [row["fixed_sample_to_grid_mean_normalized_wasserstein"] for row in fixed_sample_rows],
        marker="o",
        color="#7a5cc2",
    )
    ax.set_xlabel("reference grid size per dimension")
    ax.set_ylabel("fixed NPE sample to grid W")
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    x = np.asarray([row["sample_size"] for row in sample_summary], dtype=np.float64)
    median = np.asarray(
        [row["sample_to_grid_mean_normalized_wasserstein_median"] for row in sample_summary],
        dtype=np.float64,
    )
    q16 = np.asarray(
        [row["sample_to_grid_mean_normalized_wasserstein_q16"] for row in sample_summary],
        dtype=np.float64,
    )
    q84 = np.asarray(
        [row["sample_to_grid_mean_normalized_wasserstein_q84"] for row in sample_summary],
        dtype=np.float64,
    )
    ax.plot(x, median, marker="o", color="#2f6fbb")
    ax.fill_between(x, q16, q84, color="#2f6fbb", alpha=0.16)
    ax.set_xscale("log")
    ax.set_xlabel("NPE posterior samples")
    ax.set_ylabel("NPE sample to grid W")
    ax.grid(alpha=0.22)

    ax = axes[1, 1]
    if self_summary:
        x = np.asarray([row["sample_size"] for row in self_summary], dtype=np.float64)
        median = np.asarray(
            [row["sample_to_sample_mean_normalized_wasserstein_median"] for row in self_summary],
            dtype=np.float64,
        )
        q16 = np.asarray(
            [row["sample_to_sample_mean_normalized_wasserstein_q16"] for row in self_summary],
            dtype=np.float64,
        )
        q84 = np.asarray(
            [row["sample_to_sample_mean_normalized_wasserstein_q84"] for row in self_summary],
            dtype=np.float64,
        )
        ax.plot(x, median, marker="o", color="#b85c38")
        ax.fill_between(x, q16, q84, color="#b85c38", alpha=0.16)
        ax.set_xscale("log")
    ax.set_xlabel("NPE posterior samples")
    ax.set_ylabel("same-model sample-to-sample W")
    ax.grid(alpha=0.22)

    figure.suptitle("NPE Wasserstein metric noise-floor probe", y=0.995, fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether finite NPE posterior samples or grid resolution explain a Wasserstein floor.",
    )
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--model-pt", type=Path, default=None)
    parser.add_argument("--samples-npz", type=Path, default=None)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-sizes", type=parse_int_list, default=(60, 90, 120, 150))
    parser.add_argument("--metric-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument(
        "--sample-sizes",
        type=parse_int_list,
        default=(10_000, 25_000, 50_000, 100_000, 180_000, 300_000, 500_000),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    model_path = args.model_pt or Path(run_summary["outputs"]["model_pt"])
    samples_path = args.samples_npz or Path(run_summary["outputs"]["samples_npz"])
    samples_npz = np.load(samples_path, allow_pickle=True)
    fixed_theta_samples = np.asarray(samples_npz["theta_samples"])
    observed_context = np.asarray(samples_npz["observed_context"])

    mcmc = decay.load_samples(args.mcmc_samples, "MCMC")
    hmc = decay.load_samples(args.hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])

    references: dict[int, dict[str, object]] = {}
    reference_caches: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    grid_rows: list[dict[str, object]] = []
    fixed_sample_rows: list[dict[str, object]] = []
    grid_sizes = tuple(sorted(set(args.grid_sizes)))
    if args.metric_grid_size not in grid_sizes:
        grid_sizes = tuple(sorted((*grid_sizes, args.metric_grid_size)))
    max_grid_size = max(grid_sizes)

    for grid_size in grid_sizes:
        start = time.perf_counter()
        reference = decay.build_grid_reference(
            t=mcmc["t"],
            y=mcmc["y"],
            combined_z_samples=combined_z,
            true_theta=mcmc["true_theta"],
            grid_size=grid_size,
            chunk_size=args.reference_chunk_size,
        )
        build_seconds = time.perf_counter() - start
        references[grid_size] = reference
        reference_caches[grid_size] = build_reference_cache(reference)
        mcmc_to_grid = decay.compare_to_reference(mcmc["posterior_theta"], reference)
        hmc_to_grid = decay.compare_to_reference(hmc["posterior_theta"], reference)
        grid_rows.append({
            "grid_size": grid_size,
            "grid_points": int(reference["grid_points"]),
            "build_seconds": build_seconds,
            "max_edge_mass": max_edge_mass(reference),
            "mcmc_to_grid_mean_normalized_wasserstein": decay.mean_normalized_wasserstein_value(mcmc_to_grid),
            "hmc_to_grid_mean_normalized_wasserstein": decay.mean_normalized_wasserstein_value(hmc_to_grid),
        })

    max_reference = references[max_grid_size]
    for row in grid_rows:
        grid_size = int(row["grid_size"])
        distance = mean_normalized_reference_distance(references[grid_size], max_reference)
        row["grid_to_max_mean_normalized_wasserstein"] = decay.mean_normalized_wasserstein_value(distance)

    for grid_size in grid_sizes:
        start = time.perf_counter()
        metrics = compare_samples_to_reference_fast(
            fixed_theta_samples,
            references[grid_size],
            reference_caches[grid_size],
        )
        fixed_sample_rows.append({
            "grid_size": grid_size,
            "grid_points": int(references[grid_size]["grid_points"]),
            "fixed_sample_count": int(fixed_theta_samples.shape[0]),
            "fixed_sample_to_grid_mean_normalized_wasserstein": decay.mean_normalized_wasserstein_value(metrics),
            "metric_seconds": time.perf_counter() - start,
        })

    metric_reference = references[int(args.metric_grid_size)]
    metric_cache = reference_caches[int(args.metric_grid_size)]
    device = decay.choose_device(args.device)
    model, state = load_flow_checkpoint(model_path, device)
    sample_sizes = tuple(sorted(set(args.sample_sizes)))
    max_sample_size = max(sample_sizes)
    sample_rows: list[dict[str, object]] = []
    sampled_repeats: list[np.ndarray] = []

    for repeat in range(args.repeats):
        torch.manual_seed(args.seed + repeat)
        start = time.perf_counter()
        _z_samples, theta_samples = decay.sample_flow_posterior(
            model=model,
            observed_context=observed_context,
            context_mean=np.asarray(state["context_mean"]),
            context_std=np.asarray(state["context_std"]),
            z_mean=np.asarray(state["z_mean"]),
            z_std=np.asarray(state["z_std"]),
            linear_adjustment=state["linear_adjustment"],
            n=max_sample_size,
            device=device,
        )
        sample_seconds = time.perf_counter() - start
        sampled_repeats.append(theta_samples)
        for sample_size in sample_sizes:
            start = time.perf_counter()
            metrics = compare_samples_to_reference_fast(
                theta_samples[:sample_size],
                metric_reference,
                metric_cache,
            )
            sample_rows.append({
                "repeat": repeat,
                "sample_size": sample_size,
                "metric_grid_size": int(args.metric_grid_size),
                "sample_seconds_for_max_size": sample_seconds,
                "sample_seconds_scaled": sample_seconds * sample_size / max_sample_size,
                "sample_to_grid_mean_normalized_wasserstein": decay.mean_normalized_wasserstein_value(metrics),
                "metric_seconds": time.perf_counter() - start,
            })

    self_rows: list[dict[str, object]] = []
    if len(sampled_repeats) >= 2:
        for left_index in range(len(sampled_repeats)):
            right_index = (left_index + 1) % len(sampled_repeats)
            if right_index == left_index:
                continue
            for sample_size in sample_sizes:
                start = time.perf_counter()
                metrics = compare_samples_to_samples(
                    sampled_repeats[left_index][:sample_size],
                    sampled_repeats[right_index][:sample_size],
                    metric_reference,
                )
                self_rows.append({
                    "left_repeat": left_index,
                    "right_repeat": right_index,
                    "sample_size": sample_size,
                    "metric_grid_size": int(args.metric_grid_size),
                    "sample_to_sample_mean_normalized_wasserstein": decay.mean_normalized_wasserstein_value(metrics),
                    "metric_seconds": time.perf_counter() - start,
                })

    sample_summary = summarize_by_key(
        sample_rows,
        key="sample_size",
        value="sample_to_grid_mean_normalized_wasserstein",
    )
    self_summary = summarize_by_key(
        self_rows,
        key="sample_size",
        value="sample_to_sample_mean_normalized_wasserstein",
    )

    grid_csv = results_dir / "grid_sensitivity.csv"
    fixed_csv = results_dir / "fixed_sample_grid_sensitivity.csv"
    sample_csv = results_dir / "posterior_sample_sensitivity.csv"
    sample_summary_csv = results_dir / "posterior_sample_sensitivity_summary.csv"
    self_csv = results_dir / "posterior_sample_self_noise.csv"
    self_summary_csv = results_dir / "posterior_sample_self_noise_summary.csv"
    summary_json = results_dir / "metric_noise_probe_summary.json"
    figure_path = figures_dir / "metric_noise_probe.png"

    write_csv(grid_rows, grid_csv)
    write_csv(fixed_sample_rows, fixed_csv)
    write_csv(sample_rows, sample_csv)
    write_csv(sample_summary, sample_summary_csv)
    write_csv(self_rows, self_csv)
    write_csv(self_summary, self_summary_csv)
    plot_probe(
        grid_rows=grid_rows,
        fixed_sample_rows=fixed_sample_rows,
        sample_summary=sample_summary,
        self_summary=self_summary,
        output_path=figure_path,
    )

    output = {
        "config": {
            **vars(args),
            "model_pt": model_path,
            "samples_npz": samples_path,
        },
        "source_run": {
            "summary_json": args.run_summary,
            "reported_wasserstein": run_summary["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"],
            "reported_posterior_samples": run_summary["config"]["posterior_samples"],
            "reported_reference_grid_size": run_summary["config"]["reference_grid_size"],
            "reported_training_seconds": run_summary["training"]["training_seconds"],
            "reported_total_seconds": run_summary["timing_seconds"]["total"],
        },
        "grid_sensitivity": grid_rows,
        "fixed_sample_grid_sensitivity": fixed_sample_rows,
        "posterior_sample_sensitivity": sample_rows,
        "posterior_sample_sensitivity_summary": sample_summary,
        "posterior_sample_self_noise": self_rows,
        "posterior_sample_self_noise_summary": self_summary,
        "timing_seconds": {
            "total": time.perf_counter() - total_start,
        },
        "outputs": {
            "grid_sensitivity_csv": grid_csv,
            "fixed_sample_grid_sensitivity_csv": fixed_csv,
            "posterior_sample_sensitivity_csv": sample_csv,
            "posterior_sample_sensitivity_summary_csv": sample_summary_csv,
            "posterior_sample_self_noise_csv": self_csv,
            "posterior_sample_self_noise_summary_csv": self_summary_csv,
            "summary_json": summary_json,
            "figure": figure_path,
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")
    print(f"total_seconds: {output['timing_seconds']['total']:.2f}")


if __name__ == "__main__":
    main()
