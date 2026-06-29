from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from compare_decay_samplers import compare_to_reference
from evaluate_decay_amortization_panel import (
    build_adaptive_grid_reference,
    initial_z_ranges,
    max_edge_mass,
    sample_prior_predictive_observations,
)
from mcmc_decay_inference import PARAMETER_NAMES
from npe_stage1_decay import sample_posterior_for_observation
from evaluate_decay_amortization_panel import load_stage1_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_STAGE1_DIR = Path("runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/results")
DEFAULT_OUTPUT_DIR = Path("runs/01_exponential_decay/11_convergence_benchmarks/01_stage1_prior")


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


def parse_int_list(value: str) -> list[int]:
    values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Values must be positive.")
    return values


def weighted_grid_to_grid_wasserstein(
    reference: dict[str, object],
    baseline: dict[str, object],
) -> float:
    values = []
    left_theta = np.asarray(reference["theta_grid"], dtype=np.float64)
    right_theta = np.asarray(baseline["theta_grid"], dtype=np.float64)
    left_weights = np.asarray(reference["weights"], dtype=np.float64)
    right_weights = np.asarray(baseline["weights"], dtype=np.float64)
    for index, name in enumerate(PARAMETER_NAMES):
        sd = max(float(baseline["summary"][name]["sd"]), 1e-12)
        values.append(
            wasserstein_distance(
                left_theta[:, index],
                right_theta[:, index],
                u_weights=left_weights,
                v_weights=right_weights,
            )
            / sd
        )
    return float(np.mean(values))


def sample_to_sample_normalized_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    baseline_reference: dict[str, object],
) -> float:
    values = []
    for index, name in enumerate(PARAMETER_NAMES):
        sd = max(float(baseline_reference["summary"][name]["sd"]), 1e-12)
        values.append(wasserstein_distance(left[:, index], right[:, index]) / sd)
    return float(np.mean(values))


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "q16": float(np.quantile(arr, 0.16)),
        "median": float(np.median(arr)),
        "q84": float(np.quantile(arr, 0.84)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def median_by_key(rows: list[dict[str, object]], key: str, value_key: str) -> list[dict[str, float | int]]:
    output = []
    for item in sorted({int(row[key]) for row in rows}):
        values = [float(row[value_key]) for row in rows if int(row[key]) == item]
        times = [float(row["seconds"]) for row in rows if int(row[key]) == item]
        output.append({
            key: item,
            "value_median": float(np.median(values)),
            "value_q16": float(np.quantile(values, 0.16)),
            "value_q84": float(np.quantile(values, 0.84)),
            "seconds_median": float(np.median(times)),
            "seconds_q16": float(np.quantile(times, 0.16)),
            "seconds_q84": float(np.quantile(times, 0.84)),
            "n": len(values),
        })
    return output


def plot_convergence(
    *,
    grid_rows: list[dict[str, object]],
    npe_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    outfile: Path,
) -> None:
    grid_summary = median_by_key(grid_rows, "grid_size", "grid_to_finest_w")
    npe_summary = median_by_key(npe_rows, "posterior_samples", "npe_sample_to_large_sample_w")
    stability_summary = median_by_key(stability_rows, "posterior_samples", "npe_to_finest_grid_w")

    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))

    grid_x = np.asarray([row["grid_size"] for row in grid_summary], dtype=np.float64)
    grid_y = np.asarray([row["value_median"] for row in grid_summary], dtype=np.float64)
    grid_q16 = np.asarray([row["value_q16"] for row in grid_summary], dtype=np.float64)
    grid_q84 = np.asarray([row["value_q84"] for row in grid_summary], dtype=np.float64)
    axes[0].plot(grid_x, grid_y, marker="o", color="#2f6fbb", linewidth=2.0)
    axes[0].fill_between(grid_x, grid_q16, grid_q84, color="#2f6fbb", alpha=0.16)
    axes[0].set_xlabel("grid size per dimension")
    axes[0].set_ylabel("mean normalized W to finest grid")
    axes[0].set_title("Grid convergence")
    axes[0].grid(alpha=0.22)

    npe_x = np.asarray([row["posterior_samples"] for row in npe_summary], dtype=np.float64)
    npe_y = np.asarray([row["value_median"] for row in npe_summary], dtype=np.float64)
    npe_q16 = np.asarray([row["value_q16"] for row in npe_summary], dtype=np.float64)
    npe_q84 = np.asarray([row["value_q84"] for row in npe_summary], dtype=np.float64)
    axes[1].plot(npe_x, npe_y, marker="o", color="#b85c38", linewidth=2.0)
    axes[1].fill_between(npe_x, npe_q16, npe_q84, color="#b85c38", alpha=0.16)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("NPE posterior samples")
    axes[1].set_ylabel("mean normalized W to 1M NPE sample")
    axes[1].set_title("NPE Monte Carlo convergence")
    axes[1].grid(alpha=0.22)

    stability_x = np.asarray([row["posterior_samples"] for row in stability_summary], dtype=np.float64)
    stability_y = np.asarray([row["value_median"] for row in stability_summary], dtype=np.float64)
    stability_q16 = np.asarray([row["value_q16"] for row in stability_summary], dtype=np.float64)
    stability_q84 = np.asarray([row["value_q84"] for row in stability_summary], dtype=np.float64)
    axes[2].plot(stability_x, stability_y, marker="o", color="#2f855a", linewidth=2.0)
    axes[2].fill_between(stability_x, stability_q16, stability_q84, color="#2f855a", alpha=0.16)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("NPE posterior samples")
    axes[2].set_ylabel("NPE-to-finest-grid mean normalized W")
    axes[2].set_title("Estimator error stability")
    axes[2].grid(alpha=0.22)

    for ax in axes:
        ax.axhline(0.034, color="#172033", linestyle="--", linewidth=1.2, alpha=0.75)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decay NPE sampling and grid-reference convergence.",
    )
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--family", default="mdn")
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument("--grid-sizes", type=parse_int_list, default=parse_int_list("25,35,45,60,75,90,120"))
    parser.add_argument("--posterior-sample-counts", type=parse_int_list, default=parse_int_list("1000,3000,10000,30000,100000,300000,1000000"))
    parser.add_argument("--large-npe-samples", type=int, default=1_000_000)
    parser.add_argument("--grid-chunk-size", type=int, default=120_000)
    parser.add_argument("--grid-range-padding", type=float, default=0.45)
    parser.add_argument("--grid-min-padding", type=float, default=0.16)
    parser.add_argument("--edge-mass-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-grid-expansions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    model, stats = load_stage1_model(
        family=args.family,
        checkpoint_path=args.stage1_dir / f"{args.family}_model.pt",
        device=device,
    )
    t, x_panel, z_true, _, panel_metadata = sample_prior_predictive_observations(
        n=args.num_observations,
        seed=args.seed,
        n_observations=40,
    )
    finest_grid_size = max(args.grid_sizes)

    grid_rows: list[dict[str, object]] = []
    npe_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []

    for obs_index in range(args.num_observations):
        observed_x = x_panel[obs_index]
        true_z = z_true[obs_index]
        print(f"observation {obs_index + 1}/{args.num_observations}", flush=True)

        large_start = time.perf_counter()
        large_z, large_theta = sample_posterior_for_observation(
            model=model,
            observed_x=observed_x,
            x_mean=stats["x_mean"],
            x_std=stats["x_std"],
            z_mean=stats["z_mean"],
            z_std=stats["z_std"],
            n=args.large_npe_samples,
            device=device,
        )
        large_seconds = time.perf_counter() - large_start
        z_ranges = initial_z_ranges(
            z_samples_by_model={args.family: large_z},
            true_z=true_z,
            padding_fraction=args.grid_range_padding,
            min_padding=args.grid_min_padding,
        )

        references: dict[int, dict[str, object]] = {}
        for grid_size in args.grid_sizes:
            grid_start = time.perf_counter()
            reference, expansions = build_adaptive_grid_reference(
                t=t,
                y=observed_x,
                z_ranges=z_ranges,
                grid_size=grid_size,
                chunk_size=args.grid_chunk_size,
                edge_mass_tolerance=args.edge_mass_tolerance,
                max_expansions=args.max_grid_expansions,
                restricted_region=None,
            )
            seconds = time.perf_counter() - grid_start
            references[grid_size] = reference
            grid_rows.append({
                "observation_index": obs_index,
                "grid_size": grid_size,
                "grid_points": int(reference["grid_points"]),
                "seconds": seconds,
                "grid_expansions": expansions,
                "max_edge_mass": max_edge_mass(reference),
                "grid_to_finest_w": np.nan,
            })
            print(f"  grid {grid_size}^3 seconds={seconds:.3f}", flush=True)

        finest_reference = references[finest_grid_size]
        for row in grid_rows:
            if int(row["observation_index"]) != obs_index:
                continue
            grid_size = int(row["grid_size"])
            if grid_size == finest_grid_size:
                row["grid_to_finest_w"] = 0.0
            else:
                row["grid_to_finest_w"] = weighted_grid_to_grid_wasserstein(
                    references[grid_size],
                    finest_reference,
                )

        for n in args.posterior_sample_counts:
            sample_start = time.perf_counter()
            _, theta_samples = sample_posterior_for_observation(
                model=model,
                observed_x=observed_x,
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
                z_mean=stats["z_mean"],
                z_std=stats["z_std"],
                n=n,
                device=device,
            )
            sample_seconds = time.perf_counter() - sample_start
            npe_mc_w = sample_to_sample_normalized_wasserstein(
                theta_samples,
                large_theta,
                finest_reference,
            )
            npe_to_grid_start = time.perf_counter()
            npe_to_grid = float(compare_to_reference(theta_samples, finest_reference)["mean_normalized_wasserstein"]["value"])
            compare_seconds = time.perf_counter() - npe_to_grid_start
            npe_rows.append({
                "observation_index": obs_index,
                "posterior_samples": n,
                "seconds": sample_seconds,
                "npe_sample_to_large_sample_w": npe_mc_w,
            })
            stability_rows.append({
                "observation_index": obs_index,
                "posterior_samples": n,
                "seconds": sample_seconds,
                "compare_seconds": compare_seconds,
                "npe_to_finest_grid_w": npe_to_grid,
            })
            print(f"  npe n={n} seconds={sample_seconds:.3f} mc_w={npe_mc_w:.4f}", flush=True)

        observation_rows.append({
            "observation_index": obs_index,
            "large_npe_samples": args.large_npe_samples,
            "large_npe_seconds": large_seconds,
            "theta_true": np.exp(true_z).tolist(),
        })

    figure_path = figure_dir / "decay_npe_grid_convergence.png"
    grid_csv = results_dir / "grid_convergence.csv"
    npe_csv = results_dir / "npe_sampling_convergence.csv"
    stability_csv = results_dir / "npe_to_grid_stability.csv"
    json_path = results_dir / "decay_npe_grid_convergence_summary.json"
    plot_convergence(
        grid_rows=grid_rows,
        npe_rows=npe_rows,
        stability_rows=stability_rows,
        outfile=figure_path,
    )
    write_csv(grid_rows, grid_csv)
    write_csv(npe_rows, npe_csv)
    write_csv(stability_rows, stability_csv)

    grid_summary = median_by_key(grid_rows, "grid_size", "grid_to_finest_w")
    npe_summary = median_by_key(npe_rows, "posterior_samples", "npe_sample_to_large_sample_w")
    stability_summary = median_by_key(stability_rows, "posterior_samples", "npe_to_finest_grid_w")
    output = {
        "config": json_ready(vars(args)),
        "panel_metadata": panel_metadata,
        "observation_count": args.num_observations,
        "grid_summary": grid_summary,
        "npe_sampling_summary": npe_summary,
        "npe_to_grid_stability_summary": stability_summary,
        "observation_rows": observation_rows,
        "outputs": {
            "figure": str(figure_path),
            "grid_csv": str(grid_csv),
            "npe_csv": str(npe_csv),
            "stability_csv": str(stability_csv),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")

    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    print("grid convergence medians:")
    for row in grid_summary:
        print(
            f"  {row['grid_size']}^3: W_to_finest={row['value_median']:.5f}, "
            f"seconds={row['seconds_median']:.3f}"
        )
    print("NPE MC convergence medians:")
    for row in npe_summary:
        print(
            f"  n={row['posterior_samples']}: W_to_1M={row['value_median']:.5f}, "
            f"seconds={row['seconds_median']:.3f}"
        )


if __name__ == "__main__":
    main()
