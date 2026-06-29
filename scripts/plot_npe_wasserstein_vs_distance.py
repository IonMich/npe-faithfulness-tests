from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from abc_faithfulness_decay import make_k_grid
from compare_decay_samplers import compare_to_reference
from evaluate_decay_amortization_panel import (
    build_adaptive_grid_reference,
    initial_z_ranges,
    max_edge_mass,
)
from mcmc_decay_inference import PARAMETER_NAMES
from npe_flow_decay import (
    ConditionalSplineFlow,
    context_distances,
    make_context_summaries,
    sample_flow_posterior,
    sample_prior_z,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL = Path(
    "runs/01_exponential_decay/03_npe_flow_search/"
    "11_npe_flow_local_q0005_linear_150k_t8_seed20260706/"
    "results/npe_flow_decay_model.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/08_distance_sweep/"
    "01_flow_local_q0005_wasserstein_vs_distance"
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


def parse_float_edges(value: str) -> list[float]:
    edges = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("At least two comma-separated bin edges are required.")
    if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise argparse.ArgumentTypeError("Bin edges must be strictly increasing.")
    if edges[0] < 0:
        raise argparse.ArgumentTypeError("Bin edges must be non-negative.")
    return edges


def simulate_x_from_z(z: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.exp(z)
    mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
    return mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)


def load_flow_checkpoint(path: Path, device: torch.device) -> tuple[ConditionalSplineFlow, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConditionalSplineFlow(
        z_dim=3,
        context_dim=len(np.asarray(checkpoint["context_mean"])),
        transforms=int(config["transforms"]),
        hidden_features=tuple(int(v) for v in config["hidden_features"]),
        bins=int(config["bins"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state = {
        "config": config,
        "context_mean": np.asarray(checkpoint["context_mean"], dtype=np.float64),
        "context_std": np.asarray(checkpoint["context_std"], dtype=np.float64),
        "z_mean": np.asarray(checkpoint["z_mean"], dtype=np.float64),
        "z_std": np.asarray(checkpoint["z_std"], dtype=np.float64),
        "linear_adjustment": checkpoint.get("linear_adjustment"),
    }
    return model, state


def collect_distance_binned_observations(
    *,
    rng: np.random.Generator,
    t: np.ndarray,
    k_grid: np.ndarray,
    context_kind: str,
    observed_context: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    radius: float,
    edges: list[float],
    observations_per_bin: int,
    chunk_size: int,
    max_candidates: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bins = list(zip(edges[:-1], edges[1:], strict=True))
    accepted: list[list[dict[str, object]]] = [[] for _ in bins]
    candidate_count = 0
    start = time.perf_counter()
    while candidate_count < max_candidates and any(len(items) < observations_per_bin for items in accepted):
        current = min(chunk_size, max_candidates - candidate_count)
        z = sample_prior_z(current, rng)
        x = simulate_x_from_z(z, t, rng)
        context = make_context_summaries(
            x,
            t,
            k_grid,
            kind=context_kind,
            chunk_size=min(current, chunk_size),
        )
        distances = context_distances(context, observed_context, center, scale)
        normalized = distances / radius
        for bin_index, (lower, upper) in enumerate(bins):
            need = observations_per_bin - len(accepted[bin_index])
            if need <= 0:
                continue
            indices = np.flatnonzero((normalized >= lower) & (normalized < upper))
            if indices.size == 0:
                continue
            if indices.size > need:
                indices = rng.choice(indices, size=need, replace=False)
            for index in indices:
                accepted[bin_index].append({
                    "bin_index": bin_index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "z_true": z[index].copy(),
                    "theta_true": np.exp(z[index]).copy(),
                    "x": x[index].copy(),
                    "distance": float(distances[index]),
                    "distance_over_radius": float(normalized[index]),
                })
        candidate_count += current
        counts = [len(items) for items in accepted]
        print(f"candidate_count={candidate_count} accepted_by_bin={counts}", flush=True)

    missing = {
        f"{lower:g}-{upper:g}": observations_per_bin - len(items)
        for (lower, upper), items in zip(bins, accepted, strict=True)
        if len(items) < observations_per_bin
    }
    if missing:
        raise RuntimeError(
            f"Could not fill all distance bins after {candidate_count} candidates. Missing: {missing}"
        )

    observations = []
    for items in accepted:
        observations.extend(items)
    metadata = {
        "candidate_count": candidate_count,
        "collection_seconds": time.perf_counter() - start,
        "accepted_by_bin": [len(items) for items in accepted],
    }
    return observations, metadata


def evaluate_observation(
    *,
    model: ConditionalSplineFlow,
    state: dict[str, object],
    observation: dict[str, object],
    t: np.ndarray,
    k_grid: np.ndarray,
    context_kind: str,
    posterior_samples: int,
    grid_size: int,
    grid_chunk_size: int,
    grid_range_padding: float,
    grid_min_padding: float,
    edge_mass_tolerance: float,
    max_grid_expansions: int,
    device: torch.device,
) -> dict[str, object]:
    observed_x = np.asarray(observation["x"], dtype=np.float64)
    true_z = np.asarray(observation["z_true"], dtype=np.float64)
    context = make_context_summaries(
        observed_x[None, :],
        t,
        k_grid,
        kind=context_kind,
        chunk_size=1,
    )[0]
    sample_start = time.perf_counter()
    z_samples, theta_samples = sample_flow_posterior(
        model=model,
        observed_context=context,
        context_mean=state["context_mean"],
        context_std=state["context_std"],
        z_mean=state["z_mean"],
        z_std=state["z_std"],
        linear_adjustment=state["linear_adjustment"],
        n=posterior_samples,
        device=device,
    )
    sample_seconds = time.perf_counter() - sample_start

    grid_start = time.perf_counter()
    z_ranges = initial_z_ranges(
        z_samples_by_model={"flow_decay": z_samples},
        true_z=true_z,
        padding_fraction=grid_range_padding,
        min_padding=grid_min_padding,
    )
    reference, grid_expansions = build_adaptive_grid_reference(
        t=t,
        y=observed_x,
        z_ranges=z_ranges,
        grid_size=grid_size,
        chunk_size=grid_chunk_size,
        edge_mass_tolerance=edge_mass_tolerance,
        max_expansions=max_grid_expansions,
        restricted_region=None,
    )
    grid_seconds = time.perf_counter() - grid_start
    metrics = compare_to_reference(theta_samples, reference)
    mean_w = float(metrics["mean_normalized_wasserstein"]["value"])
    return {
        "mean_normalized_wasserstein": mean_w,
        "metrics": metrics,
        "grid_reference": {
            "grid_size": int(reference["grid_size"]),
            "grid_points": int(reference["grid_points"]),
            "grid_expansions": int(grid_expansions),
            "max_edge_mass": float(max_edge_mass(reference)),
            "edge_mass": reference["edge_mass"],
            "z_ranges": reference["z_ranges"],
        },
        "timing_seconds": {
            "npe_sampling": sample_seconds,
            "grid_reference": grid_seconds,
            "grid_points_per_second": int(reference["grid_points"]) / max(grid_seconds, 1e-12),
        },
    }


def summarize_by_bin(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for bin_index in sorted({int(row["bin_index"]) for row in rows}):
        group = [row for row in rows if int(row["bin_index"]) == bin_index]
        values = np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
        distances = np.asarray([row["distance_over_radius"] for row in group], dtype=np.float64)
        output.append({
            "bin_index": bin_index,
            "bin_lower": float(group[0]["bin_lower"]),
            "bin_upper": float(group[0]["bin_upper"]),
            "n": len(group),
            "distance_over_radius_mean": float(np.mean(distances)),
            "w_mean": float(np.mean(values)),
            "w_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "w_q25": float(np.quantile(values, 0.25)),
            "w_median": float(np.median(values)),
            "w_q75": float(np.quantile(values, 0.75)),
            "w_min": float(np.min(values)),
            "w_max": float(np.max(values)),
        })
    return output


def plot_distance_sweep(rows: list[dict[str, object]], bin_summary: list[dict[str, object]], outfile: Path) -> None:
    distance = np.asarray([row["distance_over_radius"] for row in rows], dtype=np.float64)
    wasserstein = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    bin_x = np.asarray([item["distance_over_radius_mean"] for item in bin_summary], dtype=np.float64)
    bin_median = np.asarray([item["w_median"] for item in bin_summary], dtype=np.float64)
    bin_q25 = np.asarray([item["w_q25"] for item in bin_summary], dtype=np.float64)
    bin_q75 = np.asarray([item["w_q75"] for item in bin_summary], dtype=np.float64)

    figure, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.axvspan(0.0, 1.0, color="#2f6fbb", alpha=0.08, label="trained local region")
    ax.axvline(1.0, color="#172033", linestyle="--", linewidth=1.5, label="training radius")
    ax.scatter(distance, wasserstein, s=36, color="#2f6fbb", alpha=0.72, label="held-out signals")
    ax.plot(bin_x, bin_median, color="#b85c38", linewidth=2.1, marker="o", label="bin median")
    ax.fill_between(bin_x, bin_q25, bin_q75, color="#b85c38", alpha=0.18, label="bin IQR")
    if np.nanmax(wasserstein) / max(np.nanmin(wasserstein), 1e-12) > 25:
        ax.set_yscale("log")
    ax.set_xlabel("summary distance from x0 / training radius")
    ax.set_ylabel("NPE to grid mean normalized Wasserstein")
    ax.set_title("Decay NPE degradation with distance from the trained local region")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "index",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "distance",
        "distance_over_radius",
        "mean_normalized_wasserstein",
        "grid_size",
        "grid_points",
        "grid_expansions",
        "max_edge_mass",
        "npe_sampling_seconds",
        "grid_reference_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "index": row["index"],
                "bin_index": row["bin_index"],
                "bin_lower": row["bin_lower"],
                "bin_upper": row["bin_upper"],
                "distance": row["distance"],
                "distance_over_radius": row["distance_over_radius"],
                "mean_normalized_wasserstein": row["mean_normalized_wasserstein"],
                "grid_size": row["grid_reference"]["grid_size"],
                "grid_points": row["grid_reference"]["grid_points"],
                "grid_expansions": row["grid_reference"]["grid_expansions"],
                "max_edge_mass": row["grid_reference"]["max_edge_mass"],
                "npe_sampling_seconds": row["timing_seconds"]["npe_sampling"],
                "grid_reference_seconds": row["timing_seconds"]["grid_reference"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NPE-to-grid Wasserstein as signals move away from the local NPE training center.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=parse_float_edges, default=parse_float_edges("0,1,2,3,4,6,8,12"))
    parser.add_argument("--observations-per-bin", type=int, default=4)
    parser.add_argument("--posterior-samples", type=int, default=10_000)
    parser.add_argument("--grid-size", type=int, default=90)
    parser.add_argument("--grid-chunk-size", type=int, default=120_000)
    parser.add_argument("--grid-range-padding", type=float, default=0.65)
    parser.add_argument("--grid-min-padding", type=float, default=0.24)
    parser.add_argument("--edge-mass-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-grid-expansions", type=int, default=3)
    parser.add_argument("--candidate-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.observations_per_bin < 1:
        raise ValueError("--observations-per-bin must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.summary if args.summary is not None else args.model.parent / "npe_flow_decay_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    local_region = summary["local_training"]["region"]
    observed_context = np.asarray(summary["observed_context"], dtype=np.float64)
    center = np.asarray(local_region["center"], dtype=np.float64)
    scale = np.asarray(local_region["scale"], dtype=np.float64)
    radius = float(local_region["radius"])

    device = torch.device(args.device)
    model, state = load_flow_checkpoint(args.model, device)
    config = state["config"]
    context_kind = str(config.get("context_kind", "indirect"))
    k_grid = make_k_grid(
        int(config.get("k_grid_points", 260)),
        float(config.get("k_min", 0.04)),
        float(config.get("k_max", 3.0)),
    )
    t = np.linspace(0.0, 6.0, 40)
    rng = np.random.default_rng(args.seed)

    observations, collection = collect_distance_binned_observations(
        rng=rng,
        t=t,
        k_grid=k_grid,
        context_kind=context_kind,
        observed_context=observed_context,
        center=center,
        scale=scale,
        radius=radius,
        edges=args.bins,
        observations_per_bin=args.observations_per_bin,
        chunk_size=args.candidate_chunk_size,
        max_candidates=args.max_candidates,
    )

    rows = []
    for index, observation in enumerate(observations):
        print(
            "evaluating "
            f"{index + 1}/{len(observations)} "
            f"d/r={observation['distance_over_radius']:.3f}",
            flush=True,
        )
        result = evaluate_observation(
            model=model,
            state=state,
            observation=observation,
            t=t,
            k_grid=k_grid,
            context_kind=context_kind,
            posterior_samples=args.posterior_samples,
            grid_size=args.grid_size,
            grid_chunk_size=args.grid_chunk_size,
            grid_range_padding=args.grid_range_padding,
            grid_min_padding=args.grid_min_padding,
            edge_mass_tolerance=args.edge_mass_tolerance,
            max_grid_expansions=args.max_grid_expansions,
            device=device,
        )
        rows.append({
            "index": index,
            **{
                key: json_ready(value)
                for key, value in observation.items()
                if key not in {"x", "z_true", "theta_true"}
            },
            "theta_true": {
                name: float(np.asarray(observation["theta_true"])[param_index])
                for param_index, name in enumerate(PARAMETER_NAMES)
            },
            **json_ready(result),
        })
        print(
            "  W="
            f"{result['mean_normalized_wasserstein']:.4f} "
            f"edge={result['grid_reference']['max_edge_mass']:.2e} "
            f"grid_s={result['timing_seconds']['grid_reference']:.2f}",
            flush=True,
        )

    bin_summary = summarize_by_bin(rows)
    figure_path = figure_dir / "npe_wasserstein_vs_distance.png"
    json_path = results_dir / "npe_wasserstein_vs_distance_summary.json"
    csv_path = results_dir / "npe_wasserstein_vs_distance_observations.csv"
    plot_distance_sweep(rows, bin_summary, figure_path)
    write_csv(rows, csv_path)

    output = {
        "config": {
            **{
                key: json_ready(value)
                for key, value in vars(args).items()
            },
            "summary_path": str(summary_path),
            "context_kind": context_kind,
        },
        "local_region": {
            "radius": radius,
            "normalized_training_boundary": 1.0,
            "center_definition": "observed_context x0 in local-region summary-distance metric",
            "region": local_region,
        },
        "collection": collection,
        "bin_summary": bin_summary,
        "observations": rows,
        "outputs": {
            "figure": str(figure_path),
            "summary_json": str(json_path),
            "observations_csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    print(f"observations_csv: {csv_path}")
    print("bin medians:")
    for item in bin_summary:
        print(
            f"  [{item['bin_lower']:g}, {item['bin_upper']:g}): "
            f"median W={item['w_median']:.4f}, "
            f"IQR=[{item['w_q25']:.4f}, {item['w_q75']:.4f}], n={item['n']}"
        )


if __name__ == "__main__":
    main()
