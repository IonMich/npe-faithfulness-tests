from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import build_grid_reference, compare_to_reference, summarize_samples
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    load_reference_samples,
    plot_npe_corner_overlay,
    plot_npe_predictive_overlay,
    plot_training_curves,
    sample_grid_reference,
    sample_posterior_for_observation,
    standardize,
    train_one_model,
)
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def sample_decay_pairs_from_generator(
    *,
    n: int,
    generator: torch.Generator,
    n_observations: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    prior_mean = PRIOR_LOG_MEAN.to(dtype=torch.float64)
    prior_std = PRIOR_LOG_STD.to(dtype=torch.float64)
    z = prior_mean[None, :] + torch.randn(n, 3, generator=generator, dtype=torch.float64) * prior_std[None, :]
    theta = torch.exp(z)
    mean = theta[:, 0:1] * torch.exp(-theta[:, 1:2] * t[None, :])
    x = mean + torch.randn(n, n_observations, generator=generator, dtype=torch.float64) * theta[:, 2:3]
    return x.numpy(), z.numpy(), t.numpy()


def curve_summary(x: np.ndarray, n_bins: int) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
    bins = np.array_split(np.arange(x.shape[1]), n_bins)
    bin_means = np.column_stack([x[:, indices].mean(axis=1) for indices in bins])
    first_diff = np.diff(x, axis=1)
    rough_noise = np.log(np.maximum(first_diff.std(axis=1) / math.sqrt(2.0), 1e-6))
    rough_scale = np.log(np.maximum(x.std(axis=1), 1e-6))
    early_mean = x[:, : max(1, x.shape[1] // 8)].mean(axis=1)
    late_mean = x[:, -max(1, x.shape[1] // 8) :].mean(axis=1)
    return np.column_stack([bin_means, rough_noise, rough_scale, early_mean - late_mean])


def standardized_summary_distance(
    summaries: np.ndarray,
    target_summary: np.ndarray,
    summary_mean: np.ndarray,
    summary_std: np.ndarray,
) -> np.ndarray:
    standardized = (summaries - summary_mean[None, :]) / summary_std[None, :]
    standardized_target = (target_summary - summary_mean) / summary_std
    return np.sqrt(np.mean((standardized - standardized_target[None, :]) ** 2, axis=1))


def fit_local_region(
    *,
    observed_x: np.ndarray,
    pilot_simulations: int,
    summary_bins: int,
    summary_quantile: float,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pilot_x, _, _ = sample_decay_pairs_from_generator(n=pilot_simulations, generator=generator)
    summaries = curve_summary(pilot_x, summary_bins)
    summary_mean = summaries.mean(axis=0)
    summary_std = np.maximum(summaries.std(axis=0), 1e-6)
    target_summary = curve_summary(observed_x, summary_bins)[0]
    distances = standardized_summary_distance(summaries, target_summary, summary_mean, summary_std)
    radius = float(np.quantile(distances, summary_quantile))
    return {
        "radius": radius,
        "target_summary": target_summary,
        "summary_mean": summary_mean,
        "summary_std": summary_std,
        "pilot_distance_summary": {
            "min": float(distances.min()),
            "q01": float(np.quantile(distances, 0.01)),
            "q05": float(np.quantile(distances, 0.05)),
            "q10": float(np.quantile(distances, 0.10)),
            "median": float(np.median(distances)),
            "q90": float(np.quantile(distances, 0.90)),
        },
    }


def collect_local_pairs(
    *,
    target_count: int,
    radius: float,
    target_summary: np.ndarray,
    summary_mean: np.ndarray,
    summary_std: np.ndarray,
    summary_bins: int,
    seed: int,
    chunk_size: int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted_x: list[np.ndarray] = []
    accepted_z: list[np.ndarray] = []
    accepted_distances: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    start = time.perf_counter()
    while accepted_count < target_count and candidate_count < max_candidates:
        current_chunk = min(chunk_size, max_candidates - candidate_count)
        x_chunk, z_chunk, _ = sample_decay_pairs_from_generator(n=current_chunk, generator=generator)
        candidate_count += current_chunk
        summaries = curve_summary(x_chunk, summary_bins)
        distances = standardized_summary_distance(summaries, target_summary, summary_mean, summary_std)
        mask = distances <= radius
        if np.any(mask):
            accepted_x.append(x_chunk[mask])
            accepted_z.append(z_chunk[mask])
            accepted_distances.append(distances[mask])
            accepted_count += int(mask.sum())

    if accepted_count < target_count:
        raise RuntimeError(
            f"Only accepted {accepted_count} local simulations after {candidate_count} candidates; "
            f"need {target_count}. Increase --max-candidates or --summary-quantile."
        )

    x = np.concatenate(accepted_x, axis=0)[:target_count]
    z = np.concatenate(accepted_z, axis=0)[:target_count]
    distances = np.concatenate(accepted_distances, axis=0)[:target_count]
    diagnostics = {
        "candidate_count": int(candidate_count),
        "accepted_count": int(target_count),
        "raw_accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / candidate_count),
        "collection_seconds": float(time.perf_counter() - start),
        "accepted_distance_summary": {
            "min": float(distances.min()),
            "median": float(np.median(distances)),
            "max": float(distances.max()),
        },
    }
    return x, z, distances, diagnostics


def split_train_val(
    *,
    x: np.ndarray,
    z: np.ndarray,
    distances: np.ndarray,
    train_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    train_index = order[:train_count]
    val_index = order[train_count:]
    return (
        x[train_index],
        z[train_index],
        distances[train_index],
        x[val_index],
        z[val_index],
        distances[val_index],
    )


def reference_for_observation(
    *,
    t: np.ndarray,
    observed_x: np.ndarray,
    true_theta: np.ndarray,
    fallback_z: np.ndarray,
    mcmc_samples: Path,
    hmc_samples: Path,
    grid_size: int,
    chunk_size: int,
) -> tuple[dict[str, object], str]:
    reference_z, reference_t, reference_y, reference_true_theta = load_reference_samples(mcmc_samples, hmc_samples)
    if (
        reference_t.shape == t.shape
        and reference_y.shape == observed_x.shape
        and np.allclose(reference_t, t)
        and np.allclose(reference_y, observed_x)
        and np.allclose(reference_true_theta, true_theta)
    ):
        combined_z = reference_z
        source = "mcmc_hmc_samples"
    else:
        combined_z = fallback_z
        source = "local_accepted_z_fallback"
    reference = build_grid_reference(
        t=t,
        y=observed_x,
        combined_z_samples=combined_z,
        true_theta=true_theta,
        grid_size=grid_size,
        chunk_size=chunk_size,
    )
    return reference, source


def plot_local_region(
    *,
    t: np.ndarray,
    observed_x: np.ndarray,
    true_theta: np.ndarray,
    local_x: np.ndarray,
    distances: np.ndarray,
    radius: float,
    summary_quantile: float,
    outfile: Path,
    max_curves: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if local_x.shape[0] > max_curves:
        index = rng.choice(local_x.shape[0], size=max_curves, replace=False)
        local_plot = local_x[index]
    else:
        local_plot = local_x
    t_grid = np.linspace(float(t.min()), float(t.max()), 220)
    true_mean = true_theta[0] * np.exp(-true_theta[1] * t_grid)

    figure, (ax_curves, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )
    for curve in local_plot:
        ax_curves.plot(t, curve, color="#9aa6b2", alpha=0.07, linewidth=0.8)
    ax_curves.scatter(t, observed_x, color="#111827", s=26, zorder=5, label="observed x0")
    ax_curves.plot(t_grid, true_mean, color="#111827", linestyle="--", linewidth=1.8, label="true mean")
    ax_curves.set_title("Accepted local simulations")
    ax_curves.set_xlabel("time t")
    ax_curves.set_ylabel("observation y")
    ax_curves.grid(alpha=0.22)
    ax_curves.legend(loc="upper right")

    ax_dist.hist(distances, bins=40, color="#4f6f7d", alpha=0.75)
    ax_dist.axvline(radius, color="#111827", linestyle="--", linewidth=1.8)
    ax_dist.set_title(f"summary-distance region\nprior quantile {summary_quantile:.3f}")
    ax_dist.set_xlabel("accepted summary distance")
    ax_dist.set_ylabel("count")
    ax_dist.grid(alpha=0.22)

    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local-x-region NPE models on the decay simulator.")
    parser.add_argument("--train-simulations", type=int, default=20_000)
    parser.add_argument("--val-simulations", type=int, default=5_000)
    parser.add_argument("--pilot-simulations", type=int, default=100_000)
    parser.add_argument("--summary-quantile", type=float, default=0.02)
    parser.add_argument("--summary-bins", type=int, default=8)
    parser.add_argument("--collection-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=5_000_000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--mdn-components", type=int, default=8)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-context-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--families", type=parse_families, default=["mdn"])
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_LOCAL_REGION_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_LOCAL_REGION_FIGURES)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    args.target_wasserstein, args.target_source, args.recommended_targets = resolve_target_wasserstein(
        args.target_wasserstein,
        summary_path=args.target_summary,
    )
    if args.target_summary is not None:
        args.target_summary = str(args.target_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = choose_training_device(args.device)
    np.random.seed(args.seed + 11)
    torch.manual_seed(args.seed + 13)

    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    t = t_obs.numpy()
    observed_x = y_obs.numpy()
    true_theta_np = true_theta.numpy()

    region_start = time.perf_counter()
    region = fit_local_region(
        observed_x=observed_x,
        pilot_simulations=args.pilot_simulations,
        summary_bins=args.summary_bins,
        summary_quantile=args.summary_quantile,
        seed=args.seed,
    )
    total_local_count = args.train_simulations + args.val_simulations
    local_x, local_z, local_distances, collection = collect_local_pairs(
        target_count=total_local_count,
        radius=float(region["radius"]),
        target_summary=np.asarray(region["target_summary"]),
        summary_mean=np.asarray(region["summary_mean"]),
        summary_std=np.asarray(region["summary_std"]),
        summary_bins=args.summary_bins,
        seed=args.seed + 1,
        chunk_size=args.collection_chunk_size,
        max_candidates=args.max_candidates,
    )
    region_seconds = time.perf_counter() - region_start

    train_x, train_z, train_distances, val_x, val_z, val_distances = split_train_val(
        x=local_x,
        z=local_z,
        distances=local_distances,
        train_count=args.train_simulations,
        seed=args.seed + 2,
    )

    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 1e-6)
    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)

    train_x_std = standardize(train_x, x_mean, x_std).astype(np.float32)
    val_x_std = standardize(val_x, x_mean, x_std).astype(np.float32)
    train_z_std = standardize(train_z, z_mean, z_std).astype(np.float32)
    val_z_std = standardize(val_z, z_mean, z_std).astype(np.float32)

    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_x_tensor = torch.from_numpy(val_x_std)
    val_z_tensor = torch.from_numpy(val_z_std)

    config = Stage1Config(
        train_simulations=args.train_simulations,
        val_simulations=args.val_simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        mdn_components=args.mdn_components,
        flow_layers=args.flow_layers,
        flow_context_dim=args.flow_context_dim,
        seed=args.seed,
        observed_seed=args.observed_seed,
        requested_device=args.device,
        families=args.families,
        posterior_samples=args.posterior_samples,
        reference_grid_size=args.reference_grid_size,
    )

    results: dict[str, dict[str, object]] = {}
    z_samples_by_family: dict[str, np.ndarray] = {}
    theta_samples_by_family: dict[str, np.ndarray] = {}
    model_paths: dict[str, str] = {}
    for family in args.families:
        print(f"training {family} on {device}")
        model, metrics = train_one_model(
            family=family,
            config=config,
            train_loader=train_loader,
            val_x=val_x_tensor,
            val_z=val_z_tensor,
            device=device,
            x_dim=train_x_std.shape[1],
            z_dim=train_z_std.shape[1],
        )
        z_samples, theta_samples = sample_posterior_for_observation(
            model=model,
            observed_x=observed_x,
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            n=args.posterior_samples,
            device=device,
        )
        z_samples_by_family[family] = z_samples
        theta_samples_by_family[family] = theta_samples
        metrics["posterior_summary"] = summarize_samples(theta_samples)
        results[family] = metrics

        model_path = args.output_dir / f"{family}_model.pt"
        torch.save(
            {
                "family": family,
                "state_dict": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "z_mean": z_mean,
                "z_std": z_std,
                "config": asdict(config),
                "local_region": {
                    "summary_quantile": args.summary_quantile,
                    "summary_bins": args.summary_bins,
                    "radius": float(region["radius"]),
                    "summary_mean": np.asarray(region["summary_mean"]).tolist(),
                    "summary_std": np.asarray(region["summary_std"]).tolist(),
                    "target_summary": np.asarray(region["target_summary"]).tolist(),
                },
            },
            model_path,
        )
        model_paths[family] = str(model_path)

    reference, reference_source = reference_for_observation(
        t=t,
        observed_x=observed_x,
        true_theta=true_theta_np,
        fallback_z=np.vstack([train_z, val_z]),
        mcmc_samples=args.mcmc_samples,
        hmc_samples=args.hmc_samples,
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.posterior_samples, 80_000),
        seed=args.seed + 88,
    )
    for family, theta_samples in theta_samples_by_family.items():
        metrics = compare_to_reference(theta_samples, reference)
        value = metrics["mean_normalized_wasserstein"]["value"]
        results[family]["faithfulness_to_grid_reference"] = metrics
        results[family]["target_wasserstein"] = args.target_wasserstein
        results[family]["target_pass"] = bool(value <= args.target_wasserstein)
        results[family]["target_ratio"] = float(value / args.target_wasserstein)

    samples_npz = args.output_dir / "npe_local_region_samples.npz"
    np.savez_compressed(
        samples_npz,
        observed_x=observed_x,
        t=t,
        y=observed_x,
        true_theta=true_theta_np,
        train_summary_distances=train_distances,
        val_summary_distances=val_distances,
        x_mean=x_mean,
        x_std=x_std,
        z_mean=z_mean,
        z_std=z_std,
        **{f"z_samples_{family}": samples for family, samples in z_samples_by_family.items()},
        **{f"theta_samples_{family}": samples for family, samples in theta_samples_by_family.items()},
    )

    local_region_png = args.figure_dir / "npe_local_region_curves.png"
    training_curve_png = args.figure_dir / "npe_local_region_training_curves.png"
    corner_png = args.figure_dir / "npe_local_region_corner_overlay.png"
    predictive_png = args.figure_dir / "npe_local_region_predictive_overlay.png"
    plot_local_region(
        t=t,
        observed_x=observed_x,
        true_theta=true_theta_np,
        local_x=train_x,
        distances=train_distances,
        radius=float(region["radius"]),
        summary_quantile=args.summary_quantile,
        outfile=local_region_png,
        max_curves=450,
        seed=args.seed + 5,
    )
    plot_training_curves(results, training_curve_png)
    plot_npe_corner_overlay(theta_samples_by_family, reference_samples, true_theta_np, corner_png)
    plot_npe_predictive_overlay(
        samples_by_family=theta_samples_by_family,
        t=t,
        y=observed_x,
        true_theta=true_theta_np,
        outfile=predictive_png,
    )

    summary = {
        "config": asdict(config),
        "device": str(device),
        "target_wasserstein": args.target_wasserstein,
        "timing_seconds": {
            "region": float(region_seconds),
            "total": float(time.perf_counter() - total_start),
        },
        "local_region": {
            "summary_quantile": args.summary_quantile,
            "summary_bins": args.summary_bins,
            "radius": float(region["radius"]),
            "pilot_simulations": args.pilot_simulations,
            "pilot_distance_summary": region["pilot_distance_summary"],
            "collection": collection,
            "train_distance_summary": {
                "min": float(train_distances.min()),
                "median": float(np.median(train_distances)),
                "max": float(train_distances.max()),
            },
            "val_distance_summary": {
                "min": float(val_distances.min()),
                "median": float(np.median(val_distances)),
                "max": float(val_distances.max()),
            },
            "summary_mean": np.asarray(region["summary_mean"]).tolist(),
            "summary_std": np.asarray(region["summary_std"]).tolist(),
            "target_summary": np.asarray(region["target_summary"]).tolist(),
        },
        "standardization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "z_mean": z_mean.tolist(),
            "z_std": z_std.tolist(),
        },
        "model_paths": model_paths,
        "samples_npz": str(samples_npz),
        "figures": {
            "local_region": str(local_region_png),
            "training_curves": str(training_curve_png),
            "corner_overlay": str(corner_png),
            "predictive_overlay": str(predictive_png),
        },
        "grid_reference": {
            "source": reference_source,
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "results": results,
    }
    summary_json = args.output_dir / "npe_local_region_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"local_region: {local_region_png}")
    print(f"corner_overlay: {corner_png}")
    print(f"predictive_overlay: {predictive_png}")
    print(
        "local region: "
        f"radius={float(region['radius']):.4f}, "
        f"acceptance_rate={collection['acceptance_rate']:.4f}, "
        f"candidates={collection['candidate_count']}"
    )
    print("mean normalized Wasserstein to grid reference:")
    for family in args.families:
        value = results[family]["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(
            f"  {family}: {value:.5f} "
            f"(target_ratio={results[family]['target_ratio']:.2f}x, "
            f"pass={results[family]['target_pass']})"
        )


if __name__ == "__main__":
    main()
