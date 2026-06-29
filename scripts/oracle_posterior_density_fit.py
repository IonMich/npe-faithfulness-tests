from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, summarize_samples
from npe_stage1_decay import (
    FAMILIES,
    Stage1Config,
    choose_training_device,
    plot_npe_corner_overlay,
    plot_training_curves,
    sample_grid_reference,
    sample_posterior_for_observation,
    train_one_model,
)
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def split_posterior_z(
    *,
    z: np.ndarray,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_count + val_count > len(z):
        raise ValueError(f"Requested {train_count + val_count} posterior samples but only have {len(z)}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(z))
    train = z[indices[:train_count]]
    val = z[indices[train_count : train_count + val_count]]
    return train, val


def empirical_gaussian_baselines(
    *,
    train_z: np.ndarray,
    reference: dict[str, object],
    sample_count: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    baselines = {}

    mean = train_z.mean(axis=0)
    std = train_z.std(axis=0)
    diag_z = rng.normal(loc=mean[None, :], scale=std[None, :], size=(sample_count, 3))
    diag_theta = np.exp(diag_z)
    baselines["empirical_diag_gaussian_z"] = {
        "posterior_summary": summarize_samples(diag_theta),
        "faithfulness_to_grid_reference": compare_to_reference(diag_theta, reference),
    }

    cov = np.cov(train_z.T) + np.eye(3) * 1e-8
    full_z = rng.multivariate_normal(mean, cov, size=sample_count)
    full_theta = np.exp(full_z)
    baselines["empirical_full_gaussian_z"] = {
        "posterior_summary": summarize_samples(full_theta),
        "faithfulness_to_grid_reference": compare_to_reference(full_theta, reference),
    }
    return baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit neural density families directly to posterior samples at x0.")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--families", type=parse_families, default=["diag_gaussian", "full_gaussian", "mdn"])
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--val-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--mdn-components", type=int, default=8)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-context-dim", type=int, default=96)
    parser.add_argument("--posterior-samples", type=int, default=80_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ap.ORACLE_POSTERIOR_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.ORACLE_POSTERIOR_FIGURES)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    train_z, val_z = split_posterior_z(
        z=combined_z,
        train_count=args.train_samples,
        val_count=args.val_samples,
        seed=args.seed + 2,
    )
    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=combined_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.posterior_samples, 80_000),
        seed=args.seed + 3,
    )

    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)
    train_z_std = ((train_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    val_z_std = ((val_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    train_x = np.zeros((len(train_z), 1), dtype=np.float32)
    val_x = np.zeros((len(val_z), 1), dtype=np.float32)
    x_mean = np.zeros(1, dtype=np.float64)
    x_std = np.ones(1, dtype=np.float64)

    config = Stage1Config(
        train_simulations=args.train_samples,
        val_simulations=args.val_samples,
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
        observed_seed=0,
        requested_device=args.device,
        families=args.families,
        posterior_samples=args.posterior_samples,
        reference_grid_size=args.reference_grid_size,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_z_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    results = {}
    theta_samples_by_family = {}
    for family in args.families:
        print(f"oracle posterior fit {family} on {device}")
        model, metrics = train_one_model(
            family=family,
            config=config,
            train_loader=train_loader,
            val_x=torch.from_numpy(val_x),
            val_z=torch.from_numpy(val_z_std),
            device=device,
            x_dim=1,
            z_dim=3,
        )
        z_samples, theta_samples = sample_posterior_for_observation(
            model=model,
            observed_x=np.zeros(1, dtype=np.float64),
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            n=args.posterior_samples,
            device=device,
        )
        metrics["posterior_summary"] = summarize_samples(theta_samples)
        metrics["faithfulness_to_grid_reference"] = compare_to_reference(theta_samples, reference)
        value = metrics["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        metrics["target_wasserstein"] = args.target_wasserstein
        metrics["target_ratio"] = float(value / args.target_wasserstein)
        metrics["target_pass"] = bool(value <= args.target_wasserstein)
        results[family] = metrics
        theta_samples_by_family[family] = theta_samples

    baselines = empirical_gaussian_baselines(
        train_z=train_z,
        reference=reference,
        sample_count=args.posterior_samples,
        seed=args.seed + 5,
    )
    for metrics in baselines.values():
        value = metrics["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        metrics["target_wasserstein"] = args.target_wasserstein
        metrics["target_ratio"] = float(value / args.target_wasserstein)
        metrics["target_pass"] = bool(value <= args.target_wasserstein)

    samples_npz = args.output_dir / "oracle_posterior_fit_samples.npz"
    np.savez_compressed(
        samples_npz,
        true_theta=mcmc["true_theta"],
        train_z=train_z,
        val_z=val_z,
        **{f"theta_samples_{family}": samples for family, samples in theta_samples_by_family.items()},
    )

    training_curve_png = args.figure_dir / "oracle_posterior_fit_training_curves.png"
    corner_png = args.figure_dir / "oracle_posterior_fit_corner_overlay.png"
    plot_training_curves(results, training_curve_png)
    plot_npe_corner_overlay(theta_samples_by_family, reference_samples, mcmc["true_theta"], corner_png)

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "device": str(device),
        "target_wasserstein": args.target_wasserstein,
        "samples_npz": str(samples_npz),
        "figures": {
            "training_curves": str(training_curve_png),
            "corner_overlay": str(corner_png),
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "empirical_baselines": baselines,
        "results": results,
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }
    summary_json = args.output_dir / "oracle_posterior_fit_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"corner_overlay: {corner_png}")
    print("oracle mean normalized Wasserstein to grid reference:")
    for family, metrics in results.items():
        value = metrics["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(f"  {family}: {value:.5f} (target_ratio={metrics['target_ratio']:.2f}x, pass={metrics['target_pass']})")
    print("empirical Gaussian baselines:")
    for name, metrics in baselines.items():
        value = metrics["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(f"  {name}: {value:.5f} (target_ratio={metrics['target_ratio']:.2f}x, pass={metrics['target_pass']})")


if __name__ == "__main__":
    main()
