from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, summarize_samples
from mcmc_decay_inference import simulate_decay_data
from npe_local_region_decay import (
    collect_local_pairs,
    curve_summary,
    fit_local_region,
    sample_decay_pairs_from_generator,
    split_train_val,
)
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    make_model,
    plot_npe_corner_overlay,
    plot_npe_predictive_overlay,
    plot_training_curves,
    sample_grid_reference,
    sample_posterior_for_observation,
    standardize,
    synchronize_device,
)
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def weighted_summary(weights: np.ndarray) -> dict[str, float]:
    normalized = weights / weights.sum()
    ess = 1.0 / np.sum(normalized**2)
    return {
        "min": float(weights.min()),
        "median": float(np.median(weights)),
        "mean": float(weights.mean()),
        "max": float(weights.max()),
        "sum": float(weights.sum()),
        "ess": float(ess),
        "ess_fraction": float(ess / len(weights)),
    }


def train_one_weighted_model(
    *,
    family: str,
    config: Stage1Config,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_z: torch.Tensor,
    val_w: torch.Tensor,
    device: torch.device,
    x_dim: int,
    z_dim: int,
) -> tuple[torch.nn.Module, dict[str, object]]:
    torch.manual_seed(config.seed + 2000 + FAMILIES.index(family))
    model = make_model(family, config, x_dim, z_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = {"train_nll": [], "val_nll": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = max(20, config.epochs // 5)
    epochs_since_best = 0

    synchronize_device(device)
    start = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for batch_x, batch_z, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            log_prob = model.log_prob(batch_z, batch_x)
            loss = -(log_prob * batch_w).sum() / batch_w.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            optimizer.step()
            weight_sum = float(batch_w.detach().cpu().sum())
            train_loss_sum += float(loss.detach().cpu()) * weight_sum
            train_weight_sum += weight_sum

        model.eval()
        with torch.no_grad():
            val_x_dev = val_x.to(device)
            val_z_dev = val_z.to(device)
            val_w_dev = val_w.to(device)
            val_loss = -(model.log_prob(val_z_dev, val_x_dev) * val_w_dev).sum() / val_w_dev.sum().clamp_min(1e-12)
            val_loss_float = float(val_loss.detach().cpu())
        train_loss = train_loss_sum / max(train_weight_sum, 1e-12)
        history["train_nll"].append(train_loss)
        history["val_nll"].append(val_loss_float)

        if val_loss_float < best_val:
            best_val = val_loss_float
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            break

    synchronize_device(device)
    runtime = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "family": family,
        "label": FAMILY_LABELS[family],
        "epochs_completed": len(history["train_nll"]),
        "best_val_nll": best_val,
        "final_train_nll": history["train_nll"][-1],
        "final_val_nll": history["val_nll"][-1],
        "training_seconds": runtime,
        "history": history,
    }


def make_broad_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    val_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_x, train_z, _ = sample_decay_pairs_from_generator(n=args.train_simulations, generator=train_generator)
    val_x, val_z, _ = sample_decay_pairs_from_generator(n=args.val_simulations, generator=val_generator)
    return (
        train_x,
        train_z,
        np.ones(args.train_simulations, dtype=np.float32),
        val_x,
        val_z,
        np.ones(args.val_simulations, dtype=np.float32),
        {"mode": "broad"},
    )


def make_local_data(
    args: argparse.Namespace,
    observed_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    region = fit_local_region(
        observed_x=observed_x,
        pilot_simulations=args.pilot_simulations,
        summary_bins=args.summary_bins,
        summary_quantile=args.summary_quantile,
        seed=args.seed,
    )
    local_x, local_z, distances, collection = collect_local_pairs(
        target_count=args.train_simulations + args.val_simulations,
        radius=float(region["radius"]),
        target_summary=np.asarray(region["target_summary"]),
        summary_mean=np.asarray(region["summary_mean"]),
        summary_std=np.asarray(region["summary_std"]),
        summary_bins=args.summary_bins,
        seed=args.seed + 1,
        chunk_size=args.collection_chunk_size,
        max_candidates=args.max_candidates,
    )
    train_x, train_z, train_distances, val_x, val_z, val_distances = split_train_val(
        x=local_x,
        z=local_z,
        distances=distances,
        train_count=args.train_simulations,
        seed=args.seed + 2,
    )
    metadata = {
        "mode": "hard_local",
        "summary_quantile": args.summary_quantile,
        "radius": float(region["radius"]),
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
    }
    return (
        train_x,
        train_z,
        np.ones(args.train_simulations, dtype=np.float32),
        val_x,
        val_z,
        np.ones(args.val_simulations, dtype=np.float32),
        metadata,
    )


def make_kernel_data(
    args: argparse.Namespace,
    observed_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    region = fit_local_region(
        observed_x=observed_x,
        pilot_simulations=args.pilot_simulations,
        summary_bins=args.summary_bins,
        summary_quantile=args.kernel_bandwidth_quantile,
        seed=args.seed,
    )
    bandwidth = float(region["radius"])
    target_summary = np.asarray(region["target_summary"])
    summary_mean = np.asarray(region["summary_mean"])
    summary_std = np.asarray(region["summary_std"])

    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10)
    val_generator = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    train_x, train_z, _ = sample_decay_pairs_from_generator(n=args.train_simulations, generator=train_generator)
    val_x, val_z, _ = sample_decay_pairs_from_generator(n=args.val_simulations, generator=val_generator)

    from npe_local_region_decay import standardized_summary_distance

    train_distances = standardized_summary_distance(
        curve_summary(train_x, args.summary_bins),
        target_summary,
        summary_mean,
        summary_std,
    )
    val_distances = standardized_summary_distance(
        curve_summary(val_x, args.summary_bins),
        target_summary,
        summary_mean,
        summary_std,
    )
    train_w = np.exp(-0.5 * (train_distances / bandwidth) ** 2).astype(np.float32)
    val_w = np.exp(-0.5 * (val_distances / bandwidth) ** 2).astype(np.float32)
    metadata = {
        "mode": "kernel",
        "kernel_bandwidth_quantile": args.kernel_bandwidth_quantile,
        "bandwidth": bandwidth,
        "pilot_distance_summary": region["pilot_distance_summary"],
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
        "train_weight_summary": weighted_summary(train_w),
        "val_weight_summary": weighted_summary(val_w),
    }
    return train_x, train_z, train_w, val_x, val_z, val_w, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NPE with curve-summary context and optional local weighting.")
    parser.add_argument("--mode", choices=["broad", "hard_local", "kernel"], default="broad")
    parser.add_argument("--context", choices=["summary", "raw"], default="summary")
    parser.add_argument("--families", type=parse_families, default=["mdn"])
    parser.add_argument("--train-simulations", type=int, default=50_000)
    parser.add_argument("--val-simulations", type=int, default=10_000)
    parser.add_argument("--pilot-simulations", type=int, default=120_000)
    parser.add_argument("--summary-bins", type=int, default=8)
    parser.add_argument("--summary-quantile", type=float, default=0.005)
    parser.add_argument("--kernel-bandwidth-quantile", type=float, default=0.005)
    parser.add_argument("--collection-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=10_000_000)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--mdn-components", type=int, default=8)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-context-dim", type=int, default=96)
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_SUMMARY_CONTEXT_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_SUMMARY_CONTEXT_FIGURES)
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

    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    observed_x_raw = y_obs.numpy()
    observed_context = (
        curve_summary(observed_x_raw, args.summary_bins)[0]
        if args.context == "summary"
        else observed_x_raw
    )
    true_theta_np = true_theta.numpy()

    data_start = time.perf_counter()
    if args.mode == "broad":
        train_x_raw, train_z, train_w, val_x_raw, val_z, val_w, data_metadata = make_broad_data(args)
    elif args.mode == "hard_local":
        train_x_raw, train_z, train_w, val_x_raw, val_z, val_w, data_metadata = make_local_data(args, observed_x_raw)
    else:
        train_x_raw, train_z, train_w, val_x_raw, val_z, val_w, data_metadata = make_kernel_data(args, observed_x_raw)
    data_seconds = time.perf_counter() - data_start

    if args.context == "summary":
        train_context = curve_summary(train_x_raw, args.summary_bins)
        val_context = curve_summary(val_x_raw, args.summary_bins)
    else:
        train_context = train_x_raw
        val_context = val_x_raw
    x_mean = train_context.mean(axis=0)
    x_std = np.maximum(train_context.std(axis=0), 1e-6)
    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)
    train_x_std = standardize(train_context, x_mean, x_std).astype(np.float32)
    val_x_std = standardize(val_context, x_mean, x_std).astype(np.float32)
    train_z_std = ((train_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    val_z_std = ((val_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)

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
    loader_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x_std),
            torch.from_numpy(train_z_std),
            torch.from_numpy(train_w.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    reference_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=reference_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.posterior_samples, 80_000),
        seed=args.seed + 3,
    )

    results = {}
    theta_samples_by_family = {}
    z_samples_by_family = {}
    for family in args.families:
        print(f"summary-context {args.mode} training {family} on {device}")
        model, metrics = train_one_weighted_model(
            family=family,
            config=config,
            train_loader=train_loader,
            val_x=torch.from_numpy(val_x_std),
            val_z=torch.from_numpy(val_z_std),
            val_w=torch.from_numpy(val_w.astype(np.float32)),
            device=device,
            x_dim=train_x_std.shape[1],
            z_dim=train_z_std.shape[1],
        )
        z_samples, theta_samples = sample_posterior_for_observation(
            model=model,
            observed_x=observed_context,
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
        z_samples_by_family[family] = z_samples

    samples_npz = args.output_dir / "npe_summary_context_samples.npz"
    np.savez_compressed(
        samples_npz,
        t=t_obs.numpy(),
        y=observed_x_raw,
        observed_context=observed_context,
        true_theta=true_theta_np,
        train_weights=train_w,
        val_weights=val_w,
        x_mean=x_mean,
        x_std=x_std,
        z_mean=z_mean,
        z_std=z_std,
        **{f"z_samples_{family}": samples for family, samples in z_samples_by_family.items()},
        **{f"theta_samples_{family}": samples for family, samples in theta_samples_by_family.items()},
    )

    training_curve_png = args.figure_dir / "npe_summary_context_training_curves.png"
    corner_png = args.figure_dir / "npe_summary_context_corner_overlay.png"
    predictive_png = args.figure_dir / "npe_summary_context_predictive_overlay.png"
    plot_training_curves(results, training_curve_png)
    plot_npe_corner_overlay(theta_samples_by_family, reference_samples, true_theta_np, corner_png)
    plot_npe_predictive_overlay(
        samples_by_family=theta_samples_by_family,
        t=t_obs.numpy(),
        y=observed_x_raw,
        true_theta=true_theta_np,
        outfile=predictive_png,
    )

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "mode": args.mode,
        "device": str(device),
        "target_wasserstein": args.target_wasserstein,
        "data_seconds": data_seconds,
        "data_metadata": data_metadata,
        "context": {
            "type": args.context,
            "summary_bins": args.summary_bins,
            "dimension": int(train_context.shape[1]),
            "observed_context": observed_context.tolist(),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
        },
        "weights": {
            "train": weighted_summary(train_w),
            "val": weighted_summary(val_w),
        },
        "samples_npz": str(samples_npz),
        "figures": {
            "training_curves": str(training_curve_png),
            "corner_overlay": str(corner_png),
            "predictive_overlay": str(predictive_png),
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "results": results,
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }
    summary_json = args.output_dir / "npe_summary_context_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"corner_overlay: {corner_png}")
    print("summary-context mean normalized Wasserstein to grid reference:")
    for family, metrics in results.items():
        value = metrics["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(f"  {family}: {value:.5f} (target_ratio={metrics['target_ratio']:.2f}x, pass={metrics['target_pass']})")
    if args.mode == "kernel":
        print(f"kernel ESS fraction: {weighted_summary(train_w)['ess_fraction']:.4f}")


if __name__ == "__main__":
    main()
