from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit

import npe_flow_decay as decay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/12_local_scaling/01_local_data_scaling")

PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "train_simulations": (64, 128),
        "seeds": (20269001,),
        "val_simulations": 64,
        "epochs": 1,
        "patience": 1,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip": 20.0,
        "transforms": 2,
        "hidden_features": (16, 16),
        "bins": 4,
        "posterior_samples": 256,
        "reference_grid_size": 15,
        "reference_chunk_size": 50_000,
        "local_pilot_simulations": 512,
        "local_quantile": 0.20,
        "local_max_candidates": 10_000,
        "simulate_chunk_size": 512,
        "summary_chunk_size": 512,
        "print_every": 1,
    },
    "pilot": {
        "train_simulations": (5_000, 10_000, 20_000),
        "seeds": (20260701, 20260702, 20260703),
        "val_simulations": 5_000,
        "epochs": 80,
        "patience": 25,
        "batch_size": 1024,
        "learning_rate": 6e-4,
        "weight_decay": 1e-6,
        "grad_clip": 20.0,
        "transforms": 6,
        "hidden_features": (128, 128),
        "bins": 12,
        "posterior_samples": 40_000,
        "reference_grid_size": 70,
        "reference_chunk_size": 120_000,
        "local_pilot_simulations": 120_000,
        "local_quantile": 0.005,
        "local_max_candidates": 10_000_000,
        "simulate_chunk_size": 80_000,
        "summary_chunk_size": 40_000,
        "print_every": 10,
    },
    "full": {
        "train_simulations": (10_000, 20_000, 40_000, 80_000, 150_000, 300_000),
        "seeds": (20260701, 20260702, 20260703, 20260704, 20260705),
        "val_simulations": 35_000,
        "epochs": 220,
        "patience": 55,
        "batch_size": 4096,
        "learning_rate": 6e-4,
        "weight_decay": 1e-6,
        "grad_clip": 20.0,
        "transforms": 8,
        "hidden_features": (192, 192),
        "bins": 16,
        "posterior_samples": 100_000,
        "reference_grid_size": 90,
        "reference_chunk_size": 120_000,
        "local_pilot_simulations": 400_000,
        "local_quantile": 0.005,
        "local_max_candidates": 90_000_000,
        "simulate_chunk_size": 100_000,
        "summary_chunk_size": 50_000,
        "print_every": 10,
    },
}


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


def fill_from_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.train_simulations = tuple(sorted(set(args.train_simulations)))
    args.seeds = tuple(args.seeds)
    args.hidden_features = tuple(args.hidden_features)
    return args


def load_region_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "local_training" in summary:
        return dict(summary["local_training"]["region"])
    if "region" in summary:
        return dict(summary["region"])
    raise KeyError(f"{path} does not contain local_training.region or region")


def build_reference(args: argparse.Namespace) -> dict[str, object]:
    mcmc = decay.load_samples(args.mcmc_samples, "MCMC")
    hmc = decay.load_samples(args.hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    reference = decay.build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=combined_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    mcmc_to_grid = decay.compare_to_reference(mcmc["posterior_theta"], reference)
    hmc_to_grid = decay.compare_to_reference(hmc["posterior_theta"], reference)
    recommended_target = max(
        decay.mean_normalized_wasserstein_value(mcmc_to_grid),
        decay.mean_normalized_wasserstein_value(hmc_to_grid),
    )
    target_wasserstein = (
        float(args.target_wasserstein)
        if args.target_wasserstein is not None
        else float(recommended_target)
    )
    return {
        "reference": reference,
        "mcmc_to_grid": mcmc_to_grid,
        "hmc_to_grid": hmc_to_grid,
        "recommended_target": recommended_target,
        "target_wasserstein": target_wasserstein,
        "mcmc_samples": mcmc,
    }


def prepare_observation(args: argparse.Namespace) -> dict[str, object]:
    t_obs, y_obs, true_theta = decay.simulate_decay_data(seed=args.observed_seed)
    t = t_obs.numpy()
    observed_x = y_obs.numpy()
    k_grid = decay.make_k_grid(args.k_grid_points, args.k_min, args.k_max)
    observed_context = decay.make_context_summaries(
        observed_x[None, :],
        t,
        k_grid,
        kind=args.context_kind,
        chunk_size=1,
    )[0]
    return {
        "t": t,
        "observed_x": observed_x,
        "true_theta": true_theta.numpy(),
        "k_grid": k_grid,
        "observed_context": observed_context,
    }


def fit_or_load_region(
    args: argparse.Namespace,
    observation: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    if args.local_region_summary is not None:
        return load_region_summary(args.local_region_summary)
    rng = np.random.default_rng(args.region_seed)
    region = decay.fit_local_region(
        observed_context=np.asarray(observation["observed_context"]),
        t=np.asarray(observation["t"]),
        k_grid=np.asarray(observation["k_grid"]),
        simulations=args.local_pilot_simulations,
        quantile=args.local_quantile,
        kernel_quantile=0.0,
        rng=rng,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
        context_kind=args.context_kind,
    )
    region_path = output_root / "results" / "local_region.json"
    region_path.write_text(json.dumps(json_ready({"region": region}), indent=2), encoding="utf-8")
    return region


def make_train_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        transforms=args.transforms,
        hidden_features=args.hidden_features,
        bins=args.bins,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        patience=args.patience,
        print_every=args.print_every,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run_scale_point(
    *,
    args: argparse.Namespace,
    seed: int,
    train_count: int,
    pool: dict[str, object],
    observation: dict[str, object],
    reference_pack: dict[str, object],
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    run_root = output_root / "runs" / f"n{train_count}_seed{seed}"
    results_dir = run_root / "results"
    figures_dir = run_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "local_scaling_run_summary.json"
    if args.skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    z_pool = np.asarray(pool["z"])
    context_pool = np.asarray(pool["context"])
    train_indices = np.asarray(pool["train_indices"])[:train_count]
    val_indices = np.asarray(pool["val_indices"])
    train_z = z_pool[train_indices]
    val_z = z_pool[val_indices]
    train_context = context_pool[train_indices]
    val_context = context_pool[val_indices]
    train_weights = np.ones(train_count, dtype=np.float32)
    val_weights = np.ones(len(val_indices), dtype=np.float32)

    context_mean, context_std = decay.weighted_moments(train_context, train_weights.astype(np.float64))
    train_context_std = ((train_context - context_mean[None, :]) / context_std[None, :]).astype(np.float32)
    val_context_std = ((val_context - context_mean[None, :]) / context_std[None, :]).astype(np.float32)
    observed_context_std = (
        (np.asarray(observation["observed_context"]) - context_mean) / context_std
    ).astype(np.float64)

    linear_adjustment = None
    train_target_z = train_z
    val_target_z = val_z
    if args.linear_target_adjustment:
        fitted_adjustment = decay.fit_linear_target_adjustment(
            z=train_z,
            context_std=train_context_std.astype(np.float64),
            observed_context_std=observed_context_std,
            weights=train_weights,
            ridge=args.linear_adjustment_ridge,
        )
        slope = np.asarray(fitted_adjustment["slope"])
        train_target_z = decay.apply_linear_target_adjustment(
            train_z,
            train_context_std.astype(np.float64),
            observed_context_std,
            slope,
        )
        val_target_z = decay.apply_linear_target_adjustment(
            val_z,
            val_context_std.astype(np.float64),
            observed_context_std,
            slope,
        )
        linear_adjustment = {
            "slope": slope,
            "intercept": np.asarray(fitted_adjustment["intercept"]),
            "observed_context_std": observed_context_std,
            "ridge": np.asarray(args.linear_adjustment_ridge),
        }

    z_mean, z_std = decay.weighted_moments(train_target_z, train_weights.astype(np.float64))
    train_z_std = ((train_target_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    val_z_std = ((val_target_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)

    start = time.perf_counter()
    model, training = decay.train_flow(
        train_context=train_context_std,
        train_z=train_z_std,
        train_weights=train_weights,
        val_context=val_context_std,
        val_z=val_z_std,
        val_weights=val_weights,
        args=make_train_args(args, seed),
        device=device,
    )
    model_parameters = count_parameters(model)
    torch.manual_seed(seed + train_count + 10_000)
    z_samples, theta_samples = decay.sample_flow_posterior(
        model=model,
        observed_context=np.asarray(observation["observed_context"]),
        context_mean=context_mean,
        context_std=context_std,
        z_mean=z_mean,
        z_std=z_std,
        linear_adjustment=linear_adjustment,
        n=args.posterior_samples,
        device=device,
    )
    faithfulness = decay.compare_to_reference(theta_samples, reference_pack["reference"])
    mean_w = decay.mean_normalized_wasserstein_value(faithfulness)
    target_wasserstein = float(reference_pack["target_wasserstein"])
    total_seconds = time.perf_counter() - start

    if args.save_run_figures:
        decay.plot_training(training, figures_dir / "local_scaling_training.png")
    if args.save_samples:
        np.savez_compressed(
            results_dir / "local_scaling_samples.npz",
            z_samples=z_samples,
            theta_samples=theta_samples,
        )

    z_log_det = float(np.log(z_std).sum())
    pool_target = int(pool["target_count"])
    pool_candidate_count = int(pool["collection"]["candidate_count"])
    standalone_candidate_count_estimate = pool_candidate_count * (
        (train_count + args.val_simulations) / max(pool_target, 1)
    )
    pool_collection_seconds = float(pool["collection"]["collection_seconds"])
    standalone_collection_seconds_estimate = pool_collection_seconds * (
        (train_count + args.val_simulations) / max(pool_target, 1)
    )
    total_with_standalone_collection_estimate = (
        total_seconds + standalone_collection_seconds_estimate
    )
    summary = {
        "seed": int(seed),
        "train_simulations": int(train_count),
        "val_simulations": int(args.val_simulations),
        "preset": args.preset,
        "device": str(device),
        "model_parameters": model_parameters,
        "config": {
            "observed_seed": args.observed_seed,
            "context_kind": args.context_kind,
            "local_quantile": args.local_quantile,
            "linear_target_adjustment": bool(args.linear_target_adjustment),
            "linear_adjustment_ridge": args.linear_adjustment_ridge,
            "transforms": args.transforms,
            "hidden_features": list(args.hidden_features),
            "bins": args.bins,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "posterior_samples": args.posterior_samples,
            "reference_grid_size": args.reference_grid_size,
        },
        "pool": {
            "target_count": pool_target,
            "pool_candidate_count": pool_candidate_count,
            "standalone_candidate_count_estimate": float(standalone_candidate_count_estimate),
            "pool_collection_seconds": pool_collection_seconds,
            "standalone_collection_seconds_estimate": float(standalone_collection_seconds_estimate),
            "acceptance_rate": float(pool["collection"]["acceptance_rate"]),
        },
        "standardization": {
            "context_mean": context_mean,
            "context_std": context_std,
            "z_mean": z_mean,
            "z_std": z_std,
            "z_log_det": z_log_det,
        },
        "training": {
            **training,
            "best_val_nll_target_z": float(training["best_val_nll"] + z_log_det),
            "final_val_nll_target_z": float(training["final_val_nll"] + z_log_det),
            "total_train_eval_seconds": float(total_seconds),
            "total_with_standalone_collection_estimate": float(
                total_with_standalone_collection_estimate
            ),
        },
        "weight_diagnostics": {
            "train": decay.weight_diagnostics(train_weights),
            "validation": decay.weight_diagnostics(val_weights),
        },
        "faithfulness_to_grid_reference": faithfulness,
        "target_wasserstein": target_wasserstein,
        "target_ratio": float(mean_w / target_wasserstein),
        "target_pass": bool(mean_w <= target_wasserstein),
        "posterior_summary": decay.summarize_samples(theta_samples),
        "outputs": {
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    return json_ready(summary)


def collect_seed_pool(
    *,
    args: argparse.Namespace,
    seed: int,
    region: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    max_train = max(args.train_simulations)
    target_count = max_train + args.val_simulations
    z, context, distances, collection = decay.collect_local_prior_data(
        target_count=target_count,
        observed_context=np.asarray(observation["observed_context"]),
        t=np.asarray(observation["t"]),
        k_grid=np.asarray(observation["k_grid"]),
        region=region,
        rng=rng,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
        context_kind=args.context_kind,
        max_candidates=args.local_max_candidates,
    )
    order = rng.permutation(target_count)
    train_indices = order[:max_train]
    val_indices = order[max_train:]
    return {
        "seed": int(seed),
        "target_count": int(target_count),
        "z": z,
        "context": context,
        "distances": distances,
        "collection": collection,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def region_cache_hash(region: dict[str, object]) -> str:
    keys = ("radius", "quantile", "kernel_quantile", "center", "scale")
    payload = {key: json_ready(region.get(key)) for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def pool_cache_path(
    *,
    args: argparse.Namespace,
    output_root: Path,
    seed: int,
    region: dict[str, object],
) -> Path:
    cache_dir = args.pool_cache_dir or (output_root / "results" / "pool_cache")
    max_train = max(args.train_simulations)
    target_count = max_train + args.val_simulations
    region_hash = region_cache_hash(region)
    return (
        cache_dir
        / f"local_pool_seed{seed}_target{target_count}_val{args.val_simulations}_"
        f"q{float(args.local_quantile):.6g}_region{region_hash}.npz"
    )


def save_seed_pool_cache(
    *,
    path: Path,
    pool: dict[str, object],
    args: argparse.Namespace,
    region: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": int(pool["seed"]),
        "target_count": int(pool["target_count"]),
        "train_simulations": list(args.train_simulations),
        "val_simulations": int(args.val_simulations),
        "observed_seed": int(args.observed_seed),
        "context_kind": args.context_kind,
        "local_quantile": float(args.local_quantile),
        "region_hash": region_cache_hash(region),
        "collection": pool["collection"],
    }
    np.savez_compressed(
        path,
        z=np.asarray(pool["z"]),
        context=np.asarray(pool["context"]),
        distances=np.asarray(pool["distances"]),
        train_indices=np.asarray(pool["train_indices"]),
        val_indices=np.asarray(pool["val_indices"]),
        metadata=np.asarray(json.dumps(json_ready(metadata))),
    )


def load_seed_pool_cache(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    return {
        "seed": int(metadata["seed"]),
        "target_count": int(metadata["target_count"]),
        "z": np.asarray(data["z"]),
        "context": np.asarray(data["context"]),
        "distances": np.asarray(data["distances"]),
        "collection": metadata["collection"],
        "train_indices": np.asarray(data["train_indices"]),
        "val_indices": np.asarray(data["val_indices"]),
        "cache_path": str(path),
    }


def collect_or_load_seed_pool(
    *,
    args: argparse.Namespace,
    seed: int,
    region: dict[str, object],
    observation: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    if args.cache_pools:
        path = pool_cache_path(args=args, output_root=output_root, seed=seed, region=region)
        if path.exists():
            print(f"loading cached local pool for seed={seed}: {path}")
            return load_seed_pool_cache(path)
    pool = collect_seed_pool(args=args, seed=seed, region=region, observation=observation)
    if args.cache_pools:
        path = pool_cache_path(args=args, output_root=output_root, seed=seed, region=region)
        print(f"saving cached local pool for seed={seed}: {path}")
        save_seed_pool_cache(path=path, pool=pool, args=args, region=region)
        pool["cache_path"] = str(path)
    return pool


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q16": float(np.quantile(finite, 0.16)),
        "median": float(np.median(finite)),
        "q84": float(np.quantile(finite, 0.84)),
        "max": float(np.max(finite)),
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for train_count in sorted({int(row["train_simulations"]) for row in rows}):
        group = [row for row in rows if int(row["train_simulations"]) == train_count]
        output.append({
            "train_simulations": train_count,
            "seed_count": len(group),
            "wasserstein": quantile_summary(
                np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
            ),
            "target_ratio": quantile_summary(
                np.asarray([row["target_ratio"] for row in group], dtype=np.float64)
            ),
            "best_val_nll_target_z": quantile_summary(
                np.asarray([row["best_val_nll_target_z"] for row in group], dtype=np.float64)
            ),
            "training_seconds": quantile_summary(
                np.asarray([row["training_seconds"] for row in group], dtype=np.float64)
            ),
            "total_with_standalone_collection_estimate": quantile_summary(
                np.asarray(
                    [row["total_with_standalone_collection_estimate"] for row in group],
                    dtype=np.float64,
                )
            ),
            "standalone_candidate_count_estimate": quantile_summary(
                np.asarray(
                    [row["standalone_candidate_count_estimate"] for row in group],
                    dtype=np.float64,
                )
            ),
        })
    return output


def flatten_summary(summary: dict[str, object], prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_summary(value, f"{name}."))
        else:
            output[name] = value
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def power_law(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -alpha)


def fit_wasserstein_power_law(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    x = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
    y = np.asarray([row["wasserstein"]["median"] for row in summary_rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return {"status": "skipped", "reason": "need at least three positive scale points"}
    y_min = float(np.min(y))
    try:
        popt, _pcov = curve_fit(
            power_law,
            x,
            y,
            p0=(0.8 * y_min, max(float(np.max(y) - 0.8 * y_min), 1e-6) * x[0] ** 0.5, 0.5),
            bounds=([0.0, 0.0, 0.0], [max(y_min * 0.999, 1e-12), np.inf, 5.0]),
            maxfev=20_000,
        )
    except Exception as exc:  # pragma: no cover - fit failures depend on data shape
        return {"status": "failed", "reason": str(exc)}
    fitted = power_law(x, *popt)
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "status": "ok",
        "floor": float(popt[0]),
        "amplitude": float(popt[1]),
        "alpha": float(popt[2]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "x": x.tolist(),
        "median_wasserstein": y.tolist(),
        "fitted_wasserstein": fitted.tolist(),
    }


def row_from_run_summary(summary: dict[str, object]) -> dict[str, object]:
    mean_w = decay.mean_normalized_wasserstein_value(summary["faithfulness_to_grid_reference"])
    return {
        "summary_json": summary["outputs"]["summary_json"],
        "seed": int(summary["seed"]),
        "train_simulations": int(summary["train_simulations"]),
        "val_simulations": int(summary["val_simulations"]),
        "model_parameters": int(summary["model_parameters"]),
        "mean_normalized_wasserstein": float(mean_w),
        "target_wasserstein": float(summary["target_wasserstein"]),
        "target_ratio": float(summary["target_ratio"]),
        "target_pass": bool(summary["target_pass"]),
        "best_val_nll": float(summary["training"]["best_val_nll"]),
        "best_val_nll_target_z": float(summary["training"]["best_val_nll_target_z"]),
        "final_val_nll": float(summary["training"]["final_val_nll"]),
        "final_val_nll_target_z": float(summary["training"]["final_val_nll_target_z"]),
        "epochs_completed": int(summary["training"]["epochs_completed"]),
        "best_epoch": int(summary["training"]["best_epoch"]),
        "training_seconds": float(summary["training"]["training_seconds"]),
        "total_train_eval_seconds": float(summary["training"]["total_train_eval_seconds"]),
        "total_with_standalone_collection_estimate": float(
            summary["training"]["total_with_standalone_collection_estimate"]
        ),
        "pool_candidate_count": int(summary["pool"]["pool_candidate_count"]),
        "standalone_candidate_count_estimate": float(
            summary["pool"]["standalone_candidate_count_estimate"]
        ),
        "pool_collection_seconds": float(summary["pool"]["pool_collection_seconds"]),
        "standalone_collection_seconds_estimate": float(
            summary["pool"]["standalone_collection_seconds_estimate"]
        ),
        "pool_acceptance_rate": float(summary["pool"]["acceptance_rate"]),
    }


def collect_existing_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_root / "runs").glob("*/results/local_scaling_run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row_from_run_summary(summary))
    return sorted(rows, key=lambda row: (int(row["train_simulations"]), int(row["seed"])))


def plot_scaling(
    *,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    fit: dict[str, object],
    target_wasserstein: float | None,
    outfile: Path,
) -> None:
    colors = {
        "wasserstein": "#2f6fbb",
        "target_ratio": "#b85c38",
        "nll": "#2f855a",
        "seconds": "#7a5cc2",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))
    axes_flat = axes.ravel()

    x_all = np.asarray([row["train_simulations"] for row in rows], dtype=np.float64)

    panels = [
        ("mean_normalized_wasserstein", "Mean normalized Wasserstein", axes_flat[0], colors["wasserstein"]),
        ("target_ratio", "Target ratio", axes_flat[1], colors["target_ratio"]),
        ("best_val_nll_target_z", "Best validation NLL in target-z units", axes_flat[2], colors["nll"]),
        ("training_seconds", "Training seconds", axes_flat[3], colors["seconds"]),
    ]

    for metric, ylabel, ax, color in panels:
        y_all = np.asarray([row[metric] for row in rows], dtype=np.float64)
        ax.scatter(x_all, y_all, color=color, alpha=0.42, s=28, label="seed")
        x_summary = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
        median = np.asarray([row[metric_name(metric)]["median"] for row in summary_rows], dtype=np.float64)
        q16 = np.asarray([row[metric_name(metric)]["q16"] for row in summary_rows], dtype=np.float64)
        q84 = np.asarray([row[metric_name(metric)]["q84"] for row in summary_rows], dtype=np.float64)
        ax.plot(x_summary, median, color=color, linewidth=2.1, marker="o", label="median")
        ax.fill_between(x_summary, q16, q84, color=color, alpha=0.16, label="q16-q84")
        ax.set_xscale("log")
        ax.set_xlabel("accepted local training simulations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)

    if target_wasserstein is not None:
        axes_flat[0].axhline(target_wasserstein, color="#172033", linestyle="--", linewidth=1.4)
        axes_flat[1].axhline(1.0, color="#172033", linestyle="--", linewidth=1.4)

    if fit.get("status") == "ok":
        fit_x = np.asarray(fit["x"], dtype=np.float64)
        dense_x = np.geomspace(float(fit_x.min()), float(fit_x.max()), 120)
        dense_y = power_law(dense_x, float(fit["floor"]), float(fit["amplitude"]), float(fit["alpha"]))
        axes_flat[0].plot(
            dense_x,
            dense_y,
            color="#172033",
            linewidth=1.8,
            linestyle=":",
            label=f"power fit alpha={float(fit['alpha']):.2f}",
        )
        axes_flat[0].legend(frameon=False)

    figure.suptitle("Single-decay local NPE data scaling", y=0.995, fontsize=15)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def metric_name(metric: str) -> str:
    if metric == "mean_normalized_wasserstein":
        return "wasserstein"
    if metric == "best_val_nll_target_z":
        return "best_val_nll_target_z"
    return metric


def aggregate_outputs(
    *,
    args: argparse.Namespace,
    output_root: Path,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    rows = collect_existing_rows(output_root)
    summary_rows = summarize_rows(rows) if rows else []
    fit = fit_wasserstein_power_law(summary_rows)
    rows_csv = results_dir / "local_data_scaling_rows.csv"
    summary_csv = results_dir / "local_data_scaling_summary.csv"
    summary_json = results_dir / "local_data_scaling_summary.json"
    figure_path = figures_dir / "local_data_scaling.png"

    write_csv(rows, rows_csv)
    write_csv([flatten_summary(row) for row in summary_rows], summary_csv)
    target_values = [float(row["target_wasserstein"]) for row in rows]
    target_wasserstein = float(np.median(target_values)) if target_values else None
    if rows:
        plot_scaling(
            rows=rows,
            summary_rows=summary_rows,
            fit=fit,
            target_wasserstein=target_wasserstein,
            outfile=figure_path,
        )

    output = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"mcmc_samples", "hmc_samples"}
        },
        "rows": rows,
        "scale_summary": summary_rows,
        "wasserstein_power_law_fit": fit,
        "metadata": extra_metadata or {},
        "outputs": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "figure": str(figure_path) if rows else None,
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    return json_ready(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and aggregate a nested local-data scaling sweep for single-decay NPE.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-simulations", type=parse_int_list, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument("--val-simulations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--transforms", type=int, default=None)
    parser.add_argument("--hidden-features", type=decay.parse_int_list, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--posterior-samples", type=int, default=None)
    parser.add_argument("--reference-grid-size", type=int, default=None)
    parser.add_argument("--reference-chunk-size", type=int, default=None)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--region-seed", type=int, default=20260700)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--context-kind", choices=["indirect", "enhanced"], default="indirect")
    parser.add_argument("--k-grid-points", type=int, default=260)
    parser.add_argument("--k-min", type=float, default=0.04)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--local-pilot-simulations", type=int, default=None)
    parser.add_argument("--local-quantile", type=float, default=None)
    parser.add_argument("--local-max-candidates", type=int, default=None)
    parser.add_argument("--local-region-summary", type=Path, default=None)
    parser.add_argument("--simulate-chunk-size", type=int, default=None)
    parser.add_argument("--summary-chunk-size", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--linear-target-adjustment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear-adjustment-ridge", type=float, default=1e-4)
    parser.add_argument("--mcmc-samples", type=Path, default=decay.ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=decay.ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--save-run-figures", action="store_true")
    parser.add_argument(
        "--cache-pools",
        action="store_true",
        help="Save/load accepted local pools per seed so interrupted reruns avoid local rejection sampling.",
    )
    parser.add_argument(
        "--pool-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for accepted local-pool caches. Defaults to output-root/results/pool_cache.",
    )
    return fill_from_preset(parser.parse_args())


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(json_ready(vars(args)), indent=2))
        return

    metadata: dict[str, object] = {}
    if not args.aggregate_only:
        reference_pack = build_reference(args)
        observation = prepare_observation(args)
        region = fit_or_load_region(args, observation, output_root)
        metadata = {
            "region": region,
            "target_wasserstein": reference_pack["target_wasserstein"],
            "recommended_target": reference_pack["recommended_target"],
            "mcmc_to_grid": reference_pack["mcmc_to_grid"],
            "hmc_to_grid": reference_pack["hmc_to_grid"],
        }
        device = decay.choose_device(args.device)
        for seed in args.seeds:
            print(f"collecting nested local pool for seed={seed}")
            pool = collect_or_load_seed_pool(
                args=args,
                seed=seed,
                region=region,
                observation=observation,
                output_root=output_root,
            )
            for train_count in args.train_simulations:
                print(f"training local scaling point seed={seed} train={train_count}")
                run_scale_point(
                    args=args,
                    seed=seed,
                    train_count=train_count,
                    pool=pool,
                    observation=observation,
                    reference_pack=reference_pack,
                    device=device,
                    output_root=output_root,
                )

    aggregate = aggregate_outputs(args=args, output_root=output_root, extra_metadata=metadata)
    print(f"summary_json: {aggregate['outputs']['summary_json']}")
    print(f"rows_csv: {aggregate['outputs']['rows_csv']}")
    if aggregate["outputs"].get("figure") is not None:
        print(f"figure: {aggregate['outputs']['figure']}")


if __name__ == "__main__":
    main()
