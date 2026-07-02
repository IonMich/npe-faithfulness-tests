from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

import npe_stage1_decay as stage1
from mcmc_decay_inference import PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_flow_decay import mean_normalized_wasserstein_value
from npe_metric_noise_floor_probe import (
    build_reference_cache,
    compare_samples_to_reference_fast,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot")
DEFAULT_REFERENCE_NPZ = Path(
    "runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/"
    "decay_x0_grid300_reference.npz"
)
DEFAULT_REFERENCE_METADATA = Path(
    "runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/"
    "decay_x0_grid300_reference_metadata.json"
)

PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "train_simulations": (128, 256),
        "seeds": (20260901,),
        "val_simulations": 256,
        "standardization_simulations": 512,
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 2e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 32,
        "hidden_layers": 1,
        "mdn_components": 2,
        "posterior_samples": 512,
        "print_every": 1,
    },
    "pilot": {
        "train_simulations": (1_000, 2_000, 4_000, 8_000, 16_000, 32_000),
        "seeds": (20260901, 20260902),
        "val_simulations": 12_000,
        "standardization_simulations": 60_000,
        "epochs": 90,
        "batch_size": 512,
        "learning_rate": 2e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 128,
        "hidden_layers": 3,
        "mdn_components": 5,
        "posterior_samples": 20_000,
        "print_every": 10,
    },
}

CONTEXT_VARIANTS = ("real", "zero_x", "shuffled_x")
BASELINE_GATE_METRICS = (
    ("full_val_nll_z_units", "final_validation_nll"),
    ("panel_marginal_wasserstein_mean", "panel_mean_marginal_wasserstein"),
    ("x0_grid300_wasserstein", "x0_grid300_wasserstein"),
)
CONTEXT_CACHE_MODES = {
    "fit_summary",
    "laplace_summary",
    "profile_summary",
    "raw_fit_summary",
    "raw_laplace_summary",
    "raw_profile_summary",
    "raw_decay_fit_summary",
    "raw_decay_laplace_summary",
    "raw_decay_profile_summary",
    "asinh_fit_summary",
    "rms_normalized_fit_summary",
}


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def parse_context_variants(value: str) -> tuple[str, ...]:
    items = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one context variant.")
    invalid = sorted(set(items) - set(CONTEXT_VARIANTS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown context variants: {invalid}. Expected a comma-separated subset of {CONTEXT_VARIANTS}."
        )
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


def runtime_metadata() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def fill_from_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def load_reference(npz_path: Path, metadata_path: Path) -> dict[str, object]:
    arrays = np.load(npz_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "grid_size": int(metadata["grid_size"]),
        "grid_points": int(metadata["grid_points"]),
        "theta_grid": np.asarray(arrays["theta_grid"], dtype=np.float64),
        "weights": np.asarray(arrays["weights"], dtype=np.float64),
        "summary": metadata["summary"],
        "z_ranges": metadata["z_ranges"],
        "edge_mass": metadata["edge_mass"],
        "metadata": metadata,
    }


def load_validation_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    arrays = np.load(path, allow_pickle=False)
    x_val = np.asarray(arrays["x_val"], dtype=np.float32)
    z_val = np.asarray(arrays["z_val"], dtype=np.float32)
    metadata = {
        "path": str(path),
        "simulations": int(x_val.shape[0]),
        "x_shape": list(x_val.shape),
        "z_shape": list(z_val.shape),
        "file_bytes": int(path.stat().st_size),
        "file_mib": path.stat().st_size / (1024**2),
    }
    for key in ("seed", "n_observations", "dtype"):
        if key in arrays.files:
            value = arrays[key]
            metadata[key] = value.item() if getattr(value, "shape", ()) == () else value.tolist()
    return x_val, z_val, metadata


def load_panel_marginal_cache(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    arrays = np.load(path, allow_pickle=False)
    panel = {
        "x_panel": np.asarray(arrays["x_panel"], dtype=np.float64),
        "z_panel": np.asarray(arrays["z_panel"], dtype=np.float64),
        "theta_panel": np.asarray(arrays["theta_panel"], dtype=np.float64),
        "theta_axes": np.asarray(arrays["theta_axes"], dtype=np.float64),
        "marginal_weights": np.asarray(arrays["marginal_weights"], dtype=np.float64),
        "target_wasserstein": np.asarray(arrays["target_wasserstein"], dtype=np.float64),
        "labels": np.asarray(arrays["labels"]).astype(str).tolist(),
    }
    metadata_path = path.with_suffix(".json")
    metadata = {
        "path": str(path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "panel_size": int(panel["x_panel"].shape[0]),
        "grid_size": int(panel["theta_axes"].shape[-1]),
        "file_bytes": int(path.stat().st_size),
        "file_mib": path.stat().st_size / (1024**2),
    }
    if metadata_path.exists():
        metadata["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return panel, metadata


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "min": float("nan"),
            "q16": float("nan"),
            "median": float("nan"),
            "q84": float("nan"),
            "max": float("nan"),
        }
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


def make_stage1_config(args: argparse.Namespace, seed: int, train_simulations: int) -> stage1.Stage1Config:
    return stage1.Stage1Config(
        train_simulations=int(train_simulations),
        val_simulations=int(args.val_simulations),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        lr_schedule=str(args.lr_schedule),
        lr_eta_min=float(args.lr_eta_min),
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_decay_epochs=int(args.lr_decay_epochs),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_eps=float(args.adam_eps),
        validation_every_epochs=int(args.validation_every_epochs),
        skip_training_validation=bool(args.skip_training_validation),
        torch_compile=str(args.torch_compile),
        grad_clip_norm=float(args.grad_clip_norm),
        ema_decay=float(args.ema_decay),
        batching_mode=str(args.batching_mode),
        max_optimizer_steps=int(args.max_optimizer_steps),
        loss_weight_mode=str(args.loss_weight_mode),
        loss_tail_weight=float(args.loss_tail_weight),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        mdn_components=int(args.mdn_components),
        flow_layers=int(args.flow_layers),
        flow_context_dim=int(args.flow_context_dim),
        flow_activation=str(args.flow_activation),
        flow_residual=bool(args.flow_residual),
        flow_randperm=bool(args.flow_randperm),
        flow_passes=int(args.flow_passes),
        flow_kind=str(args.flow_kind),
        seed=int(seed),
        observed_seed=int(args.observed_seed),
        requested_device=str(args.device),
        families=[str(args.family)],
        posterior_samples=int(args.posterior_samples),
        reference_grid_size=300,
        train_sampler=str(args.train_sampler),
        context_features=str(args.context_features),
        spline_bins=int(args.spline_bins),
        target_transform=str(args.target_transform),
        target_ridge=float(args.target_ridge),
    )


def standardization_stats(args: argparse.Namespace) -> dict[str, np.ndarray]:
    x_std_sample, _, _ = stage1.sample_decay_pairs(
        n=int(args.standardization_simulations),
        seed=int(args.standardization_seed),
    )
    x_std_sample = stage1.transform_context_features(x_std_sample, str(args.context_features))
    z_mean = PRIOR_LOG_MEAN.detach().cpu().numpy().astype(np.float64)
    z_std = PRIOR_LOG_STD.detach().cpu().numpy().astype(np.float64)
    return {
        "x_mean": np.mean(x_std_sample, axis=0),
        "x_std": np.maximum(np.std(x_std_sample, axis=0), 1e-6),
        "z_mean": z_mean,
        "z_std": np.maximum(z_std, 1e-8),
    }


def context_variant_description(variant: str) -> str:
    if variant == "real":
        return "Paired real standardized context."
    if variant == "zero_x":
        return "All standardized context values are zero; input-independent baseline."
    if variant == "shuffled_x":
        return "Training contexts are shuffled against targets; evaluation contexts stay real."
    raise ValueError(f"Unknown context variant: {variant}")


def run_dir_name(train_simulations: int, seed: int, context_variant: str) -> str:
    base = f"n{int(train_simulations)}_seed{int(seed)}"
    if context_variant == "real":
        return base
    return f"{base}_{context_variant}"


def context_cache_path(source_path: Path, mode: str) -> Path:
    safe_mode = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in mode)
    return source_path.with_name(f"{source_path.stem}_{safe_mode}_context_float32.npy")


def transform_context_features_cached(
    x: np.ndarray,
    mode: str,
    *,
    source_path: Path | None = None,
) -> np.ndarray:
    if source_path is None or mode not in CONTEXT_CACHE_MODES:
        return stage1.transform_context_features(x, mode)
    cache_path = context_cache_path(source_path, mode)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == x.shape[0]:
            return np.asarray(cached)
    context = stage1.transform_context_features(x, mode).astype(np.float32, copy=False)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, context)
    try:
        tmp_path.replace(cache_path)
    except OSError:
        if not cache_path.exists():
            raise
    return context


def apply_training_context_variant(x_std: np.ndarray, context_variant: str, seed: int) -> np.ndarray:
    if context_variant == "real":
        return x_std
    if context_variant == "zero_x":
        return np.zeros_like(x_std)
    if context_variant == "shuffled_x":
        if x_std.shape[0] <= 1:
            return x_std.copy()
        rng = np.random.default_rng(int(seed))
        permutation = rng.permutation(x_std.shape[0])
        if np.array_equal(permutation, np.arange(x_std.shape[0])):
            permutation = np.roll(permutation, 1)
        return x_std[permutation].copy()
    raise ValueError(f"Unknown context variant: {context_variant}")


def apply_evaluation_context_variant(x_std: np.ndarray, context_variant: str) -> np.ndarray:
    if context_variant in {"real", "shuffled_x"}:
        return x_std
    if context_variant == "zero_x":
        return np.zeros_like(x_std)
    raise ValueError(f"Unknown context variant: {context_variant}")


@torch.no_grad()
def sample_posterior_for_standardized_context(
    *,
    model: torch.nn.Module,
    x_standardized: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    n: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_tensor = torch.from_numpy(x_standardized[None, :].astype(np.float32)).to(device)
    z_standardized = model.sample(n, x_tensor).detach().cpu().numpy()
    z = z_standardized * z_std[None, :] + z_mean[None, :]
    theta = np.exp(z)
    return z, theta


def evaluate_val_nll_z_units(
    *,
    model: torch.nn.Module,
    val_x_std: np.ndarray,
    val_z_std: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x_std), torch.from_numpy(val_z_std)),
        batch_size=batch_size,
        shuffle=False,
    )
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_z in loader:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)
            loss = -model.log_prob(batch_z, batch_x)
            total += float(loss.detach().cpu().sum())
            count += int(batch_x.shape[0])
    nll_standardized = total / max(count, 1)
    return float(nll_standardized + np.log(z_std).sum())


def evaluate_val_nll_z_values(
    *,
    model: torch.nn.Module,
    val_x_std: np.ndarray,
    val_z_std: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x_std), torch.from_numpy(val_z_std)),
        batch_size=batch_size,
        shuffle=False,
    )
    values = np.empty(val_x_std.shape[0], dtype=np.float32)
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_z in loader:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)
            loss = -model.log_prob(batch_z, batch_x).detach().cpu().numpy().astype(np.float32)
            stop = offset + loss.shape[0]
            values[offset:stop] = loss
            offset = stop
    return values[:offset].astype(np.float64, copy=False) + float(np.log(z_std).sum())


def summarize_val_nll_z_values(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def evaluate_val_nll_z_summary(
    *,
    model: torch.nn.Module,
    val_x_std: np.ndarray,
    val_z_std: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    values = evaluate_val_nll_z_values(
        model=model,
        val_x_std=val_x_std,
        val_z_std=val_z_std,
        z_std=z_std,
        device=device,
        batch_size=batch_size,
    )
    return summarize_val_nll_z_values(values)


def plot_top_validation_failures(
    *,
    x_values: np.ndarray,
    theta_values: np.ndarray,
    nll_values: np.ndarray,
    output_path: Path,
) -> None:
    if x_values.size == 0:
        return
    panel_count = min(int(x_values.shape[0]), 12)
    cols = min(3, panel_count)
    rows = int(math.ceil(panel_count / cols))
    t = np.linspace(0.0, 6.0, x_values.shape[1])
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= panel_count:
            ax.axis("off")
            continue
        x = x_values[index]
        theta = theta_values[index]
        mean = theta[0] * np.exp(-theta[1] * t)
        ax.plot(t, x, color="#2f6fbb", linewidth=1.3, label="x")
        ax.plot(t, mean, color="#b85c38", linewidth=1.2, linestyle="--", label="mean")
        ax.set_title(f"rank {index + 1} NLL={nll_values[index]:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_top_validation_failures(
    *,
    results_dir: Path,
    nll_values: np.ndarray,
    val_x_std_original: np.ndarray,
    val_z_std: np.ndarray,
    stats: dict[str, np.ndarray],
    top_k: int,
    context_variant: str,
) -> dict[str, object] | None:
    if top_k <= 0 or nll_values.size == 0:
        return None
    count = min(int(top_k), int(nll_values.size))
    ranking_values = np.nan_to_num(nll_values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    candidate = np.argpartition(ranking_values, -count)[-count:]
    order = candidate[np.argsort(ranking_values[candidate])[::-1]]
    x_values = (
        val_x_std_original[order].astype(np.float64) * stats["x_std"][None, :]
        + stats["x_mean"][None, :]
    )
    z_values = (
        val_z_std[order].astype(np.float64) * stats["z_std"][None, :]
        + stats["z_mean"][None, :]
    )
    theta_values = np.exp(z_values)
    selected_nll = np.asarray(nll_values[order], dtype=np.float64)
    npz_path = results_dir / "validation_top_failures.npz"
    json_path = results_dir / "validation_top_failures.json"
    figure_path = results_dir / "validation_top_failures.png"
    np.savez_compressed(
        npz_path,
        index=order.astype(np.int64),
        nll_z_units=selected_nll,
        x=x_values,
        z=z_values,
        theta=theta_values,
        context_variant=np.asarray(context_variant),
    )
    plot_top_validation_failures(
        x_values=x_values,
        theta_values=theta_values,
        nll_values=selected_nll,
        output_path=figure_path,
    )
    summary = {
        "context_variant": context_variant,
        "top_k": count,
        "source_count": int(nll_values.size),
        "worst_nll_z_units": float(selected_nll[0]),
        "best_of_top_nll_z_units": float(selected_nll[-1]),
        "indices": order.astype(int).tolist(),
        "nll_z_units": selected_nll.tolist(),
        "npz": str(npz_path),
        "figure": str(figure_path),
    }
    json_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    summary["json"] = str(json_path)
    return summary


def compare_samples_to_marginals(
    *,
    theta_samples: np.ndarray,
    theta_axes: np.ndarray,
    marginal_weights: np.ndarray,
) -> float:
    values = []
    for axis in range(3):
        ref_axis = theta_axes[axis]
        ref_weights = marginal_weights[axis] / np.sum(marginal_weights[axis])
        mean = float(np.sum(ref_axis * ref_weights))
        sd = float(np.sqrt(max(np.sum((ref_axis - mean) ** 2 * ref_weights), 0.0)))
        w = wasserstein_distance(theta_samples[:, axis], ref_axis, v_weights=ref_weights)
        values.append(float(w / max(sd, 1e-12)))
    return float(np.mean(values))


def evaluate_panel_marginal_wasserstein(
    *,
    model: torch.nn.Module,
    panel: dict[str, object],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    n: int,
    device: torch.device,
    context_variant: str = "real",
    context_features: str = "raw",
) -> dict[str, object]:
    x_panel = stage1.transform_context_features(
        np.asarray(panel["x_panel"], dtype=np.float64),
        context_features,
    )
    theta_axes = np.asarray(panel["theta_axes"], dtype=np.float64)
    marginal_weights = np.asarray(panel["marginal_weights"], dtype=np.float64)
    targets = np.asarray(panel["target_wasserstein"], dtype=np.float64)
    labels = list(panel["labels"])
    rows = []
    w_values = []
    ratio_values = []
    for index, label in enumerate(labels):
        x_standardized = stage1.standardize(
            x_panel[index][None, :],
            x_mean,
            x_std,
        ).astype(np.float32)[0]
        x_standardized = apply_evaluation_context_variant(
            x_standardized[None, :],
            context_variant,
        )[0]
        _, theta_samples = sample_posterior_for_standardized_context(
            model=model,
            z_mean=z_mean,
            z_std=z_std,
            x_standardized=x_standardized,
            n=n,
            device=device,
        )
        w_value = compare_samples_to_marginals(
            theta_samples=theta_samples,
            theta_axes=theta_axes[index],
            marginal_weights=marginal_weights[index],
        )
        target = float(targets[index])
        ratio = float(w_value / target) if target > 0 else float("nan")
        w_values.append(w_value)
        ratio_values.append(ratio)
        rows.append({
            "index": int(index),
            "label": str(label),
            "wasserstein": float(w_value),
            "target_wasserstein": target,
            "target_ratio": ratio,
        })
    return {
        "context_variant": context_variant,
        "posterior_samples_per_signal": int(n),
        "signal_count": int(len(rows)),
        "per_signal": rows,
        "wasserstein": quantile_summary(np.asarray(w_values, dtype=np.float64)),
        "target_ratio": quantile_summary(np.asarray(ratio_values, dtype=np.float64)),
    }


def power_law_with_floor(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -alpha)


def fit_decreasing_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, object] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if x.size < 4:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = max(float(np.max(y) - np.min(y)), 1e-8)
    floor_upper = float(np.min(y) - 1e-8)
    floor_lower = float(np.min(y) - max(10.0 * span, 1.0))
    initial_floor = float(np.min(y) - 0.25 * span)
    initial_alpha = 0.25
    initial_amplitude = float(max(np.max(y) - initial_floor, 1e-8) * (np.min(x) ** initial_alpha))
    try:
        params, _ = curve_fit(
            power_law_with_floor,
            x,
            y,
            p0=[initial_floor, initial_amplitude, initial_alpha],
            bounds=([floor_lower, 1e-12, 1e-4], [floor_upper, 1e12, 5.0]),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    fitted = power_law_with_floor(x, *params)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    floor, amplitude, alpha = [float(value) for value in params]
    excess = y - floor
    log_valid = excess > 0
    log_r2 = float("nan")
    if np.count_nonzero(log_valid) >= 3:
        log_x = np.log(x[log_valid])
        log_y = np.log(excess[log_valid])
        log_fit = np.log(np.maximum(fitted[log_valid] - floor, 1e-300))
        ss_log_res = float(np.sum((log_y - log_fit) ** 2))
        ss_log_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
        log_r2 = 1.0 - ss_log_res / ss_log_tot if ss_log_tot > 0 else float("nan")
    return {
        "floor": floor,
        "amplitude": amplitude,
        "alpha": alpha,
        "r2_raw": r2,
        "r2_log_excess": log_r2,
        "x": x.tolist(),
        "y": y.tolist(),
        "fitted": fitted.tolist(),
    }


def row_context_variant(row: dict[str, object]) -> str:
    return str(row.get("context_variant", "real"))


def context_variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in CONTEXT_VARIANTS:
        return (CONTEXT_VARIANTS.index(variant), variant)
    return (len(CONTEXT_VARIANTS), variant)


def summarize_rows(rows: list[dict[str, object]], *, context_variant: str | None = None) -> list[dict[str, object]]:
    output = []
    for train_count in sorted({int(row["train_simulations"]) for row in rows}):
        group = [row for row in rows if int(row["train_simulations"]) == train_count]
        item = {
            "context_variant": context_variant if context_variant is not None else "mixed",
            "train_simulations": train_count,
            "seed_count": len(group),
            "x0_grid300_wasserstein": quantile_summary(
                np.asarray([row["x0_grid300_wasserstein"] for row in group], dtype=np.float64)
            ),
            "x0_grid300_target_ratio": quantile_summary(
                np.asarray([row["x0_grid300_target_ratio"] for row in group], dtype=np.float64)
            ),
            "best_val_nll_z_units": quantile_summary(
                np.asarray([row["best_val_nll_z_units"] for row in group], dtype=np.float64)
            ),
            "initial_train_batch_nll_z_units": quantile_summary(
                np.asarray([row.get("initial_train_batch_nll_z_units", float("nan")) for row in group], dtype=np.float64)
            ),
            "initial_val_nll_z_units": quantile_summary(
                np.asarray([row.get("initial_val_nll_z_units", float("nan")) for row in group], dtype=np.float64)
            ),
            "final_train_nll_z_units": quantile_summary(
                np.asarray([row.get("final_train_nll_z_units", float("nan")) for row in group], dtype=np.float64)
            ),
            "full_val_nll_z_units": quantile_summary(
                np.asarray([row["full_val_nll_z_units"] for row in group], dtype=np.float64)
            ),
            "training_seconds": quantile_summary(
                np.asarray([row["training_seconds"] for row in group], dtype=np.float64)
            ),
            "optimizer_steps": quantile_summary(
                np.asarray([row.get("optimizer_steps", float("nan")) for row in group], dtype=np.float64)
            ),
            "batches_per_epoch": quantile_summary(
                np.asarray([row.get("batches_per_epoch", float("nan")) for row in group], dtype=np.float64)
            ),
        }
        if all("panel_marginal_wasserstein_mean" in row for row in group):
            item["panel_marginal_wasserstein_mean"] = quantile_summary(
                np.asarray([row["panel_marginal_wasserstein_mean"] for row in group], dtype=np.float64)
            )
            item["panel_marginal_wasserstein_median"] = quantile_summary(
                np.asarray([row["panel_marginal_wasserstein_median"] for row in group], dtype=np.float64)
            )
            item["panel_marginal_target_ratio_mean"] = quantile_summary(
                np.asarray([row["panel_marginal_target_ratio_mean"] for row in group], dtype=np.float64)
            )
            item["panel_marginal_target_ratio_median"] = quantile_summary(
                np.asarray([row["panel_marginal_target_ratio_median"] for row in group], dtype=np.float64)
            )
        output.append(item)
    return output


def summary_by_train(summary_rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["train_simulations"]): row for row in summary_rows}


def metric_median(row: dict[str, object], metric: str) -> float:
    value = row.get(metric)
    if not isinstance(value, dict):
        return float("nan")
    return float(value.get("median", float("nan")))


def build_baseline_gate_summary(
    summary_rows_by_variant: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    if "real" not in summary_rows_by_variant:
        return {
            "reference_variant": "real",
            "rows": [],
            "all_pass": None,
            "reason": "No real context variant rows available.",
        }
    real_by_train = summary_by_train(summary_rows_by_variant["real"])
    rows = []
    for variant in sorted(summary_rows_by_variant, key=context_variant_sort_key):
        if variant == "real":
            continue
        baseline_by_train = summary_by_train(summary_rows_by_variant[variant])
        for train_count in sorted(set(real_by_train) & set(baseline_by_train)):
            checks = []
            real_row = real_by_train[train_count]
            baseline_row = baseline_by_train[train_count]
            for metric, label in BASELINE_GATE_METRICS:
                if metric not in real_row or metric not in baseline_row:
                    continue
                real_value = metric_median(real_row, metric)
                baseline_value = metric_median(baseline_row, metric)
                finite = bool(np.isfinite(real_value) and np.isfinite(baseline_value))
                checks.append({
                    "metric": metric,
                    "label": label,
                    "real_median": real_value,
                    "baseline_median": baseline_value,
                    "delta_baseline_minus_real": (
                        float(baseline_value - real_value) if finite else float("nan")
                    ),
                    "passed": bool(real_value < baseline_value) if finite else None,
                })
            checked = [check for check in checks if check["passed"] is not None]
            rows.append({
                "context_variant": variant,
                "train_simulations": int(train_count),
                "check_count": int(len(checked)),
                "passed": all(bool(check["passed"]) for check in checked) if checked else None,
                "checks": checks,
            })
    checked_rows = [row for row in rows if row["passed"] is not None]
    return {
        "reference_variant": "real",
        "checked_metrics": [
            {"metric": metric, "label": label, "direction": "lower_is_better"}
            for metric, label in BASELINE_GATE_METRICS
        ],
        "rows": rows,
        "all_pass": all(bool(row["passed"]) for row in checked_rows) if checked_rows else None,
    }


def flatten_baseline_gate_rows(gates: dict[str, object]) -> list[dict[str, object]]:
    output = []
    for row in gates.get("rows", []):
        if not isinstance(row, dict):
            continue
        for check in row.get("checks", []):
            if not isinstance(check, dict):
                continue
            output.append({
                "context_variant": row["context_variant"],
                "train_simulations": row["train_simulations"],
                "row_passed": row["passed"],
                **check,
            })
    return output


def fit_summary(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    x = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
    nll = np.asarray(
        [row["full_val_nll_z_units"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    best_nll = np.asarray(
        [row["best_val_nll_z_units"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    output = {
        "full_val_nll_z_units": fit_decreasing_power_law(x, nll),
        "best_val_nll_z_units": fit_decreasing_power_law(x, best_nll),
    }
    x0_w = np.asarray(
        [row["x0_grid300_wasserstein"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    output["x0_grid300_wasserstein"] = fit_decreasing_power_law(x, x0_w)
    if all("panel_marginal_wasserstein_mean" in row for row in summary_rows):
        panel_w = np.asarray(
            [row["panel_marginal_wasserstein_mean"]["median"] for row in summary_rows],
            dtype=np.float64,
        )
        output["panel_marginal_wasserstein_mean"] = fit_decreasing_power_law(x, panel_w)
    return output


def primary_wasserstein_metrics(summary_rows: list[dict[str, object]]) -> tuple[str, str, str, str]:
    if summary_rows and "panel_marginal_wasserstein_mean" in summary_rows[0]:
        return (
            "panel_marginal_wasserstein_mean",
            "panel mean marginal W",
            "panel_marginal_target_ratio_mean",
            "panel mean target ratio",
        )
    return (
        "x0_grid300_wasserstein",
        "x0 W to grid-300",
        "x0_grid300_target_ratio",
        "x0 grid-300 target ratio",
    )


def plot_scaling(
    *,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    fits: dict[str, object],
    target_wasserstein: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    x_summary = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
    x_dense = np.geomspace(np.min(x_summary), np.max(x_summary), 200)
    w_metric, w_label, ratio_metric, ratio_label = primary_wasserstein_metrics(summary_rows)

    panels = [
        (w_metric, w_label, axes[0, 0], "#2f6fbb", True),
        (ratio_metric, ratio_label, axes[0, 1], "#b85c38", True),
        ("full_val_nll_z_units", "validation NLL in z units", axes[1, 0], "#2f855a", False),
        ("best_val_nll_z_units", "best validation NLL in z units", axes[1, 1], "#5f4bb6", False),
    ]
    x_all = np.asarray([row["train_simulations"] for row in rows], dtype=np.float64)
    for metric, ylabel, ax, color, log_y in panels:
        y_all = np.asarray([row.get(metric, float("nan")) for row in rows], dtype=np.float64)
        median = np.asarray([row[metric]["median"] for row in summary_rows], dtype=np.float64)
        q16 = np.asarray([row[metric]["q16"] for row in summary_rows], dtype=np.float64)
        q84 = np.asarray([row[metric]["q84"] for row in summary_rows], dtype=np.float64)
        plotted_values = np.concatenate([y_all, median, q16, q84])
        finite = np.isfinite(plotted_values)
        positive = plotted_values > 0
        if log_y and not np.any(finite & positive):
            ax.set_xscale("log")
            ax.set_xlabel("broad prior-predictive training signals")
            ax.set_ylabel(ylabel)
            ax.text(
                0.5,
                0.5,
                "metric unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(alpha=0.22)
            continue
        ax.scatter(x_all, y_all, color=color, alpha=0.35, s=22, label="seed")
        ax.plot(x_summary, median, color=color, marker="o", linewidth=2.0, label="median")
        ax.fill_between(x_summary, q16, q84, color=color, alpha=0.16, label="q16-q84")
        fit = fits.get(metric)
        if isinstance(fit, dict):
            fit_y = power_law_with_floor(
                x_dense,
                float(fit["floor"]),
                float(fit["amplitude"]),
                float(fit["alpha"]),
            )
            label = f"fit alpha={float(fit['alpha']):.2f}, R2={float(fit['r2_raw']):.2f}"
            ax.plot(x_dense, fit_y, color="#172033", linestyle="--", linewidth=1.25, label=label)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("broad prior-predictive training signals")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)

    if w_metric == "x0_grid300_wasserstein":
        axes[0, 0].axhline(target_wasserstein, color="#172033", linestyle=":", linewidth=1.2)
    axes[0, 1].axhline(1.0, color="#172033", linestyle=":", linewidth=1.2)
    figure.suptitle("Broad MDN scaling sweep", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_log_excess(
    *,
    summary_rows: list[dict[str, object]],
    fits: dict[str, object],
    output_path: Path,
) -> None:
    panels = [
        (primary_wasserstein_metrics(summary_rows)[0], f"{primary_wasserstein_metrics(summary_rows)[1]} minus fitted floor", "#2f6fbb"),
        ("full_val_nll_z_units", "validation NLL minus fitted floor", "#2f855a"),
        ("best_val_nll_z_units", "best validation NLL minus fitted floor", "#5f4bb6"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    x = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
    x_dense = np.geomspace(np.min(x), np.max(x), 200)
    for metric, ylabel, color, ax in zip(
        [panel[0] for panel in panels],
        [panel[1] for panel in panels],
        [panel[2] for panel in panels],
        axes,
        strict=True,
    ):
        fit = fits.get(metric)
        if not isinstance(fit, dict):
            ax.text(0.5, 0.5, "fit unavailable", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        floor = float(fit["floor"])
        y = np.asarray([row[metric]["median"] for row in summary_rows], dtype=np.float64)
        excess = y - floor
        fit_excess = power_law_with_floor(
            x_dense,
            floor,
            float(fit["amplitude"]),
            float(fit["alpha"]),
        ) - floor
        valid = excess > 0
        ax.scatter(x[valid], excess[valid], color=color, s=30, label="median excess")
        ax.plot(
            x_dense,
            fit_excess,
            color="#172033",
            linestyle="--",
            linewidth=1.35,
            label=f"alpha={float(fit['alpha']):.2f}, log R2={float(fit['r2_log_excess']):.2f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("broad prior-predictive training signals")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    figure.suptitle("Broad MDN fitted-floor log-log excess checks", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def collect_run_rows(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for summary_path in sorted((output_root / "runs").glob("*/results/broad_scaling_run_summary.json")):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return rows


def set_torch_threads(thread_count: int | None) -> None:
    if thread_count is None:
        return
    torch.set_num_threads(int(thread_count))
    try:
        torch.set_num_interop_threads(max(1, int(thread_count)))
    except RuntimeError:
        pass


def format_int_list(values: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def format_context_variants(values: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(value) for value in values)


def training_loss_weights(z: np.ndarray, mode: str, tail_weight: float) -> np.ndarray | None:
    if mode == "none":
        return None
    z_unit = (np.asarray(z, dtype=np.float64) - PRIOR_LOG_MEAN.numpy()[None, :]) / PRIOR_LOG_STD.numpy()[None, :]
    if mode == "low_noise_exp":
        score = -z_unit[:, 2]
        weights = np.exp(float(tail_weight) * score)
        weights = np.clip(weights, 0.05, 30.0)
        weights /= max(float(weights.mean()), 1e-12)
        return weights.astype(np.float32)
    if mode == "low_noise_hard":
        weights = (z_unit[:, 2] < -float(tail_weight)).astype(np.float64)
        weights /= max(float(weights.mean()), 1e-12)
        return weights.astype(np.float32)
    if mode == "snr_exp":
        z_raw = np.asarray(z, dtype=np.float64)
        mean = PRIOR_LOG_MEAN.numpy()
        std = PRIOR_LOG_STD.numpy()
        score = (z_raw[:, 0] - z_raw[:, 2] - float(mean[0] - mean[2])) / float(
            math.sqrt(std[0] * std[0] + std[2] * std[2])
        )
        weights = np.exp(float(tail_weight) * score)
        weights = np.clip(weights, 0.05, 30.0)
        weights /= max(float(weights.mean()), 1e-12)
        return weights.astype(np.float32)
    if mode == "snr_hard":
        z_raw = np.asarray(z, dtype=np.float64)
        mean = PRIOR_LOG_MEAN.numpy()
        std = PRIOR_LOG_STD.numpy()
        score = (z_raw[:, 0] - z_raw[:, 2] - float(mean[0] - mean[2])) / float(
            math.sqrt(std[0] * std[0] + std[2] * std[2])
        )
        weights = (score > float(tail_weight)).astype(np.float64)
        weights /= max(float(weights.mean()), 1e-12)
        return weights.astype(np.float32)
    if mode != "tail_balanced":
        raise ValueError(f"Unknown loss_weight_mode: {mode}")
    tail_score = np.zeros(z_unit.shape[0], dtype=np.float64)
    tail_score += z_unit[:, 2] > 1.5  # high noise
    tail_score += z_unit[:, 1] < -1.5  # very slow decay
    tail_score += z_unit[:, 0] > 1.5  # high amplitude
    tail_score += z_unit[:, 0] < -1.5  # low amplitude
    tail_score += (z_unit[:, 2] > 1.0) & (z_unit[:, 1] < -1.0)
    weights = 1.0 + float(tail_weight) * tail_score
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def build_parallel_child_command(args: argparse.Namespace, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        str(args.preset),
        "--output-root",
        str(args.output_root),
        "--train-simulations",
        format_int_list(args.train_simulations),
        "--seeds",
        str(seed),
        "--val-simulations",
        str(args.val_simulations),
        "--standardization-simulations",
        str(args.standardization_simulations),
        "--train-sampler",
        str(args.train_sampler),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--lr-schedule",
        str(args.lr_schedule),
        "--lr-eta-min",
        str(args.lr_eta_min),
        "--lr-warmup-steps",
        str(args.lr_warmup_steps),
        "--lr-decay-epochs",
        str(args.lr_decay_epochs),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-eps",
        str(args.adam_eps),
        "--validation-every-epochs",
        str(args.validation_every_epochs),
        *(["--skip-training-validation"] if args.skip_training_validation else []),
        "--max-optimizer-steps",
        str(args.max_optimizer_steps),
        "--torch-compile",
        str(args.torch_compile),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--ema-decay",
        str(args.ema_decay),
        "--batching-mode",
        str(args.batching_mode),
        "--loss-weight-mode",
        str(args.loss_weight_mode),
        "--loss-tail-weight",
        str(args.loss_tail_weight),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--mdn-components",
        str(args.mdn_components),
        "--flow-layers",
        str(args.flow_layers),
        "--flow-context-dim",
        str(args.flow_context_dim),
        "--flow-activation",
        str(args.flow_activation),
        *(["--flow-residual"] if args.flow_residual else []),
        *(["--flow-randperm"] if args.flow_randperm else []),
        "--flow-kind",
        str(args.flow_kind),
        "--flow-passes",
        str(args.flow_passes),
        "--spline-bins",
        str(args.spline_bins),
        "--target-transform",
        str(args.target_transform),
        "--target-ridge",
        str(args.target_ridge),
        "--context-features",
        str(args.context_features),
        "--family",
        str(args.family),
        "--context-variants",
        format_context_variants(args.context_variants),
        "--posterior-samples",
        str(args.posterior_samples),
        "--observed-seed",
        str(args.observed_seed),
        "--validation-seed",
        str(args.validation_seed),
        "--standardization-seed",
        str(args.standardization_seed),
        "--device",
        str(args.device),
        "--jobs",
        "1",
        "--tail-top-k",
        str(args.tail_top_k),
        "--no-aggregate",
        "--print-every",
        str(args.print_every),
    ]
    if args.torch_threads is not None:
        command.extend(["--torch-threads", str(args.torch_threads)])
    if args.skip_x0_reference:
        command.append("--skip-x0-reference")
    else:
        command.extend(["--reference-npz", str(args.reference_npz)])
        command.extend(["--reference-metadata", str(args.reference_metadata)])
    if args.panel_marginal_cache is not None:
        command.extend(["--panel-marginal-cache", str(args.panel_marginal_cache)])
    if args.panel_posterior_samples is not None:
        command.extend(["--panel-posterior-samples", str(args.panel_posterior_samples)])
    if args.validation_cache is not None:
        command.extend(["--validation-cache", str(args.validation_cache)])
    command.extend(["--early-val-cache-simulations", str(args.early_val_cache_simulations)])
    if args.skip_existing:
        command.append("--skip-existing")
    command.append("--save-models" if args.save_models else "--no-save-models")
    if args.train_only:
        command.append("--train-only")
    return command


def run_parallel_by_seed(args: argparse.Namespace) -> None:
    max_workers = max(1, int(args.jobs))
    seeds = [int(seed) for seed in args.seeds]
    logs_dir = args.output_root / "logs" / "parallel"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.torch_threads is not None:
        threads = str(int(args.torch_threads))
        env.update({
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        })

    pending = list(seeds)
    active: dict[int, tuple[subprocess.Popen[bytes], object, Path]] = {}
    failed: list[tuple[int, int, Path]] = []
    print(f"parallel seeds={seeds} max_workers={max_workers}", flush=True)

    def reap_finished_processes() -> None:
        for seed, (process, log_handle, log_path) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[seed]
            if return_code != 0:
                failed.append((seed, int(return_code), log_path))
                print(f"failed seed={seed} rc={return_code} log={log_path}", flush=True)
            else:
                print(f"finished seed={seed} log={log_path}", flush=True)

    try:
        while pending or active:
            while pending and len(active) < max_workers:
                seed = pending.pop(0)
                command = build_parallel_child_command(args, seed)
                log_path = logs_dir / f"seed{seed}.log"
                log_handle = log_path.open("wb")
                log_handle.write((" ".join(command) + "\n\n").encode("utf-8"))
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                active[seed] = (process, log_handle, log_path)
                print(f"started seed={seed} pid={process.pid} log={log_path}", flush=True)

            time.sleep(1.0)
            reap_finished_processes()
    except KeyboardInterrupt:
        for process, log_handle, _ in active.values():
            process.terminate()
            log_handle.close()
        raise

    metadata = {
        "mode": "parallel_by_seed",
        "runtime": runtime_metadata(),
        "max_workers": max_workers,
        "seeds": seeds,
        "log_dir": str(logs_dir),
        "failed": [
            {"seed": seed, "return_code": return_code, "log": str(log_path)}
            for seed, return_code, log_path in failed
        ],
    }
    (args.output_root / "results").mkdir(parents=True, exist_ok=True)
    (args.output_root / "results" / "parallel_metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2),
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(1)


def print_run_row(row: dict[str, object]) -> None:
    if "panel_marginal_wasserstein_mean" in row:
        w_log_label = "panelW"
        w_log_value = float(row["panel_marginal_wasserstein_mean"])
        ratio_log_label = "panelRatio"
        ratio_log_value = float(row["panel_marginal_target_ratio_mean"])
    else:
        w_log_label = "x0W"
        w_log_value = float(row["x0_grid300_wasserstein"])
        ratio_log_label = "x0Ratio"
        ratio_log_value = float(row["x0_grid300_target_ratio"])
    nll_value = row.get("full_val_nll_z_units")
    if nll_value is None or not math.isfinite(float(nll_value)):
        nll_text = "skipped"
    else:
        nll_text = f"{float(nll_value):.4f}"
    print(
        "  "
        f"{w_log_label}={w_log_value:.5f} "
        f"{ratio_log_label}={ratio_log_value:.2f} "
        f"NLLz={nll_text} "
        f"seconds={row['training_seconds']:.1f}",
        flush=True,
    )


def run_one(
    *,
    args: argparse.Namespace,
    seed: int,
    train_simulations: int,
    context_variant: str,
    train_x_pool: np.ndarray,
    train_z_pool: np.ndarray,
    val_x_std: np.ndarray,
    val_z_std: np.ndarray,
    nll_val_x_std: np.ndarray,
    nll_val_z_std: np.ndarray,
    nll_validation_metadata: dict[str, object],
    observed_x: np.ndarray,
    stats: dict[str, np.ndarray],
    reference: dict[str, object] | None,
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    panel_reference: dict[str, object] | None,
    target_wasserstein: float,
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    run_dir = output_root / "runs" / run_dir_name(train_simulations, seed, context_variant)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "broad_scaling_run_summary.json"
    samples_path = results_dir / "broad_scaling_samples.npz"
    if args.skip_existing and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_model = existing.get("model_pt")
        existing_model_path = Path(str(existing_model)) if existing_model is not None else None
        if samples_path.exists() or (
            bool(args.train_only)
            and existing_model_path is not None
            and existing_model_path.exists()
        ):
            return existing

    z_log_det = float(np.log(stats["z_std"]).sum())
    config = replace(
        make_stage1_config(args, seed=seed, train_simulations=train_simulations),
        progress_jsonl=results_dir / "training_progress.jsonl",
        progress_nll_offset=z_log_det,
    )
    train_x_raw = train_x_pool[:train_simulations]
    train_z = train_z_pool[:train_simulations]
    train_x = stage1.transform_context_features(train_x_raw, str(args.context_features))
    train_x_std = stage1.standardize(train_x, stats["x_mean"], stats["x_std"]).astype(np.float32)
    train_z_std = stage1.standardize(train_z, stats["z_mean"], stats["z_std"]).astype(np.float32)
    train_x_model_std = apply_training_context_variant(
        train_x_std,
        context_variant,
        seed=int(seed + 30_000 + train_simulations),
    ).astype(np.float32)
    train_weights = training_loss_weights(
        train_z,
        mode=str(args.loss_weight_mode),
        tail_weight=float(args.loss_tail_weight),
    )
    val_x_model_std = apply_evaluation_context_variant(val_x_std, context_variant).astype(np.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2 + train_simulations)
    train_tensors = [
        torch.from_numpy(train_x_model_std),
        torch.from_numpy(train_z_std),
    ]
    if train_weights is not None:
        train_tensors.append(torch.from_numpy(train_weights))
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )
    model, metrics = stage1.train_one_model(
        family=str(args.family),
        config=config,
        train_loader=train_loader,
        val_x=torch.from_numpy(val_x_model_std),
        val_z=torch.from_numpy(val_z_std),
        device=device,
        x_dim=train_x_std.shape[1],
        z_dim=train_z_std.shape[1],
    )
    best_val_nll_z = float(metrics["best_val_nll"] + z_log_det)
    initial_train_batch_nll_z = float(metrics["initial_train_batch_nll"] + z_log_det)
    initial_val_nll_z = float(metrics["initial_val_nll"] + z_log_det)
    final_train_nll_z = float(metrics["final_train_nll"] + z_log_det)
    model_path = None
    if args.save_models:
        model_path = results_dir / f"{args.family}_model.pt"
        torch.save(
            {
                "family": args.family,
                "context_variant": context_variant,
                "context_variant_description": context_variant_description(context_variant),
                "state_dict": model.state_dict(),
                "x_mean": stats["x_mean"],
                "x_std": stats["x_std"],
                "z_mean": stats["z_mean"],
                "z_std": stats["z_std"],
                "config": asdict(config),
                "runtime": runtime_metadata(),
            },
            model_path,
        )
    if args.train_only:
        row = {
            "seed": int(seed),
            "family": str(args.family),
            "context_variant": context_variant,
            "context_variant_description": context_variant_description(context_variant),
            "context_features": str(args.context_features),
            "train_simulations": int(train_simulations),
            "val_simulations": int(args.val_simulations),
            "nll_val_simulations": 0,
            "nll_validation": {
                "skipped": True,
                "use": "exact_saved_checkpoint_ensemble_evaluator",
            },
            "posterior_samples": 0,
            "model_parameters": int(sum(param.numel() for param in model.parameters())),
            "runtime": runtime_metadata(),
            "epochs_completed": int(metrics["epochs_completed"]),
            "optimizer_steps": int(metrics["optimizer_steps"]),
            "batches_per_epoch": int(metrics["batches_per_epoch"]),
            "initial_train_batch_nll_standardized": float(metrics["initial_train_batch_nll"]),
            "initial_train_batch_nll_z_units": initial_train_batch_nll_z,
            "initial_val_nll_standardized": float(metrics["initial_val_nll"]),
            "initial_val_nll_z_units": initial_val_nll_z,
            "initial_losses_finite": bool(metrics["initial_losses_finite"]),
            "initial_eval_seconds": float(metrics["initial_eval_seconds"]),
            "validation_every_epochs": int(metrics["validation_every_epochs"]),
            "validation_evaluations": int(metrics["validation_evaluations"]),
            "max_optimizer_steps": int(metrics["max_optimizer_steps"]),
            "loss_weight_mode": str(metrics["loss_weight_mode"]),
            "loss_tail_weight": float(metrics["loss_tail_weight"]),
            "torch_compile": str(metrics["torch_compile"]),
            "grad_clip_norm": float(metrics["grad_clip_norm"]),
            "ema_decay": float(metrics["ema_decay"]),
            "batching_mode": str(metrics["batching_mode"]),
            "progress_jsonl": metrics["progress_jsonl"],
            "lr_warmup_steps": int(metrics["lr_warmup_steps"]),
            "lr_decay_epochs": int(metrics["lr_decay_epochs"]),
            "adam_beta1": float(metrics["adam_beta1"]),
            "adam_beta2": float(metrics["adam_beta2"]),
            "adam_eps": float(metrics["adam_eps"]),
            "flow_activation": str(metrics["flow_activation"]),
            "flow_residual": bool(metrics["flow_residual"]),
            "flow_randperm": bool(metrics["flow_randperm"]),
            "flow_passes": int(metrics["flow_passes"]),
            "flow_kind": str(metrics["flow_kind"]),
            "best_val_nll_standardized": float(metrics["best_val_nll"]),
            "best_val_nll_z_units": best_val_nll_z,
            "final_train_nll_standardized": float(metrics["final_train_nll"]),
            "final_train_nll_z_units": final_train_nll_z,
            "final_val_nll_standardized": float(metrics["final_val_nll"]),
            "full_val_nll_z_units": float("nan"),
            "full_val_nll_z_summary": None,
            "training_seconds": float(metrics["training_seconds"]),
            "x0_grid300_wasserstein": float("nan"),
            "x0_grid300_target": float(target_wasserstein),
            "x0_grid300_target_ratio": float("nan"),
            "summary_json": str(summary_path),
            "samples_npz": None,
            "model_pt": str(model_path) if model_path is not None else None,
            "validation_top_failures": None,
            "wasserstein_metrics": None,
            "history": metrics["history"],
            "config": asdict(config),
            "train_only": True,
        }
        summary_path.write_text(json.dumps(json_ready(row), indent=2), encoding="utf-8")
        return json_ready(row)
    nll_val_x_model_std = apply_evaluation_context_variant(nll_val_x_std, context_variant).astype(np.float32)
    full_val_nll_z_values = evaluate_val_nll_z_values(
        model=model,
        val_x_std=nll_val_x_model_std,
        val_z_std=nll_val_z_std,
        z_std=stats["z_std"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    full_val_nll_z_summary = summarize_val_nll_z_values(full_val_nll_z_values)
    full_val_nll_z = float(full_val_nll_z_summary["mean"])
    observed_x_context = stage1.transform_context_features(observed_x[None, :], str(args.context_features))[0]
    observed_x_std = stage1.standardize(
        observed_x_context[None, :],
        stats["x_mean"],
        stats["x_std"],
    ).astype(np.float32)
    observed_x_model_std = apply_evaluation_context_variant(observed_x_std, context_variant)[0]
    z_samples, theta_samples = sample_posterior_for_standardized_context(
        model=model,
        x_standardized=observed_x_model_std,
        z_mean=stats["z_mean"],
        z_std=stats["z_std"],
        n=int(args.posterior_samples),
        device=device,
    )
    w_metrics = None
    w_value = float("nan")
    target_ratio = float("nan")
    if reference is not None and reference_cache is not None and math.isfinite(target_wasserstein):
        w_metrics = compare_samples_to_reference_fast(theta_samples, reference, reference_cache)
        w_value = mean_normalized_wasserstein_value(w_metrics)
        target_ratio = float(w_value / target_wasserstein) if target_wasserstein > 0 else float("nan")
    panel_metrics = None
    if panel_reference is not None:
        panel_metrics = evaluate_panel_marginal_wasserstein(
            model=model,
            panel=panel_reference,
            x_mean=stats["x_mean"],
            x_std=stats["x_std"],
            z_mean=stats["z_mean"],
            z_std=stats["z_std"],
            n=int(args.panel_posterior_samples or args.posterior_samples),
            device=device,
            context_variant=context_variant,
            context_features=str(args.context_features),
        )
    tail_failures = save_top_validation_failures(
        results_dir=results_dir,
        nll_values=full_val_nll_z_values,
        val_x_std_original=nll_val_x_std,
        val_z_std=nll_val_z_std,
        stats=stats,
        top_k=int(args.tail_top_k),
        context_variant=context_variant,
    )
    np.savez_compressed(
        samples_path,
        z_samples=z_samples,
        theta_samples=theta_samples,
        observed_x=observed_x,
        observed_x_model_std=observed_x_model_std,
        context_variant=np.asarray(context_variant),
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        z_mean=stats["z_mean"],
        z_std=stats["z_std"],
    )
    row = {
        "seed": int(seed),
        "family": str(args.family),
        "context_variant": context_variant,
        "context_variant_description": context_variant_description(context_variant),
        "context_features": str(args.context_features),
        "train_simulations": int(train_simulations),
        "val_simulations": int(args.val_simulations),
        "nll_val_simulations": int(nll_val_x_std.shape[0]),
        "nll_validation": nll_validation_metadata,
        "posterior_samples": int(args.posterior_samples),
        "model_parameters": int(sum(param.numel() for param in model.parameters())),
        "runtime": runtime_metadata(),
        "epochs_completed": int(metrics["epochs_completed"]),
        "optimizer_steps": int(metrics["optimizer_steps"]),
        "batches_per_epoch": int(metrics["batches_per_epoch"]),
        "initial_train_batch_nll_standardized": float(metrics["initial_train_batch_nll"]),
        "initial_train_batch_nll_z_units": initial_train_batch_nll_z,
        "initial_val_nll_standardized": float(metrics["initial_val_nll"]),
        "initial_val_nll_z_units": initial_val_nll_z,
        "initial_losses_finite": bool(metrics["initial_losses_finite"]),
        "initial_eval_seconds": float(metrics["initial_eval_seconds"]),
        "validation_every_epochs": int(metrics["validation_every_epochs"]),
        "validation_evaluations": int(metrics["validation_evaluations"]),
        "max_optimizer_steps": int(metrics["max_optimizer_steps"]),
        "loss_weight_mode": str(metrics["loss_weight_mode"]),
        "loss_tail_weight": float(metrics["loss_tail_weight"]),
        "torch_compile": str(metrics["torch_compile"]),
        "grad_clip_norm": float(metrics["grad_clip_norm"]),
        "ema_decay": float(metrics["ema_decay"]),
        "batching_mode": str(metrics["batching_mode"]),
        "progress_jsonl": metrics["progress_jsonl"],
        "lr_warmup_steps": int(metrics["lr_warmup_steps"]),
        "lr_decay_epochs": int(metrics["lr_decay_epochs"]),
        "adam_beta1": float(metrics["adam_beta1"]),
        "adam_beta2": float(metrics["adam_beta2"]),
        "adam_eps": float(metrics["adam_eps"]),
        "flow_activation": str(metrics["flow_activation"]),
        "flow_residual": bool(metrics["flow_residual"]),
        "flow_randperm": bool(metrics["flow_randperm"]),
        "flow_passes": int(metrics["flow_passes"]),
        "flow_kind": str(metrics["flow_kind"]),
        "best_val_nll_standardized": float(metrics["best_val_nll"]),
        "best_val_nll_z_units": best_val_nll_z,
        "final_train_nll_standardized": float(metrics["final_train_nll"]),
        "final_train_nll_z_units": final_train_nll_z,
        "final_val_nll_standardized": float(metrics["final_val_nll"]),
        "full_val_nll_z_units": float(full_val_nll_z),
        "full_val_nll_z_summary": full_val_nll_z_summary,
        "training_seconds": float(metrics["training_seconds"]),
        "x0_grid300_wasserstein": float(w_value),
        "x0_grid300_target": float(target_wasserstein),
        "x0_grid300_target_ratio": target_ratio,
        "summary_json": str(summary_path),
        "samples_npz": str(samples_path),
        "model_pt": str(model_path) if model_path is not None else None,
        "validation_top_failures": tail_failures,
        "wasserstein_metrics": w_metrics,
        "history": metrics["history"],
        "config": asdict(config),
    }
    if panel_metrics is not None:
        row["panel_marginal_wasserstein_mean"] = float(panel_metrics["wasserstein"]["mean"])
        row["panel_marginal_wasserstein_median"] = float(panel_metrics["wasserstein"]["median"])
        row["panel_marginal_target_ratio_mean"] = float(panel_metrics["target_ratio"]["mean"])
        row["panel_marginal_target_ratio_median"] = float(panel_metrics["target_ratio"]["median"])
        row["panel_marginal_metrics"] = panel_metrics
    summary_path.write_text(json.dumps(json_ready(row), indent=2), encoding="utf-8")
    return json_ready(row)


def aggregate_and_write(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    reference: dict[str, object] | None,
    output_root: Path,
) -> dict[str, object]:
    if not rows:
        raise ValueError(f"No broad scaling run summaries found under {output_root / 'runs'}")
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    target_wasserstein = (
        float(reference["metadata"]["recommended_target"]) if reference is not None else float("nan")
    )
    variants = sorted({row_context_variant(row) for row in rows}, key=context_variant_sort_key)
    rows_by_variant = {
        variant: [row for row in rows if row_context_variant(row) == variant]
        for variant in variants
    }
    summary_rows_by_variant = {
        variant: summarize_rows(variant_rows, context_variant=variant)
        for variant, variant_rows in rows_by_variant.items()
    }
    primary_variant = "real" if "real" in summary_rows_by_variant else variants[0]
    summary_rows = summary_rows_by_variant[primary_variant]
    fits_by_variant = {
        variant: fit_summary(variant_summary_rows)
        for variant, variant_summary_rows in summary_rows_by_variant.items()
        if variant_summary_rows
    }
    fits = fits_by_variant[primary_variant]
    baseline_gates = build_baseline_gate_summary(summary_rows_by_variant)
    rows_csv = results_dir / "broad_scaling_rows.csv"
    summary_csv = results_dir / "broad_scaling_summary.csv"
    baseline_gates_csv = results_dir / "broad_scaling_baseline_gates.csv"
    summary_json = results_dir / "broad_scaling_summary.json"
    figure_path = figures_dir / "broad_scaling_law.png"
    log_excess_path = figures_dir / "broad_scaling_log_excess.png"
    write_csv([flatten_summary(row) for row in rows], rows_csv)
    all_summary_rows = [
        row
        for variant in variants
        for row in summary_rows_by_variant[variant]
    ]
    write_csv([flatten_summary(row) for row in all_summary_rows], summary_csv)
    write_csv(flatten_baseline_gate_rows(baseline_gates), baseline_gates_csv)
    plot_scaling(
        rows=rows_by_variant[primary_variant],
        summary_rows=summary_rows,
        fits=fits,
        target_wasserstein=target_wasserstein,
        output_path=figure_path,
    )
    plot_log_excess(
        summary_rows=summary_rows,
        fits=fits,
        output_path=log_excess_path,
    )
    output = {
        "config": json_ready(vars(args)),
        "runtime": runtime_metadata(),
        "reference": (
            {
                "grid_size": int(reference["grid_size"]),
                "grid_points": int(reference["grid_points"]),
                "target_wasserstein": target_wasserstein,
            }
            if reference is not None
            else None
        ),
        "rows": rows,
        "summary_rows": summary_rows,
        "summary_rows_by_context_variant": summary_rows_by_variant,
        "primary_context_variant": primary_variant,
        "power_law_fits": fits,
        "power_law_fits_by_context_variant": fits_by_variant,
        "baseline_gates": baseline_gates,
        "outputs": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "baseline_gates_csv": str(baseline_gates_csv),
            "summary_json": str(summary_json),
            "figure": str(figure_path),
            "log_excess_figure": str(log_excess_path),
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    return json_ready(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled broad-prior NPE scaling sweep for decay.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-simulations", type=parse_int_list, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument("--val-simulations", type=int, default=None)
    parser.add_argument("--standardization-simulations", type=int, default=None)
    parser.add_argument(
        "--train-sampler",
        choices=stage1.TRAIN_SAMPLERS,
        default="random",
        help="Sampler for training simulations. Validation and standardization stay random.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for final validation-NLL evaluation. Defaults to --batch-size.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine_epoch", "cosine_step", "one_cycle"),
        default="constant",
        help="Learning-rate schedule for broad-stage training.",
    )
    parser.add_argument(
        "--lr-eta-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine schedules. Default 0.0 preserves prior behavior.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Optimizer steps for linear LR warmup before cosine_step decay.",
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=0,
        help=(
            "Epochs over which cosine_epoch decays to --lr-eta-min. "
            "Use 0 to decay over --epochs, preserving prior behavior."
        ),
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Evaluate early-stop validation every N epochs, plus epoch 1 and final epoch.",
    )
    parser.add_argument(
        "--skip-training-validation",
        action="store_true",
        help="Skip initial/epoch validation during training and save the final state.",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="Stop training after this many optimizer steps. Use 0 to train full epochs.",
    )
    parser.add_argument(
        "--torch-compile",
        choices=("none", "default", "reduce_overhead"),
        default="none",
        help="Optional torch.compile mode for the NPE model.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=20.0,
        help="Gradient clipping norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional weight EMA decay in [0, 1). Use 0 to disable EMA.",
    )
    parser.add_argument(
        "--batching-mode",
        choices=("dataloader", "pre_shuffle", "sequential"),
        default="dataloader",
        help=(
            "Batch source for training. pre_shuffle uses contiguous tensor slices after "
            "per-epoch shuffling; sequential slices the existing random simulation order."
        ),
    )
    parser.add_argument(
        "--loss-weight-mode",
        choices=("none", "tail_balanced", "low_noise_exp", "low_noise_hard", "snr_exp", "snr_hard"),
        default="none",
        help="Optional per-simulation loss weighting mode.",
    )
    parser.add_argument(
        "--loss-tail-weight",
        type=float,
        default=3.0,
        help="Additional weight applied per active tail condition for tail_balanced loss.",
    )
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--mdn-components", type=int, default=None)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--flow-context-dim", type=int, default=64)
    parser.add_argument("--flow-activation", choices=stage1.FLOW_ACTIVATIONS, default="relu")
    parser.add_argument("--flow-residual", action="store_true")
    parser.add_argument("--flow-randperm", action="store_true")
    parser.add_argument("--flow-kind", choices=stage1.ZUKO_FLOW_KINDS, default="nsf")
    parser.add_argument(
        "--flow-passes",
        type=int,
        default=0,
        help="NSF autoregressive passes. Use 0 for Zuko's fully autoregressive default.",
    )
    parser.add_argument("--spline-bins", type=int, default=12)
    parser.add_argument(
        "--target-transform",
        choices=("none", "linear_residual", "fit_summary_residual"),
        default="none",
        help="Optional deterministic target transform before density training.",
    )
    parser.add_argument(
        "--target-ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty for residual target transforms.",
    )
    parser.add_argument(
        "--context-features",
        choices=stage1.CONTEXT_FEATURE_MODES,
        default="raw",
        help="Context representation used by the NPE.",
    )
    parser.add_argument(
        "--family",
        choices=("mdn", "affine_flow", "spline_flow", "full_gaussian", "diag_gaussian"),
        default="mdn",
    )
    parser.add_argument(
        "--context-variants",
        type=parse_context_variants,
        default=("real",),
        help=(
            "Comma-separated context variants to run. "
            "real keeps paired contexts, zero_x uses an input-independent baseline, "
            "and shuffled_x breaks training context-target pairing."
        ),
    )
    parser.add_argument("--posterior-samples", type=int, default=None)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--validation-seed", type=int, default=20260990)
    parser.add_argument("--standardization-seed", type=int, default=20260991)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Limit PyTorch CPU threads for this process; useful when running several workers.",
    )
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--reference-metadata", type=Path, default=DEFAULT_REFERENCE_METADATA)
    parser.add_argument(
        "--skip-x0-reference",
        action="store_true",
        help=(
            "Skip the large x0 grid-reference comparison. Panel marginal W and validation NLL "
            "still run when their caches are supplied."
        ),
    )
    parser.add_argument(
        "--panel-marginal-cache",
        type=Path,
        default=None,
        help=(
            "Optional cached panel of per-signal 1D marginal references. "
            "When supplied, panel marginal W replaces x0 W as the primary W plot."
        ),
    )
    parser.add_argument(
        "--panel-posterior-samples",
        type=int,
        default=None,
        help="Posterior samples per panel signal. Defaults to --posterior-samples.",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=None,
        help=(
            "Optional cached broad prior-predictive validation npz with x_val and z_val arrays. "
            "Used for the final reported NLL only; early stopping still uses --val-simulations."
        ),
    )
    parser.add_argument(
        "--early-val-cache-simulations",
        type=int,
        default=0,
        help=(
            "Use the first N validation-cache examples for checkpoint selection. "
            "Use 0 to keep the generated --val-simulations set."
        ),
    )
    parser.add_argument(
        "--tail-top-k",
        type=int,
        default=20,
        help="Save the top-k final-validation NLL failures for each run. Use 0 to disable.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Run independent seeds in parallel. Each worker still trains all requested D values "
            "serially for its seed, preserving the nested per-seed training pool."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("subprocess", "threads"),
        default="subprocess",
        help=(
            "Backend for --jobs with multiple seeds. subprocess preserves isolation; "
            "threads share precomputed context transforms, useful for expensive features."
        ),
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save trained model checkpoints. Use --no-save-models to keep only metrics and samples.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Train and save checkpoints without final full-cache NLL, posterior sampling, "
            "panel metrics, or failure plots. Use a separate checkpoint evaluator for proof NLL."
        ),
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-every", type=int, default=None)
    args = fill_from_preset(parser.parse_args())
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    return args


def main() -> None:
    args = parse_args()
    set_torch_threads(args.torch_threads)
    output_root = args.output_root
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(json_ready(vars(args)), indent=2))
        return

    if (
        args.jobs > 1
        and args.parallel_backend == "subprocess"
        and not args.aggregate_only
        and not args.no_aggregate
        and len(args.seeds) > 1
    ):
        run_parallel_by_seed(args)
        if args.train_only:
            print("train_only: parallel checkpoints written; skipping aggregate summary")
            return
        reference = None if args.skip_x0_reference else load_reference(args.reference_npz, args.reference_metadata)
        rows = collect_run_rows(output_root)
        output = aggregate_and_write(args=args, rows=rows, reference=reference, output_root=output_root)
        print(f"summary_json: {output['outputs']['summary_json']}")
        print(f"figure: {output['outputs']['figure']}")
        return

    if args.jobs > 1 and not args.aggregate_only and len(args.seeds) <= 1:
        print("--jobs > 1 requested, but only one seed is present; running serially.", flush=True)

    reference = None if args.skip_x0_reference else load_reference(args.reference_npz, args.reference_metadata)
    rows: list[dict[str, object]] = []
    if not args.aggregate_only:
        start = time.perf_counter()
        reference_cache = build_reference_cache(reference) if reference is not None and not args.train_only else None
        target_wasserstein = (
            float(reference["metadata"]["recommended_target"]) if reference is not None else float("nan")
        )
        panel_reference = None
        panel_reference_metadata = None
        if args.panel_marginal_cache is not None and not args.train_only:
            panel_reference, panel_reference_metadata = load_panel_marginal_cache(args.panel_marginal_cache)
        stats = standardization_stats(args)
        val_x, val_z, _ = stage1.sample_decay_pairs(
            n=int(args.val_simulations),
            seed=int(args.validation_seed),
        )
        val_x_context = stage1.transform_context_features(val_x, str(args.context_features))
        early_stop_validation_metadata = {
            "path": None,
            "simulations": int(args.val_simulations),
            "seed": int(args.validation_seed),
            "generated_in_sweep": True,
            "use": "early_stopping",
        }
        val_x_std = stage1.standardize(val_x_context, stats["x_mean"], stats["x_std"]).astype(np.float32)
        val_z_std = stage1.standardize(val_z, stats["z_mean"], stats["z_std"]).astype(np.float32)

        load_final_validation_cache = args.validation_cache is not None and (
            not args.train_only or int(args.early_val_cache_simulations) > 0
        )
        if load_final_validation_cache:
            nll_val_x, nll_val_z, nll_validation_metadata = load_validation_cache(args.validation_cache)
            nll_validation_metadata["use"] = "final_reported_nll"
            nll_val_x_context = transform_context_features_cached(
                nll_val_x,
                str(args.context_features),
                source_path=args.validation_cache,
            )
            nll_val_x_std = stage1.standardize(nll_val_x_context, stats["x_mean"], stats["x_std"]).astype(np.float32)
            nll_val_z_std = stage1.standardize(nll_val_z, stats["z_mean"], stats["z_std"]).astype(np.float32)
            del nll_val_x, nll_val_x_context, nll_val_z
            if int(args.early_val_cache_simulations) > 0:
                early_count = min(int(args.early_val_cache_simulations), int(nll_val_x_std.shape[0]))
                val_x_std = nll_val_x_std[:early_count]
                val_z_std = nll_val_z_std[:early_count]
                early_stop_validation_metadata = {
                    "path": str(args.validation_cache),
                    "simulations": int(early_count),
                    "source_simulations": int(nll_val_x_std.shape[0]),
                    "seed": None,
                    "generated_in_sweep": False,
                    "use": "early_stopping",
                }
        elif args.train_only:
            nll_val_x_std = val_x_std[:0]
            nll_val_z_std = val_z_std[:0]
            nll_validation_metadata = {
                "path": None,
                "simulations": 0,
                "skipped": True,
                "use": "exact_saved_checkpoint_ensemble_evaluator",
            }
        else:
            nll_val_x_std = val_x_std
            nll_val_z_std = val_z_std
            nll_validation_metadata = {
                "path": None,
                "simulations": int(args.val_simulations),
                "seed": int(args.validation_seed),
                "generated_in_sweep": True,
                "use": "final_reported_nll",
            }
        _, y_obs, _ = simulate_decay_data(seed=int(args.observed_seed))
        observed_x = y_obs.detach().cpu().numpy()
        device = stage1.choose_training_device(str(args.device))
        max_train = max(int(value) for value in args.train_simulations)
        total_runs = len(args.seeds) * len(args.train_simulations) * len(args.context_variants)
        completed = 0
        if args.jobs > 1 and args.parallel_backend == "threads" and len(args.seeds) > 1:
            train_pools = {
                int(seed): stage1.sample_decay_pairs(
                    n=max_train,
                    seed=int(seed),
                    sampler=str(args.train_sampler),
                )[:2]
                for seed in args.seeds
            }
            tasks: list[tuple[int, int, str]] = [
                (int(seed), int(train_simulations), str(context_variant))
                for seed in args.seeds
                for train_simulations in args.train_simulations
                for context_variant in args.context_variants
            ]
            print(
                f"threaded seeds={list(train_pools)} total_runs={len(tasks)} "
                f"max_workers={int(args.jobs)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
                future_to_task = {
                    executor.submit(
                        run_one,
                        args=args,
                        seed=seed,
                        train_simulations=train_simulations,
                        context_variant=context_variant,
                        train_x_pool=train_pools[seed][0],
                        train_z_pool=train_pools[seed][1],
                        val_x_std=val_x_std,
                        val_z_std=val_z_std,
                        nll_val_x_std=nll_val_x_std,
                        nll_val_z_std=nll_val_z_std,
                        nll_validation_metadata=nll_validation_metadata,
                        observed_x=observed_x,
                        stats=stats,
                        reference=reference,
                        reference_cache=reference_cache,
                        panel_reference=panel_reference,
                        target_wasserstein=target_wasserstein,
                        device=device,
                        output_root=output_root,
                    ): (seed, train_simulations, context_variant)
                    for seed, train_simulations, context_variant in tasks
                }
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    seed, train_simulations, context_variant = future_to_task[future]
                    row = future.result()
                    rows.append(row)
                    print(
                        f"[{completed}/{len(tasks)}] finished {args.family} "
                        f"variant={context_variant} seed={seed} train={train_simulations}",
                        flush=True,
                    )
                    print_run_row(row)
        else:
            for seed in args.seeds:
                train_x_pool, train_z_pool, _ = stage1.sample_decay_pairs(
                    n=max_train,
                    seed=int(seed),
                    sampler=str(args.train_sampler),
                )
                for train_simulations in args.train_simulations:
                    for context_variant in args.context_variants:
                        completed += 1
                        print(
                            f"[{completed}/{total_runs}] training {args.family} "
                            f"variant={context_variant} seed={seed} train={train_simulations}",
                            flush=True,
                        )
                        row = run_one(
                            args=args,
                            seed=int(seed),
                            train_simulations=int(train_simulations),
                            context_variant=str(context_variant),
                            train_x_pool=train_x_pool,
                            train_z_pool=train_z_pool,
                            val_x_std=val_x_std,
                            val_z_std=val_z_std,
                            nll_val_x_std=nll_val_x_std,
                            nll_val_z_std=nll_val_z_std,
                            nll_validation_metadata=nll_validation_metadata,
                            observed_x=observed_x,
                            stats=stats,
                            reference=reference,
                            reference_cache=reference_cache,
                            panel_reference=panel_reference,
                            target_wasserstein=target_wasserstein,
                            device=device,
                            output_root=output_root,
                        )
                        rows.append(row)
                        print_run_row(row)
        metadata = {
            "total_seconds_before_aggregation": time.perf_counter() - start,
            "runtime": runtime_metadata(),
            "early_stop_validation": early_stop_validation_metadata,
            "nll_validation": nll_validation_metadata,
            "panel_reference": panel_reference_metadata,
            "x0_reference": (
                {
                    "npz": str(args.reference_npz),
                    "metadata": str(args.reference_metadata),
                    "target_wasserstein": target_wasserstein,
                }
                if reference is not None
                else None
            ),
            "torch_threads": torch.get_num_threads(),
            "parallel_backend": str(args.parallel_backend),
            "eval_batch_size": int(args.eval_batch_size),
            "context_variants": list(args.context_variants),
            "context_features": str(args.context_features),
            "tail_top_k": int(args.tail_top_k),
            "early_val_cache_simulations": int(args.early_val_cache_simulations),
            "train_sampler": str(args.train_sampler),
            "lr_eta_min": float(args.lr_eta_min),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "lr_decay_epochs": int(args.lr_decay_epochs),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "adam_eps": float(args.adam_eps),
            "flow_activation": str(args.flow_activation),
            "flow_residual": bool(args.flow_residual),
            "flow_randperm": bool(args.flow_randperm),
            "flow_passes": int(args.flow_passes),
            "flow_kind": str(args.flow_kind),
            "validation_every_epochs": int(args.validation_every_epochs),
            "skip_training_validation": bool(args.skip_training_validation),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "torch_compile": str(args.torch_compile),
            "grad_clip_norm": float(args.grad_clip_norm),
            "ema_decay": float(args.ema_decay),
            "batching_mode": str(args.batching_mode),
            "standardization": {key: value for key, value in stats.items()},
        }
        metadata_name = (
            "broad_scaling_metadata.json"
            if not args.no_aggregate
            else f"broad_scaling_metadata_seeds_{format_int_list(args.seeds)}.json"
        )
        (results_dir / metadata_name).write_text(
            json.dumps(json_ready(metadata), indent=2),
            encoding="utf-8",
        )
    else:
        rows = collect_run_rows(output_root)

    if args.no_aggregate:
        print(f"no_aggregate: true rows_written={len(rows)}")
        return

    output = aggregate_and_write(args=args, rows=rows, reference=reference, output_root=output_root)
    print(f"summary_json: {output['outputs']['summary_json']}")
    print(f"figure: {output['outputs']['figure']}")


if __name__ == "__main__":
    main()
