from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance

import npe_stage1_decay as stage1
from evaluate_npe_ensemble_nll import evaluate_ensemble
from mcmc_decay_inference import PARAMETER_NAMES
from npe_posterior_viewer import load_stage1_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/201_flow2_ensemble_data_scaling"
)
DEFAULT_VALIDATION_CACHE = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
)
DEFAULT_PANEL_CACHE = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel64_grid180_refined_marginals.npz"
)
POPULATION_ENTROPY_NLL = -3.6386545787958
POPULATION_ENTROPY_NLL_UNCERTAINTY = 0.0026


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
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


def run_summary_path(train_root: Path, train_simulations: int, seed: int) -> Path:
    return train_root / "runs" / f"n{train_simulations}_seed{seed}" / "results" / "broad_scaling_run_summary.json"


def collect_member_records(
    *,
    train_roots: list[Path],
    train_simulations: int,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = []
    missing = []
    for seed in seeds:
        summary_path = None
        for train_root in train_roots:
            candidate = run_summary_path(train_root, train_simulations, seed)
            if candidate.exists():
                summary_path = candidate
                break
        if summary_path is None:
            missing.append(run_summary_path(train_roots[0], train_simulations, seed))
            continue
        if not summary_path.exists():
            missing.append(summary_path)
            continue
        record = read_json(summary_path)
        model_path = record.get("model_pt")
        if not model_path:
            missing.append(summary_path)
            continue
        resolved_model = resolve_path(Path(str(model_path)))
        if not resolved_model.exists():
            missing.append(resolved_model)
            continue
        record["_summary_path"] = str(summary_path)
        record["_model_path"] = str(resolved_model)
        records.append(record)
    if missing:
        missing_text = "\n".join(str(path) for path in missing[:12])
        if len(missing) > 12:
            missing_text += f"\n... {len(missing) - 12} more"
        raise FileNotFoundError(
            f"Missing member summaries/checkpoints for D={train_simulations}:\n{missing_text}"
        )
    return records


def load_panel_marginal_cache(path: Path, max_signals: int) -> tuple[dict[str, object], dict[str, object]]:
    arrays = np.load(path, allow_pickle=False)
    limit = int(max_signals)
    panel_size = int(arrays["x_panel"].shape[0])
    if limit <= 0 or limit > panel_size:
        limit = panel_size
    panel = {
        "x_panel": np.asarray(arrays["x_panel"][:limit], dtype=np.float64),
        "theta_panel": np.asarray(arrays["theta_panel"][:limit], dtype=np.float64),
        "theta_axes": np.asarray(arrays["theta_axes"][:limit], dtype=np.float64),
        "marginal_weights": np.asarray(arrays["marginal_weights"][:limit], dtype=np.float64),
        "target_wasserstein": np.asarray(arrays["target_wasserstein"][:limit], dtype=np.float64),
        "labels": np.asarray(arrays["labels"][:limit]).astype(str).tolist(),
    }
    metadata_path = path.with_suffix(".json")
    metadata: dict[str, object] = {
        "path": str(path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "panel_size": limit,
        "source_panel_size": panel_size,
        "grid_size": int(panel["theta_axes"].shape[-1]),
        "target_wasserstein": quantile_summary(panel["target_wasserstein"]),
    }
    if metadata_path.exists():
        metadata["metadata"] = read_json(metadata_path)
    return panel, metadata


def observed_features_for_state(state: dict[str, object], x: np.ndarray) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float64)
    x_mean = np.asarray(state["x_mean"], dtype=np.float64)
    if raw.shape[0] == x_mean.shape[0]:
        return raw
    config = state.get("config", {})
    mode = str(config.get("context_features", "raw")) if isinstance(config, dict) else "raw"
    features = stage1.transform_context_features(raw[None, :], mode)[0]
    if features.shape[0] != x_mean.shape[0]:
        raise ValueError(
            f"Context feature shape mismatch for context_features={mode!r}: "
            f"got {features.shape[0]}, expected {x_mean.shape[0]}"
        )
    return features


def equal_member_counts(total: int, member_count: int) -> np.ndarray:
    if total < member_count:
        raise ValueError("--posterior-samples must be at least the ensemble size.")
    counts = np.full(member_count, total // member_count, dtype=int)
    counts[: total - int(counts.sum())] += 1
    return counts


def compare_samples_to_marginals(
    *,
    theta_samples: np.ndarray,
    theta_axes: np.ndarray,
    marginal_weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    values = []
    per_axis = {}
    for axis, name in enumerate(PARAMETER_NAMES):
        ref_axis = theta_axes[axis]
        ref_weights = marginal_weights[axis] / np.sum(marginal_weights[axis])
        mean = float(np.sum(ref_axis * ref_weights))
        sd = float(np.sqrt(max(np.sum((ref_axis - mean) ** 2 * ref_weights), 0.0)))
        w_value = wasserstein_distance(
            theta_samples[:, axis],
            ref_axis,
            v_weights=ref_weights,
        )
        normalized = float(w_value / max(sd, 1e-12))
        values.append(normalized)
        per_axis[f"w_{name}"] = normalized
    return float(np.mean(values)), per_axis


def evaluate_panel_wasserstein(
    *,
    model_paths: list[Path],
    panel: dict[str, object],
    posterior_samples: int,
    seed: int,
    device: torch.device,
    print_prefix: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loaded = [load_stage1_checkpoint(path, device) for path in model_paths]
    counts = equal_member_counts(int(posterior_samples), len(loaded))
    x_panel = np.asarray(panel["x_panel"], dtype=np.float64)
    theta_panel = np.asarray(panel["theta_panel"], dtype=np.float64)
    theta_axes = np.asarray(panel["theta_axes"], dtype=np.float64)
    marginal_weights = np.asarray(panel["marginal_weights"], dtype=np.float64)
    targets = np.asarray(panel["target_wasserstein"], dtype=np.float64)
    labels = list(panel["labels"])
    rows = []
    for index, label in enumerate(labels):
        theta_parts = []
        for member_index, ((model, state), count) in enumerate(zip(loaded, counts, strict=True)):
            torch.manual_seed(int(seed) + 1000 * index + member_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed) + 1000 * index + member_index)
            observed_features = observed_features_for_state(state, x_panel[index])
            _, theta_member = stage1.sample_posterior_for_observation(
                model=model,
                observed_x=observed_features,
                x_mean=np.asarray(state["x_mean"], dtype=np.float64),
                x_std=np.asarray(state["x_std"], dtype=np.float64),
                z_mean=np.asarray(state["z_mean"], dtype=np.float64),
                z_std=np.asarray(state["z_std"], dtype=np.float64),
                n=int(count),
                device=device,
            )
            theta_parts.append(theta_member)
        theta_samples = np.vstack(theta_parts)
        w_value, per_axis = compare_samples_to_marginals(
            theta_samples=theta_samples,
            theta_axes=theta_axes[index],
            marginal_weights=marginal_weights[index],
        )
        target = float(targets[index])
        row = {
            "index": int(index),
            "label": str(label),
            "A": float(theta_panel[index, 0]),
            "k": float(theta_panel[index, 1]),
            "sigma": float(theta_panel[index, 2]),
            "target_wasserstein": target,
            "wasserstein": w_value,
            "target_ratio": float(w_value / target) if target > 0 else float("nan"),
        }
        row.update(per_axis)
        rows.append(row)
        if index == 0 or index + 1 == len(labels) or (index + 1) % 16 == 0:
            print(
                f"{print_prefix} panel [{index + 1}/{len(labels)}] "
                f"W={w_value:.5f} ratio={row['target_ratio']:.2f}",
                flush=True,
            )
    w_values = np.asarray([row["wasserstein"] for row in rows], dtype=np.float64)
    ratio_values = np.asarray([row["target_ratio"] for row in rows], dtype=np.float64)
    summary = {
        "posterior_samples_per_signal": int(posterior_samples),
        "member_sample_counts": counts.tolist(),
        "signal_count": len(rows),
        "wasserstein": quantile_summary(w_values),
        "target_ratio": quantile_summary(ratio_values),
    }
    return summary, rows


def power_with_floor(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -alpha)


def fit_power_with_floor(x: np.ndarray, y: np.ndarray, *, min_floor: float = 0.0) -> dict[str, object] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if x.size < 4 or np.any(y <= 0.0):
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    floor_upper = min(float(np.min(y)) - 1e-10, float(np.max(y)) - 1e-10)
    if floor_upper < min_floor:
        return None
    initial_floor = max(float(np.min(y)) * 0.5, min_floor)
    initial_alpha = 0.25
    initial_amplitude = float((np.max(y) - initial_floor) * np.min(x) ** initial_alpha)
    try:
        params, _ = curve_fit(
            power_with_floor,
            x,
            y,
            p0=[initial_floor, max(initial_amplitude, 1e-12), initial_alpha],
            bounds=([min_floor, 1e-12, 1e-4], [floor_upper, 1e12, 5.0]),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    fitted = power_with_floor(x, *params)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    floor, amplitude, alpha = [float(value) for value in params]
    excess = y - floor
    log_valid = excess > 0.0
    log_r2 = float("nan")
    if np.count_nonzero(log_valid) >= 3:
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


def fit_loss_with_asymptote(x: np.ndarray, y: np.ndarray) -> dict[str, object] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if x.size < 4:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    asymptote_upper = float(np.min(y)) - 1e-10
    span = max(float(np.max(y) - np.min(y)), 1e-6)
    initial_asymptote = float(np.min(y) - 0.5 * span)
    initial_alpha = 0.25
    initial_amplitude = float((np.max(y) - initial_asymptote) * np.min(x) ** initial_alpha)
    try:
        params, _ = curve_fit(
            power_with_floor,
            x,
            y,
            p0=[initial_asymptote, max(initial_amplitude, 1e-12), initial_alpha],
            bounds=(
                [float(np.min(y) - 10.0 * span), 1e-12, 1e-4],
                [asymptote_upper, 1e12, 5.0],
            ),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    fitted = power_with_floor(x, *params)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    asymptote, amplitude, alpha = [float(value) for value in params]
    return {
        "asymptote": asymptote,
        "amplitude": amplitude,
        "alpha": alpha,
        "r2_raw": r2,
        "x": x.tolist(),
        "y": y.tolist(),
        "fitted": fitted.tolist(),
    }


def fit_power_no_floor(x: np.ndarray, y: np.ndarray) -> dict[str, object] | None:
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if x.size < 3:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    alpha = float(-slope)
    amplitude = float(np.exp(intercept))
    fitted = amplitude * np.power(x, -alpha)
    ss_log_res = float(np.sum((log_y - np.log(fitted)) ** 2))
    ss_log_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    log_r2 = 1.0 - ss_log_res / ss_log_tot if ss_log_tot > 0 else float("nan")
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    raw_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "amplitude": amplitude,
        "alpha": alpha,
        "r2_raw": raw_r2,
        "r2_log": log_r2,
        "x": x.tolist(),
        "y": y.tolist(),
        "fitted": fitted.tolist(),
    }


def entropy_floor_sensitivity(
    x: np.ndarray,
    nll: np.ndarray,
    *,
    entropy_floor: float,
    entropy_uncertainty: float,
) -> list[dict[str, object]]:
    results = []
    for delta in (-entropy_uncertainty, 0.0, entropy_uncertainty):
        floor = float(entropy_floor + delta)
        excess = np.asarray(nll, dtype=np.float64) - floor
        fit = fit_power_no_floor(x, excess)
        results.append(
            {
                "entropy_floor": floor,
                "entropy_floor_delta": float(delta),
                "min_excess": float(np.min(excess)),
                "fit_nll_excess_no_floor": fit,
            }
        )
    return results


def update_scaling_fits(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    x = np.asarray([row["train_simulations_per_member"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"] for row in rows], dtype=np.float64)
    nll_excess = np.asarray([row["nll_excess_over_entropy_floor"] for row in rows], dtype=np.float64)
    panel_w = np.asarray([row["panel_wasserstein_mean"] for row in rows], dtype=np.float64)
    entropy_floor = float(summary.get("population_entropy_nll", POPULATION_ENTROPY_NLL))
    entropy_uncertainty = float(
        summary.get("population_entropy_nll_uncertainty", POPULATION_ENTROPY_NLL_UNCERTAINTY)
    )
    panel_floor = float(summary["panel_floor_wasserstein_mean"])
    summary["population_entropy_nll_uncertainty"] = entropy_uncertainty
    fits = dict(summary.get("fits", {}))
    fits["full_val_nll_z_units"] = fit_loss_with_asymptote(x, nll)
    fits["nll_excess_fixed_entropy_no_floor"] = fit_power_no_floor(x, nll_excess)
    fits["nll_excess_fixed_entropy_floor_sensitivity"] = entropy_floor_sensitivity(
        x,
        nll,
        entropy_floor=entropy_floor,
        entropy_uncertainty=entropy_uncertainty,
    )
    fits["nll_excess_over_entropy_floor"] = fit_power_with_floor(
        x,
        nll_excess,
        min_floor=0.0,
    )
    fits["panel_wasserstein_mean"] = fit_power_with_floor(x, panel_w, min_floor=panel_floor)
    summary["fits"] = fits


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_region_labels(ax: plt.Axes, x_values: np.ndarray) -> None:
    if x_values.size < 4:
        return
    left = float(x_values[1])
    right = float(x_values[-2])
    ax.axvline(left, color="#8a8a8a", linewidth=1.1)
    ax.axvline(right, color="#8a8a8a", linewidth=1.1)
    centers = [
        math.sqrt(float(x_values[0]) * left),
        math.sqrt(left * right),
        math.sqrt(right * float(x_values[-1])),
    ]
    labels = ["Small Data", "Power-law Probe", "Floor Probe"]
    for center, label in zip(centers, labels, strict=True):
        ax.text(
            center,
            0.965,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )


def plot_scaling(summary: dict[str, object], rows: list[dict[str, object]], output_path: Path) -> None:
    x = np.asarray([row["train_simulations_per_member"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"] for row in rows], dtype=np.float64)
    nll_excess = np.asarray([row["nll_excess_over_entropy_floor"] for row in rows], dtype=np.float64)
    entropy_floor = float(summary.get("population_entropy_nll", POPULATION_ENTROPY_NLL))
    entropy_uncertainty = float(
        summary.get("population_entropy_nll_uncertainty", POPULATION_ENTROPY_NLL_UNCERTAINTY)
    )
    fits = summary.get("fits", {})
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, nll, color="#0f766e", marker="o", linewidth=2.3, label=r"measured $L(D)$")
    fit = fits.get("full_val_nll_z_units") if isinstance(fits, dict) else None
    if isinstance(fit, dict):
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(
            x_dense,
            power_with_floor(
                x_dense,
                float(fit["asymptote"]),
                float(fit["amplitude"]),
                float(fit["alpha"]),
            ),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=rf"fit $L_{{free}}+A D^{{-\alpha}}$; $\alpha={float(fit['alpha']):.2f}$",
        )
        ax.axhline(
            float(fit["asymptote"]),
            color="#d97706",
            linestyle=":",
            linewidth=1.3,
            label=rf"free fitted $L_{{free}}={float(fit['asymptote']):.5f}$",
        )
    ax.axhspan(
        entropy_floor - entropy_uncertainty,
        entropy_floor + entropy_uncertainty,
        color="#b42318",
        alpha=0.12,
        label=r"independent $\hat H \pm s_H$",
    )
    ax.set_xscale("log")
    y_min = min(float(np.nanmin(nll)), entropy_floor - entropy_uncertainty)
    y_max = max(float(np.nanmax(nll)), entropy_floor + entropy_uncertainty)
    margin = max(0.015, 0.15 * (y_max - y_min))
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("training simulations per ensemble member D")
    ax.set_ylabel(r"$L(D)$: validation NLL (z units)")
    ax.set_title("Raw loss with free asymptote")
    ax.text(
        0.02,
        0.04,
        rf"Left fit: $L(D)=L_{{free}}+A D^{{-\alpha}}$"
        "\n"
        rf"Independent floor: $\hat H={entropy_floor:.5f}\pm {entropy_uncertainty:.3f}$",
        transform=ax.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)

    ax = axes[1]
    ax.plot(
        x,
        nll_excess,
        color="#0f766e",
        marker="o",
        linewidth=2.3,
        label=r"$\Delta_{\hat H}(D)=L(D)-\hat H$",
    )
    fit = fits.get("nll_excess_fixed_entropy_no_floor") if isinstance(fits, dict) else None
    if isinstance(fit, dict):
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(
            x_dense,
            float(fit["amplitude"]) * np.power(x_dense, -float(fit["alpha"])),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=rf"fit $B D^{{-\beta}}$; $\beta={float(fit['alpha']):.2f}$",
        )
    lower_excess = nll - (entropy_floor + entropy_uncertainty)
    upper_excess = nll - (entropy_floor - entropy_uncertainty)
    if np.all(upper_excess > 0.0):
        ax.fill_between(
            x,
            np.maximum(lower_excess, 1e-5),
            upper_excess,
            color="#b42318",
            alpha=0.10,
            label=r"$\hat H$ uncertainty propagated",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    lower = max(
        min(float(np.nanmin(nll_excess)) * 0.55, float(np.nanmin(upper_excess)) * 0.55),
        1e-4,
    )
    ax.set_ylim(
        lower,
        float(np.nanmax(upper_excess)) * 1.75,
    )
    ax.set_xlabel("training simulations per ensemble member D")
    ax.set_ylabel(r"$\Delta_{\hat H}(D)$: validation NLL minus $\hat H$")
    ax.set_title("Fixed-floor excess loss")
    ax.text(
        0.02,
        0.04,
        rf"Right fit: $\Delta_{{\hat H}}(D)=B D^{{-\beta}}$"
        "\n"
        r"Band: $L(D)-(\hat H\pm s_H)$",
        transform=ax.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)
    fig.suptitle("Single-decay Flow2 residual NSF ensemble data scaling", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_panel_diagnostic(summary: dict[str, object], rows: list[dict[str, object]], output_path: Path) -> None:
    x = np.asarray([row["train_simulations_per_member"] for row in rows], dtype=np.float64)
    panel_w = np.asarray([row["panel_wasserstein_mean"] for row in rows], dtype=np.float64)
    panel_floor = float(summary["panel_floor_wasserstein_mean"])
    fits = summary.get("fits", {})
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.8), constrained_layout=True)
    ax.plot(x, panel_w, color="#5b56b3", marker="o", linewidth=2.3)
    fit = fits.get("panel_wasserstein_mean") if isinstance(fits, dict) else None
    if isinstance(fit, dict):
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(
            x_dense,
            power_with_floor(
                x_dense,
                float(fit["floor"]),
                float(fit["amplitude"]),
                float(fit["alpha"]),
            ),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=f"diagnostic fit alpha={float(fit['alpha']):.2f}",
        )
    ax.axhline(panel_floor, color="#b42318", linestyle=":", linewidth=1.3, label="panel eval floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(
        max(panel_floor * 0.75, float(np.nanmin(panel_w)) * 0.65),
        float(np.nanmax(panel_w)) * 1.75,
    )
    ax.set_xlabel("training simulations per ensemble member D")
    ax.set_ylabel("panel mean normalized marginal Wasserstein")
    ax.set_title("Posterior faithfulness diagnostic")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed 4-member Flow2 residual NSF ensemble data-scaling sweep."
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument(
        "--extra-train-root",
        type=Path,
        action="append",
        default=[],
        help="Additional train roots to search for completed D/seed checkpoints.",
    )
    parser.add_argument(
        "--train-simulations",
        type=parse_int_list,
        default=(64_000, 128_000, 256_000, 512_000, 1_024_000, 2_048_000),
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=(20260901, 20260902, 20260903, 20260904),
    )
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--max-panel-signals", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--plot-only-summary",
        type=Path,
        default=None,
        help="Regenerate plots from an existing summary JSON without recomputing metrics.",
    )
    args = parser.parse_args()

    train_root = resolve_path(args.train_root)
    train_roots = [train_root] + [resolve_path(path) for path in args.extra_train_root]
    validation_cache = resolve_path(args.validation_cache)
    panel_cache = resolve_path(args.panel_cache)
    device = torch.device(str(args.device))
    panel, panel_metadata = load_panel_marginal_cache(panel_cache, int(args.max_panel_signals))
    results_dir = train_root / "results"
    figures_dir = train_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only_summary is not None:
        summary_path = resolve_path(args.plot_only_summary)
        summary = read_json(summary_path)
        rows = list(summary["rows"])
        update_scaling_fits(summary, rows)
        outputs = summary.get("outputs", {})
        figure_path = Path(str(outputs.get("figure", figures_dir / "flow2_ensemble_data_scaling_weng_style.png")))
        diagnostic_path = figure_path.with_name("flow2_ensemble_panel_w_diagnostic.png")
        plot_scaling(summary, rows, figure_path)
        plot_panel_diagnostic(summary, rows, diagnostic_path)
        summary["outputs"]["panel_diagnostic_figure"] = str(diagnostic_path)
        summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
        print(json.dumps(summary["outputs"], indent=2), flush=True)
        return

    rows = []
    per_signal_rows = []
    for d_index, train_count in enumerate(args.train_simulations):
        records = collect_member_records(
            train_roots=train_roots,
            train_simulations=int(train_count),
            seeds=args.seeds,
        )
        model_paths = [Path(record["_model_path"]) for record in records]
        print(f"evaluating D={train_count} with {len(model_paths)} members", flush=True)
        nll = evaluate_ensemble(
            model_paths=model_paths,
            validation_cache=validation_cache,
            device=device,
            batch_size=int(args.batch_size),
            max_examples=int(args.max_validation_examples),
        )
        nll_path = results_dir / f"flow2_ensemble_nll_n{train_count}.json"
        nll_path.write_text(json.dumps(json_ready(nll), indent=2), encoding="utf-8")
        panel_summary, panel_rows = evaluate_panel_wasserstein(
            model_paths=model_paths,
            panel=panel,
            posterior_samples=int(args.posterior_samples),
            seed=int(args.seed) + 100_000 * d_index,
            device=device,
            print_prefix=f"D={train_count}",
        )
        panel_path = results_dir / f"flow2_ensemble_panel_w_n{train_count}.json"
        panel_path.write_text(
            json.dumps(json_ready({"summary": panel_summary, "rows": panel_rows}), indent=2),
            encoding="utf-8",
        )
        for panel_row in panel_rows:
            per_signal_rows.append({
                "train_simulations_per_member": int(train_count),
                **panel_row,
            })
        member_seconds = np.asarray(
            [record.get("training_seconds", float("nan")) for record in records],
            dtype=np.float64,
        )
        row = {
            "train_simulations_per_member": int(train_count),
            "aggregate_train_simulations": int(train_count) * len(model_paths),
            "ensemble_size": len(model_paths),
            "seeds": list(args.seeds),
            "model_parameters_per_member": int(records[0].get("model_parameters", 0)),
            "full_val_nll_z_units": float(nll["ensemble_full_val_nll_z_units"]),
            "nll_excess_over_entropy_floor": float(
                nll["ensemble_full_val_nll_z_units"] - POPULATION_ENTROPY_NLL
            ),
            "best_individual_full_val_nll_z_units": float(
                nll["best_individual_full_val_nll_z_units"]
            ),
            "panel_wasserstein_mean": float(panel_summary["wasserstein"]["mean"]),
            "panel_wasserstein_median": float(panel_summary["wasserstein"]["median"]),
            "panel_target_ratio_mean": float(panel_summary["target_ratio"]["mean"]),
            "panel_target_ratio_median": float(panel_summary["target_ratio"]["median"]),
            "member_training_seconds_sum": float(np.nansum(member_seconds)),
            "member_training_seconds_max": float(np.nanmax(member_seconds)),
            "member_summary_paths": [record["_summary_path"] for record in records],
            "model_paths": [str(path) for path in model_paths],
            "nll_json": str(nll_path),
            "panel_json": str(panel_path),
        }
        rows.append(row)

    rows.sort(key=lambda row: int(row["train_simulations_per_member"]))
    panel_floor = float(panel_metadata["target_wasserstein"]["mean"])
    outputs = {
        "rows_csv": str(results_dir / "flow2_ensemble_data_scaling_rows.csv"),
        "per_signal_csv": str(results_dir / "flow2_ensemble_data_scaling_panel_rows.csv"),
        "summary_json": str(results_dir / "flow2_ensemble_data_scaling_summary.json"),
        "figure": str(figures_dir / "flow2_ensemble_data_scaling_weng_style.png"),
        "panel_diagnostic_figure": str(figures_dir / "flow2_ensemble_panel_w_diagnostic.png"),
    }
    summary = {
        "description": "Fixed-recipe 4-member Flow2 residual NSF ensemble data scaling.",
        "train_root": str(train_root),
        "train_roots": [str(path) for path in train_roots],
        "validation_cache": str(validation_cache),
        "panel": panel_metadata,
        "population_entropy_nll": POPULATION_ENTROPY_NLL,
        "population_entropy_nll_uncertainty": POPULATION_ENTROPY_NLL_UNCERTAINTY,
        "panel_floor_wasserstein_mean": panel_floor,
        "train_simulations_axis": "per_member",
        "rows": rows,
        "outputs": outputs,
    }
    update_scaling_fits(summary, rows)
    write_csv(rows, Path(outputs["rows_csv"]))
    write_csv(per_signal_rows, Path(outputs["per_signal_csv"]))
    Path(outputs["summary_json"]).write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    plot_scaling(summary, rows, Path(outputs["figure"]))
    plot_panel_diagnostic(summary, rows, Path(outputs["panel_diagnostic_figure"]))
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
