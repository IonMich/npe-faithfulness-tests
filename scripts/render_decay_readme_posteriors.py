from __future__ import annotations

import argparse
import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp

import npe_stage1_decay as stage1
from calibrate_sign_target import build_grid_reference, compare_samples_to_reference, mode_summary
from npe_flow_stress_tests import make_banana_case, make_label_switch_case, make_linear6_case, make_sign_case, run_random_walk_mcmc
from npe_posterior_viewer import (
    DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
    DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
    DEFAULT_BEST_BROAD_MODEL,
    DEFAULT_BEST_BROAD_SPLINE_MODEL,
    DEFAULT_BROAD_MODEL,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
    NPEPosteriorViewer,
    SampleCornerLayer,
    WeightedCornerLayer,
    render_corner_layers,
)
from train_sign_population_npe import (
    BANANA_B,
    BANANA_C,
    BANANA_PRIOR_STD,
    BANANA_SIGMA,
    banana_context,
    banana_log_evidence,
    banana_log_likelihood,
    label_log_likelihood_np,
    label_raw_prior_logpdf,
    linear6_log_py_given_log_sigma,
    linear6_sufficient_stats,
    sort_label_target,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_decay_posteriors"
SUMMARY_PATH = OUTPUT_DIR / "decay_population_readme_posteriors_summary.json"
SIGN_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_sign_posteriors"
BANANA_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_banana_posteriors"
LABEL_SWITCH_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_label_switch_posteriors"
LINEAR6_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_linear6_posteriors"
SIGN_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/"
    "results/sign_population_ensemble_summary.json"
)
BANANA_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/03_stress_banana/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/"
    "results/banana_population_ensemble_summary.json"
)
LABEL_SWITCH_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/04_stress_label_switch/03_population_npe/02_flow2_residual_full_prior_512k_ensemble4_e30/"
    "results/label_switch_population_ensemble_summary.json"
)
LINEAR6_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/"
    "results/linear6_population_ensemble_summary.json"
)
LOG_2PI = np.log(2.0 * np.pi)

MODEL_ID_MAP = {
    "broad_fresh_e15_ensemble4": "flow2_residual_nsf_ensemble4",
    "broad_weighted_checkpoint_pool": "convex_weighted_checkpoint_ensemble",
}

CASES = [
    {
        "key": "population_prior_predictive",
        "mode": "prior",
        "corner_path": OUTPUT_DIR / "decay_population_posterior_corner.png",
        "signal_path": OUTPUT_DIR / "decay_population_posterior_signal.png",
    },
    {
        "key": "low_prior_stress",
        "mode": "low_prior_very_low",
        "corner_path": OUTPUT_DIR / "decay_low_prior_stress_posterior_corner.png",
        "signal_path": OUTPUT_DIR / "decay_low_prior_stress_posterior_signal.png",
    },
]


def save_data_uri(uri: str, path: Path) -> None:
    prefix, payload = uri.split(",", 1)
    if not prefix.startswith("data:image/png;base64"):
        raise ValueError(f"Unexpected image data URI prefix: {prefix[:80]}")
    path.write_bytes(base64.b64decode(payload))


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


def metric_value(metrics: dict[str, object] | None) -> float | None:
    if metrics is None:
        return None
    value = metrics.get("mean_normalized_wasserstein")
    if not isinstance(value, dict):
        return None
    return float(value["value"])


def clean_model_id(model_id: str) -> str:
    return MODEL_ID_MAP.get(model_id, model_id)


def selected_model_summary(model: dict[str, object]) -> dict[str, object]:
    return {
        "id": clean_model_id(str(model["id"])),
        "label": str(model["label"]),
        "ensemble_size": model.get("ensemble_size"),
        "full_val_nll_z_units": model.get("full_val_nll_z_units"),
        "training_seconds": model.get("training_seconds"),
    }


def summarize_mcmc_diagnostics(diagnostics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "max_rhat": max(float(value["rhat"]) for value in diagnostics.values()),
        "min_bulk_ess": min(float(value["ess_bulk"]) for value in diagnostics.values()),
        "min_tail_ess": min(float(value["ess_tail"]) for value in diagnostics.values()),
    }


def sample_sign_prior_predictive_signal(*, seed: int, draw_index: int) -> tuple[np.ndarray, np.ndarray]:
    case = make_sign_case()
    rng = np.random.default_rng(seed)
    theta = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(draw_index + 1, case.z_dim),
    )
    x = case.simulate_x(theta, rng)
    return theta[draw_index].astype(np.float64), x[draw_index].astype(np.float64)


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean[None, :]) / std[None, :]).astype(np.float32)


def stage1_config_from_checkpoint(config_dict: dict[str, Any]) -> stage1.Stage1Config:
    config = stage1.Stage1Config(**config_dict)
    if config.progress_jsonl is not None:
        config = replace(config, progress_jsonl=Path(config.progress_jsonl))
    return config


def load_sign_member(member_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(member_path, map_location="cpu", weights_only=False)
    config = stage1_config_from_checkpoint(checkpoint["config"])
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float64)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float64)
    model = stage1.make_model(
        "spline_flow",
        config,
        x_dim=int(x_mean.shape[0]),
        z_dim=int(z_mean.shape[0]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "path": member_path,
    }


def load_sign_ensemble(summary_path: Path, device: torch.device) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return [load_sign_member(Path(item["model_pt"]), device) for item in summary["members"]]


@torch.no_grad()
def sample_stage1_ensemble(
    *,
    members: list[dict[str, Any]],
    x_context: np.ndarray,
    samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    base_count = samples // len(members)
    remainder = samples % len(members)
    for index, member in enumerate(members):
        count = base_count + (1 if index < remainder else 0)
        x_standardized = standardize(
            x_context[None, :],
            np.asarray(member["x_mean"], dtype=np.float64),
            np.asarray(member["x_std"], dtype=np.float64),
        )
        x_tensor = torch.from_numpy(x_standardized).to(device)
        torch.manual_seed(int(seed) + index)
        standardized_samples = member["model"].sample(count, x_tensor).detach().cpu().numpy()
        samples_raw = (
            standardized_samples * np.asarray(member["z_std"], dtype=np.float64)[None, :]
            + np.asarray(member["z_mean"], dtype=np.float64)[None, :]
        )
        chunks.append(samples_raw)
    result = np.concatenate(chunks, axis=0)
    rng.shuffle(result, axis=0)
    return result


@torch.no_grad()
def sample_sign_population_npe(
    *,
    members: list[dict[str, Any]],
    x: np.ndarray,
    samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folded = sample_stage1_ensemble(
        members=members,
        x_context=x,
        samples=samples,
        seed=seed,
        device=device,
    )
    sign = np.where(rng.random(folded.shape[0]) < 0.5, -1.0, 1.0)
    samples_raw = np.column_stack([sign * np.maximum(folded[:, 0], 0.0), folded[:, 1]])
    rng.shuffle(samples_raw, axis=0)
    return samples_raw


def sample_linear6_prior_predictive_signal(*, seed: int, draw_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_linear6_case()
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(draw_index + 1, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return z[draw_index].astype(np.float64), x[draw_index].astype(np.float64), case.context(x)[draw_index].astype(np.float64)


def sample_banana_prior_predictive_signal(*, seed: int, draw_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_banana_case()
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(draw_index + 1, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return z[draw_index].astype(np.float64), x[draw_index].astype(np.float64), banana_context(x)[draw_index]


def sample_label_switch_prior_predictive_signal(
    *,
    seed: int,
    draw_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    case = make_label_switch_case()
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(draw_index + 1, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return (
        z[draw_index].astype(np.float64),
        sort_label_target(z)[draw_index].astype(np.float64),
        x[draw_index].astype(np.float64),
        case.context(x)[draw_index].astype(np.float64),
    )


def normal_logpdf(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized * standardized - np.log(std) - 0.5 * LOG_2PI


def banana_exact_grid_layer(
    *,
    x_context: np.ndarray,
    grid_size: int,
    grid_limit: float,
) -> WeightedCornerLayer:
    x_context_2d = x_context[None, :]
    x_raw = x_context[:2]
    center = np.array([x_raw[0], x_raw[1] - BANANA_B * (x_raw[0] * x_raw[0] - BANANA_C)])
    theta1 = np.linspace(center[0] - grid_limit, center[0] + grid_limit, grid_size)
    theta2 = np.linspace(center[1] - grid_limit, center[1] + grid_limit, grid_size)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="ij")
    values = np.column_stack([theta1_grid.ravel(), theta2_grid.ravel()])
    log_prior = normal_logpdf(values[:, 0], 0.0, BANANA_PRIOR_STD) + normal_logpdf(
        values[:, 1],
        0.0,
        BANANA_PRIOR_STD,
    )
    repeated_context = np.repeat(x_context_2d, values.shape[0], axis=0)
    log_post = log_prior + banana_log_likelihood(repeated_context, values)
    log_post -= float(
        banana_log_evidence(
            x_context_2d,
            quadrature_order=64,
            chunk_size=1,
        )[0]
    )
    log_mass = log_post.reshape(grid_size, grid_size)
    step1 = float(theta1[1] - theta1[0]) if grid_size > 1 else 1.0
    step2 = float(theta2[1] - theta2[0]) if grid_size > 1 else 1.0
    weights = np.exp(log_mass - logsumexp(log_mass))

    def widths(axis: np.ndarray) -> np.ndarray:
        if axis.size <= 1:
            return np.ones_like(axis)
        return np.full_like(axis, float(axis[1] - axis[0]), dtype=np.float64)

    return WeightedCornerLayer(
        label="Exact grid",
        color="#172033",
        values=values,
        weights=weights.ravel(),
        grid_shape=weights.shape,
        axes=(theta1, theta2),
        widths=(widths(theta1), widths(theta2)),
        hist_lw=2.0,
        contour_lw=1.55,
    )


def uniform_axis_widths(axis: np.ndarray) -> np.ndarray:
    if axis.size <= 1:
        return np.ones_like(axis, dtype=np.float64)
    return np.full_like(axis, float(axis[1] - axis[0]), dtype=np.float64)


def label_switch_exact_grid_layer(
    *,
    x_raw: np.ndarray,
    reference_samples: np.ndarray,
    true_values: np.ndarray,
    grid_size: int,
    pad_fraction: float = 0.25,
    chunk_size: int = 30_000,
) -> tuple[WeightedCornerLayer, dict[str, object]]:
    reference = np.asarray(reference_samples, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)
    axes = []
    axis_ranges = []
    for index in range(3):
        low, high = np.quantile(reference[:, index], [0.001, 0.999])
        low = min(float(low), float(true_values[index]))
        high = max(float(high), float(true_values[index]))
        width = max(high - low, 1e-6)
        low -= pad_fraction * width
        high += pad_fraction * width
        axes.append(np.linspace(low, high, grid_size, dtype=np.float64))
        axis_ranges.append([low, high])

    mu_low, mu_high, log_sigma = np.meshgrid(*axes, indexing="ij")
    values = np.column_stack([mu_low.ravel(), mu_high.ravel(), log_sigma.ravel()])
    log_post = np.full(values.shape[0], -np.inf, dtype=np.float64)
    x_raw = np.asarray(x_raw, dtype=np.float64)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        z = values[start:stop]
        valid = z[:, 0] < z[:, 1]
        if not np.any(valid):
            continue
        z_valid = z[valid]
        x_batch = np.broadcast_to(x_raw[None, :], (z_valid.shape[0], x_raw.size))
        log_post_chunk = log_post[start:stop]
        log_post_chunk[valid] = (
            np.log(2.0)
            + label_raw_prior_logpdf(z_valid)
            + label_log_likelihood_np(x_batch, z_valid)
        )

    log_norm = float(logsumexp(log_post))
    if not np.isfinite(log_norm):
        raise RuntimeError("Label-switching exact grid has no finite posterior mass.")
    weights = np.exp(log_post - log_norm)
    weight_grid = weights.reshape((grid_size, grid_size, grid_size))
    edge_mask = np.zeros_like(weight_grid, dtype=bool)
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = 0
        edge_mask[tuple(sl)] = True
        sl[axis] = -1
        edge_mask[tuple(sl)] = True

    metadata = {
        "grid_size": int(grid_size),
        "grid_points": int(values.shape[0]),
        "axis_ranges": axis_ranges,
        "edge_mass": float(weight_grid[edge_mask].sum()),
        "pad_fraction": float(pad_fraction),
    }
    return (
        WeightedCornerLayer(
            label="Exact grid",
            color="#172033",
            values=values,
            weights=weights,
            grid_shape=weight_grid.shape,
            axes=tuple(axes),
            widths=tuple(uniform_axis_widths(axis) for axis in axes),
            hist_lw=2.0,
            contour_lw=1.55,
        ),
        metadata,
    )


def sample_banana_exact_posterior(
    *,
    x_context: np.ndarray,
    samples: int,
    seed: int,
    grid_size: int = 2400,
    grid_limit: float = 6.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_raw = x_context[:2]
    center = float(x_raw[0])
    theta1_grid = np.linspace(center - grid_limit, center + grid_limit, grid_size)
    log_theta1_prior = normal_logpdf(theta1_grid, 0.0, BANANA_PRIOR_STD)
    log_x1 = normal_logpdf(x_raw[0], theta1_grid, BANANA_SIGMA[0])
    theta2_integrated_std = np.sqrt(BANANA_PRIOR_STD * BANANA_PRIOR_STD + BANANA_SIGMA[1] * BANANA_SIGMA[1])
    x2_mean = BANANA_B * (theta1_grid * theta1_grid - BANANA_C)
    log_x2_integrated = normal_logpdf(x_raw[1], x2_mean, theta2_integrated_std)
    log_theta1_post = log_theta1_prior + log_x1 + log_x2_integrated
    probabilities = np.exp(log_theta1_post - logsumexp(log_theta1_post))
    indices = rng.choice(grid_size, size=samples, replace=True, p=probabilities)
    step = float(theta1_grid[1] - theta1_grid[0])
    theta1 = np.clip(
        theta1_grid[indices] + rng.uniform(-0.5 * step, 0.5 * step, size=samples),
        theta1_grid[0],
        theta1_grid[-1],
    )
    posterior_var = 1.0 / (1.0 / (BANANA_PRIOR_STD * BANANA_PRIOR_STD) + 1.0 / (BANANA_SIGMA[1] * BANANA_SIGMA[1]))
    posterior_mean = posterior_var * (x_raw[1] - BANANA_B * (theta1 * theta1 - BANANA_C)) / (
        BANANA_SIGMA[1] * BANANA_SIGMA[1]
    )
    theta2 = rng.normal(posterior_mean, np.sqrt(posterior_var), size=samples)
    return np.column_stack([theta1, theta2])


def sample_linear6_exact_posterior(
    *,
    x_context: np.ndarray,
    samples: int,
    seed: int,
    grid_size: int = 2400,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coef, projected_sq, residual_sq = linear6_sufficient_stats(x_context[None, :])
    log_sigma_mean = np.log(0.25)
    log_sigma_std = 0.50
    grid = np.linspace(log_sigma_mean - 5.0 * log_sigma_std, log_sigma_mean + 5.0 * log_sigma_std, grid_size)
    log_post = normal_logpdf(grid, log_sigma_mean, log_sigma_std) + linear6_log_py_given_log_sigma(
        projected_sq=projected_sq[:, None],
        residual_sq=residual_sq[:, None],
        log_sigma=grid[None, :],
    )[0]
    probabilities = np.exp(log_post - logsumexp(log_post))
    indices = rng.choice(grid_size, size=samples, replace=True, p=probabilities)
    step = float(grid[1] - grid[0])
    log_sigma = np.clip(grid[indices] + rng.uniform(-0.5 * step, 0.5 * step, size=samples), grid[0], grid[-1])
    n_obs = 32
    prior_std_w = 1.25
    sigma2 = np.exp(2.0 * log_sigma)
    posterior_var = 1.0 / (1.0 / (prior_std_w * prior_std_w) + n_obs / sigma2)
    shrink = posterior_var * n_obs / sigma2
    mean = shrink[:, None] * coef[0][None, :]
    weights = rng.normal(mean, np.sqrt(posterior_var)[:, None])
    return np.column_stack([weights, log_sigma])


def compare_sample_marginals(estimate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    from scipy.stats import wasserstein_distance

    names = [f"w{i}" for i in range(1, 7)] + ["log_sigma"]
    rows = {}
    normalized = []
    for index, name in enumerate(names):
        scale = max(float(np.std(reference[:, index], ddof=1)), 1e-12)
        distance = float(wasserstein_distance(reference[:, index], estimate[:, index]))
        rows[name] = {
            "wasserstein": distance,
            "reference_sd": scale,
            "normalized_wasserstein": distance / scale,
        }
        normalized.append(distance / scale)
    return {
        "mean_normalized_wasserstein": float(np.mean(normalized)),
        "parameters": rows,
    }


def compare_sample_marginals_named(
    estimate: np.ndarray,
    reference: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    from scipy.stats import wasserstein_distance

    rows = {}
    normalized = []
    for index, name in enumerate(names):
        scale = max(float(np.std(reference[:, index], ddof=1)), 1e-12)
        distance = float(wasserstein_distance(reference[:, index], estimate[:, index]))
        rows[name] = {
            "wasserstein": distance,
            "reference_sd": scale,
            "normalized_wasserstein": distance / scale,
        }
        normalized.append(distance / scale)
    return {
        "mean_normalized_wasserstein": float(np.mean(normalized)),
        "parameters": rows,
    }


def compare_samples_to_weighted_grid_named(
    estimate: np.ndarray,
    reference: WeightedCornerLayer,
    names: list[str],
) -> dict[str, object]:
    from scipy.stats import wasserstein_distance

    estimate = np.asarray(estimate, dtype=np.float64)
    values = np.asarray(reference.values, dtype=np.float64)
    weights = np.asarray(reference.weights, dtype=np.float64)
    rows = {}
    normalized = []
    for index, name in enumerate(names):
        reference_values = values[:, index]
        mean = float(np.sum(reference_values * weights))
        scale = max(float(np.sqrt(np.sum(weights * (reference_values - mean) ** 2))), 1e-12)
        distance = float(wasserstein_distance(reference_values, estimate[:, index], u_weights=weights))
        rows[name] = {
            "wasserstein": distance,
            "reference_sd": scale,
            "normalized_wasserstein": distance / scale,
        }
        normalized.append(distance / scale)
    return {
        "mean_normalized_wasserstein": float(np.mean(normalized)),
        "parameters": rows,
    }


def sign_reference_layer(reference: dict[str, object]) -> WeightedCornerLayer:
    theta1 = np.asarray(reference["theta1_grid"], dtype=np.float64)
    theta2 = np.asarray(reference["theta2_grid"], dtype=np.float64)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="ij")
    weights = np.asarray(reference["weights"], dtype=np.float64)

    def widths(axis: np.ndarray) -> np.ndarray:
        if axis.size <= 1:
            return np.ones_like(axis)
        return np.full_like(axis, float(axis[1] - axis[0]), dtype=np.float64)

    return WeightedCornerLayer(
        label="Exact grid",
        color="#172033",
        values=np.column_stack([theta1_grid.ravel(), theta2_grid.ravel()]),
        weights=weights.ravel(),
        grid_shape=weights.shape,
        axes=(theta1, theta2),
        widths=(widths(theta1), widths(theta2)),
        hist_lw=2.0,
        contour_lw=1.55,
    )


def render_case(viewer: NPEPosteriorViewer, case: dict[str, object]) -> dict[str, object]:
    result = viewer.render(
        model_ids=["broad_fresh_e15_ensemble4", "broad_weighted_checkpoint_pool"],
        mode=str(case["mode"]),
        draw_id=None,
        reuse_current=False,
        refresh_layers=set(),
        npe_render_mode="sample",
        posterior_samples=7000,
        include_grid=True,
        include_mcmc=True,
        grid_size=60,
        npe_grid_size=None,
    )
    save_data_uri(str(result["corner"]), Path(case["corner_path"]))
    save_data_uri(str(result["signal"]), Path(case["signal_path"]))

    npe_metrics_raw = result["npe_grid_metrics"]
    assert isinstance(npe_metrics_raw, dict)
    mcmc_diagnostics = summarize_mcmc_diagnostics(result["mcmc_metadata"]["diagnostics"])
    return {
        "mode": case["mode"],
        "mode_metadata": result["mode_metadata"],
        "true_theta": result["true_theta"],
        "posterior_samples": result["posterior_samples"],
        "corner_path": str(Path(case["corner_path"]).relative_to(ROOT)),
        "signal_path": str(Path(case["signal_path"]).relative_to(ROOT)),
        "grid": {
            "grid_size": result["grid_metadata"]["grid_size"],
            "grid_points": result["grid_metadata"]["grid_points"],
            "max_edge_mass": result["grid_metadata"]["max_edge_mass"],
        },
        "mcmc": {
            "mean_normalized_wasserstein": metric_value(result["mcmc_grid_metrics"]),
            "acceptance_rate": result["mcmc_metadata"]["acceptance_rate"],
            "convergence_ok": result["mcmc_metadata"]["convergence_ok"],
            **mcmc_diagnostics,
        },
        "npe_mean_normalized_wasserstein": {
            clean_model_id(model_id): metric_value(metrics)
            for model_id, metrics in npe_metrics_raw.items()
        },
        "selected_models": [
            selected_model_summary(model)
            for model in result["selected_npe_models"]
        ],
    }


def render_decay_cases() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viewer = NPEPosteriorViewer(
        DEFAULT_MODEL,
        DEFAULT_BROAD_MODEL,
        DEFAULT_BEST_BROAD_MODEL,
        DEFAULT_BEST_BROAD_SPLINE_MODEL,
        DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
        DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
        DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
        seed=20260702,
        device="cpu",
        mcmc_device="cpu",
        mcmc_chains=8,
        mcmc_steps=24_000,
        mcmc_burn_in=6_000,
        mcmc_proposal_scale=(0.030, 0.030, 0.040),
    )
    summary = {
        str(case["key"]): render_case(viewer, case)
        for case in CASES
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)


def render_sign_population_case(args: argparse.Namespace) -> None:
    SIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    theta, x = sample_sign_prior_predictive_signal(
        seed=args.signal_seed,
        draw_index=args.draw_index,
    )
    reference = build_grid_reference(x0=x, grid_size=args.grid_size, grid_limit=args.grid_limit)
    members = load_sign_ensemble(args.sign_ensemble_summary, device)
    npe_samples = sample_sign_population_npe(
        members=members,
        x=x,
        samples=args.npe_samples,
        seed=args.seed + 1000,
        device=device,
    )

    case = make_sign_case()
    mcmc_samples, mcmc_accept, mcmc_seconds = run_random_walk_mcmc(
        case,
        x,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        seed=args.seed,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    mcmc_post = mcmc_samples[:, args.mcmc_burn_in :, :].reshape(-1, case.z_dim)

    figure = render_corner_layers(
        labels=[r"$\theta_1$", r"$\theta_2$"],
        true_values=theta,
        weighted_layers=[sign_reference_layer(reference)],
        sample_layers=[
            SampleCornerLayer("MCMC", "#b85c38", mcmc_post, hist_lw=1.5, contour_lw=1.35),
            SampleCornerLayer("Population NPE", "#0f766e", npe_samples, hist_lw=1.5, contour_lw=1.35),
        ],
        true_color="#172033",
        title="Sign population posterior: exact grid vs MCMC vs NPE",
        rng=np.random.default_rng(args.seed + 2000),
    )
    figure_path = SIGN_OUTPUT_DIR / "sign_population_prior_signal_corner.png"
    summary_path = SIGN_OUTPUT_DIR / "sign_population_prior_signal_summary.json"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    summary = {
        "description": (
            "Fresh full-prior sign posterior check for the population-trained "
            "folded-target NPE ensemble. This is not the old fixed-x0 sign run."
        ),
        "signal": {
            "seed": int(args.signal_seed),
            "draw_index": int(args.draw_index),
            "theta": theta,
            "x": x,
        },
        "grid": {
            "grid_size": int(args.grid_size),
            "grid_limit": float(args.grid_limit),
            "edge_mass": reference["edge_mass"],
        },
        "mcmc": {
            "chains": int(args.mcmc_chains),
            "steps": int(args.mcmc_steps),
            "burn_in": int(args.mcmc_burn_in),
            "posterior_samples": int(mcmc_post.shape[0]),
            "acceptance_rate": float(np.mean(mcmc_accept)),
            "seconds": float(mcmc_seconds),
            "to_grid_raw": compare_samples_to_reference(mcmc_post, reference, diagnostic=False),
            "to_grid_diagnostic": compare_samples_to_reference(mcmc_post, reference, diagnostic=True),
            "mode_mass": mode_summary(mcmc_post),
        },
        "npe": {
            "ensemble_summary": args.sign_ensemble_summary,
            "members": int(len(members)),
            "posterior_samples": int(npe_samples.shape[0]),
            "to_grid_raw": compare_samples_to_reference(npe_samples, reference, diagnostic=False),
            "to_grid_diagnostic": compare_samples_to_reference(npe_samples, reference, diagnostic=True),
            "mode_mass": mode_summary(npe_samples),
        },
        "outputs": {
            "figure": figure_path,
            "summary": summary_path,
        },
    }
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def render_banana_population_case(args: argparse.Namespace) -> None:
    BANANA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    case = make_banana_case()
    theta, x_raw, x_context = sample_banana_prior_predictive_signal(
        seed=args.signal_seed,
        draw_index=args.draw_index,
    )
    members = load_sign_ensemble(args.banana_ensemble_summary, device)
    exact_samples = sample_banana_exact_posterior(
        x_context=x_context,
        samples=args.exact_samples,
        seed=args.seed + 3000,
        grid_limit=args.grid_limit,
    )
    npe_samples = sample_stage1_ensemble(
        members=members,
        x_context=x_context,
        samples=args.npe_samples,
        seed=args.seed + 1000,
        device=device,
    )
    mcmc_samples, mcmc_accept, mcmc_seconds = run_random_walk_mcmc(
        case,
        x_raw,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        seed=args.seed,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    mcmc_post = mcmc_samples[:, args.mcmc_burn_in :, :].reshape(-1, case.z_dim)
    grid_layer = banana_exact_grid_layer(
        x_context=x_context,
        grid_size=args.grid_size,
        grid_limit=args.grid_limit,
    )
    weight_grid = grid_layer.weights.reshape(grid_layer.grid_shape)
    edge_mass = float(
        np.sum(weight_grid[0, :])
        + np.sum(weight_grid[-1, :])
        + np.sum(weight_grid[1:-1, 0])
        + np.sum(weight_grid[1:-1, -1])
    )

    figure = render_corner_layers(
        labels=[r"$\theta_1$", r"$\theta_2$"],
        true_values=theta,
        weighted_layers=[grid_layer],
        sample_layers=[
            SampleCornerLayer("MCMC", "#b85c38", mcmc_post, hist_lw=1.5, contour_lw=1.35),
            SampleCornerLayer("Population NPE", "#0f766e", npe_samples, hist_lw=1.5, contour_lw=1.35),
        ],
        true_color="#172033",
        title="Banana posterior: exact grid vs MCMC vs NPE",
        rng=np.random.default_rng(args.seed + 2000),
    )
    figure_path = BANANA_OUTPUT_DIR / "banana_population_prior_signal_corner.png"
    summary_path = BANANA_OUTPUT_DIR / "banana_population_prior_signal_summary.json"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    names = ["theta1", "theta2"]
    summary = {
        "description": (
            "Fresh full-prior Banana posterior check for the population-trained "
            "Flow2 residual NSF ensemble. The comparison is in raw theta coordinates."
        ),
        "signal": {
            "seed": int(args.signal_seed),
            "draw_index": int(args.draw_index),
            "theta": theta,
            "x": x_raw,
            "context": x_context,
        },
        "grid": {
            "grid_size": int(args.grid_size),
            "grid_limit": float(args.grid_limit),
            "edge_mass": edge_mass,
        },
        "exact": {
            "posterior_samples": int(exact_samples.shape[0]),
        },
        "mcmc": {
            "chains": int(args.mcmc_chains),
            "steps": int(args.mcmc_steps),
            "burn_in": int(args.mcmc_burn_in),
            "posterior_samples": int(mcmc_post.shape[0]),
            "acceptance_rate": float(np.mean(mcmc_accept)),
            "seconds": float(mcmc_seconds),
            "to_exact": compare_sample_marginals_named(mcmc_post, exact_samples, names),
        },
        "npe": {
            "ensemble_summary": args.banana_ensemble_summary,
            "members": int(len(members)),
            "posterior_samples": int(npe_samples.shape[0]),
            "to_exact": compare_sample_marginals_named(npe_samples, exact_samples, names),
        },
        "outputs": {
            "figure": figure_path,
            "summary": summary_path,
        },
    }
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def render_label_switch_population_case(args: argparse.Namespace) -> None:
    LABEL_SWITCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    case = make_label_switch_case()
    z_raw, z_sorted, x_raw, x_context = sample_label_switch_prior_predictive_signal(
        seed=args.signal_seed,
        draw_index=args.draw_index,
    )
    members = load_sign_ensemble(args.label_switch_ensemble_summary, device)
    npe_samples = sample_stage1_ensemble(
        members=members,
        x_context=x_context,
        samples=args.npe_samples,
        seed=args.seed + 1000,
        device=device,
    )
    mcmc_samples, mcmc_accept, mcmc_seconds = run_random_walk_mcmc(
        case,
        x_raw,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        seed=args.seed,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    mcmc_sorted = sort_label_target(mcmc_samples[:, args.mcmc_burn_in :, :].reshape(-1, case.z_dim))
    label_grid_size = min(int(args.grid_size), 96)
    grid_layer, grid_metadata = label_switch_exact_grid_layer(
        x_raw=x_raw,
        reference_samples=mcmc_sorted,
        true_values=z_sorted,
        grid_size=label_grid_size,
    )

    figure = render_corner_layers(
        labels=[r"$\mu_{low}$", r"$\mu_{high}$", r"$\log\sigma$"],
        true_values=z_sorted,
        weighted_layers=[grid_layer],
        sample_layers=[
            SampleCornerLayer("MCMC reference", "#b85c38", mcmc_sorted, hist_lw=1.5, contour_lw=1.35),
            SampleCornerLayer("Population NPE", "#0f766e", npe_samples, hist_lw=1.5, contour_lw=1.35),
        ],
        true_color="#172033",
        title="Label switching posterior: exact grid vs MCMC vs NPE",
        rng=np.random.default_rng(args.seed + 2000),
    )
    figure_path = LABEL_SWITCH_OUTPUT_DIR / "label_switch_population_prior_signal_corner.png"
    summary_path = LABEL_SWITCH_OUTPUT_DIR / "label_switch_population_prior_signal_summary.json"
    figure.savefig(figure_path, dpi=170, bbox_inches="tight")

    names = ["mu_low", "mu_high", "log_sigma"]
    summary = {
        "description": (
            "Fresh full-prior label-switching posterior check for the "
            "population-trained Flow2 residual NSF ensemble. Samples are "
            "shown in sorted NLL target coordinates."
        ),
        "signal": {
            "seed": int(args.signal_seed),
            "draw_index": int(args.draw_index),
            "z_raw": z_raw,
            "z_sorted": z_sorted,
            "x_summary_context": x_context,
        },
        "mcmc": {
            "chains": int(args.mcmc_chains),
            "steps": int(args.mcmc_steps),
            "burn_in": int(args.mcmc_burn_in),
            "posterior_samples": int(mcmc_sorted.shape[0]),
            "acceptance_rate": float(np.mean(mcmc_accept)),
            "seconds": float(mcmc_seconds),
            "to_grid": compare_samples_to_weighted_grid_named(mcmc_sorted, grid_layer, names),
        },
        "grid": grid_metadata,
        "npe": {
            "ensemble_summary": args.label_switch_ensemble_summary,
            "members": int(len(members)),
            "posterior_samples": int(npe_samples.shape[0]),
            "to_grid": compare_samples_to_weighted_grid_named(npe_samples, grid_layer, names),
            "to_mcmc": compare_sample_marginals_named(npe_samples, mcmc_sorted, names),
        },
        "outputs": {
            "figure": figure_path,
            "summary": summary_path,
        },
    }
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def render_linear6_population_case(args: argparse.Namespace) -> None:
    LINEAR6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    z_true, x_raw, x_context = sample_linear6_prior_predictive_signal(
        seed=args.signal_seed,
        draw_index=args.draw_index,
    )
    members = load_sign_ensemble(args.linear6_ensemble_summary, device)
    exact_samples = sample_linear6_exact_posterior(
        x_context=x_context,
        samples=args.exact_samples,
        seed=args.seed + 3000,
    )
    npe_samples = sample_stage1_ensemble(
        members=members,
        x_context=x_context,
        samples=args.npe_samples,
        seed=args.seed + 1000,
        device=device,
    )

    figure = render_corner_layers(
        labels=[
            r"$w_1$",
            r"$w_2$",
            r"$w_3$",
            r"$w_4$",
            r"$w_5$",
            r"$w_6$",
            r"$\log\sigma$",
        ],
        true_values=z_true,
        weighted_layers=[],
        sample_layers=[
            SampleCornerLayer("Exact posterior", "#172033", exact_samples, hist_lw=1.55, contour_lw=1.20),
            SampleCornerLayer("Population NPE", "#0f766e", npe_samples, hist_lw=1.45, contour_lw=1.15),
        ],
        true_color="#172033",
        title="Linear6 posterior: exact reference vs NPE",
        max_sample_plot=18_000,
        rng=np.random.default_rng(args.seed + 2000),
    )
    figure_path = LINEAR6_OUTPUT_DIR / "linear6_population_prior_signal_corner.png"
    summary_path = LINEAR6_OUTPUT_DIR / "linear6_population_prior_signal_summary.json"
    figure.savefig(figure_path, dpi=130, bbox_inches="tight")

    summary = {
        "description": (
            "Fresh full-prior Linear6 posterior check for the population-trained "
            "Flow2 residual NSF ensemble. The comparison is in the NLL target "
            "coordinates, including log_sigma."
        ),
        "signal": {
            "seed": int(args.signal_seed),
            "draw_index": int(args.draw_index),
            "z": z_true,
            "x": x_raw,
            "context": x_context,
        },
        "exact": {
            "posterior_samples": int(exact_samples.shape[0]),
        },
        "npe": {
            "ensemble_summary": args.linear6_ensemble_summary,
            "members": int(len(members)),
            "posterior_samples": int(npe_samples.shape[0]),
            "to_exact": compare_sample_marginals(npe_samples, exact_samples),
        },
        "outputs": {
            "figure": figure_path,
            "summary": summary_path,
        },
    }
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render README posterior comparison figures.")
    parser.add_argument(
        "--mode",
        choices=("single_decay", "sign_population", "banana_population", "label_switch_population", "linear6_population"),
        default="single_decay",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--sign-ensemble-summary", type=Path, default=SIGN_ENSEMBLE_SUMMARY)
    parser.add_argument("--banana-ensemble-summary", type=Path, default=BANANA_ENSEMBLE_SUMMARY)
    parser.add_argument("--label-switch-ensemble-summary", type=Path, default=LABEL_SWITCH_ENSEMBLE_SUMMARY)
    parser.add_argument("--linear6-ensemble-summary", type=Path, default=LINEAR6_ENSEMBLE_SUMMARY)
    parser.add_argument("--signal-seed", type=int, default=20260707)
    parser.add_argument("--draw-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--grid-limit", type=float, default=4.0)
    parser.add_argument("--npe-samples", type=int, default=80_000)
    parser.add_argument("--exact-samples", type=int, default=80_000)
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=12_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=3_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single_decay":
        render_decay_cases()
    elif args.mode == "sign_population":
        render_sign_population_case(args)
    elif args.mode == "banana_population":
        render_banana_population_case(args)
    elif args.mode == "label_switch_population":
        render_label_switch_population_case(args)
    else:
        render_linear6_population_case(args)


if __name__ == "__main__":
    main()
