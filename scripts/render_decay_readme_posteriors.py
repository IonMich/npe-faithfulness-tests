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
from npe_flow_stress_tests import (
    arviz_diagnostics as stress_arviz_diagnostics,
    make_banana_case,
    make_label_switch_case,
    make_linear6_case,
    make_sign_case,
    make_two_exp_case,
    run_random_walk_mcmc,
)
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
    evaluate_model_log_prob,
    label_log_likelihood_np,
    label_raw_prior_logpdf,
    linear6_log_py_given_log_sigma,
    linear6_sufficient_stats,
    sample_two_exp_population_raw,
    sort_label_target,
    TWO_EXP_PRIOR_MEAN,
    TWO_EXP_PRIOR_STD,
    two_exp_exact_posterior_nll,
    two_exp_log_likelihood_batched,
    two_exp_log_proposal_density,
    two_exp_raw_prior_logpdf_batched,
    two_exp_sample_proposal,
    two_exp_target_transform,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_decay_posteriors"
SUMMARY_PATH = OUTPUT_DIR / "decay_population_readme_posteriors_summary.json"
SIGN_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_sign_posteriors"
BANANA_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_banana_posteriors"
LABEL_SWITCH_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_label_switch_posteriors"
LINEAR6_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_linear6_posteriors"
TWO_EXP_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_two_exp_posteriors"
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
TWO_EXP_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/06_two_exponential/03_population_npe/21_flow2_e30_plus_high_snr_weighted_equal5_eval/"
    "results/two_exp_population_ensemble_summary.json"
)
TWO_EXP_TARGET_LABELS = [
    r"$\log(A_1+A_2)$",
    r"$\log(A_1/A_2)$",
    r"$\log k_1$",
    r"$\log\Delta k$",
    r"$\log\sigma$",
]
TWO_EXP_TARGET_NAMES = [
    "log_amplitude_sum",
    "log_amplitude_ratio",
    "log_k1",
    "log_delta_k",
    "log_sigma",
]
TWO_EXP_LOW_PRIOR_OFFSET = np.array([2.0, -2.0, 2.0, -2.0, 1.5], dtype=np.float64)
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


def repo_relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


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


@torch.no_grad()
def evaluate_stage1_ensemble_log_prob(
    *,
    members: list[dict[str, Any]],
    x_context: np.ndarray,
    z_target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not members:
        raise ValueError("Cannot evaluate an empty ensemble.")
    member_log_probs = []
    for member in members:
        chunks = []
        for start in range(0, x_context.shape[0], batch_size):
            stop = min(start + batch_size, x_context.shape[0])
            log_prob = evaluate_model_log_prob(
                model=member["model"],
                x_raw=x_context[start:stop],
                z_raw=z_target[start:stop],
                x_mean=np.asarray(member["x_mean"], dtype=np.float64),
                x_std=np.asarray(member["x_std"], dtype=np.float64),
                z_mean=np.asarray(member["z_mean"], dtype=np.float64),
                z_std=np.asarray(member["z_std"], dtype=np.float64),
                device=device,
            )
            chunks.append(log_prob.detach().cpu().numpy().astype(np.float64))
        member_log_probs.append(np.concatenate(chunks, axis=0))
    return logsumexp(np.stack(member_log_probs, axis=0), axis=0) - np.log(len(members))


def build_two_exp_readme_cases(
    *,
    args: argparse.Namespace,
    members: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, object]:
    print("Building two-exp posterior cases: one prior-predictive, one low-prior stress draw...")
    draw_index = int(args.draw_index)
    if draw_index < 0:
        raise ValueError("--draw-index must be nonnegative.")

    x_prior_all, x_context_prior_all, z_prior_all = sample_two_exp_population_raw(
        n=draw_index + 1,
        seed=int(args.signal_seed),
    )
    case = make_two_exp_case(ordered=True)
    low_prior_offset = np.asarray(args.two_exp_low_prior_offset, dtype=np.float64)
    low_prior_z = TWO_EXP_PRIOR_MEAN + TWO_EXP_PRIOR_STD * low_prior_offset
    low_prior_rng = np.random.default_rng(int(args.two_exp_low_prior_noise_seed))
    low_prior_x = case.simulate_x(low_prior_z[None, :], low_prior_rng)[0].astype(np.float32)
    low_prior_context = case.context(low_prior_x[None, :])[0].astype(np.float32)
    prior_mahalanobis = float(np.linalg.norm(low_prior_offset))

    raw_cases = {
        "easy": {
            "mode": "prior_predictive",
            "source": {
                "signal_seed": int(args.signal_seed),
                "draw_index": draw_index,
            },
            "z_raw": z_prior_all[draw_index],
            "x_raw": x_prior_all[draw_index],
            "x_context": x_context_prior_all[draw_index],
        },
        "difficult": {
            "mode": "low_prior_stress",
            "source": {
                "standardized_prior_offset": low_prior_offset,
                "prior_mahalanobis": prior_mahalanobis,
                "log_prior_density_delta_vs_mean": float(-0.5 * prior_mahalanobis * prior_mahalanobis),
                "noise_seed": int(args.two_exp_low_prior_noise_seed),
            },
            "z_raw": low_prior_z.astype(np.float32),
            "x_raw": low_prior_x,
            "x_context": low_prior_context,
        },
    }

    case_names = list(raw_cases)
    x_raw = np.stack([np.asarray(raw_cases[name]["x_raw"], dtype=np.float32) for name in case_names], axis=0)
    x_context = np.stack([np.asarray(raw_cases[name]["x_context"], dtype=np.float32) for name in case_names], axis=0)
    z_raw = np.stack([np.asarray(raw_cases[name]["z_raw"], dtype=np.float32) for name in case_names], axis=0)
    z_target = two_exp_target_transform(z_raw, target=str(args.two_exp_target))
    ensemble_log_prob = evaluate_stage1_ensemble_log_prob(
        members=members,
        x_context=x_context,
        z_target=z_target,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    ensemble_nll = -ensemble_log_prob
    exact_nll, diagnostics = two_exp_exact_posterior_nll(
        x_raw=x_raw,
        x_context=x_context,
        z_raw=z_raw,
        floor_method="importance",
        importance_samples=int(args.two_exp_selection_importance_samples),
        importance_seed=int(args.two_exp_importance_seed),
        importance_batch_size=int(args.two_exp_importance_batch_size),
        prior_mixture=float(args.two_exp_prior_mixture),
        proposal_inflation=float(args.two_exp_proposal_inflation),
        smc_particles=int(args.two_exp_smc_particles),
        smc_beta_steps=int(args.two_exp_smc_beta_steps),
        smc_mh_steps=int(args.two_exp_smc_mh_steps),
        smc_seed=int(args.two_exp_smc_seed),
        smc_batch_size=int(args.two_exp_smc_batch_size),
        smc_step_scale=float(args.two_exp_smc_step_scale),
    )
    gap = ensemble_nll - exact_nll

    cases = {}
    for index, name in enumerate(case_names):
        raw_case = raw_cases[name]
        cases[name] = {
            "mode": raw_case["mode"],
            "source": raw_case["source"],
            "npe_nll": float(ensemble_nll[index]),
            "reference_nll": float(exact_nll[index]),
            "paired_gap": float(gap[index]),
            "z_raw": z_raw[index],
            "z_target": z_target[index],
            "x_raw": x_raw[index],
            "x_context": x_context[index],
        }
        print(
            f"Built {name} case mode={raw_case['mode']} "
            f"NPE={ensemble_nll[index]:.5f} reference={exact_nll[index]:.5f} gap={gap[index]:.5f}"
        )

    return {
        "case_definition": (
            "The easy case is an ordinary prior-predictive draw. The difficult case "
            "matches the single-decay convention: a deterministic low-prior-density "
            "draw in raw prior coordinates, not the largest held-out NLL miss."
        ),
        "case_importance_samples": int(args.two_exp_selection_importance_samples),
        "importance_diagnostics": diagnostics,
        "gap_summary": {
            "mean": float(np.mean(gap)),
            "median": float(np.median(gap)),
            "min": float(np.min(gap)),
            "max": float(np.max(gap)),
            "mean_abs": float(np.mean(np.abs(gap))),
        },
        "cases": cases,
    }


def sample_two_exp_importance_reference(
    *,
    x_raw: np.ndarray,
    x_context: np.ndarray,
    z_raw: np.ndarray,
    target: str,
    samples: int,
    resamples: int,
    seed: int,
    prior_mixture: float,
    proposal_inflation: float,
) -> tuple[np.ndarray, dict[str, object]]:
    proposal, component_weights = two_exp_sample_proposal(
        np.asarray(x_context, dtype=np.float64)[None, :],
        np.asarray(z_raw, dtype=np.float64)[None, :],
        samples=int(samples),
        seed=int(seed),
        prior_mixture=float(prior_mixture),
        inflation=float(proposal_inflation),
    )
    log_integrand = two_exp_raw_prior_logpdf_batched(proposal) + two_exp_log_likelihood_batched(
        np.asarray(x_raw, dtype=np.float64)[None, :],
        proposal,
    )
    log_q = two_exp_log_proposal_density(
        proposal,
        np.asarray(x_context, dtype=np.float64)[None, :],
        np.asarray(z_raw, dtype=np.float64)[None, :],
        weights=component_weights,
        prior_mixture=float(prior_mixture),
        inflation=float(proposal_inflation),
    )
    log_weights = log_integrand[0] - log_q[0]
    log_norm = float(logsumexp(log_weights))
    probabilities = np.exp(log_weights - log_norm)
    rng = np.random.default_rng(seed + 17)
    indices = rng.choice(int(samples), size=int(resamples), replace=True, p=probabilities)
    target_samples = two_exp_target_transform(proposal[0, indices], target=target).astype(np.float64)
    ess = float(1.0 / np.sum(probabilities * probabilities))
    return target_samples, {
        "proposal_samples": int(samples),
        "resampled_posterior_samples": int(resamples),
        "importance_seed": int(seed),
        "log_evidence": float(log_norm - np.log(samples)),
        "ess": ess,
        "relative_ess": float(ess / samples),
        "log_weight_std": float(np.std(log_weights)),
        "prior_mixture": float(prior_mixture),
        "proposal_inflation": float(proposal_inflation),
    }


def sample_two_exp_mcmc_reference(
    *,
    x_raw: np.ndarray,
    z_raw: np.ndarray,
    target: str,
    chains: int,
    steps: int,
    burn_in: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    base_case = make_two_exp_case(ordered=True)
    z_center = np.asarray(z_raw, dtype=np.float64)

    def initial_z(chains_count: int, _x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = np.asarray(base_case.mcmc_proposal_scale, dtype=np.float64) * 2.5
        return rng.normal(z_center[None, :], scale[None, :], size=(chains_count, base_case.z_dim))

    case = replace(base_case, true_z=z_center, initial_z=initial_z)
    samples_raw, accept, seconds = run_random_walk_mcmc(
        case,
        np.asarray(x_raw, dtype=np.float64),
        chains=int(chains),
        steps=int(steps),
        seed=int(seed),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    if burn_in >= steps:
        raise ValueError("--mcmc-burn-in must be smaller than --mcmc-steps.")
    post_raw = samples_raw[:, int(burn_in) :, :].reshape(-1, base_case.z_dim)
    post_target = two_exp_target_transform(post_raw, target=target).astype(np.float64)
    diagnostics = stress_arviz_diagnostics(
        samples_raw,
        int(burn_in),
        lambda z: two_exp_target_transform(z, target=target),
        tuple(TWO_EXP_TARGET_NAMES),
    )
    return post_target, {
        "kind": "mcmc",
        "chains": int(chains),
        "steps": int(steps),
        "burn_in": int(burn_in),
        "posterior_samples": int(post_target.shape[0]),
        "acceptance_rate": float(np.mean(accept)),
        "seconds": float(seconds),
        "diagnostics": diagnostics,
        **summarize_mcmc_diagnostics(diagnostics),
    }


def render_two_exp_population_cases(args: argparse.Namespace) -> None:
    TWO_EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    members = load_sign_ensemble(args.two_exp_ensemble_summary, device)
    selection = build_two_exp_readme_cases(args=args, members=members, device=device)

    outputs = {}
    case_summaries = {}
    for offset, (case_name, case_info) in enumerate(selection["cases"].items()):
        if args.two_exp_posterior_reference == "importance":
            reference_samples, reference_metadata = sample_two_exp_importance_reference(
                x_raw=np.asarray(case_info["x_raw"], dtype=np.float64),
                x_context=np.asarray(case_info["x_context"], dtype=np.float64),
                z_raw=np.asarray(case_info["z_raw"], dtype=np.float64),
                target=str(args.two_exp_target),
                samples=int(args.two_exp_reference_samples),
                resamples=int(args.exact_samples),
                seed=int(args.seed) + 5000 + 101 * offset,
                prior_mixture=float(args.two_exp_prior_mixture),
                proposal_inflation=float(args.two_exp_proposal_inflation),
            )
            reference_label = "Importance reference"
        else:
            reference_samples, reference_metadata = sample_two_exp_mcmc_reference(
                x_raw=np.asarray(case_info["x_raw"], dtype=np.float64),
                z_raw=np.asarray(case_info["z_raw"], dtype=np.float64),
                target=str(args.two_exp_target),
                chains=int(args.mcmc_chains),
                steps=int(args.mcmc_steps),
                burn_in=int(args.mcmc_burn_in),
                seed=int(args.seed) + 5000 + 101 * offset,
            )
            reference_label = "MCMC reference"
        npe_samples = sample_stage1_ensemble(
            members=members,
            x_context=np.asarray(case_info["x_context"], dtype=np.float64),
            samples=int(args.npe_samples),
            seed=int(args.seed) + 1000 + 101 * offset,
            device=device,
        )
        true_target = np.asarray(case_info["z_target"], dtype=np.float64)
        figure = render_corner_layers(
            labels=TWO_EXP_TARGET_LABELS,
            true_values=true_target,
            weighted_layers=[],
            sample_layers=[
                SampleCornerLayer(
                    reference_label,
                    "#172033",
                    reference_samples,
                    hist_lw=1.55,
                    contour_lw=1.20,
                ),
                SampleCornerLayer("Best-NLL NPE", "#0f766e", npe_samples, hist_lw=1.45, contour_lw=1.15),
            ],
            true_color="#172033",
            title=f"Two-exponential {case_name} full-prior posterior\n{reference_label} vs best-NLL NPE",
            max_sample_plot=18_000,
            rng=np.random.default_rng(int(args.seed) + 2000 + offset),
        )
        figure_path = TWO_EXP_OUTPUT_DIR / f"two_exp_best_nll_{case_name}_posterior_corner.png"
        figure.savefig(figure_path, dpi=130, bbox_inches="tight")
        outputs[case_name] = repo_relative(figure_path)
        case_summaries[case_name] = {
            **case_info,
            "posterior_reference": reference_metadata,
            "npe": {
                "posterior_samples": int(npe_samples.shape[0]),
                "to_posterior_reference": compare_sample_marginals_named(
                    npe_samples,
                    reference_samples,
                    TWO_EXP_TARGET_NAMES,
                ),
            },
            "outputs": {
                "figure": repo_relative(figure_path),
            },
        }

    summary_path = TWO_EXP_OUTPUT_DIR / "two_exp_best_nll_posterior_summary.json"
    summary = {
        "description": (
            "Two representative full-prior posterior checks for the current "
            "best-NLL two-exponential population NPE, the equal-5 mixture of "
            "the Flow2 ridge ensemble and high-SNR weighted member."
        ),
        "ensemble_summary": repo_relative(args.two_exp_ensemble_summary),
        "target": str(args.two_exp_target),
        "labels": TWO_EXP_TARGET_LABELS,
        "selection": {
            key: value
            for key, value in selection.items()
            if key != "cases"
        },
        "cases": case_summaries,
        "outputs": outputs | {"summary": repo_relative(summary_path)},
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
        choices=(
            "single_decay",
            "sign_population",
            "banana_population",
            "label_switch_population",
            "linear6_population",
            "two_exp_population",
        ),
        default="single_decay",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--sign-ensemble-summary", type=Path, default=SIGN_ENSEMBLE_SUMMARY)
    parser.add_argument("--banana-ensemble-summary", type=Path, default=BANANA_ENSEMBLE_SUMMARY)
    parser.add_argument("--label-switch-ensemble-summary", type=Path, default=LABEL_SWITCH_ENSEMBLE_SUMMARY)
    parser.add_argument("--linear6-ensemble-summary", type=Path, default=LINEAR6_ENSEMBLE_SUMMARY)
    parser.add_argument("--two-exp-ensemble-summary", type=Path, default=TWO_EXP_ENSEMBLE_SUMMARY)
    parser.add_argument("--signal-seed", type=int, default=20260707)
    parser.add_argument("--draw-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--grid-limit", type=float, default=4.0)
    parser.add_argument("--npe-samples", type=int, default=80_000)
    parser.add_argument("--exact-samples", type=int, default=80_000)
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=12_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=3_000)
    parser.add_argument("--two-exp-target", choices=("amplitude_sum_delta", "amplitude_sum_rate"), default="amplitude_sum_delta")
    parser.add_argument(
        "--two-exp-case-importance-samples",
        "--two-exp-selection-importance-samples",
        dest="two_exp_selection_importance_samples",
        type=int,
        default=4096,
    )
    parser.add_argument("--two-exp-reference-samples", type=int, default=100_000)
    parser.add_argument("--two-exp-posterior-reference", choices=("mcmc", "importance"), default="mcmc")
    parser.add_argument("--two-exp-low-prior-offset", type=float, nargs=5, default=TWO_EXP_LOW_PRIOR_OFFSET.tolist())
    parser.add_argument("--two-exp-low-prior-noise-seed", type=int, default=2026070203)
    parser.add_argument("--two-exp-importance-seed", type=int, default=20260723)
    parser.add_argument("--two-exp-importance-batch-size", type=int, default=16)
    parser.add_argument("--two-exp-prior-mixture", type=float, default=0.02)
    parser.add_argument("--two-exp-proposal-inflation", type=float, default=1.0)
    parser.add_argument("--two-exp-smc-particles", type=int, default=4096)
    parser.add_argument("--two-exp-smc-beta-steps", type=int, default=96)
    parser.add_argument("--two-exp-smc-mh-steps", type=int, default=2)
    parser.add_argument("--two-exp-smc-seed", type=int, default=20260723)
    parser.add_argument("--two-exp-smc-batch-size", type=int, default=8)
    parser.add_argument("--two-exp-smc-step-scale", type=float, default=0.85)
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
    elif args.mode == "linear6_population":
        render_linear6_population_case(args)
    else:
        render_two_exp_population_cases(args)


if __name__ == "__main__":
    main()
