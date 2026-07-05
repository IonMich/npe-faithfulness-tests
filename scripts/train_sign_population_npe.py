from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp, roots_hermitenorm
from torch.utils.data import DataLoader, TensorDataset

import npe_stage1_decay as stage1
from npe_flow_stress_tests import (
    StressCase,
    make_banana_case,
    make_label_switch_case,
    make_linear6_case,
    make_two_exp_case,
)


DEFAULT_OUTPUT_ROOT = Path("runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior")
DEFAULT_BANANA_OUTPUT_ROOT = Path("runs/03_stress_banana/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4")
DEFAULT_LABEL_SWITCH_OUTPUT_ROOT = (
    Path("runs/04_stress_label_switch/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4")
)
DEFAULT_LINEAR6_OUTPUT_ROOT = Path("runs/05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4")
DEFAULT_TWO_EXP_OUTPUT_ROOT = Path("runs/06_two_exponential/03_population_npe/00_entropy_floor_full_prior_probe")
FOLDED_SIGN_FLOOR = -1.426941782495585
FOLDED_SIGN_FLOOR_SE = 0.0011526154301947824
LOG_2PI = math.log(2.0 * math.pi)
BANANA_SIGMA = np.array([0.20, 0.18], dtype=np.float64)
BANANA_B = 0.65
BANANA_C = 0.70
BANANA_PRIOR_STD = 1.8
LABEL_N_OBS = 80
LABEL_PRIOR_MEAN = np.array([0.0, 0.0, math.log(0.45)], dtype=np.float64)
LABEL_PRIOR_STD = np.array([2.2, 2.2, 0.55], dtype=np.float64)
TWO_EXP_N_OBS = 45
TWO_EXP_T = np.linspace(0.0, 6.0, TWO_EXP_N_OBS)
TWO_EXP_PRIOR_MEAN = np.array(
    [math.log(2.5), math.log(0.35), math.log(1.4), math.log(0.75), math.log(0.25)],
    dtype=np.float64,
)
TWO_EXP_PRIOR_STD = np.array([0.60, 0.55, 0.65, 0.60, 0.45], dtype=np.float64)
TWO_EXP_CONTEXT_CHUNK_SIZE = 32_768
TWO_EXP_TARGETS = ("amplitude_sum_delta", "amplitude_sum_rate")
TWO_EXP_TARGET_MODES = ("direct", "profile_residual")
POPULATION_LOSS_WEIGHT_MODES = ("none", "two_exp_high_snr", "two_exp_low_noise", "two_exp_high_snr_low_noise")


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def summarize(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "std_error": float(np.std(finite, ddof=1) / math.sqrt(finite.size))
        if finite.size > 1
        else 0.0,
        "min": float(np.min(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def summarize_diagnostics(items: list[dict[str, Any]]) -> dict[str, object] | None:
    if not items:
        return None
    output: dict[str, object] = {}
    keys = sorted(set().union(*(item.keys() for item in items)))
    for key in keys:
        values = [item[key] for item in items if key in item]
        if values and all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            output[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
        elif values and all(value == values[0] for value in values):
            output[key] = values[0]
    return output


def runtime_metadata() -> dict[str, object]:
    return {
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def default_output_root(model: str) -> Path:
    if model == "sign":
        return DEFAULT_OUTPUT_ROOT
    if model == "banana":
        return DEFAULT_BANANA_OUTPUT_ROOT
    if model == "label_switch":
        return DEFAULT_LABEL_SWITCH_OUTPUT_ROOT
    if model == "linear6":
        return DEFAULT_LINEAR6_OUTPUT_ROOT
    if model == "two_exp":
        return DEFAULT_TWO_EXP_OUTPUT_ROOT
    raise ValueError(f"Unsupported population model: {model}")


def two_exp_target_description(target: str) -> str:
    if target == "amplitude_sum_delta":
        return "(log(A1 + A2), log(A1/A2), log k1, log Delta k, log sigma)"
    if target == "amplitude_sum_rate":
        return "(log(A1 + A2), log(A1/A2), log k2, log(k1/Delta k), log sigma)"
    raise ValueError(f"Unsupported two_exp target: {target}")


def population_target_description(model: str, *, two_exp_target: str = "amplitude_sum_delta") -> str:
    if model == "sign":
        return "(abs(theta1), theta2)"
    if model == "banana":
        return "(theta1, theta2)"
    if model == "label_switch":
        return "(mu_low, mu_high, log_sigma)"
    if model == "linear6":
        return "(w1, ..., w6, log_sigma)"
    if model == "two_exp":
        return two_exp_target_description(two_exp_target)
    raise ValueError(f"Unsupported population model: {model}")


def population_kind(model: str) -> str:
    if model == "sign":
        return "sign_population_flow2_residual_nsf_ensemble"
    if model == "banana":
        return "banana_population_flow2_residual_nsf_ensemble"
    if model == "label_switch":
        return "label_switch_population_flow2_residual_nsf_ensemble"
    if model == "linear6":
        return "linear6_population_flow2_residual_nsf_ensemble"
    if model == "two_exp":
        return "two_exp_population_flow2_residual_nsf_ensemble"
    raise ValueError(f"Unsupported population model: {model}")


def population_description(model: str, *, two_exp_target: str = "amplitude_sum_delta") -> str:
    if model == "sign":
        return (
            "Full-prior sign-symmetry population NPE using the single-decay "
            "Flow2 residual NSF/randperm training recipe, with folded target "
            "(abs(theta1), theta2)."
        )
    if model == "banana":
        return (
            "Full-prior Banana population NPE using the single-decay Flow2 "
            "residual NSF/randperm training recipe, with raw target "
            "(theta1, theta2)."
        )
    if model == "label_switch":
        return (
            "Full-prior label-switching population NPE using the single-decay "
            "Flow2 residual NSF/randperm training recipe, with sorted target "
            "(mu_low, mu_high, log_sigma)."
        )
    if model == "linear6":
        return (
            "Full-prior Linear6 population NPE using the single-decay Flow2 "
            "residual NSF/randperm training recipe, with target "
            "(w1, ..., w6, log_sigma)."
        )
    if model == "two_exp":
        return (
            "Full-prior ordered two-exponential population NPE using the "
            "single-decay Flow2 residual NSF/randperm training recipe, with "
            f"invertible ridge target {two_exp_target_description(two_exp_target)}."
        )
    raise ValueError(f"Unsupported population model: {model}")


def sample_sign_population(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.8, size=(n, 2)).astype(np.float64)
    x = np.column_stack(
        [
            theta[:, 0] * theta[:, 0] + rng.normal(0.0, 0.22, size=n),
            theta[:, 1] + rng.normal(0.0, 0.16, size=n),
        ]
    )
    folded = np.column_stack([np.abs(theta[:, 0]), theta[:, 1]])
    return x.astype(np.float32), folded.astype(np.float32)


def banana_context(x: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x, dtype=np.float64)
    x1 = x_raw[:, 0]
    x2 = x_raw[:, 1]
    curvature = x1 * x1 - BANANA_C
    dewarped_theta2 = x2 - BANANA_B * curvature
    return np.column_stack([x1, x2, dewarped_theta2, curvature])


def sample_banana_population(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    case = make_banana_case()
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(n, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return banana_context(x).astype(np.float32), z.astype(np.float32)


def sort_label_target(z: np.ndarray) -> np.ndarray:
    low = np.minimum(z[:, 0], z[:, 1])
    high = np.maximum(z[:, 0], z[:, 1])
    return np.column_stack([low, high, z[:, 2]])


def sample_label_switch_population(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, context, z_sorted = sample_label_switch_population_raw(n=n, seed=seed)
    return context, z_sorted


def sample_label_switch_population_raw(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_label_switch_case()
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(n, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return x.astype(np.float32), case.context(x).astype(np.float32), sort_label_target(z).astype(np.float32)


def sample_two_exp_population_raw(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_two_exp_case(ordered=True)
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(n, case.z_dim),
    )
    x_chunks = []
    context_chunks = []
    for start in range(0, n, TWO_EXP_CONTEXT_CHUNK_SIZE):
        stop = min(start + TWO_EXP_CONTEXT_CHUNK_SIZE, n)
        x_chunk = case.simulate_x(z[start:stop], rng)
        x_chunks.append(x_chunk.astype(np.float32))
        context_chunks.append(case.context(x_chunk).astype(np.float32))
    return np.concatenate(x_chunks, axis=0), np.concatenate(context_chunks, axis=0), z.astype(np.float32)


def two_exp_target_transform(z_raw: np.ndarray, *, target: str = "amplitude_sum_delta") -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    log_sum_amplitude = np.logaddexp(z[:, 0], z[:, 2])
    log_amplitude_ratio = z[:, 0] - z[:, 2]
    if target == "amplitude_sum_delta":
        rate_columns = (z[:, 1], z[:, 3])
    elif target == "amplitude_sum_rate":
        rate_columns = (np.logaddexp(z[:, 1], z[:, 3]), z[:, 1] - z[:, 3])
    else:
        raise ValueError(f"Unsupported two_exp target: {target}")
    transformed = np.column_stack(
        [
            log_sum_amplitude,
            log_amplitude_ratio,
            rate_columns[0],
            rate_columns[1],
            z[:, 4],
        ]
    )
    return transformed.astype(np.float32)


def two_exp_target_inverse(z_target: np.ndarray, *, target: str = "amplitude_sum_delta") -> np.ndarray:
    z = np.asarray(z_target, dtype=np.float64)
    amplitude_log_normalizer = np.logaddexp(0.0, z[:, 1])
    if target == "amplitude_sum_delta":
        log_k1 = z[:, 2]
        log_delta = z[:, 3]
    elif target == "amplitude_sum_rate":
        rate_log_normalizer = np.logaddexp(0.0, z[:, 3])
        log_k1 = z[:, 2] + z[:, 3] - rate_log_normalizer
        log_delta = z[:, 2] - rate_log_normalizer
    else:
        raise ValueError(f"Unsupported two_exp target: {target}")
    raw = np.column_stack(
        [
            z[:, 0] + z[:, 1] - amplitude_log_normalizer,
            log_k1,
            z[:, 0] - amplitude_log_normalizer,
            log_delta,
            z[:, 4],
        ]
    )
    return raw.astype(np.float32)


def sample_two_exp_population(
    *,
    n: int,
    seed: int,
    target: str = "amplitude_sum_delta",
) -> tuple[np.ndarray, np.ndarray]:
    _x_raw, context, z = sample_two_exp_population_raw(n=n, seed=seed)
    return context, two_exp_target_transform(z, target=target)


def sample_stress_population(
    case: StressCase,
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(n, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return case.context(x).astype(np.float32), z.astype(np.float32)


def sample_population(
    *,
    model: str,
    n: int,
    seed: int,
    two_exp_target: str = "amplitude_sum_delta",
) -> tuple[np.ndarray, np.ndarray]:
    if model == "sign":
        return sample_sign_population(n=n, seed=seed)
    if model == "banana":
        return sample_banana_population(n=n, seed=seed)
    if model == "label_switch":
        return sample_label_switch_population(n=n, seed=seed)
    if model == "linear6":
        return sample_stress_population(make_linear6_case(), n=n, seed=seed)
    if model == "two_exp":
        return sample_two_exp_population(n=n, seed=seed, target=two_exp_target)
    raise ValueError(f"Unsupported population model: {model}")


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean[None, :]) / std[None, :]).astype(np.float32)


def two_exp_loss_weights(
    z_raw: np.ndarray,
    *,
    mode: str,
    tail_weight: float,
    tail_quantile: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in (0, 1).")
    if tail_weight <= 0.0:
        raise ValueError("tail_weight must be positive.")
    z = np.asarray(z_raw, dtype=np.float64)
    log_total_amplitude = np.logaddexp(z[:, 0], z[:, 2])
    log_snr = log_total_amplitude - z[:, 4]
    low_noise_score = -z[:, 4]
    if mode == "two_exp_high_snr":
        score = log_snr
    elif mode == "two_exp_low_noise":
        score = low_noise_score
    elif mode == "two_exp_high_snr_low_noise":
        score = (
            (log_snr - np.mean(log_snr)) / max(float(np.std(log_snr)), 1e-8)
            + (low_noise_score - np.mean(low_noise_score)) / max(float(np.std(low_noise_score)), 1e-8)
        )
    else:
        raise ValueError(f"Unsupported loss weight mode: {mode}")
    threshold = float(np.quantile(score, tail_quantile))
    weights = np.ones(z.shape[0], dtype=np.float32)
    weights[score >= threshold] = float(tail_weight)
    weights /= float(np.mean(weights))
    return weights.astype(np.float32)


def loss_weight_summary(
    weights: np.ndarray | None,
    *,
    mode: str,
    tail_weight: float,
    tail_quantile: float,
) -> dict[str, object]:
    output: dict[str, object] = {
        "mode": mode,
        "tail_weight": float(tail_weight),
        "tail_quantile": float(tail_quantile),
        "applied": weights is not None,
    }
    if weights is None:
        return output
    weight_array = np.asarray(weights, dtype=np.float64)
    output.update(
        {
            "summary": summarize(weight_array),
            "effective_sample_size": float(weight_array.sum() ** 2 / np.sum(weight_array * weight_array)),
            "upweighted_fraction": float(np.mean(weight_array > 1.0)),
        }
    )
    return output


def make_config(args: argparse.Namespace, *, seed: int, train_simulations: int) -> stage1.Stage1Config:
    return stage1.Stage1Config(
        train_simulations=int(train_simulations),
        val_simulations=int(args.val_simulations),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        mdn_components=int(args.mdn_components),
        flow_layers=int(args.flow_layers),
        flow_context_dim=64,
        seed=int(seed),
        observed_seed=int(seed),
        requested_device=str(args.device),
        families=[str(args.family)],
        posterior_samples=0,
        reference_grid_size=0,
        train_sampler="random",
        context_features="raw",
        spline_bins=int(args.spline_bins),
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
        target_transform=str(args.target_transform),
        target_ridge=float(args.target_ridge),
        flow_activation=str(args.flow_activation),
        flow_residual=bool(args.flow_residual),
        flow_randperm=bool(args.flow_randperm),
        flow_passes=int(args.flow_passes),
        flow_kind=str(args.flow_kind),
    )


def linear6_sufficient_stats(x_context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_linear6_case()
    d_w = case.z_dim - 1
    coef = np.asarray(x_context[:, :d_w], dtype=np.float64)
    sigma_hat = np.exp(np.asarray(x_context[:, d_w], dtype=np.float64))
    n_obs = 32
    residual_sq = n_obs * sigma_hat * sigma_hat
    projected_sq = n_obs * np.sum(coef * coef, axis=1)
    return coef, projected_sq, residual_sq


def linear6_log_py_given_log_sigma(
    *,
    projected_sq: np.ndarray,
    residual_sq: np.ndarray,
    log_sigma: np.ndarray,
) -> np.ndarray:
    n_obs = 32
    d_w = 6
    prior_std_w = 1.25
    sigma2 = np.exp(2.0 * log_sigma)
    projected_var = sigma2 + n_obs * prior_std_w * prior_std_w
    return -0.5 * (
        n_obs * LOG_2PI
        + d_w * np.log(projected_var)
        + (n_obs - d_w) * np.log(sigma2)
        + projected_sq / projected_var
        + residual_sq / sigma2
    )


def normal_logpdf_1d(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized * standardized - math.log(std) - 0.5 * LOG_2PI


def normal_logpdf(value: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized * standardized - np.log(std) - 0.5 * LOG_2PI


def banana_raw_x(x_context: np.ndarray) -> np.ndarray:
    return np.asarray(x_context[:, :2], dtype=np.float64)


def banana_log_evidence(
    x_context: np.ndarray,
    *,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    x_raw = banana_raw_x(x_context)
    x1 = x_raw[:, 0]
    x2 = x_raw[:, 1]
    prior_var = BANANA_PRIOR_STD * BANANA_PRIOR_STD
    sigma1_var = BANANA_SIGMA[0] * BANANA_SIGMA[0]
    x1_var = prior_var + sigma1_var
    theta1_var_given_x1 = 1.0 / (1.0 / prior_var + 1.0 / sigma1_var)
    theta1_std_given_x1 = math.sqrt(theta1_var_given_x1)
    theta1_mean_given_x1 = theta1_var_given_x1 * x1 / sigma1_var
    x2_std_given_theta1 = math.sqrt(prior_var + BANANA_SIGMA[1] * BANANA_SIGMA[1])
    log_px1 = normal_logpdf(x1, 0.0, math.sqrt(x1_var))
    nodes, weights = roots_hermitenorm(int(quadrature_order))
    log_weights = np.log(weights) - 0.5 * LOG_2PI
    result = np.empty(x_context.shape[0], dtype=np.float64)
    for start in range(0, x_context.shape[0], chunk_size):
        stop = min(start + chunk_size, x_context.shape[0])
        theta1_nodes = (
            theta1_mean_given_x1[start:stop, None]
            + theta1_std_given_x1 * np.asarray(nodes, dtype=np.float64)[None, :]
        )
        x2_mean = BANANA_B * (theta1_nodes * theta1_nodes - BANANA_C)
        log_terms = log_weights[None, :] + normal_logpdf(
            x2[start:stop, None],
            x2_mean,
            x2_std_given_theta1,
        )
        result[start:stop] = log_px1[start:stop] + logsumexp(log_terms, axis=1)
    return result


def banana_log_likelihood(x_context: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
    x_raw = banana_raw_x(x_context)
    z = np.asarray(z_raw, dtype=np.float64)
    mean1 = z[:, 0]
    mean2 = z[:, 1] + BANANA_B * (z[:, 0] * z[:, 0] - BANANA_C)
    return normal_logpdf(x_raw[:, 0], mean1, BANANA_SIGMA[0]) + normal_logpdf(
        x_raw[:, 1],
        mean2,
        BANANA_SIGMA[1],
    )


def banana_exact_posterior_nll(
    *,
    x_context: np.ndarray,
    z_raw: np.ndarray,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    log_prior = normal_logpdf(z[:, 0], 0.0, BANANA_PRIOR_STD) + normal_logpdf(
        z[:, 1],
        0.0,
        BANANA_PRIOR_STD,
    )
    log_likelihood = banana_log_likelihood(x_context, z)
    log_evidence = banana_log_evidence(
        x_context,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
    )
    return -(log_prior + log_likelihood - log_evidence)


def label_raw_prior_logpdf(z_raw: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    return (
        -0.5 * ((z - LABEL_PRIOR_MEAN[None, :]) / LABEL_PRIOR_STD[None, :]) ** 2
        - np.log(LABEL_PRIOR_STD[None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=1)


def label_raw_prior_logpdf_batched(z_raw: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    return (
        -0.5 * ((z - LABEL_PRIOR_MEAN[None, None, :]) / LABEL_PRIOR_STD[None, None, :]) ** 2
        - np.log(LABEL_PRIOR_STD[None, None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=2)


def label_log_likelihood_np(x_raw: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    z = np.asarray(z_raw, dtype=np.float64)
    mu1 = z[:, 0]
    mu2 = z[:, 1]
    log_sigma = z[:, 2]
    sigma = np.exp(log_sigma)
    log_a = -0.5 * ((x - mu1[:, None]) / sigma[:, None]) ** 2 - log_sigma[:, None] - 0.5 * LOG_2PI
    log_b = -0.5 * ((x - mu2[:, None]) / sigma[:, None]) ** 2 - log_sigma[:, None] - 0.5 * LOG_2PI
    return (np.logaddexp(log_a, log_b) - math.log(2.0)).sum(axis=1)


def label_log_likelihood_batched(x_raw: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    z = np.asarray(z_raw, dtype=np.float64)
    mu1 = z[:, :, 0]
    mu2 = z[:, :, 1]
    log_sigma = z[:, :, 2]
    sigma = np.exp(log_sigma)
    y = x[:, None, :]
    log_a = -0.5 * ((y - mu1[:, :, None]) / sigma[:, :, None]) ** 2 - log_sigma[:, :, None] - 0.5 * LOG_2PI
    log_b = -0.5 * ((y - mu2[:, :, None]) / sigma[:, :, None]) ** 2 - log_sigma[:, :, None] - 0.5 * LOG_2PI
    return (np.logaddexp(log_a, log_b) - math.log(2.0)).sum(axis=2)


def label_gaussian_logpdf_batched(z_raw: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    return (
        -0.5 * ((z - c[:, None, :]) / s[:, None, :]) ** 2
        - np.log(s[:, None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=2)


def label_proposal_parameters(
    x_context: np.ndarray,
    *,
    inflation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    context = np.asarray(x_context, dtype=np.float64)
    em_low = context[:, -3]
    em_high = context[:, -2]
    em_log_sigma = context[:, -1]
    center_a = np.column_stack([em_low, em_high, em_log_sigma])
    center_b = np.column_stack([em_high, em_low, em_log_sigma])
    sigma = np.exp(em_log_sigma)
    mu_scale = np.maximum(sigma / math.sqrt(LABEL_N_OBS / 2.0), 0.055) * inflation
    log_scale = np.full_like(mu_scale, 0.060 * inflation)
    narrow = np.column_stack([mu_scale, mu_scale, log_scale])
    wide = narrow * 3.0
    return center_a, center_b, narrow, wide, sigma


def label_sample_proposal(
    x_context: np.ndarray,
    *,
    samples: int,
    seed: int,
    prior_mixture: float,
    inflation: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = x_context.shape[0]
    center_a, center_b, narrow, wide, _ = label_proposal_parameters(x_context, inflation=inflation)
    local_weight = max(1.0 - prior_mixture, 1e-12)
    weights = np.array(
        [
            prior_mixture,
            0.35 * local_weight,
            0.35 * local_weight,
            0.15 * local_weight,
            0.15 * local_weight,
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    component = rng.choice(weights.size, size=(batch, samples), p=weights)
    z = np.empty((batch, samples, 3), dtype=np.float64)
    for index in range(batch):
        for component_index in range(weights.size):
            mask = component[index] == component_index
            count = int(mask.sum())
            if count == 0:
                continue
            if component_index == 0:
                z[index, mask] = rng.normal(LABEL_PRIOR_MEAN, LABEL_PRIOR_STD, size=(count, 3))
            elif component_index == 1:
                z[index, mask] = rng.normal(center_a[index], narrow[index], size=(count, 3))
            elif component_index == 2:
                z[index, mask] = rng.normal(center_b[index], narrow[index], size=(count, 3))
            elif component_index == 3:
                z[index, mask] = rng.normal(center_a[index], wide[index], size=(count, 3))
            else:
                z[index, mask] = rng.normal(center_b[index], wide[index], size=(count, 3))
    return z, weights


def label_log_proposal_density(
    z_raw: np.ndarray,
    x_context: np.ndarray,
    *,
    weights: np.ndarray,
    prior_mixture: float,
    inflation: float,
) -> np.ndarray:
    center_a, center_b, narrow, wide, _ = label_proposal_parameters(x_context, inflation=inflation)
    terms = [
        math.log(max(prior_mixture, 1e-300)) + label_raw_prior_logpdf_batched(z_raw),
        math.log(float(weights[1])) + label_gaussian_logpdf_batched(z_raw, center_a, narrow),
        math.log(float(weights[2])) + label_gaussian_logpdf_batched(z_raw, center_b, narrow),
        math.log(float(weights[3])) + label_gaussian_logpdf_batched(z_raw, center_a, wide),
        math.log(float(weights[4])) + label_gaussian_logpdf_batched(z_raw, center_b, wide),
    ]
    return logsumexp(np.stack(terms, axis=0), axis=0)


def label_log_evidence_importance(
    x_raw: np.ndarray,
    x_context: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int,
    prior_mixture: float,
    inflation: float,
) -> tuple[np.ndarray, dict[str, float]]:
    log_evidence = np.empty(x_context.shape[0], dtype=np.float64)
    ess_values = []
    log_weight_std = []
    for start in range(0, x_context.shape[0], batch_size):
        stop = min(start + batch_size, x_context.shape[0])
        batch_context = np.asarray(x_context[start:stop], dtype=np.float64)
        batch_x = np.asarray(x_raw[start:stop], dtype=np.float64)
        proposal, weights = label_sample_proposal(
            batch_context,
            samples=samples,
            seed=seed + start,
            prior_mixture=prior_mixture,
            inflation=inflation,
        )
        log_integrand = label_raw_prior_logpdf_batched(proposal) + label_log_likelihood_batched(batch_x, proposal)
        log_q = label_log_proposal_density(
            proposal,
            batch_context,
            weights=weights,
            prior_mixture=prior_mixture,
            inflation=inflation,
        )
        log_w = log_integrand - log_q
        log_evidence[start:stop] = logsumexp(log_w, axis=1) - math.log(samples)
        normalized = np.exp(log_w - logsumexp(log_w, axis=1)[:, None])
        ess = 1.0 / np.sum(normalized * normalized, axis=1)
        ess_values.append(ess)
        log_weight_std.append(np.std(log_w, axis=1))
    ess_all = np.concatenate(ess_values)
    log_weight_std_all = np.concatenate(log_weight_std)
    diagnostics = {
        "importance_samples": int(samples),
        "importance_batch_size": int(batch_size),
        "prior_mixture": float(prior_mixture),
        "proposal_inflation": float(inflation),
        "ess_mean": float(np.mean(ess_all)),
        "ess_median": float(np.median(ess_all)),
        "ess_q05": float(np.quantile(ess_all, 0.05)),
        "ess_min": float(np.min(ess_all)),
        "relative_ess_mean": float(np.mean(ess_all) / samples),
        "relative_ess_q05": float(np.quantile(ess_all, 0.05) / samples),
        "log_weight_std_mean": float(np.mean(log_weight_std_all)),
        "log_weight_std_q95": float(np.quantile(log_weight_std_all, 0.95)),
    }
    return log_evidence, diagnostics


def label_switch_exact_posterior_nll(
    *,
    x_raw: np.ndarray,
    x_context: np.ndarray,
    z_sorted: np.ndarray,
    importance_samples: int,
    importance_seed: int,
    importance_batch_size: int,
    prior_mixture: float,
    proposal_inflation: float,
) -> tuple[np.ndarray, dict[str, float]]:
    z = np.asarray(z_sorted, dtype=np.float64)
    log_prior = label_raw_prior_logpdf(z)
    log_likelihood = label_log_likelihood_np(np.asarray(x_raw, dtype=np.float64), z)
    log_evidence, diagnostics = label_log_evidence_importance(
        x_raw=x_raw,
        x_context=x_context,
        samples=importance_samples,
        seed=importance_seed,
        batch_size=importance_batch_size,
        prior_mixture=prior_mixture,
        inflation=proposal_inflation,
    )
    return -(math.log(2.0) + log_prior + log_likelihood - log_evidence), diagnostics


def two_exp_profile_center_from_context(x_context: np.ndarray) -> np.ndarray:
    context = np.asarray(x_context, dtype=np.float64)
    log_a1 = context[:, 0]
    log_k1 = context[:, 1]
    log_a2 = context[:, 2]
    log_k2 = context[:, 3]
    log_sigma = context[:, 4]
    k1 = np.exp(log_k1)
    k2 = np.exp(log_k2)
    log_delta = np.log(np.maximum(k2 - k1, 1e-8))
    return np.column_stack([log_a1, log_k1, log_a2, log_delta, log_sigma])


def two_exp_target_center_from_context(
    x_context: np.ndarray,
    *,
    target: str,
) -> np.ndarray:
    profile_raw = two_exp_profile_center_from_context(x_context)
    return two_exp_target_transform(profile_raw, target=target)


def two_exp_model_target_from_raw(
    z_raw: np.ndarray,
    x_context: np.ndarray,
    *,
    target: str,
    mode: str,
) -> np.ndarray:
    target_z = two_exp_target_transform(z_raw, target=target)
    if mode == "direct":
        return target_z
    if mode == "profile_residual":
        return target_z - two_exp_target_center_from_context(x_context, target=target)
    raise ValueError(f"Unsupported two_exp target mode: {mode}")


def two_exp_raw_prior_logpdf(z_raw: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    return (
        -0.5 * ((z - TWO_EXP_PRIOR_MEAN[None, :]) / TWO_EXP_PRIOR_STD[None, :]) ** 2
        - np.log(TWO_EXP_PRIOR_STD[None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=1)


def two_exp_raw_prior_logpdf_batched(z_raw: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    return (
        -0.5 * ((z - TWO_EXP_PRIOR_MEAN[None, None, :]) / TWO_EXP_PRIOR_STD[None, None, :]) ** 2
        - np.log(TWO_EXP_PRIOR_STD[None, None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=2)


def two_exp_log_likelihood_np(x_raw: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    z = np.asarray(z_raw, dtype=np.float64)
    a1 = np.exp(z[:, 0])
    k1 = np.exp(z[:, 1])
    a2 = np.exp(z[:, 2])
    k2 = k1 + np.exp(z[:, 3])
    log_sigma = z[:, 4]
    mean = (
        a1[:, None] * np.exp(-k1[:, None] * TWO_EXP_T[None, :])
        + a2[:, None] * np.exp(-k2[:, None] * TWO_EXP_T[None, :])
    )
    residual = x - mean
    return (-0.5 * residual * residual * np.exp(-2.0 * log_sigma[:, None]) - log_sigma[:, None] - 0.5 * LOG_2PI).sum(axis=1)


def two_exp_log_likelihood_batched(x_raw: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    z = np.asarray(z_raw, dtype=np.float64)
    a1 = np.exp(z[:, :, 0])
    k1 = np.exp(z[:, :, 1])
    a2 = np.exp(z[:, :, 2])
    k2 = k1 + np.exp(z[:, :, 3])
    log_sigma = z[:, :, 4]
    mean = (
        a1[:, :, None] * np.exp(-k1[:, :, None] * TWO_EXP_T[None, None, :])
        + a2[:, :, None] * np.exp(-k2[:, :, None] * TWO_EXP_T[None, None, :])
    )
    residual = x[:, None, :] - mean
    return (
        -0.5 * residual * residual * np.exp(-2.0 * log_sigma[:, :, None])
        - log_sigma[:, :, None]
        - 0.5 * LOG_2PI
    ).sum(axis=2)


def two_exp_gaussian_logpdf_batched(z_raw: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.asarray(z_raw, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    return (
        -0.5 * ((z - c[:, None, :]) / s[:, None, :]) ** 2
        - np.log(s[:, None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=2)


def two_exp_gauss_newton_scale(center: np.ndarray) -> np.ndarray:
    z = np.asarray(center, dtype=np.float64)
    a1 = np.exp(z[:, 0])
    k1 = np.exp(z[:, 1])
    a2 = np.exp(z[:, 2])
    delta = np.exp(z[:, 3])
    k2 = k1 + delta
    sigma2 = np.exp(2.0 * z[:, 4])
    e1 = np.exp(-k1[:, None] * TWO_EXP_T[None, :])
    e2 = np.exp(-k2[:, None] * TWO_EXP_T[None, :])
    jac = np.empty((z.shape[0], TWO_EXP_N_OBS, 4), dtype=np.float64)
    jac[:, :, 0] = a1[:, None] * e1
    jac[:, :, 1] = -TWO_EXP_T[None, :] * k1[:, None] * (a1[:, None] * e1 + a2[:, None] * e2)
    jac[:, :, 2] = a2[:, None] * e2
    jac[:, :, 3] = -TWO_EXP_T[None, :] * delta[:, None] * a2[:, None] * e2
    prior_precision = 1.0 / (TWO_EXP_PRIOR_STD * TWO_EXP_PRIOR_STD)
    precision = np.einsum("nti,ntj,n->nij", jac, jac, 1.0 / sigma2)
    precision += np.eye(4)[None, :, :] * prior_precision[:4][None, None, :]
    covariance = np.linalg.inv(precision + np.eye(4)[None, :, :] * 1e-8)
    scale = np.empty((z.shape[0], 5), dtype=np.float64)
    scale[:, :4] = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-10))
    scale[:, 4] = 1.0 / np.sqrt(2.0 * TWO_EXP_N_OBS + prior_precision[4])
    return np.clip(scale, 0.015, TWO_EXP_PRIOR_STD[None, :] * 1.25)


def two_exp_proposal_parameters(
    x_context: np.ndarray,
    z_true: np.ndarray,
    *,
    inflation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profile_center = two_exp_profile_center_from_context(x_context)
    true_center = np.asarray(z_true, dtype=np.float64)
    true_scale = two_exp_gauss_newton_scale(true_center) * inflation
    profile_scale = two_exp_gauss_newton_scale(profile_center) * inflation
    return true_center, profile_center, true_scale, profile_scale


def two_exp_sample_proposal(
    x_context: np.ndarray,
    z_true: np.ndarray,
    *,
    samples: int,
    seed: int,
    prior_mixture: float,
    inflation: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = x_context.shape[0]
    true_center, profile_center, true_scale, profile_scale = two_exp_proposal_parameters(
        x_context,
        z_true,
        inflation=inflation,
    )
    local_weight = max(1.0 - prior_mixture, 1e-12)
    weights = np.array(
        [
            prior_mixture,
            0.30 * local_weight,
            0.25 * local_weight,
            0.25 * local_weight,
            0.20 * local_weight,
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    component = rng.choice(weights.size, size=(batch, samples), p=weights)
    z = np.empty((batch, samples, 5), dtype=np.float64)
    for index in range(batch):
        for component_index in range(weights.size):
            mask = component[index] == component_index
            count = int(mask.sum())
            if count == 0:
                continue
            if component_index == 0:
                z[index, mask] = rng.normal(TWO_EXP_PRIOR_MEAN, TWO_EXP_PRIOR_STD, size=(count, 5))
            elif component_index == 1:
                z[index, mask] = rng.normal(true_center[index], true_scale[index], size=(count, 5))
            elif component_index == 2:
                z[index, mask] = rng.normal(profile_center[index], profile_scale[index], size=(count, 5))
            elif component_index == 3:
                z[index, mask] = rng.normal(true_center[index], true_scale[index] * 3.0, size=(count, 5))
            else:
                z[index, mask] = rng.normal(profile_center[index], profile_scale[index] * 3.0, size=(count, 5))
    return z, weights


def two_exp_log_proposal_density(
    z_raw: np.ndarray,
    x_context: np.ndarray,
    z_true: np.ndarray,
    *,
    weights: np.ndarray,
    prior_mixture: float,
    inflation: float,
) -> np.ndarray:
    true_center, profile_center, true_scale, profile_scale = two_exp_proposal_parameters(
        x_context,
        z_true,
        inflation=inflation,
    )
    terms = [
        math.log(max(prior_mixture, 1e-300)) + two_exp_raw_prior_logpdf_batched(z_raw),
        math.log(float(weights[1])) + two_exp_gaussian_logpdf_batched(z_raw, true_center, true_scale),
        math.log(float(weights[2])) + two_exp_gaussian_logpdf_batched(z_raw, profile_center, profile_scale),
        math.log(float(weights[3])) + two_exp_gaussian_logpdf_batched(z_raw, true_center, true_scale * 3.0),
        math.log(float(weights[4])) + two_exp_gaussian_logpdf_batched(z_raw, profile_center, profile_scale * 3.0),
    ]
    return logsumexp(np.stack(terms, axis=0), axis=0)


def two_exp_log_evidence_importance(
    x_raw: np.ndarray,
    x_context: np.ndarray,
    z_true: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int,
    prior_mixture: float,
    inflation: float,
) -> tuple[np.ndarray, dict[str, float]]:
    log_evidence = np.empty(x_context.shape[0], dtype=np.float64)
    ess_values = []
    log_weight_std = []
    for start in range(0, x_context.shape[0], batch_size):
        stop = min(start + batch_size, x_context.shape[0])
        batch_context = np.asarray(x_context[start:stop], dtype=np.float64)
        batch_x = np.asarray(x_raw[start:stop], dtype=np.float64)
        batch_z = np.asarray(z_true[start:stop], dtype=np.float64)
        proposal, weights = two_exp_sample_proposal(
            batch_context,
            batch_z,
            samples=samples,
            seed=seed + start,
            prior_mixture=prior_mixture,
            inflation=inflation,
        )
        log_integrand = two_exp_raw_prior_logpdf_batched(proposal) + two_exp_log_likelihood_batched(batch_x, proposal)
        log_q = two_exp_log_proposal_density(
            proposal,
            batch_context,
            batch_z,
            weights=weights,
            prior_mixture=prior_mixture,
            inflation=inflation,
        )
        log_w = log_integrand - log_q
        log_evidence[start:stop] = logsumexp(log_w, axis=1) - math.log(samples)
        normalized = np.exp(log_w - logsumexp(log_w, axis=1)[:, None])
        ess = 1.0 / np.sum(normalized * normalized, axis=1)
        ess_values.append(ess)
        log_weight_std.append(np.std(log_w, axis=1))
    ess_all = np.concatenate(ess_values)
    log_weight_std_all = np.concatenate(log_weight_std)
    diagnostics = {
        "importance_samples": int(samples),
        "importance_batch_size": int(batch_size),
        "prior_mixture": float(prior_mixture),
        "proposal_inflation": float(inflation),
        "ess_mean": float(np.mean(ess_all)),
        "ess_median": float(np.median(ess_all)),
        "ess_q05": float(np.quantile(ess_all, 0.05)),
        "ess_min": float(np.min(ess_all)),
        "relative_ess_mean": float(np.mean(ess_all) / samples),
        "relative_ess_q05": float(np.quantile(ess_all, 0.05) / samples),
        "log_weight_std_mean": float(np.mean(log_weight_std_all)),
        "log_weight_std_q95": float(np.quantile(log_weight_std_all, 0.95)),
    }
    return log_evidence, diagnostics


def two_exp_beta_schedule(steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("SMC beta steps must be positive.")
    t = np.linspace(0.0, 1.0, steps + 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def systematic_resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)
    batch, particles = normalized.shape
    positions = (rng.random((batch, 1)) + np.arange(particles, dtype=np.float64)[None, :]) / particles
    cdf = np.cumsum(normalized, axis=1)
    cdf[:, -1] = 1.0
    indices = np.empty((batch, particles), dtype=np.int64)
    for row in range(batch):
        indices[row] = np.searchsorted(cdf[row], positions[row], side="right")
    return np.minimum(indices, particles - 1)


def two_exp_log_evidence_smc(
    x_raw: np.ndarray,
    z_true: np.ndarray,
    *,
    particles: int,
    beta_steps: int,
    mh_steps: int,
    seed: int,
    batch_size: int,
    step_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if particles < 2:
        raise ValueError("SMC needs at least two particles.")
    if batch_size < 1:
        raise ValueError("SMC batch size must be positive.")
    betas = two_exp_beta_schedule(beta_steps)
    x = np.asarray(x_raw, dtype=np.float64)
    true_z = np.asarray(z_true, dtype=np.float64)
    log_evidence = np.empty(x.shape[0], dtype=np.float64)
    ess_batches = []
    acceptance_batches = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        rng = np.random.default_rng(seed + start)
        batch_x = x[start:stop]
        batch_z = true_z[start:stop]
        batch = batch_x.shape[0]
        z = rng.normal(
            TWO_EXP_PRIOR_MEAN[None, None, :],
            TWO_EXP_PRIOR_STD[None, None, :],
            size=(batch, particles, 5),
        )
        log_prior = two_exp_raw_prior_logpdf_batched(z)
        log_likelihood = two_exp_log_likelihood_batched(batch_x, z)
        local_scale = two_exp_gauss_newton_scale(batch_z)
        row_index = np.arange(batch)[:, None]
        batch_log_evidence = np.zeros(batch, dtype=np.float64)
        batch_ess = []
        batch_acceptance = []
        for beta_previous, beta in zip(betas[:-1], betas[1:]):
            log_increment = (float(beta) - float(beta_previous)) * log_likelihood
            log_norm = logsumexp(log_increment, axis=1)
            batch_log_evidence += log_norm - math.log(particles)
            normalized = np.exp(log_increment - log_norm[:, None])
            ess = 1.0 / np.sum(normalized * normalized, axis=1)
            batch_ess.append(ess)
            indices = systematic_resample_indices(normalized, rng)
            z = z[row_index, indices, :]
            log_prior = log_prior[row_index, indices]
            log_likelihood = log_likelihood[row_index, indices]
            target = log_prior + float(beta) * log_likelihood
            if mh_steps <= 0:
                continue
            beta_scale = max(float(beta), 0.02)
            proposal_scale = local_scale * float(step_scale) / math.sqrt(beta_scale)
            proposal_scale = np.minimum(proposal_scale, TWO_EXP_PRIOR_STD[None, :] * 0.75)
            for _ in range(mh_steps):
                proposal = z + rng.normal(size=z.shape) * proposal_scale[:, None, :]
                proposal_log_prior = two_exp_raw_prior_logpdf_batched(proposal)
                proposal_log_likelihood = two_exp_log_likelihood_batched(batch_x, proposal)
                proposal_target = proposal_log_prior + float(beta) * proposal_log_likelihood
                accept = np.log(rng.random((batch, particles))) < (proposal_target - target)
                z[accept] = proposal[accept]
                log_prior[accept] = proposal_log_prior[accept]
                log_likelihood[accept] = proposal_log_likelihood[accept]
                target[accept] = proposal_target[accept]
                batch_acceptance.append(np.mean(accept, axis=1))
        log_evidence[start:stop] = batch_log_evidence
        ess_batches.append(np.stack(batch_ess, axis=1))
        if batch_acceptance:
            acceptance_batches.append(np.stack(batch_acceptance, axis=1))
    ess_matrix = np.concatenate(ess_batches, axis=0)
    min_relative_ess = np.min(ess_matrix / particles, axis=1)
    diagnostics = {
        "smc_particles": int(particles),
        "smc_beta_steps": int(beta_steps),
        "smc_mh_steps": int(mh_steps),
        "smc_batch_size": int(batch_size),
        "smc_step_scale": float(step_scale),
        "incremental_relative_ess_mean": float(np.mean(ess_matrix) / particles),
        "incremental_relative_ess_q05": float(np.quantile(ess_matrix / particles, 0.05)),
        "min_incremental_relative_ess_mean": float(np.mean(min_relative_ess)),
        "min_incremental_relative_ess_q05": float(np.quantile(min_relative_ess, 0.05)),
    }
    if acceptance_batches:
        acceptance_matrix = np.concatenate(acceptance_batches, axis=0)
        diagnostics.update(
            {
                "mh_acceptance_mean": float(np.mean(acceptance_matrix)),
                "mh_acceptance_q05": float(np.quantile(acceptance_matrix, 0.05)),
            }
        )
    return log_evidence, diagnostics


def two_exp_exact_posterior_nll(
    *,
    x_raw: np.ndarray,
    x_context: np.ndarray,
    z_raw: np.ndarray,
    floor_method: str,
    importance_samples: int,
    importance_seed: int,
    importance_batch_size: int,
    prior_mixture: float,
    proposal_inflation: float,
    smc_particles: int,
    smc_beta_steps: int,
    smc_mh_steps: int,
    smc_seed: int,
    smc_batch_size: int,
    smc_step_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    z = np.asarray(z_raw, dtype=np.float64)
    log_prior = two_exp_raw_prior_logpdf(z)
    log_likelihood = two_exp_log_likelihood_np(np.asarray(x_raw, dtype=np.float64), z)
    if floor_method == "importance":
        log_evidence, diagnostics = two_exp_log_evidence_importance(
            x_raw=x_raw,
            x_context=x_context,
            z_true=z,
            samples=importance_samples,
            seed=importance_seed,
            batch_size=importance_batch_size,
            prior_mixture=prior_mixture,
            inflation=proposal_inflation,
        )
    elif floor_method == "smc":
        log_evidence, diagnostics = two_exp_log_evidence_smc(
            x_raw=x_raw,
            z_true=z,
            particles=smc_particles,
            beta_steps=smc_beta_steps,
            mh_steps=smc_mh_steps,
            seed=smc_seed,
            batch_size=smc_batch_size,
            step_scale=smc_step_scale,
        )
    else:
        raise ValueError(f"Unsupported two_exp floor method: {floor_method}")
    diagnostics["floor_method"] = floor_method
    return -(log_prior + log_likelihood - log_evidence), diagnostics


def linear6_log_evidence(
    x_context: np.ndarray,
    *,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    _, projected_sq, residual_sq = linear6_sufficient_stats(x_context)
    nodes, weights = roots_hermitenorm(int(quadrature_order))
    log_weights = np.log(weights) - 0.5 * LOG_2PI
    log_sigma_mean = math.log(0.25)
    log_sigma_std = 0.50
    log_sigma_nodes = log_sigma_mean + log_sigma_std * np.asarray(nodes, dtype=np.float64)
    result = np.empty(x_context.shape[0], dtype=np.float64)
    for start in range(0, x_context.shape[0], chunk_size):
        stop = min(start + chunk_size, x_context.shape[0])
        log_terms = (
            log_weights[None, :]
            + linear6_log_py_given_log_sigma(
                projected_sq=projected_sq[start:stop, None],
                residual_sq=residual_sq[start:stop, None],
                log_sigma=log_sigma_nodes[None, :],
            )
        )
        result[start:stop] = logsumexp(log_terms, axis=1)
    return result


def linear6_exact_posterior_nll(
    *,
    x_context: np.ndarray,
    z_raw: np.ndarray,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    coef, projected_sq, residual_sq = linear6_sufficient_stats(x_context)
    d_w = 6
    n_obs = 32
    prior_std_w = 1.25
    log_sigma_mean = math.log(0.25)
    log_sigma_std = 0.50
    log_sigma = np.asarray(z_raw[:, -1], dtype=np.float64)
    sigma2 = np.exp(2.0 * log_sigma)
    posterior_var = 1.0 / (1.0 / (prior_std_w * prior_std_w) + n_obs / sigma2)
    shrink = posterior_var * n_obs / sigma2
    posterior_mean = shrink[:, None] * coef
    delta = np.asarray(z_raw[:, :d_w], dtype=np.float64) - posterior_mean
    log_w_given_sigma_x = -0.5 * (
        d_w * LOG_2PI
        + d_w * np.log(posterior_var)
        + np.sum(delta * delta, axis=1) / posterior_var
    )
    log_py_sigma = linear6_log_py_given_log_sigma(
        projected_sq=projected_sq,
        residual_sq=residual_sq,
        log_sigma=log_sigma,
    )
    log_evidence = linear6_log_evidence(
        x_context,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
    )
    log_sigma_posterior = (
        normal_logpdf_1d(log_sigma, log_sigma_mean, log_sigma_std)
        + log_py_sigma
        - log_evidence
    )
    return -(log_w_given_sigma_x + log_sigma_posterior)


def evaluate_model_log_prob(
    *,
    model: torch.nn.Module,
    x_raw: np.ndarray,
    z_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x_standardized = standardize(x_raw, x_mean, x_std)
    z_standardized = standardize(z_raw, z_mean, z_std)
    log_det = float(np.log(z_std.astype(np.float64)).sum())
    return model.log_prob(
        torch.from_numpy(z_standardized).to(device),
        torch.from_numpy(x_standardized).to(device),
    ) - log_det


def member_model_target(
    *,
    model_name: str,
    z_raw: np.ndarray,
    x_context: np.ndarray,
    member: dict[str, object],
    default_two_exp_target: str,
) -> np.ndarray:
    if model_name != "two_exp":
        return z_raw
    member_target = str(member.get("two_exp_target", default_two_exp_target))
    target_z = two_exp_target_transform(z_raw, target=member_target)
    target_mode = str(member.get("two_exp_target_mode", "direct"))
    if target_mode == "direct":
        return target_z
    if target_mode == "profile_residual":
        return target_z - two_exp_target_center_from_context(x_context, target=member_target)
    raise ValueError(f"Unsupported two_exp target mode in checkpoint: {target_mode}")


def evaluate_member_log_probs(
    *,
    model_name: str,
    members: list[dict[str, object]],
    x_context: np.ndarray,
    z_raw: np.ndarray,
    default_two_exp_target: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, x_context.shape[0], batch_size):
        stop = min(start + batch_size, x_context.shape[0])
        batch_x = x_context[start:stop]
        batch_z = z_raw[start:stop]
        batch_log_probs = []
        for member in members:
            member_z_model = member_model_target(
                model_name=model_name,
                z_raw=batch_z,
                x_context=batch_x,
                member=member,
                default_two_exp_target=default_two_exp_target,
            )
            log_prob = evaluate_model_log_prob(
                model=member["model"],
                x_raw=batch_x,
                z_raw=member_z_model,
                x_mean=member["x_mean"],
                x_std=member["x_std"],
                z_mean=member["z_mean"],
                z_std=member["z_std"],
                device=device,
            )
            batch_log_probs.append(log_prob.detach().cpu().numpy().astype(np.float32))
        chunks.append(np.stack(batch_log_probs, axis=1))
    return np.concatenate(chunks, axis=0)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def quantile_bin_summary(feature: np.ndarray, gap: np.ndarray, *, bins: int = 5) -> list[dict[str, object]]:
    values = np.asarray(feature, dtype=np.float64)
    gaps = np.asarray(gap, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(gaps)
    values = values[mask]
    gaps = gaps[mask]
    if values.size == 0:
        return []
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    rows = []
    for index in range(bins):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == bins - 1:
            bin_mask = (values >= low) & (values <= high)
        else:
            bin_mask = (values >= low) & (values < high)
        bin_gap = gaps[bin_mask]
        rows.append(
            {
                "low": low,
                "high": high,
                "n": int(bin_gap.size),
                "gap_mean": float(np.mean(bin_gap)) if bin_gap.size else None,
                "gap_std_error": float(np.std(bin_gap, ddof=1) / math.sqrt(bin_gap.size))
                if bin_gap.size > 1
                else 0.0,
            }
        )
    return rows


def two_exp_gap_features(z_raw: np.ndarray, x_context: np.ndarray, *, target: str) -> dict[str, np.ndarray]:
    z = np.asarray(z_raw, dtype=np.float64)
    profile = two_exp_profile_center_from_context(x_context)
    target_z = two_exp_target_transform(z, target=target).astype(np.float64)
    log_k2 = np.logaddexp(z[:, 1], z[:, 3])
    log_total_amplitude = np.logaddexp(z[:, 0], z[:, 2])
    profile_delta = profile - z
    features = {
        "log_A1": z[:, 0],
        "log_k1": z[:, 1],
        "log_A2": z[:, 2],
        "log_delta_k": z[:, 3],
        "log_sigma": z[:, 4],
        "log_k2": log_k2,
        "log_total_amplitude": log_total_amplitude,
        "log_amplitude_ratio": z[:, 0] - z[:, 2],
        "log_delta_over_k1": z[:, 3] - z[:, 1],
        "log_snr": log_total_amplitude - z[:, 4],
        "profile_abs_error_log_A1": np.abs(profile_delta[:, 0]),
        "profile_abs_error_log_k1": np.abs(profile_delta[:, 1]),
        "profile_abs_error_log_A2": np.abs(profile_delta[:, 2]),
        "profile_abs_error_log_delta_k": np.abs(profile_delta[:, 3]),
        "profile_abs_error_log_sigma": np.abs(profile_delta[:, 4]),
        "profile_l2_error": np.linalg.norm(profile_delta, axis=1),
    }
    for index, name in enumerate(two_exp_target_description(target).strip("()").split(", ")):
        features[f"target_{index}_{name}"] = target_z[:, index]
    return features


def write_two_exp_gap_diagnostics(
    *,
    path: Path,
    z_raw: np.ndarray,
    x_context: np.ndarray,
    exact_nll: np.ndarray,
    ensemble_nll: np.ndarray,
    two_exp_target: str,
) -> None:
    gap = np.asarray(ensemble_nll, dtype=np.float64) - np.asarray(exact_nll, dtype=np.float64)
    features = two_exp_gap_features(z_raw, x_context, target=two_exp_target)
    feature_rows = {}
    for name, values in features.items():
        values_array = np.asarray(values, dtype=np.float64)
        q90 = np.quantile(values_array, 0.90)
        q10 = np.quantile(values_array, 0.10)
        high_gap = gap[values_array >= q90]
        low_gap = gap[values_array <= q10]
        feature_rows[name] = {
            "summary": summarize(values_array),
            "corr_with_gap": safe_corrcoef(values_array, gap),
            "gap_mean_top_decile": float(np.mean(high_gap)) if high_gap.size else None,
            "gap_mean_bottom_decile": float(np.mean(low_gap)) if low_gap.size else None,
            "gap_top_minus_bottom_decile": float(np.mean(high_gap) - np.mean(low_gap))
            if high_gap.size and low_gap.size
            else None,
            "gap_by_feature_quantile": quantile_bin_summary(values_array, gap),
        }
    ranked_by_corr = sorted(
        (
            {"feature": name, "corr_with_gap": row["corr_with_gap"]}
            for name, row in feature_rows.items()
            if row["corr_with_gap"] is not None
        ),
        key=lambda item: abs(float(item["corr_with_gap"])),
        reverse=True,
    )
    ranked_by_decile = sorted(
        (
            {"feature": name, "gap_top_minus_bottom_decile": row["gap_top_minus_bottom_decile"]}
            for name, row in feature_rows.items()
            if row["gap_top_minus_bottom_decile"] is not None
        ),
        key=lambda item: abs(float(item["gap_top_minus_bottom_decile"])),
        reverse=True,
    )
    top_indices = np.argsort(gap)[-20:][::-1]
    output = {
        "target": two_exp_target_description(two_exp_target),
        "n": int(gap.size),
        "gap": summarize(gap),
        "ensemble_nll": summarize(np.asarray(ensemble_nll, dtype=np.float64)),
        "exact_nll": summarize(np.asarray(exact_nll, dtype=np.float64)),
        "ranked_by_abs_correlation": ranked_by_corr[:12],
        "ranked_by_abs_decile_contrast": ranked_by_decile[:12],
        "features": feature_rows,
        "top_gap_examples": [
            {
                "index": int(index),
                "gap": float(gap[index]),
                "ensemble_nll": float(ensemble_nll[index]),
                "exact_nll": float(exact_nll[index]),
                "z_raw": np.asarray(z_raw[index], dtype=np.float64).tolist(),
                "profile_center": two_exp_profile_center_from_context(x_context[index : index + 1])[0].tolist(),
            }
            for index in top_indices
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_population_nll(
    *,
    model_name: str,
    members: list[dict[str, object]],
    validation_examples: int,
    validation_seed: int,
    batch_size: int,
    device: torch.device,
    linear6_quadrature_order: int,
    banana_quadrature_order: int,
    label_importance_samples: int,
    label_importance_seed: int,
    label_importance_batch_size: int,
    label_prior_mixture: float,
    label_proposal_inflation: float,
    two_exp_target: str,
    two_exp_floor_method: str,
    two_exp_importance_samples: int,
    two_exp_importance_seed: int,
    two_exp_importance_batch_size: int,
    two_exp_prior_mixture: float,
    two_exp_proposal_inflation: float,
    two_exp_smc_particles: int,
    two_exp_smc_beta_steps: int,
    two_exp_smc_mh_steps: int,
    two_exp_smc_seed: int,
    two_exp_smc_batch_size: int,
    two_exp_smc_step_scale: float,
    diagnostics_output: Path | None = None,
    stacking_gate: torch.nn.Module | None = None,
    stacking_x_mean: np.ndarray | None = None,
    stacking_x_std: np.ndarray | None = None,
) -> dict[str, Any]:
    if model_name == "label_switch":
        x_raw_val, x_val, z_val = sample_label_switch_population_raw(n=validation_examples, seed=validation_seed)
    elif model_name == "two_exp":
        x_raw_val, x_val, z_val = sample_two_exp_population_raw(n=validation_examples, seed=validation_seed)
    else:
        x_raw_val = None
        x_val, z_val = sample_population(
            model=model_name,
            n=validation_examples,
            seed=validation_seed,
            two_exp_target=two_exp_target,
        )
    individual_chunks: list[list[np.ndarray]] = [[] for _ in members]
    ensemble_chunks: list[np.ndarray] = []
    stacking_weight_chunks: list[np.ndarray] = []
    exact_chunks: list[np.ndarray] = []
    exact_diagnostics: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for start in range(0, validation_examples, batch_size):
        stop = min(start + batch_size, validation_examples)
        batch_x = x_val[start:stop]
        batch_z = z_val[start:stop]
        if model_name == "banana":
            exact_chunks.append(
                banana_exact_posterior_nll(
                    x_context=batch_x,
                    z_raw=batch_z,
                    quadrature_order=banana_quadrature_order,
                    chunk_size=batch_size,
                )
            )
        elif model_name == "label_switch":
            assert x_raw_val is not None
            exact_nll, diagnostics = label_switch_exact_posterior_nll(
                x_raw=x_raw_val[start:stop],
                x_context=batch_x,
                z_sorted=batch_z,
                importance_samples=label_importance_samples,
                importance_seed=label_importance_seed + start,
                importance_batch_size=label_importance_batch_size,
                prior_mixture=label_prior_mixture,
                proposal_inflation=label_proposal_inflation,
            )
            exact_chunks.append(exact_nll)
            exact_diagnostics.append(diagnostics)
        elif model_name == "two_exp":
            assert x_raw_val is not None
            exact_nll, diagnostics = two_exp_exact_posterior_nll(
                x_raw=x_raw_val[start:stop],
                x_context=batch_x,
                z_raw=batch_z,
                floor_method=two_exp_floor_method,
                importance_samples=two_exp_importance_samples,
                importance_seed=two_exp_importance_seed + start,
                importance_batch_size=two_exp_importance_batch_size,
                prior_mixture=two_exp_prior_mixture,
                proposal_inflation=two_exp_proposal_inflation,
                smc_particles=two_exp_smc_particles,
                smc_beta_steps=two_exp_smc_beta_steps,
                smc_mh_steps=two_exp_smc_mh_steps,
                smc_seed=two_exp_smc_seed + start,
                smc_batch_size=two_exp_smc_batch_size,
                smc_step_scale=two_exp_smc_step_scale,
            )
            exact_chunks.append(exact_nll)
            exact_diagnostics.append(diagnostics)
        elif model_name == "linear6":
            exact_chunks.append(
                linear6_exact_posterior_nll(
                    x_context=batch_x,
                    z_raw=batch_z,
                    quadrature_order=linear6_quadrature_order,
                    chunk_size=batch_size,
                )
            )
        log_probs = []
        for index, member in enumerate(members):
            member_z_model = member_model_target(
                model_name=model_name,
                z_raw=batch_z,
                x_context=batch_x,
                member=member,
                default_two_exp_target=two_exp_target,
            )
            log_prob = evaluate_model_log_prob(
                model=member["model"],
                x_raw=batch_x,
                z_raw=member_z_model,
                x_mean=member["x_mean"],
                x_std=member["x_std"],
                z_mean=member["z_mean"],
                z_std=member["z_std"],
                device=device,
            )
            log_prob_np = log_prob.detach().cpu().numpy().astype(np.float64)
            individual_chunks[index].append(-log_prob_np)
            log_probs.append(log_prob_np)
        stacked = np.stack(log_probs, axis=0)
        if stacking_gate is None:
            ensemble_log_prob = logsumexp(stacked, axis=0) - math.log(len(members))
        else:
            if stacking_x_mean is None or stacking_x_std is None:
                raise ValueError("stacking_x_mean and stacking_x_std are required with stacking_gate.")
            stacking_gate.eval()
            with torch.no_grad():
                gate_x = standardize(batch_x, stacking_x_mean, stacking_x_std).astype(np.float32)
                logits = stacking_gate(torch.from_numpy(gate_x).to(device))
                log_weights = torch.log_softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            ensemble_log_prob = logsumexp(stacked.T + log_weights, axis=1)
            stacking_weight_chunks.append(np.exp(log_weights))
        ensemble_chunks.append(-ensemble_log_prob)

    individual_nll = [np.concatenate(chunks) for chunks in individual_chunks]
    ensemble_nll = np.concatenate(ensemble_chunks)
    ensemble_summary = summarize(ensemble_nll)
    output = {
        "validation_examples": int(validation_examples),
        "validation_seed": int(validation_seed),
        "evaluation_seconds": float(time.perf_counter() - start_time),
        "individual_nll": [summarize(values) for values in individual_nll],
        "best_individual_nll": float(min(np.mean(values) for values in individual_nll)),
        "ensemble_nll": ensemble_summary,
    }
    if stacking_weight_chunks:
        weights = np.concatenate(stacking_weight_chunks, axis=0)
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
        output["stacking_gate"] = {
            "mean_weights": weights.mean(axis=0).tolist(),
            "q05_weights": np.quantile(weights, 0.05, axis=0).tolist(),
            "q95_weights": np.quantile(weights, 0.95, axis=0).tolist(),
            "mean_entropy": float(np.mean(entropy)),
            "max_weight_mean": float(np.mean(np.max(weights, axis=1))),
        }
    if model_name == "sign":
        gap = float(ensemble_summary["mean"] - FOLDED_SIGN_FLOOR)
        combined_se = math.sqrt(float(ensemble_summary["std_error"]) ** 2 + FOLDED_SIGN_FLOOR_SE**2)
        output.update({
            "floor": {
            "estimate": FOLDED_SIGN_FLOOR,
            "standard_error": FOLDED_SIGN_FLOOR_SE,
            "coordinate_target": "(abs(theta1), theta2)",
            },
            "ensemble_gap_to_floor": gap,
            "combined_standard_error": combined_se,
            "gap_z_score": gap / combined_se if combined_se > 0 else None,
        })
    elif model_name in {"banana", "label_switch", "linear6", "two_exp"}:
        exact_nll = np.concatenate(exact_chunks)
        gap_samples = ensemble_nll - exact_nll
        paired_gap = summarize(gap_samples)
        floor_summary = summarize(exact_nll)
        if model_name == "banana":
            floor_target = "(theta1, theta2)"
            floor_method = (
                "Analytic theta2 integration with posterior-centered "
                f"one-dimensional Gauss-Hermite evidence integration over theta1, order {banana_quadrature_order}."
            )
            floor_diagnostics = None
        elif model_name == "label_switch":
            floor_target = "(mu_low, mu_high, log_sigma)"
            floor_method = (
                "Symmetry-folded sorted-coordinate posterior with raw evidence "
                f"estimated by symmetric Gaussian-mixture importance sampling, {label_importance_samples} samples per signal."
            )
            floor_diagnostics = summarize_diagnostics(exact_diagnostics)
        elif model_name == "two_exp":
            floor_target = two_exp_target_description(two_exp_target)
            if two_exp_floor_method == "smc":
                floor_method = (
                    "Ordered two-exponential ridge-coordinate posterior with raw-coordinate evidence "
                    "estimated by prior-to-posterior tempered SMC, "
                    f"{two_exp_smc_particles} particles, {two_exp_smc_beta_steps} beta steps."
                )
            else:
                floor_method = (
                    "Ordered two-exponential ridge-coordinate posterior with raw-coordinate evidence "
                    "estimated by Gaussian-mixture importance sampling around the "
                    f"profile fit and validation draw, {two_exp_importance_samples} samples per signal."
                )
            floor_diagnostics = summarize_diagnostics(exact_diagnostics)
        else:
            floor_target = "(w1, ..., w6, log_sigma)"
            floor_method = (
                "Linear-Gaussian conditional posterior with one-dimensional "
                f"Gauss-Hermite evidence integration, order {linear6_quadrature_order}."
            )
            floor_diagnostics = None
        output.update({
            "floor": {
                "estimate": float(floor_summary["mean"]),
                "standard_error": float(floor_summary["std_error"]),
                "coordinate_target": floor_target,
                "method": floor_method,
                "summary": floor_summary,
                "diagnostics": floor_diagnostics,
            },
            "ensemble_gap_to_floor": float(paired_gap["mean"]),
            "paired_gap_standard_error": float(paired_gap["std_error"]),
            "gap_z_score": float(paired_gap["mean"]) / float(paired_gap["std_error"])
            if float(paired_gap["std_error"]) > 0.0
            else None,
            "paired_gap_summary": paired_gap,
        })
        if model_name == "two_exp" and diagnostics_output is not None:
            write_two_exp_gap_diagnostics(
                path=diagnostics_output,
                z_raw=z_val,
                x_context=x_val,
                exact_nll=exact_nll,
                ensemble_nll=ensemble_nll,
                two_exp_target=two_exp_target,
            )
            output["gap_diagnostics_json"] = str(diagnostics_output)
    else:
        raise ValueError(f"Unsupported population model: {model_name}")
    return output


def estimate_population_floor(
    *,
    model_name: str,
    validation_examples: int,
    validation_seed: int,
    batch_size: int,
    linear6_quadrature_order: int,
    banana_quadrature_order: int,
    label_importance_samples: int,
    label_importance_seed: int,
    label_importance_batch_size: int,
    label_prior_mixture: float,
    label_proposal_inflation: float,
    two_exp_target: str,
    two_exp_floor_method: str,
    two_exp_importance_samples: int,
    two_exp_importance_seed: int,
    two_exp_importance_batch_size: int,
    two_exp_prior_mixture: float,
    two_exp_proposal_inflation: float,
    two_exp_smc_particles: int,
    two_exp_smc_beta_steps: int,
    two_exp_smc_mh_steps: int,
    two_exp_smc_seed: int,
    two_exp_smc_batch_size: int,
    two_exp_smc_step_scale: float,
) -> dict[str, Any]:
    if model_name == "sign":
        return {
            "validation_examples": 0,
            "validation_seed": int(validation_seed),
            "floor": {
                "estimate": FOLDED_SIGN_FLOOR,
                "standard_error": FOLDED_SIGN_FLOOR_SE,
                "coordinate_target": "(abs(theta1), theta2)",
                "method": "Previously computed folded sign entropy floor.",
            },
        }
    if model_name == "label_switch":
        x_raw_val, x_val, z_val = sample_label_switch_population_raw(n=validation_examples, seed=validation_seed)
    elif model_name == "two_exp":
        x_raw_val, x_val, z_val = sample_two_exp_population_raw(n=validation_examples, seed=validation_seed)
    else:
        x_raw_val = None
        x_val, z_val = sample_population(
            model=model_name,
            n=validation_examples,
            seed=validation_seed,
            two_exp_target=two_exp_target,
        )
    exact_chunks: list[np.ndarray] = []
    exact_diagnostics: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for start in range(0, validation_examples, batch_size):
        stop = min(start + batch_size, validation_examples)
        batch_x = x_val[start:stop]
        batch_z = z_val[start:stop]
        if model_name == "banana":
            exact_chunks.append(
                banana_exact_posterior_nll(
                    x_context=batch_x,
                    z_raw=batch_z,
                    quadrature_order=banana_quadrature_order,
                    chunk_size=batch_size,
                )
            )
        elif model_name == "label_switch":
            assert x_raw_val is not None
            exact_nll, diagnostics = label_switch_exact_posterior_nll(
                x_raw=x_raw_val[start:stop],
                x_context=batch_x,
                z_sorted=batch_z,
                importance_samples=label_importance_samples,
                importance_seed=label_importance_seed + start,
                importance_batch_size=label_importance_batch_size,
                prior_mixture=label_prior_mixture,
                proposal_inflation=label_proposal_inflation,
            )
            exact_chunks.append(exact_nll)
            exact_diagnostics.append(diagnostics)
        elif model_name == "two_exp":
            assert x_raw_val is not None
            exact_nll, diagnostics = two_exp_exact_posterior_nll(
                x_raw=x_raw_val[start:stop],
                x_context=batch_x,
                z_raw=batch_z,
                floor_method=two_exp_floor_method,
                importance_samples=two_exp_importance_samples,
                importance_seed=two_exp_importance_seed + start,
                importance_batch_size=two_exp_importance_batch_size,
                prior_mixture=two_exp_prior_mixture,
                proposal_inflation=two_exp_proposal_inflation,
                smc_particles=two_exp_smc_particles,
                smc_beta_steps=two_exp_smc_beta_steps,
                smc_mh_steps=two_exp_smc_mh_steps,
                smc_seed=two_exp_smc_seed + start,
                smc_batch_size=two_exp_smc_batch_size,
                smc_step_scale=two_exp_smc_step_scale,
            )
            exact_chunks.append(exact_nll)
            exact_diagnostics.append(diagnostics)
        elif model_name == "linear6":
            exact_chunks.append(
                linear6_exact_posterior_nll(
                    x_context=batch_x,
                    z_raw=batch_z,
                    quadrature_order=linear6_quadrature_order,
                    chunk_size=batch_size,
                )
            )
        else:
            raise ValueError(f"Unsupported population model: {model_name}")
    exact_nll = np.concatenate(exact_chunks)
    floor_summary = summarize(exact_nll)
    if model_name == "banana":
        floor_target = "(theta1, theta2)"
        floor_method = (
            "Analytic theta2 integration with posterior-centered "
            f"one-dimensional Gauss-Hermite evidence integration over theta1, order {banana_quadrature_order}."
        )
        floor_diagnostics = None
    elif model_name == "label_switch":
        floor_target = "(mu_low, mu_high, log_sigma)"
        floor_method = (
            "Symmetry-folded sorted-coordinate posterior with raw evidence "
            f"estimated by symmetric Gaussian-mixture importance sampling, {label_importance_samples} samples per signal."
        )
        floor_diagnostics = summarize_diagnostics(exact_diagnostics)
    elif model_name == "two_exp":
        floor_target = two_exp_target_description(two_exp_target)
        if two_exp_floor_method == "smc":
            floor_method = (
                "Ordered two-exponential ridge-coordinate posterior with raw-coordinate evidence "
                "estimated by prior-to-posterior tempered SMC, "
                f"{two_exp_smc_particles} particles, {two_exp_smc_beta_steps} beta steps."
            )
        else:
            floor_method = (
                "Ordered two-exponential ridge-coordinate posterior with raw-coordinate evidence "
                "estimated by Gaussian-mixture importance sampling around the "
                f"profile fit and validation draw, {two_exp_importance_samples} samples per signal."
            )
        floor_diagnostics = summarize_diagnostics(exact_diagnostics)
    else:
        floor_target = "(w1, ..., w6, log_sigma)"
        floor_method = (
            "Linear-Gaussian conditional posterior with one-dimensional "
            f"Gauss-Hermite evidence integration, order {linear6_quadrature_order}."
        )
        floor_diagnostics = None
    return {
        "validation_examples": int(validation_examples),
        "validation_seed": int(validation_seed),
        "evaluation_seconds": float(time.perf_counter() - start_time),
        "floor": {
            "estimate": float(floor_summary["mean"]),
            "standard_error": float(floor_summary["std_error"]),
            "coordinate_target": floor_target,
            "method": floor_method,
            "summary": floor_summary,
            "diagnostics": floor_diagnostics,
        },
    }


def resolve_existing_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    rooted = base_dir / candidate
    if rooted.exists():
        return rooted
    return candidate


def load_population_members(summary_path: Path, *, device: torch.device) -> list[dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    members = []
    repo_root = Path.cwd()
    source_two_exp_target = str(summary.get("recipe", {}).get("two_exp_target", "amplitude_sum_delta"))
    for member in summary["members"]:
        model_path = resolve_existing_path(str(member["model_pt"]), base_dir=repo_root)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config_dict = dict(checkpoint["config"])
        progress_jsonl = config_dict.get("progress_jsonl")
        if progress_jsonl is not None:
            config_dict["progress_jsonl"] = Path(progress_jsonl)
        config = stage1.Stage1Config(**config_dict)
        x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
        x_std = np.asarray(checkpoint["x_std"], dtype=np.float64)
        z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
        z_std = np.asarray(checkpoint["z_std"], dtype=np.float64)
        model = stage1.make_model(
            str(checkpoint["family"]),
            config,
            x_dim=int(x_mean.shape[0]),
            z_dim=int(z_mean.shape[0]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()
        members.append(
            {
                "model": model,
                "x_mean": x_mean,
                "x_std": x_std,
                "z_mean": z_mean,
                "z_std": z_std,
                "two_exp_target": str(
                    checkpoint.get(
                        "two_exp_target",
                        member.get("two_exp_target", source_two_exp_target),
                    )
                ),
                "two_exp_target_mode": str(checkpoint.get("two_exp_target_mode", "direct")),
                "source_summary": str(summary_path),
                "summary": member.get("member_summary", {}),
                "summary_json": member.get("summary_json"),
                "model_pt": str(model_path),
            }
        )
    return members


def load_population_members_from_summaries(
    summary_paths: list[Path],
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    seen: set[str] = set()
    for summary_path in summary_paths:
        for member in load_population_members(summary_path, device=device):
            key = str(resolve_existing_path(str(member["model_pt"]), base_dir=Path.cwd()))
            if key in seen:
                continue
            seen.add(key)
            members.append(member)
    if not members:
        raise ValueError("No population members loaded for stacking.")
    return members


def sample_population_for_stacking(
    *,
    model_name: str,
    n: int,
    seed: int,
    two_exp_target: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "two_exp":
        _x_raw, x_context, z_raw = sample_two_exp_population_raw(n=n, seed=seed)
        return x_context, z_raw
    if model_name == "label_switch":
        _x_raw, x_context, z_sorted = sample_label_switch_population_raw(n=n, seed=seed)
        return x_context, z_sorted
    return sample_population(model=model_name, n=n, seed=seed, two_exp_target=two_exp_target)


def train_stacking_gate(
    *,
    model_name: str,
    members: list[dict[str, object]],
    calibration_examples: int,
    calibration_seed: int,
    validation_fraction: float,
    batch_size: int,
    eval_batch_size: int,
    hidden_dim: int,
    hidden_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    two_exp_target: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if len(members) < 2:
        raise ValueError("Stacking needs at least two frozen members.")
    data_start = time.perf_counter()
    x_context, z_raw = sample_population_for_stacking(
        model_name=model_name,
        n=calibration_examples,
        seed=calibration_seed,
        two_exp_target=two_exp_target,
    )
    incompatible = [
        {
            "model_pt": str(member.get("model_pt", "")),
            "x_dim": int(np.asarray(member["x_mean"]).shape[0]),
        }
        for member in members
        if int(np.asarray(member["x_mean"]).shape[0]) != int(x_context.shape[1])
    ]
    if incompatible:
        raise ValueError(
            f"Stacking context dimension mismatch: calibration x_dim={x_context.shape[1]}, "
            f"incompatible_members={incompatible}"
        )
    x_mean = x_context.mean(axis=0).astype(np.float64)
    x_std = np.maximum(x_context.std(axis=0), 1e-6).astype(np.float64)
    x_gate = standardize(x_context, x_mean, x_std).astype(np.float32)
    member_log_probs = evaluate_member_log_probs(
        model_name=model_name,
        members=members,
        x_context=x_context,
        z_raw=z_raw,
        default_two_exp_target=two_exp_target,
        batch_size=eval_batch_size,
        device=device,
    ).astype(np.float32)
    data_seconds = time.perf_counter() - data_start

    n = x_gate.shape[0]
    val_count = int(round(n * float(validation_fraction)))
    val_count = min(max(1, val_count), max(1, n - 1))
    rng = np.random.default_rng(calibration_seed + 17)
    order = rng.permutation(n)
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    if train_idx.size == 0:
        raise ValueError("Stacking calibration split left no training examples.")

    gate = stage1.make_mlp(x_gate.shape[1], len(members), hidden_dim, hidden_layers).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_gate[train_idx]),
            torch.from_numpy(member_log_probs[train_idx]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(calibration_seed + 23),
    )
    val_x = torch.from_numpy(x_gate[val_idx]).to(device)
    val_log_probs = torch.from_numpy(member_log_probs[val_idx]).to(device)

    def batch_loss(batch_x: torch.Tensor, batch_log_probs: torch.Tensor) -> torch.Tensor:
        logits = gate(batch_x)
        return -torch.logsumexp(torch.log_softmax(logits, dim=-1) + batch_log_probs, dim=-1).mean()

    history = {"train_nll": [], "val_nll": [], "epoch_seconds": []}
    best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        gate.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_log_probs in train_loader:
            batch_x = batch_x.to(device)
            batch_log_probs = batch_log_probs.to(device)
            loss = batch_loss(batch_x, batch_log_probs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(batch_x.shape[0])
            train_count += int(batch_x.shape[0])
        gate.eval()
        with torch.no_grad():
            val_loss = float(batch_loss(val_x, val_log_probs).detach().cpu())
        train_loss = train_loss_sum / max(train_count, 1)
        history["train_nll"].append(float(train_loss))
        history["val_nll"].append(float(val_loss))
        history["epoch_seconds"].append(float(time.perf_counter() - epoch_start))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        logits = gate(torch.from_numpy(x_gate).to(device))
        weights = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    metadata = {
        "calibration_examples": int(calibration_examples),
        "calibration_seed": int(calibration_seed),
        "validation_fraction": float(validation_fraction),
        "train_examples": int(train_idx.size),
        "validation_examples": int(val_idx.size),
        "hidden_dim": int(hidden_dim),
        "hidden_layers": int(hidden_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "best_epoch": int(best_epoch),
        "best_val_nll": float(best_val),
        "final_train_nll": float(history["train_nll"][-1]),
        "final_val_nll": float(history["val_nll"][-1]),
        "data_seconds": float(data_seconds),
        "training_seconds": float(time.perf_counter() - start_time),
        "x_mean": x_mean,
        "x_std": x_std,
        "mean_weights": weights.mean(axis=0),
        "q05_weights": np.quantile(weights, 0.05, axis=0),
        "q95_weights": np.quantile(weights, 0.95, axis=0),
        "mean_entropy": float(np.mean(entropy)),
        "max_weight_mean": float(np.mean(np.max(weights, axis=1))),
        "history": history,
    }
    return gate, metadata


def train_member(
    *,
    args: argparse.Namespace,
    seed: int,
    member_index: int,
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    member_dir = output_root / f"member_{member_index:02d}_seed{seed}"
    results_dir = member_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = results_dir / "training_progress.jsonl"

    data_start = time.perf_counter()
    train_w = None
    if str(args.model) == "two_exp":
        _train_x_raw, train_x, train_z_raw = sample_two_exp_population_raw(n=int(args.train_simulations), seed=seed)
        _val_x_raw, val_x, val_z_raw = sample_two_exp_population_raw(n=int(args.val_simulations), seed=seed + 1)
        train_z = two_exp_model_target_from_raw(
            train_z_raw,
            train_x,
            target=str(args.two_exp_target),
            mode=str(args.two_exp_target_mode),
        )
        val_z = two_exp_model_target_from_raw(
            val_z_raw,
            val_x,
            target=str(args.two_exp_target),
            mode=str(args.two_exp_target_mode),
        )
        train_w = two_exp_loss_weights(
            train_z_raw,
            mode=str(args.loss_weight_mode),
            tail_weight=float(args.loss_tail_weight),
            tail_quantile=float(args.loss_tail_quantile),
        )
    else:
        train_x, train_z = sample_population(
            model=args.model,
            n=int(args.train_simulations),
            seed=seed,
            two_exp_target=str(args.two_exp_target),
        )
        val_x, val_z = sample_population(
            model=args.model,
            n=int(args.val_simulations),
            seed=seed + 1,
            two_exp_target=str(args.two_exp_target),
        )
    x_mean = train_x.mean(axis=0).astype(np.float64)
    x_std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float64)
    z_mean = train_z.mean(axis=0).astype(np.float64)
    z_std = np.maximum(train_z.std(axis=0), 1e-6).astype(np.float64)
    train_x_std = standardize(train_x, x_mean, x_std)
    train_z_std = standardize(train_z, z_mean, z_std)
    val_x_std = standardize(val_x, x_mean, x_std)
    val_z_std = standardize(val_z, z_mean, z_std)
    weights_metadata = loss_weight_summary(
        train_w,
        mode=str(args.loss_weight_mode),
        tail_weight=float(args.loss_tail_weight),
        tail_quantile=float(args.loss_tail_quantile),
    )
    data_seconds = time.perf_counter() - data_start

    config = replace(
        make_config(args, seed=seed, train_simulations=int(args.train_simulations)),
        progress_jsonl=progress_jsonl,
        progress_nll_offset=float(np.log(z_std).sum()),
    )
    train_tensors = [torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)]
    if train_w is not None:
        train_tensors.append(torch.from_numpy(train_w.astype(np.float32)))
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=int(args.batch_size),
        shuffle=str(args.batching_mode) == "dataloader",
        generator=torch.Generator(device="cpu").manual_seed(seed + 2),
    )
    print(
        f"{args.model} member {member_index} seed={seed} train={args.train_simulations} "
        f"family={args.family} x_dim={train_x_std.shape[1]} z_dim={train_z_std.shape[1]} "
        f"batches={len(train_loader)} device={device}",
        flush=True,
    )
    model, metrics = stage1.train_one_model(
        family=str(args.family),
        config=config,
        train_loader=train_loader,
        val_x=torch.from_numpy(val_x_std),
        val_z=torch.from_numpy(val_z_std),
        device=device,
        x_dim=train_x_std.shape[1],
        z_dim=train_z_std.shape[1],
    )
    model_path = results_dir / f"{args.model}_population_{args.family}_model.pt"
    checkpoint = {
        "family": str(args.family),
        "state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "config": asdict(config),
        "target": population_target_description(str(args.model), two_exp_target=str(args.two_exp_target)),
        "two_exp_target": str(args.two_exp_target),
        "two_exp_target_mode": str(args.two_exp_target_mode),
        "loss_weight": weights_metadata,
        "runtime": runtime_metadata(),
    }
    torch.save(checkpoint, model_path)
    z_log_det = float(np.log(z_std).sum())
    summary = {
        "seed": int(seed),
        "member_index": int(member_index),
        "model_pt": str(model_path),
        "data_seconds": float(data_seconds),
        "model_parameters": int(sum(param.numel() for param in model.parameters())),
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "z_log_det": z_log_det,
        "two_exp_target_mode": str(args.two_exp_target_mode) if str(args.model) == "two_exp" else None,
        "best_val_nll_standardized": float(metrics["best_val_nll"]),
        "best_val_nll_target_units": float(metrics["best_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["best_val_nll"]))
        else None,
        "final_train_nll_standardized": float(metrics["final_train_nll"]),
        "final_train_nll_target_units": float(metrics["final_train_nll"] + z_log_det),
        "final_val_nll_standardized": float(metrics["final_val_nll"]),
        "final_val_nll_target_units": float(metrics["final_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["final_val_nll"]))
        else None,
        "epochs_completed": int(metrics["epochs_completed"]),
        "optimizer_steps": int(metrics["optimizer_steps"]),
        "training_seconds": float(metrics["training_seconds"]),
        "history": metrics["history"],
        "config": asdict(config),
        "loss_weight": weights_metadata,
    }
    if args.model == "sign":
        summary["best_val_nll_folded_units"] = summary["best_val_nll_target_units"]
        summary["final_train_nll_folded_units"] = summary["final_train_nll_target_units"]
        summary["final_val_nll_folded_units"] = summary["final_val_nll_target_units"]
    summary_path = results_dir / f"{args.model}_population_member_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "two_exp_target_mode": str(args.two_exp_target_mode) if str(args.model) == "two_exp" else "direct",
        "summary": summary,
        "summary_json": str(summary_path),
        "model_pt": str(model_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full-prior stress-model population NPE with the single-decay Flow2 recipe."
    )
    parser.add_argument("--model", choices=("sign", "banana", "label_switch", "linear6", "two_exp"), default="sign")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--evaluate-summary", type=Path, default=None)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--stack-summaries", type=Path, nargs="+", default=None)
    parser.add_argument("--stack-calibration-examples", type=int, default=32_768)
    parser.add_argument("--stack-calibration-seed", type=int, default=20260713)
    parser.add_argument("--stack-validation-fraction", type=float, default=0.25)
    parser.add_argument("--stack-epochs", type=int, default=200)
    parser.add_argument("--stack-batch-size", type=int, default=1024)
    parser.add_argument("--stack-hidden-dim", type=int, default=32)
    parser.add_argument("--stack-hidden-layers", type=int, default=1)
    parser.add_argument("--stack-learning-rate", type=float, default=0.01)
    parser.add_argument("--stack-weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=parse_int_list, default=(20260901, 20260902, 20260903, 20260904))
    parser.add_argument("--train-simulations", type=int, default=2_048_000)
    parser.add_argument("--val-simulations", type=int, default=65_536)
    parser.add_argument("--validation-examples", type=int, default=1_000_000)
    parser.add_argument("--validation-seed", type=int, default=20260705)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.00325)
    parser.add_argument("--weight-decay", type=float, default=0.0002)
    parser.add_argument("--hidden-dim", type=int, default=80)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--family", choices=stage1.FAMILIES, default="spline_flow")
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--spline-bins", type=int, default=8)
    parser.add_argument("--flow-activation", choices=stage1.FLOW_ACTIVATIONS, default="relu")
    parser.add_argument("--flow-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-randperm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-passes", type=int, default=0)
    parser.add_argument("--flow-kind", choices=stage1.ZUKO_FLOW_KINDS, default="nsf")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine_epoch", "cosine_step", "one_cycle"), default="cosine_step")
    parser.add_argument("--lr-eta-min", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-decay-epochs", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--validation-every-epochs", type=int, default=1)
    parser.add_argument("--skip-training-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", choices=("none", "default", "reduce_overhead"), default="none")
    parser.add_argument("--grad-clip-norm", type=float, default=20.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--batching-mode", choices=("dataloader", "pre_shuffle", "sequential"), default="pre_shuffle")
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--loss-weight-mode", choices=POPULATION_LOSS_WEIGHT_MODES, default="none")
    parser.add_argument("--loss-tail-weight", type=float, default=3.0)
    parser.add_argument("--loss-tail-quantile", type=float, default=0.8)
    parser.add_argument(
        "--target-transform",
        choices=("none", "linear_residual", "fit_summary_residual"),
        default="none",
    )
    parser.add_argument("--target-ridge", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=65_536)
    parser.add_argument("--floor-only", action="store_true")
    parser.add_argument("--banana-quadrature-order", type=int, default=64)
    parser.add_argument("--linear6-quadrature-order", type=int, default=64)
    parser.add_argument("--label-importance-samples", type=int, default=4096)
    parser.add_argument("--label-importance-seed", type=int, default=20260719)
    parser.add_argument("--label-importance-batch-size", type=int, default=64)
    parser.add_argument("--label-prior-mixture", type=float, default=0.03)
    parser.add_argument("--label-proposal-inflation", type=float, default=2.0)
    parser.add_argument("--two-exp-target", choices=TWO_EXP_TARGETS, default="amplitude_sum_delta")
    parser.add_argument("--two-exp-target-mode", choices=TWO_EXP_TARGET_MODES, default="direct")
    parser.add_argument("--two-exp-floor-method", choices=("importance", "smc"), default="importance")
    parser.add_argument("--two-exp-importance-samples", type=int, default=4096)
    parser.add_argument("--two-exp-importance-seed", type=int, default=20260723)
    parser.add_argument("--two-exp-importance-batch-size", type=int, default=16)
    parser.add_argument("--two-exp-prior-mixture", type=float, default=0.02)
    parser.add_argument("--two-exp-proposal-inflation", type=float, default=1.0)
    parser.add_argument("--two-exp-smc-particles", type=int, default=2048)
    parser.add_argument("--two-exp-smc-beta-steps", type=int, default=48)
    parser.add_argument("--two-exp-smc-mh-steps", type=int, default=1)
    parser.add_argument("--two-exp-smc-seed", type=int, default=20260729)
    parser.add_argument("--two-exp-smc-batch-size", type=int, default=4)
    parser.add_argument("--two-exp-smc-step-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.loss_weight_mode) != "none" and str(args.model) != "two_exp":
        raise ValueError("--loss-weight-mode is currently only supported for --model two_exp.")
    if args.output_root is not None:
        output_root = args.output_root
    elif args.evaluate_summary is not None:
        output_root = args.evaluate_summary.parent.parent
    elif args.stack_summaries is not None:
        output_root = default_output_root(str(args.model)).parent / "26_learned_stack_full_prior"
    else:
        output_root = default_output_root(str(args.model))
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    if args.stack_summaries is not None:
        members = load_population_members_from_summaries(list(args.stack_summaries), device=device)
        gate, stack_training = train_stacking_gate(
            model_name=str(args.model),
            members=members,
            calibration_examples=int(args.stack_calibration_examples),
            calibration_seed=int(args.stack_calibration_seed),
            validation_fraction=float(args.stack_validation_fraction),
            batch_size=int(args.stack_batch_size),
            eval_batch_size=int(args.eval_batch_size),
            hidden_dim=int(args.stack_hidden_dim),
            hidden_layers=int(args.stack_hidden_layers),
            epochs=int(args.stack_epochs),
            learning_rate=float(args.stack_learning_rate),
            weight_decay=float(args.stack_weight_decay),
            two_exp_target=str(args.two_exp_target),
            device=device,
        )
        gate_path = results_dir / f"{args.model}_population_stacking_gate.pt"
        torch.save(
            {
                "state_dict": gate.state_dict(),
                "x_mean": np.asarray(stack_training["x_mean"], dtype=np.float64),
                "x_std": np.asarray(stack_training["x_std"], dtype=np.float64),
                "member_count": len(members),
                "model_name": str(args.model),
                "two_exp_target": str(args.two_exp_target),
                "training": stack_training,
                "runtime": runtime_metadata(),
            },
            gate_path,
        )
        diagnostics_output = args.diagnostics_output
        if diagnostics_output is None and str(args.model) == "two_exp":
            diagnostics_output = results_dir / f"{args.model}_population_stacked_gap_diagnostics.json"
        evaluation = evaluate_population_nll(
            model_name=str(args.model),
            members=members,
            validation_examples=int(args.validation_examples),
            validation_seed=int(args.validation_seed),
            batch_size=int(args.eval_batch_size),
            device=device,
            linear6_quadrature_order=int(args.linear6_quadrature_order),
            banana_quadrature_order=int(args.banana_quadrature_order),
            label_importance_samples=int(args.label_importance_samples),
            label_importance_seed=int(args.label_importance_seed),
            label_importance_batch_size=int(args.label_importance_batch_size),
            label_prior_mixture=float(args.label_prior_mixture),
            label_proposal_inflation=float(args.label_proposal_inflation),
            two_exp_target=str(args.two_exp_target),
            two_exp_floor_method=str(args.two_exp_floor_method),
            two_exp_importance_samples=int(args.two_exp_importance_samples),
            two_exp_importance_seed=int(args.two_exp_importance_seed),
            two_exp_importance_batch_size=int(args.two_exp_importance_batch_size),
            two_exp_prior_mixture=float(args.two_exp_prior_mixture),
            two_exp_proposal_inflation=float(args.two_exp_proposal_inflation),
            two_exp_smc_particles=int(args.two_exp_smc_particles),
            two_exp_smc_beta_steps=int(args.two_exp_smc_beta_steps),
            two_exp_smc_mh_steps=int(args.two_exp_smc_mh_steps),
            two_exp_smc_seed=int(args.two_exp_smc_seed),
            two_exp_smc_batch_size=int(args.two_exp_smc_batch_size),
            two_exp_smc_step_scale=float(args.two_exp_smc_step_scale),
            diagnostics_output=diagnostics_output,
            stacking_gate=gate,
            stacking_x_mean=np.asarray(stack_training["x_mean"], dtype=np.float64),
            stacking_x_std=np.asarray(stack_training["x_std"], dtype=np.float64),
        )
        summary = {
            "kind": f"{args.model}_population_learned_stacking",
            "description": f"Learned x-dependent stacking over frozen full-prior {args.model} population NPE members.",
            "target": population_target_description(str(args.model), two_exp_target=str(args.two_exp_target)),
            "device": str(device),
            "source_summaries": [str(path) for path in args.stack_summaries],
            "gate_pt": str(gate_path),
            "recipe": {
                "member_count": len(members),
                "validation_examples": int(args.validation_examples),
                "validation_seed": int(args.validation_seed),
                "eval_batch_size": int(args.eval_batch_size),
                "stack_calibration_examples": int(args.stack_calibration_examples),
                "stack_calibration_seed": int(args.stack_calibration_seed),
                "stack_validation_fraction": float(args.stack_validation_fraction),
                "stack_epochs": int(args.stack_epochs),
                "stack_batch_size": int(args.stack_batch_size),
                "stack_hidden_dim": int(args.stack_hidden_dim),
                "stack_hidden_layers": int(args.stack_hidden_layers),
                "stack_learning_rate": float(args.stack_learning_rate),
                "stack_weight_decay": float(args.stack_weight_decay),
                "two_exp_target": str(args.two_exp_target),
                "two_exp_floor_method": str(args.two_exp_floor_method),
                "two_exp_importance_samples": int(args.two_exp_importance_samples),
                "two_exp_importance_seed": int(args.two_exp_importance_seed),
                "two_exp_importance_batch_size": int(args.two_exp_importance_batch_size),
                "two_exp_prior_mixture": float(args.two_exp_prior_mixture),
                "two_exp_proposal_inflation": float(args.two_exp_proposal_inflation),
            },
            "members": [
                {
                    "model_pt": str(member["model_pt"]),
                    "source_summary": str(member.get("source_summary", "")),
                    "two_exp_target": str(member.get("two_exp_target", "")),
                    "two_exp_target_mode": str(member.get("two_exp_target_mode", "")),
                }
                for member in members
            ],
            "stack_training": stack_training,
            "evaluation": evaluation,
            "runtime": runtime_metadata(),
        }
        summary_path = results_dir / f"{args.model}_population_stacked_evaluation_summary.json"
        summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        print(f"summary_json: {summary_path}", flush=True)
        return
    if args.evaluate_summary is not None:
        source_summary = json.loads(args.evaluate_summary.read_text(encoding="utf-8"))
        two_exp_target = str(source_summary.get("recipe", {}).get("two_exp_target", args.two_exp_target))
        diagnostics_output = args.diagnostics_output
        if diagnostics_output is None and str(args.model) == "two_exp":
            diagnostics_output = results_dir / f"{args.model}_population_gap_diagnostics.json"
        members = load_population_members(args.evaluate_summary, device=device)
        evaluation = evaluate_population_nll(
            model_name=str(args.model),
            members=members,
            validation_examples=int(args.validation_examples),
            validation_seed=int(args.validation_seed),
            batch_size=int(args.eval_batch_size),
            device=device,
            linear6_quadrature_order=int(args.linear6_quadrature_order),
            banana_quadrature_order=int(args.banana_quadrature_order),
            label_importance_samples=int(args.label_importance_samples),
            label_importance_seed=int(args.label_importance_seed),
            label_importance_batch_size=int(args.label_importance_batch_size),
            label_prior_mixture=float(args.label_prior_mixture),
            label_proposal_inflation=float(args.label_proposal_inflation),
            two_exp_target=two_exp_target,
            two_exp_floor_method=str(args.two_exp_floor_method),
            two_exp_importance_samples=int(args.two_exp_importance_samples),
            two_exp_importance_seed=int(args.two_exp_importance_seed),
            two_exp_importance_batch_size=int(args.two_exp_importance_batch_size),
            two_exp_prior_mixture=float(args.two_exp_prior_mixture),
            two_exp_proposal_inflation=float(args.two_exp_proposal_inflation),
            two_exp_smc_particles=int(args.two_exp_smc_particles),
            two_exp_smc_beta_steps=int(args.two_exp_smc_beta_steps),
            two_exp_smc_mh_steps=int(args.two_exp_smc_mh_steps),
            two_exp_smc_seed=int(args.two_exp_smc_seed),
            two_exp_smc_batch_size=int(args.two_exp_smc_batch_size),
            two_exp_smc_step_scale=float(args.two_exp_smc_step_scale),
            diagnostics_output=diagnostics_output,
        )
        summary = {
            "kind": f"{args.model}_population_existing_evaluation",
            "description": f"Evaluation-only full-prior {args.model} population NPE run.",
            "source_summary": str(args.evaluate_summary),
            "target": population_target_description(str(args.model), two_exp_target=two_exp_target),
            "device": str(device),
            "recipe": {
                "validation_examples": int(args.validation_examples),
                "validation_seed": int(args.validation_seed),
                "eval_batch_size": int(args.eval_batch_size),
                "two_exp_target": two_exp_target,
                "two_exp_floor_method": str(args.two_exp_floor_method),
                "two_exp_importance_samples": int(args.two_exp_importance_samples),
                "two_exp_importance_seed": int(args.two_exp_importance_seed),
                "two_exp_importance_batch_size": int(args.two_exp_importance_batch_size),
                "two_exp_prior_mixture": float(args.two_exp_prior_mixture),
                "two_exp_proposal_inflation": float(args.two_exp_proposal_inflation),
            },
            "evaluation": evaluation,
            "runtime": runtime_metadata(),
        }
        summary_path = results_dir / f"{args.model}_population_existing_evaluation_summary.json"
        summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        print(f"summary_json: {summary_path}", flush=True)
        return
    if bool(args.floor_only):
        floor = estimate_population_floor(
            model_name=str(args.model),
            validation_examples=int(args.validation_examples),
            validation_seed=int(args.validation_seed),
            batch_size=int(args.eval_batch_size),
            linear6_quadrature_order=int(args.linear6_quadrature_order),
            banana_quadrature_order=int(args.banana_quadrature_order),
            label_importance_samples=int(args.label_importance_samples),
            label_importance_seed=int(args.label_importance_seed),
            label_importance_batch_size=int(args.label_importance_batch_size),
            label_prior_mixture=float(args.label_prior_mixture),
            label_proposal_inflation=float(args.label_proposal_inflation),
            two_exp_target=str(args.two_exp_target),
            two_exp_floor_method=str(args.two_exp_floor_method),
            two_exp_importance_samples=int(args.two_exp_importance_samples),
            two_exp_importance_seed=int(args.two_exp_importance_seed),
            two_exp_importance_batch_size=int(args.two_exp_importance_batch_size),
            two_exp_prior_mixture=float(args.two_exp_prior_mixture),
            two_exp_proposal_inflation=float(args.two_exp_proposal_inflation),
            two_exp_smc_particles=int(args.two_exp_smc_particles),
            two_exp_smc_beta_steps=int(args.two_exp_smc_beta_steps),
            two_exp_smc_mh_steps=int(args.two_exp_smc_mh_steps),
            two_exp_smc_seed=int(args.two_exp_smc_seed),
            two_exp_smc_batch_size=int(args.two_exp_smc_batch_size),
            two_exp_smc_step_scale=float(args.two_exp_smc_step_scale),
        )
        summary = {
            "kind": f"{args.model}_population_entropy_floor",
            "description": f"Full-prior {args.model} population entropy-floor estimate.",
            "target": population_target_description(str(args.model), two_exp_target=str(args.two_exp_target)),
            "recipe": {
                "validation_examples": int(args.validation_examples),
                "validation_seed": int(args.validation_seed),
                "eval_batch_size": int(args.eval_batch_size),
                "banana_quadrature_order": int(args.banana_quadrature_order),
                "linear6_quadrature_order": int(args.linear6_quadrature_order),
                "two_exp_target": str(args.two_exp_target),
                "label_importance_samples": int(args.label_importance_samples),
                "label_importance_seed": int(args.label_importance_seed),
                "label_importance_batch_size": int(args.label_importance_batch_size),
                "label_prior_mixture": float(args.label_prior_mixture),
                "label_proposal_inflation": float(args.label_proposal_inflation),
                "two_exp_floor_method": str(args.two_exp_floor_method),
                "two_exp_importance_samples": int(args.two_exp_importance_samples),
                "two_exp_importance_seed": int(args.two_exp_importance_seed),
                "two_exp_importance_batch_size": int(args.two_exp_importance_batch_size),
                "two_exp_prior_mixture": float(args.two_exp_prior_mixture),
                "two_exp_proposal_inflation": float(args.two_exp_proposal_inflation),
                "two_exp_smc_particles": int(args.two_exp_smc_particles),
                "two_exp_smc_beta_steps": int(args.two_exp_smc_beta_steps),
                "two_exp_smc_mh_steps": int(args.two_exp_smc_mh_steps),
                "two_exp_smc_seed": int(args.two_exp_smc_seed),
                "two_exp_smc_batch_size": int(args.two_exp_smc_batch_size),
                "two_exp_smc_step_scale": float(args.two_exp_smc_step_scale),
            },
            "evaluation": floor,
            "runtime": runtime_metadata(),
        }
        summary_path = results_dir / f"{args.model}_population_floor_summary.json"
        summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        print(f"summary_json: {summary_path}", flush=True)
        return
    started = time.perf_counter()
    members = []
    for member_index, seed in enumerate(args.seeds, start=1):
        members.append(
            train_member(
                args=args,
                seed=int(seed),
                member_index=member_index,
                device=device,
                output_root=output_root,
            )
        )
    evaluation = evaluate_population_nll(
        model_name=str(args.model),
        members=members,
        validation_examples=int(args.validation_examples),
        validation_seed=int(args.validation_seed),
        batch_size=int(args.eval_batch_size),
        device=device,
        linear6_quadrature_order=int(args.linear6_quadrature_order),
        banana_quadrature_order=int(args.banana_quadrature_order),
        label_importance_samples=int(args.label_importance_samples),
        label_importance_seed=int(args.label_importance_seed),
        label_importance_batch_size=int(args.label_importance_batch_size),
        label_prior_mixture=float(args.label_prior_mixture),
        label_proposal_inflation=float(args.label_proposal_inflation),
        two_exp_target=str(args.two_exp_target),
        two_exp_floor_method=str(args.two_exp_floor_method),
        two_exp_importance_samples=int(args.two_exp_importance_samples),
        two_exp_importance_seed=int(args.two_exp_importance_seed),
        two_exp_importance_batch_size=int(args.two_exp_importance_batch_size),
        two_exp_prior_mixture=float(args.two_exp_prior_mixture),
        two_exp_proposal_inflation=float(args.two_exp_proposal_inflation),
        two_exp_smc_particles=int(args.two_exp_smc_particles),
        two_exp_smc_beta_steps=int(args.two_exp_smc_beta_steps),
        two_exp_smc_mh_steps=int(args.two_exp_smc_mh_steps),
        two_exp_smc_seed=int(args.two_exp_smc_seed),
        two_exp_smc_batch_size=int(args.two_exp_smc_batch_size),
        two_exp_smc_step_scale=float(args.two_exp_smc_step_scale),
        diagnostics_output=args.diagnostics_output,
    )
    summary = {
        "kind": population_kind(str(args.model)),
        "description": population_description(str(args.model), two_exp_target=str(args.two_exp_target)),
        "target": population_target_description(str(args.model), two_exp_target=str(args.two_exp_target)),
        "device": str(device),
        "wall_seconds": float(time.perf_counter() - started),
        "recipe": {
            "ensemble_size": len(members),
            "seeds": [int(seed) for seed in args.seeds],
            "train_simulations_per_member": int(args.train_simulations),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "family": str(args.family),
            "mdn_components": int(args.mdn_components),
            "flow_layers": int(args.flow_layers),
            "flow_kind": str(args.flow_kind),
            "flow_residual": bool(args.flow_residual),
            "flow_randperm": bool(args.flow_randperm),
            "spline_bins": int(args.spline_bins),
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
            "lr_schedule": str(args.lr_schedule),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "batching_mode": str(args.batching_mode),
            "loss_weight_mode": str(args.loss_weight_mode),
            "loss_tail_weight": float(args.loss_tail_weight),
            "loss_tail_quantile": float(args.loss_tail_quantile),
            "target_transform": str(args.target_transform),
            "target_ridge": float(args.target_ridge),
            "banana_quadrature_order": int(args.banana_quadrature_order),
            "linear6_quadrature_order": int(args.linear6_quadrature_order),
            "two_exp_target": str(args.two_exp_target),
            "two_exp_target_mode": str(args.two_exp_target_mode),
            "label_importance_samples": int(args.label_importance_samples),
            "label_importance_seed": int(args.label_importance_seed),
            "label_importance_batch_size": int(args.label_importance_batch_size),
            "label_prior_mixture": float(args.label_prior_mixture),
            "label_proposal_inflation": float(args.label_proposal_inflation),
            "two_exp_floor_method": str(args.two_exp_floor_method),
            "two_exp_importance_samples": int(args.two_exp_importance_samples),
            "two_exp_importance_seed": int(args.two_exp_importance_seed),
            "two_exp_importance_batch_size": int(args.two_exp_importance_batch_size),
            "two_exp_prior_mixture": float(args.two_exp_prior_mixture),
            "two_exp_proposal_inflation": float(args.two_exp_proposal_inflation),
            "two_exp_smc_particles": int(args.two_exp_smc_particles),
            "two_exp_smc_beta_steps": int(args.two_exp_smc_beta_steps),
            "two_exp_smc_mh_steps": int(args.two_exp_smc_mh_steps),
            "two_exp_smc_seed": int(args.two_exp_smc_seed),
            "two_exp_smc_batch_size": int(args.two_exp_smc_batch_size),
            "two_exp_smc_step_scale": float(args.two_exp_smc_step_scale),
        },
        "members": [
            {
                "summary_json": member["summary_json"],
                "model_pt": member["model_pt"],
                "member_summary": member["summary"],
            }
            for member in members
        ],
        "evaluation": evaluation,
    }
    summary_path = results_dir / f"{args.model}_population_ensemble_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
