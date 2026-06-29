from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import artifact_paths as ap

import corner
import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from compare_decay_samplers import build_grid_reference, load_samples, subsample
from corner_truth import overplot_true_values, true_theta_legend_handle
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD
from npe_stage1_decay import posterior_predictive_band, sample_grid_reference
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOG_2PI = math.log(2.0 * math.pi)

COLORS = {
    "grid_reference": "#172033",
    "prior_abc": "#2f6fbb",
    "prior_abc_regression": "#6aa6d8",
    "proposal_abc": "#c06f2d",
    "proposal_abc_regression": "#e0a15f",
    "smc_abc": "#3f8f5f",
    "smc_abc_regression": "#79b985",
}

PROPOSAL_SOURCES = {
    "snpe_diag": {
        "path": ap.SNPE_SEQUENTIAL_GAUSSIANS_RESULTS / "snpe_sequential_samples.npz",
        "key": "z_final_corrected_diag_gaussian",
        "label": "SNPE diagonal proposal",
    },
    "snpe_mdn": {
        "path": ap.SNPE_SEQUENTIAL_MDN_RESULTS / "snpe_sequential_samples.npz",
        "key": "z_final_corrected_mdn",
        "label": "SNPE MDN proposal",
    },
    "stage1_mdn": {
        "path": ap.NPE_STAGE1_SCALED_RESULTS / "npe_stage1_samples.npz",
        "key": "z_samples_mdn",
        "label": "Broad NPE MDN proposal",
    },
    "local_mdn": {
        "path": ap.NPE_LOCAL_REGION_Q0005_MDN_20K_RESULTS / "npe_local_region_samples.npz",
        "key": "z_samples_mdn",
        "label": "Local NPE MDN proposal",
    },
}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(PROPOSAL_SOURCES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown proposal sources: {unknown}")
    return items


def prior_logpdf_z(z: np.ndarray) -> np.ndarray:
    mean = PRIOR_LOG_MEAN.numpy()[None, :]
    std = PRIOR_LOG_STD.numpy()[None, :]
    return (
        -0.5 * ((z - mean) / std) ** 2
        - np.log(std)
        - 0.5 * LOG_2PI
    ).sum(axis=1)


def sample_prior_z(n: int, rng: np.random.Generator) -> np.ndarray:
    mean = PRIOR_LOG_MEAN.numpy()
    std = PRIOR_LOG_STD.numpy()
    return rng.normal(mean[None, :], std[None, :], size=(n, 3))


def make_k_grid(k_points: int, k_min: float, k_max: float) -> np.ndarray:
    return np.exp(np.linspace(math.log(k_min), math.log(k_max), k_points))


def decay_fit_summaries(
    x: np.ndarray,
    t: np.ndarray,
    k_grid: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    basis = np.exp(-k_grid[:, None] * t[None, :])
    denom = np.sum(basis**2, axis=1)
    summaries = np.empty((x.shape[0], 3), dtype=np.float64)

    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        x_chunk = x[start:stop]
        numerator = x_chunk @ basis.T
        amplitude = np.maximum(numerator / denom[None, :], 1e-8)
        x_square = np.sum(x_chunk**2, axis=1, keepdims=True)
        sse = x_square - 2.0 * amplitude * numerator + amplitude**2 * denom[None, :]
        sse = np.maximum(sse, 1e-10)
        rows = np.arange(stop - start)

        best = np.argmin(sse, axis=1)
        best_log_k = np.log(k_grid[best])
        interior = (best > 0) & (best < len(k_grid) - 1)
        if np.any(interior):
            left = sse[rows[interior], best[interior] - 1]
            center = sse[rows[interior], best[interior]]
            right = sse[rows[interior], best[interior] + 1]
            spacing = math.log(k_grid[1]) - math.log(k_grid[0])
            curvature = left - 2.0 * center + right
            offset = 0.5 * (left - right) / np.maximum(curvature, 1e-12)
            offset = np.clip(offset, -1.0, 1.0)
            best_log_k[interior] = best_log_k[interior] + offset * spacing

        best_k = np.exp(best_log_k)
        best_basis = np.exp(-best_k[:, None] * t[None, :])
        best_denom = np.sum(best_basis**2, axis=1)
        best_numerator = np.sum(x_chunk * best_basis, axis=1)
        best_amplitude = np.maximum(best_numerator / best_denom, 1e-8)
        best_residual = x_chunk - best_amplitude[:, None] * best_basis
        best_sse = np.maximum(np.sum(best_residual**2, axis=1), 1e-10)
        best_sigma = np.sqrt(best_sse / x.shape[1])
        summaries[start:stop, 0] = np.log(np.maximum(best_amplitude, 1e-8))
        summaries[start:stop, 1] = np.log(np.maximum(best_k, 1e-8))
        summaries[start:stop, 2] = np.log(np.maximum(best_sigma, 1e-8))

    return summaries


def simulate_summaries(
    z: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    k_grid: np.ndarray,
    *,
    simulate_chunk_size: int,
    summary_chunk_size: int,
) -> np.ndarray:
    summaries = np.empty_like(z, dtype=np.float64)
    for start in range(0, z.shape[0], simulate_chunk_size):
        stop = min(start + simulate_chunk_size, z.shape[0])
        theta = np.exp(z[start:stop])
        mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
        x = mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)
        summaries[start:stop] = decay_fit_summaries(
            x,
            t,
            k_grid,
            chunk_size=summary_chunk_size,
        )
    return summaries


def summary_whitener(
    *,
    t: np.ndarray,
    y: np.ndarray,
    k_grid: np.ndarray,
    simulations: int,
    seed: int,
    simulate_chunk_size: int,
    summary_chunk_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = sample_prior_z(simulations, rng)
    summaries = simulate_summaries(
        z,
        t,
        rng,
        k_grid,
        simulate_chunk_size=simulate_chunk_size,
        summary_chunk_size=summary_chunk_size,
    )
    observed_summary = decay_fit_summaries(
        y[None, :],
        t,
        k_grid,
        chunk_size=1,
    )[0]
    cov = np.cov(summaries, rowvar=False)
    cov = cov + np.eye(3) * 1e-6
    chol = np.linalg.cholesky(cov)
    return {
        "observed_summary": observed_summary,
        "summary_mean": np.mean(summaries, axis=0),
        "summary_covariance": cov,
        "summary_cholesky": chol,
    }


def whitened_delta_and_distance(
    summaries: np.ndarray,
    observed_summary: np.ndarray,
    summary_cholesky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = summaries - observed_summary[None, :]
    whitened = np.linalg.solve(summary_cholesky, delta.T).T
    distances = np.linalg.norm(whitened, axis=1)
    return whitened, distances


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, sorted_values)


def weighted_summary(theta: np.ndarray, weights: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    weights = weights / np.sum(weights)
    for index, name in enumerate(PARAMETER_NAMES):
        values = theta[:, index]
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - mean) ** 2))
        q05, q16, q50, q84, q95 = weighted_quantile(
            values,
            weights,
            [0.05, 0.16, 0.50, 0.84, 0.95],
        )
        summary[name] = {
            "mean": mean,
            "sd": float(math.sqrt(max(variance, 0.0))),
            "q05": float(q05),
            "q16": float(q16),
            "median": float(q50),
            "q84": float(q84),
            "q95": float(q95),
        }
    return summary


def weighted_compare_to_reference(
    theta: np.ndarray,
    weights: np.ndarray,
    reference: dict[str, object],
) -> dict[str, object]:
    weights = weights / np.sum(weights)
    theta_grid = reference["theta_grid"]
    grid_weights = reference["weights"]
    ref_summary = reference["summary"]
    sample_summary = weighted_summary(theta, weights)
    metrics: dict[str, object] = {}
    normalized_values = []
    for index, name in enumerate(PARAMETER_NAMES):
        wasserstein = wasserstein_distance(
            theta[:, index],
            theta_grid[:, index],
            u_weights=weights,
            v_weights=grid_weights,
        )
        ref_sd = max(float(ref_summary[name]["sd"]), 1e-12)
        normalized = float(wasserstein / ref_sd)
        metrics[name] = {
            "wasserstein_to_grid": float(wasserstein),
            "wasserstein_to_grid_in_ref_sd": normalized,
            "median_error": float(sample_summary[name]["median"] - ref_summary[name]["median"]),
            "q05_error": float(sample_summary[name]["q05"] - ref_summary[name]["q05"]),
            "q95_error": float(sample_summary[name]["q95"] - ref_summary[name]["q95"]),
        }
        normalized_values.append(normalized)
    metrics["mean_normalized_wasserstein"] = {
        "value": float(np.mean(normalized_values)),
    }
    return metrics


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("All log weights are non-finite.")
    safe = np.where(finite, log_weights, -np.inf)
    return np.exp(safe - logsumexp(safe))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights**2))


def resample_weighted(
    theta: np.ndarray,
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = weights / np.sum(weights)
    index = rng.choice(theta.shape[0], size=n, replace=True, p=weights)
    return theta[index]


def fit_gaussian_proposal(z_samples: np.ndarray, inflation: float) -> dict[str, np.ndarray | float]:
    mean = np.mean(z_samples, axis=0)
    cov = np.cov(z_samples, rowvar=False)
    cov = cov * inflation**2 + np.eye(3) * 1e-6
    chol = np.linalg.cholesky(cov)
    inv = np.linalg.inv(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return {
        "mean": mean,
        "covariance": cov,
        "cholesky": chol,
        "inverse": inv,
        "log_det": float(log_det),
        "inflation": inflation,
    }


def sample_gaussian_proposal(
    proposal: dict[str, np.ndarray | float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(proposal["mean"])
    chol = np.asarray(proposal["cholesky"])
    return mean[None, :] + rng.normal(size=(n, 3)) @ chol.T


def gaussian_logpdf(
    z: np.ndarray,
    proposal: dict[str, np.ndarray | float],
) -> np.ndarray:
    mean = np.asarray(proposal["mean"])
    inv = np.asarray(proposal["inverse"])
    diff = z - mean[None, :]
    maha = np.sum((diff @ inv) * diff, axis=1)
    return -0.5 * (3 * LOG_2PI + float(proposal["log_det"]) + maha)


def weighted_covariance(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / np.sum(weights)
    mean = np.sum(weights[:, None] * z, axis=0)
    delta = z - mean[None, :]
    cov = (delta * weights[:, None]).T @ delta
    return cov + np.eye(z.shape[1]) * 1e-6


def gaussian_mixture_logpdf(
    z: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    covariance: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    inv = np.linalg.inv(covariance)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    log_weights = np.log(weights / np.sum(weights) + 1e-300)
    out = np.empty(z.shape[0], dtype=np.float64)
    const = -0.5 * (3 * LOG_2PI + log_det)
    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        diff = z[start:stop, None, :] - means[None, :, :]
        maha = np.einsum("bni,ij,bnj->bn", diff, inv, diff)
        out[start:stop] = logsumexp(log_weights[None, :] + const - 0.5 * maha, axis=1)
    return out


def regression_adjust_z(
    z: np.ndarray,
    whitened_delta: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    weights = weights / np.sum(weights)
    design = np.column_stack([np.ones(z.shape[0]), whitened_delta])
    sqrt_weights = np.sqrt(weights)
    xw = design * sqrt_weights[:, None]
    yw = z * sqrt_weights[:, None]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    adjustment = whitened_delta @ beta[1:, :]
    return z - adjustment


def evaluate_particles(
    *,
    name: str,
    label: str,
    z: np.ndarray,
    whitened_delta: np.ndarray,
    log_weights: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    simulations: int,
    runtime_seconds: float,
    reference: dict[str, object],
    target_wasserstein: float,
    posterior_samples: int,
    rng: np.random.Generator,
    regression_adjustment: bool,
    regression_ridge: float,
) -> tuple[dict[str, object], np.ndarray]:
    weights = normalize_log_weights(log_weights)
    z_eval = z
    if regression_adjustment:
        z_eval = regression_adjust_z(
            z,
            whitened_delta,
            weights,
            ridge=regression_ridge,
        )
    theta = np.exp(z_eval)
    metrics = weighted_compare_to_reference(theta, weights, reference)
    mean_w = metrics["mean_normalized_wasserstein"]["value"]
    resampled_theta = resample_weighted(theta, weights, posterior_samples, rng)
    result = {
        "name": name,
        "label": label,
        "particles": int(z.shape[0]),
        "simulations": int(simulations),
        "epsilon": float(epsilon),
        "distance_min": float(np.min(distances)),
        "distance_median": float(np.median(distances)),
        "distance_max": float(np.max(distances)),
        "ess": effective_sample_size(weights),
        "ess_fraction": effective_sample_size(weights) / z.shape[0],
        "runtime_seconds": float(runtime_seconds),
        "regression_adjustment": regression_adjustment,
        "faithful": bool(mean_w <= target_wasserstein),
        "target_wasserstein": target_wasserstein,
        "metrics": metrics,
        "posterior_summary": weighted_summary(theta, weights),
    }
    return result, resampled_theta


def evaluate_abc_pool(
    *,
    method_prefix: str,
    method_label: str,
    z: np.ndarray,
    whitened_delta: np.ndarray,
    distances: np.ndarray,
    base_log_weights: np.ndarray,
    threshold_quantiles: list[float],
    kernel_quantiles: list[float],
    simulations: int,
    runtime_seconds: float,
    reference: dict[str, object],
    target_wasserstein: float,
    posterior_samples: int,
    rng: np.random.Generator,
    regression_ridge: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    results: list[dict[str, object]] = []
    samples: dict[str, np.ndarray] = {}

    for quantile in threshold_quantiles:
        epsilon = float(np.quantile(distances, quantile))
        mask = distances <= epsilon
        if np.sum(mask) < 8:
            continue
        label = f"{method_label}, threshold q={quantile:g}"
        name = f"{method_prefix}_threshold_q{quantile:g}".replace(".", "p")
        result, sample = evaluate_particles(
            name=name,
            label=label,
            z=z[mask],
            whitened_delta=whitened_delta[mask],
            log_weights=base_log_weights[mask],
            distances=distances[mask],
            epsilon=epsilon,
            simulations=simulations,
            runtime_seconds=runtime_seconds,
            reference=reference,
            target_wasserstein=target_wasserstein,
            posterior_samples=posterior_samples,
            rng=rng,
            regression_adjustment=False,
            regression_ridge=regression_ridge,
        )
        results.append(result)
        samples[name] = sample

        reg_result, reg_sample = evaluate_particles(
            name=f"{name}_regression",
            label=f"{label}, regression adjusted",
            z=z[mask],
            whitened_delta=whitened_delta[mask],
            log_weights=base_log_weights[mask],
            distances=distances[mask],
            epsilon=epsilon,
            simulations=simulations,
            runtime_seconds=runtime_seconds,
            reference=reference,
            target_wasserstein=target_wasserstein,
            posterior_samples=posterior_samples,
            rng=rng,
            regression_adjustment=True,
            regression_ridge=regression_ridge,
        )
        results.append(reg_result)
        samples[reg_result["name"]] = reg_sample

    for quantile in kernel_quantiles:
        epsilon = float(np.quantile(distances, quantile))
        log_kernel = -0.5 * (distances / max(epsilon, 1e-12)) ** 2
        label = f"{method_label}, Gaussian kernel h=q{quantile:g}"
        name = f"{method_prefix}_kernel_q{quantile:g}".replace(".", "p")
        result, sample = evaluate_particles(
            name=name,
            label=label,
            z=z,
            whitened_delta=whitened_delta,
            log_weights=base_log_weights + log_kernel,
            distances=distances,
            epsilon=epsilon,
            simulations=simulations,
            runtime_seconds=runtime_seconds,
            reference=reference,
            target_wasserstein=target_wasserstein,
            posterior_samples=posterior_samples,
            rng=rng,
            regression_adjustment=False,
            regression_ridge=regression_ridge,
        )
        results.append(result)
        samples[name] = sample

        weights = normalize_log_weights(base_log_weights + log_kernel)
        keep = weights > (np.max(weights) * 1e-6)
        if np.sum(keep) >= 8:
            reg_result, reg_sample = evaluate_particles(
                name=f"{name}_regression",
                label=f"{label}, regression adjusted",
                z=z[keep],
                whitened_delta=whitened_delta[keep],
                log_weights=(base_log_weights + log_kernel)[keep],
                distances=distances[keep],
                epsilon=epsilon,
                simulations=simulations,
                runtime_seconds=runtime_seconds,
                reference=reference,
                target_wasserstein=target_wasserstein,
                posterior_samples=posterior_samples,
                rng=rng,
                regression_adjustment=True,
                regression_ridge=regression_ridge,
            )
            results.append(reg_result)
            samples[reg_result["name"]] = reg_sample

    return results, samples


def run_prior_abc(
    *,
    n: int,
    t: np.ndarray,
    rng: np.random.Generator,
    k_grid: np.ndarray,
    whitener: dict[str, np.ndarray],
    args: argparse.Namespace,
    reference: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    start = time.perf_counter()
    z = sample_prior_z(n, rng)
    summaries = simulate_summaries(
        z,
        t,
        rng,
        k_grid,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
    )
    whitened_delta, distances = whitened_delta_and_distance(
        summaries,
        whitener["observed_summary"],
        whitener["summary_cholesky"],
    )
    runtime = time.perf_counter() - start
    return evaluate_abc_pool(
        method_prefix="prior_abc",
        method_label="Prior ABC",
        z=z,
        whitened_delta=whitened_delta,
        distances=distances,
        base_log_weights=np.zeros(n, dtype=np.float64),
        threshold_quantiles=args.threshold_quantiles,
        kernel_quantiles=args.kernel_quantiles,
        simulations=n,
        runtime_seconds=runtime,
        reference=reference,
        target_wasserstein=args.target_wasserstein,
        posterior_samples=args.posterior_samples,
        rng=rng,
        regression_ridge=args.regression_ridge,
    )


def load_proposal_samples(source: str) -> np.ndarray:
    spec = PROPOSAL_SOURCES[source]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Proposal source {source} does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    key = spec["key"]
    if key not in data.files:
        raise KeyError(f"Proposal source {source} at {path} has no key {key}. Keys: {data.files}")
    return np.asarray(data[key], dtype=np.float64)


def run_proposal_abc(
    *,
    source: str,
    inflation: float,
    n: int,
    t: np.ndarray,
    rng: np.random.Generator,
    k_grid: np.ndarray,
    whitener: dict[str, np.ndarray],
    args: argparse.Namespace,
    reference: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, object]]:
    start = time.perf_counter()
    source_z = load_proposal_samples(source)
    proposal = fit_gaussian_proposal(source_z, inflation)
    z = sample_gaussian_proposal(proposal, n, rng)
    summaries = simulate_summaries(
        z,
        t,
        rng,
        k_grid,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
    )
    whitened_delta, distances = whitened_delta_and_distance(
        summaries,
        whitener["observed_summary"],
        whitener["summary_cholesky"],
    )
    base_log_weights = prior_logpdf_z(z) - gaussian_logpdf(z, proposal)
    runtime = time.perf_counter() - start
    prefix = f"proposal_abc_{source}_infl{inflation:g}".replace(".", "p")
    results, samples = evaluate_abc_pool(
        method_prefix=prefix,
        method_label=f"{PROPOSAL_SOURCES[source]['label']}, inflation {inflation:g}",
        z=z,
        whitened_delta=whitened_delta,
        distances=distances,
        base_log_weights=base_log_weights,
        threshold_quantiles=args.threshold_quantiles,
        kernel_quantiles=args.kernel_quantiles,
        simulations=n,
        runtime_seconds=runtime,
        reference=reference,
        target_wasserstein=args.target_wasserstein,
        posterior_samples=args.posterior_samples,
        rng=rng,
        regression_ridge=args.regression_ridge,
    )
    proposal_summary = {
        "source": source,
        "source_path": str(PROPOSAL_SOURCES[source]["path"]),
        "source_key": PROPOSAL_SOURCES[source]["key"],
        "inflation": inflation,
        "mean": np.asarray(proposal["mean"]).tolist(),
        "covariance": np.asarray(proposal["covariance"]).tolist(),
        "source_samples": int(source_z.shape[0]),
    }
    return results, samples, proposal_summary


def run_smc_abc(
    *,
    t: np.ndarray,
    rng: np.random.Generator,
    k_grid: np.ndarray,
    whitener: dict[str, np.ndarray],
    args: argparse.Namespace,
    reference: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    start = time.perf_counter()
    initial_z = sample_prior_z(args.smc_initial_simulations, rng)
    initial_summaries = simulate_summaries(
        initial_z,
        t,
        rng,
        k_grid,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
    )
    initial_delta, initial_distances = whitened_delta_and_distance(
        initial_summaries,
        whitener["observed_summary"],
        whitener["summary_cholesky"],
    )
    keep = np.argsort(initial_distances)[: args.smc_particles]
    particles = initial_z[keep]
    deltas = initial_delta[keep]
    distances = initial_distances[keep]
    weights = np.full(args.smc_particles, 1.0 / args.smc_particles)
    simulations = args.smc_initial_simulations

    all_results: list[dict[str, object]] = []
    all_samples: dict[str, np.ndarray] = {}

    for round_index in range(args.smc_rounds + 1):
        elapsed = time.perf_counter() - start
        result, sample = evaluate_particles(
            name=f"smc_abc_round{round_index}",
            label=f"SMC-ABC round {round_index}",
            z=particles,
            whitened_delta=deltas,
            log_weights=np.log(weights + 1e-300),
            distances=distances,
            epsilon=float(np.max(distances)),
            simulations=simulations,
            runtime_seconds=elapsed,
            reference=reference,
            target_wasserstein=args.target_wasserstein,
            posterior_samples=args.posterior_samples,
            rng=rng,
            regression_adjustment=False,
            regression_ridge=args.regression_ridge,
        )
        result["round"] = round_index
        result["acceptance_rate_last_round"] = None
        all_results.append(result)
        all_samples[result["name"]] = sample

        reg_result, reg_sample = evaluate_particles(
            name=f"smc_abc_round{round_index}_regression",
            label=f"SMC-ABC round {round_index}, regression adjusted",
            z=particles,
            whitened_delta=deltas,
            log_weights=np.log(weights + 1e-300),
            distances=distances,
            epsilon=float(np.max(distances)),
            simulations=simulations,
            runtime_seconds=elapsed,
            reference=reference,
            target_wasserstein=args.target_wasserstein,
            posterior_samples=args.posterior_samples,
            rng=rng,
            regression_adjustment=True,
            regression_ridge=args.regression_ridge,
        )
        reg_result["round"] = round_index
        reg_result["acceptance_rate_last_round"] = None
        all_results.append(reg_result)
        all_samples[reg_result["name"]] = reg_sample

        if round_index == args.smc_rounds:
            break

        epsilon = float(np.quantile(distances, args.smc_threshold_quantile))
        kernel_cov = weighted_covariance(particles, weights) * args.smc_kernel_scale**2
        kernel_cov = kernel_cov + np.eye(3) * 1e-6
        kernel_chol = np.linalg.cholesky(kernel_cov)
        accepted_z: list[np.ndarray] = []
        accepted_deltas: list[np.ndarray] = []
        accepted_distances: list[np.ndarray] = []
        attempts = 0

        while sum(item.shape[0] for item in accepted_z) < args.smc_particles:
            if attempts >= args.smc_max_simulations_per_round:
                break
            batch = min(args.smc_batch_size, args.smc_max_simulations_per_round - attempts)
            ancestor = rng.choice(args.smc_particles, size=batch, replace=True, p=weights)
            proposal_z = particles[ancestor] + rng.normal(size=(batch, 3)) @ kernel_chol.T
            summaries = simulate_summaries(
                proposal_z,
                t,
                rng,
                k_grid,
                simulate_chunk_size=args.simulate_chunk_size,
                summary_chunk_size=args.summary_chunk_size,
            )
            proposal_delta, proposal_distances = whitened_delta_and_distance(
                summaries,
                whitener["observed_summary"],
                whitener["summary_cholesky"],
            )
            mask = proposal_distances <= epsilon
            if np.any(mask):
                accepted_z.append(proposal_z[mask])
                accepted_deltas.append(proposal_delta[mask])
                accepted_distances.append(proposal_distances[mask])
            attempts += batch

        simulations += attempts
        if not accepted_z:
            break

        new_particles = np.vstack(accepted_z)[: args.smc_particles]
        new_deltas = np.vstack(accepted_deltas)[: args.smc_particles]
        new_distances = np.concatenate(accepted_distances)[: args.smc_particles]
        log_proposal = gaussian_mixture_logpdf(
            new_particles,
            particles,
            weights,
            kernel_cov,
            chunk_size=args.mixture_logpdf_chunk_size,
        )
        new_log_weights = prior_logpdf_z(new_particles) - log_proposal
        new_weights = normalize_log_weights(new_log_weights)

        particles = new_particles
        deltas = new_deltas
        distances = new_distances
        weights = new_weights

        for recent in all_results[-2:]:
            recent["next_round_epsilon"] = epsilon
            recent["next_round_attempted_simulations"] = attempts
            recent["next_round_accepted_particles"] = int(new_particles.shape[0])
            recent["next_round_acceptance_rate"] = float(new_particles.shape[0] / attempts)

    return all_results, all_samples


def plot_distance_curve(
    results: list[dict[str, object]],
    outfile: Path,
    target: float,
) -> None:
    ordered = sorted(
        results,
        key=lambda item: item["metrics"]["mean_normalized_wasserstein"]["value"],
    )
    labels = [item["name"] for item in ordered]
    values = [item["metrics"]["mean_normalized_wasserstein"]["value"] for item in ordered]
    ess_fraction = [item["ess_fraction"] for item in ordered]
    positions = np.arange(len(values))

    figure, ax = plt.subplots(figsize=(max(12, 0.42 * len(values)), 6.8))
    bars = ax.bar(positions, values, color="#607d8b", alpha=0.82)
    for bar, ess in zip(bars, ess_fraction):
        if ess < 0.05:
            bar.set_alpha(0.38)
    ax.axhline(target, color="#111827", linestyle="--", linewidth=1.8, label=f"target = {target:.3f}")
    ax.set_ylabel("mean normalized Wasserstein to grid")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_corner_overlay(
    *,
    samples_by_name: dict[str, np.ndarray],
    labels_by_name: dict[str, str],
    reference_samples: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
    max_samples: int,
) -> None:
    labels = [r"$A$", r"$k$", r"$\sigma$"]
    figure = corner.corner(
        subsample(reference_samples, max_samples, seed=501),
        labels=labels,
        color=COLORS["grid_reference"],
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.8},
        contour_kwargs={"linewidths": 1.5},
    )
    palette = ["#2f6fbb", "#c06f2d", "#3f8f5f", "#7a5cc2", "#b85c38"]
    handles = [
        plt.Line2D([0], [0], color=COLORS["grid_reference"], lw=2, label="Grid reference"),
        true_theta_legend_handle(),
    ]
    for index, (name, samples) in enumerate(samples_by_name.items()):
        color = palette[index % len(palette)]
        corner.corner(
            subsample(samples, max_samples, seed=600 + index),
            fig=figure,
            labels=labels,
            color=color,
            plot_datapoints=False,
            fill_contours=False,
            levels=(0.50, 0.90),
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
            contour_kwargs={"linewidths": 1.3},
        )
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=labels_by_name[name]))
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96), fontsize=9)
    figure.subplots_adjust(top=0.90)
    figure.suptitle("ABC-corrected posterior overlay", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictive_overlay(
    *,
    samples_by_name: dict[str, np.ndarray],
    labels_by_name: dict[str, str],
    t: np.ndarray,
    y: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
) -> None:
    t_grid = np.linspace(float(t.min()), float(t.max()), 220)
    true_mean = true_theta[0] * np.exp(-true_theta[1] * t_grid)
    figure, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(t, y, color="#172033", s=28, zorder=5, label="observed data")
    ax.plot(t_grid, true_mean, color="#172033", linestyle="--", linewidth=1.8, label="true mean")
    palette = ["#2f6fbb", "#c06f2d", "#3f8f5f", "#7a5cc2", "#b85c38"]
    for index, (name, samples) in enumerate(samples_by_name.items()):
        lower, median, upper = posterior_predictive_band(
            samples,
            t_grid,
            seed=700 + index,
            max_draws=900,
        )
        color = palette[index % len(palette)]
        ax.fill_between(t_grid, lower, upper, color=color, alpha=0.10)
        ax.plot(t_grid, median, color=color, linewidth=2.0, label=labels_by_name[name])
    ax.set_xlabel("time t")
    ax.set_ylabel("replicated observation y")
    ax.set_title("ABC-corrected posterior predictive overlay")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ABC-style repairs for NPE faithfulness on the decay model.")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.ABC_FAITHFULNESS_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.ABC_FAITHFULNESS_FIGURES)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--calibration-simulations", type=int, default=80_000)
    parser.add_argument("--prior-simulations", type=int, default=200_000)
    parser.add_argument("--proposal-simulations", type=int, default=200_000)
    parser.add_argument("--proposal-sources", type=parse_str_list, default=["snpe_diag", "snpe_mdn"])
    parser.add_argument("--proposal-inflations", type=parse_float_list, default=[1.5, 2.5])
    parser.add_argument("--threshold-quantiles", type=parse_float_list, default=[0.001, 0.0025, 0.005, 0.01])
    parser.add_argument("--kernel-quantiles", type=parse_float_list, default=[0.001, 0.0025, 0.005])
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    parser.add_argument("--max-corner-samples", type=int, default=25_000)
    parser.add_argument("--k-grid-points", type=int, default=220)
    parser.add_argument("--k-min", type=float, default=0.04)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--simulate-chunk-size", type=int, default=40_000)
    parser.add_argument("--summary-chunk-size", type=int, default=20_000)
    parser.add_argument("--regression-ridge", type=float, default=1e-4)
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--skip-proposal", action="store_true")
    parser.add_argument("--skip-smc", action="store_true")
    parser.add_argument("--smc-particles", type=int, default=2500)
    parser.add_argument("--smc-initial-simulations", type=int, default=150_000)
    parser.add_argument("--smc-rounds", type=int, default=5)
    parser.add_argument("--smc-threshold-quantile", type=float, default=0.50)
    parser.add_argument("--smc-kernel-scale", type=float, default=1.5)
    parser.add_argument("--smc-batch-size", type=int, default=8000)
    parser.add_argument("--smc-max-simulations-per-round", type=int, default=300_000)
    parser.add_argument("--mixture-logpdf-chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target_wasserstein, args.target_source, args.recommended_targets = resolve_target_wasserstein(
        args.target_wasserstein,
        summary_path=args.target_summary,
    )
    if args.target_summary is not None:
        args.target_summary = str(args.target_summary)
    total_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    t = mcmc["t"]
    y = mcmc["y"]
    true_theta = mcmc["true_theta"]
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])

    grid_start = time.perf_counter()
    reference = build_grid_reference(
        t=t,
        y=y,
        combined_z_samples=combined_z,
        true_theta=true_theta,
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    grid_seconds = time.perf_counter() - grid_start

    k_grid = make_k_grid(args.k_grid_points, args.k_min, args.k_max)
    calibration_start = time.perf_counter()
    whitener = summary_whitener(
        t=t,
        y=y,
        k_grid=k_grid,
        simulations=args.calibration_simulations,
        seed=args.seed + 10,
        simulate_chunk_size=args.simulate_chunk_size,
        summary_chunk_size=args.summary_chunk_size,
    )
    calibration_seconds = time.perf_counter() - calibration_start

    all_results: list[dict[str, object]] = []
    all_samples: dict[str, np.ndarray] = {}
    proposal_summaries: list[dict[str, object]] = []

    if not args.skip_prior:
        prior_results, prior_samples = run_prior_abc(
            n=args.prior_simulations,
            t=t,
            rng=rng,
            k_grid=k_grid,
            whitener=whitener,
            args=args,
            reference=reference,
        )
        all_results.extend(prior_results)
        all_samples.update(prior_samples)
        print(f"prior_abc_results: {len(prior_results)}")

    if not args.skip_proposal:
        for source in args.proposal_sources:
            for inflation in args.proposal_inflations:
                proposal_results, proposal_samples, proposal_summary = run_proposal_abc(
                    source=source,
                    inflation=inflation,
                    n=args.proposal_simulations,
                    t=t,
                    rng=rng,
                    k_grid=k_grid,
                    whitener=whitener,
                    args=args,
                    reference=reference,
                )
                all_results.extend(proposal_results)
                all_samples.update(proposal_samples)
                proposal_summaries.append(proposal_summary)
                best = min(
                    proposal_results,
                    key=lambda item: item["metrics"]["mean_normalized_wasserstein"]["value"],
                )
                best_w = best["metrics"]["mean_normalized_wasserstein"]["value"]
                print(f"proposal_abc {source} inflation={inflation:g} best_w={best_w:.5f}")

    if not args.skip_smc:
        smc_results, smc_samples = run_smc_abc(
            t=t,
            rng=rng,
            k_grid=k_grid,
            whitener=whitener,
            args=args,
            reference=reference,
        )
        all_results.extend(smc_results)
        all_samples.update(smc_samples)
        best = min(
            smc_results,
            key=lambda item: item["metrics"]["mean_normalized_wasserstein"]["value"],
        )
        best_w = best["metrics"]["mean_normalized_wasserstein"]["value"]
        print(f"smc_abc_results: {len(smc_results)}, best_w={best_w:.5f}")

    if not all_results:
        raise RuntimeError("No ABC results were produced.")

    sorted_results = sorted(
        all_results,
        key=lambda item: item["metrics"]["mean_normalized_wasserstein"]["value"],
    )
    best_result = sorted_results[0]
    best_names = [item["name"] for item in sorted_results[: min(5, len(sorted_results))]]
    best_samples = {name: all_samples[name] for name in best_names}
    labels_by_name = {
        item["name"]: item["label"]
        for item in all_results
    }

    reference_samples = sample_grid_reference(
        reference,
        n=args.posterior_samples,
        seed=args.seed + 99,
    )
    distance_curve = args.figure_dir / "abc_faithfulness_distance_curve.png"
    corner_overlay = args.figure_dir / "abc_faithfulness_corner_overlay.png"
    predictive_overlay = args.figure_dir / "abc_faithfulness_predictive_overlay.png"
    plot_distance_curve(sorted_results, distance_curve, args.target_wasserstein)
    plot_corner_overlay(
        samples_by_name=best_samples,
        labels_by_name=labels_by_name,
        reference_samples=reference_samples,
        true_theta=true_theta,
        outfile=corner_overlay,
        max_samples=args.max_corner_samples,
    )
    plot_predictive_overlay(
        samples_by_name=best_samples,
        labels_by_name=labels_by_name,
        t=t,
        y=y,
        true_theta=true_theta,
        outfile=predictive_overlay,
    )

    samples_path = args.output_dir / "abc_faithfulness_samples.npz"
    np.savez_compressed(
        samples_path,
        t=t,
        y=y,
        true_theta=true_theta,
        reference_theta_samples=reference_samples,
        **{f"theta_samples_{name}": samples for name, samples in best_samples.items()},
    )

    summary = {
        "config": {
            "target_wasserstein": args.target_wasserstein,
            "seed": args.seed,
            "calibration_simulations": args.calibration_simulations,
            "prior_simulations": args.prior_simulations,
            "proposal_simulations": args.proposal_simulations,
            "proposal_sources": args.proposal_sources,
            "proposal_inflations": args.proposal_inflations,
            "threshold_quantiles": args.threshold_quantiles,
            "kernel_quantiles": args.kernel_quantiles,
            "smc_particles": args.smc_particles,
            "smc_initial_simulations": args.smc_initial_simulations,
            "smc_rounds": args.smc_rounds,
            "smc_threshold_quantile": args.smc_threshold_quantile,
            "smc_kernel_scale": args.smc_kernel_scale,
            "k_grid_points": args.k_grid_points,
            "k_min": args.k_min,
            "k_max": args.k_max,
        },
        "reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "summary_statistic": {
            "type": "indirect_exponential_least_squares",
            "observed_summary_log_A_log_k_log_sigma": whitener["observed_summary"].tolist(),
            "prior_predictive_summary_mean": whitener["summary_mean"].tolist(),
            "prior_predictive_summary_covariance": whitener["summary_covariance"].tolist(),
        },
        "proposal_summaries": proposal_summaries,
        "best_result": best_result,
        "results": sorted_results,
        "outputs": {
            "summary_json": str(args.output_dir / "abc_faithfulness_summary.json"),
            "samples_npz": str(samples_path),
            "distance_curve": str(distance_curve),
            "corner_overlay": str(corner_overlay),
            "predictive_overlay": str(predictive_overlay),
        },
        "timing_seconds": {
            "grid_reference": grid_seconds,
            "summary_calibration": calibration_seconds,
            "total": time.perf_counter() - total_start,
        },
    }
    summary_json = args.output_dir / "abc_faithfulness_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_w = best_result["metrics"]["mean_normalized_wasserstein"]["value"]
    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_path}")
    print(f"distance_curve: {distance_curve}")
    print(f"corner_overlay: {corner_overlay}")
    print(f"predictive_overlay: {predictive_overlay}")
    print(f"best: {best_result['name']} W={best_w:.5f} faithful={best_result['faithful']}")


if __name__ == "__main__":
    main()
