from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcmc_decay_inference import PRIOR_LOG_MEAN, PRIOR_LOG_STD


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)
LOG_2PI = math.log(2.0 * math.pi)


def log_normal_density(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (z - mean[None, :]) / std[None, :]
    return -0.5 * np.sum(standardized * standardized, axis=1) - float(
        np.sum(np.log(std)) + 0.5 * z.shape[1] * LOG_2PI
    )


def make_grid(grid_size: int, sigma_width: float) -> tuple[np.ndarray, float]:
    mean = PRIOR_LOG_MEAN.detach().cpu().numpy().astype(np.float64)
    std = PRIOR_LOG_STD.detach().cpu().numpy().astype(np.float64)
    axes = [
        np.linspace(mean[index] - sigma_width * std[index], mean[index] + sigma_width * std[index], grid_size)
        for index in range(3)
    ]
    dz = [float(axis[1] - axis[0]) for axis in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([item.reshape(-1) for item in mesh], axis=1).astype(np.float64)
    return grid, float(np.log(dz[0] * dz[1] * dz[2]))


def log_likelihood(x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
    amplitude = np.exp(z[:, 0])
    decay_rate = np.exp(z[:, 1])
    sigma = np.exp(z[:, 2])
    phi = np.exp(-decay_rate[:, None] * t[None, :])
    mean = amplitude[:, None] * phi
    diff = x[:, None, :] - mean[None, :, :]
    sse = np.sum(diff * diff, axis=2)
    n_obs = x.shape[1]
    return -0.5 * (n_obs * LOG_2PI + 2.0 * n_obs * np.log(sigma)[None, :] + sse / (sigma[None, :] ** 2))


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.validation_cache)
    x_val = np.asarray(data["x_val"][: args.examples], dtype=np.float64)
    z_val = np.asarray(data["z_val"][: args.examples], dtype=np.float64)
    t = np.linspace(0.0, 6.0, x_val.shape[1], dtype=np.float64)
    grid, log_cell_volume = make_grid(int(args.grid_size), float(args.sigma_width))
    prior_mean = PRIOR_LOG_MEAN.detach().cpu().numpy().astype(np.float64)
    prior_std = PRIOR_LOG_STD.detach().cpu().numpy().astype(np.float64)
    log_prior_grid = log_normal_density(grid, prior_mean, prior_std)
    log_prior_true = log_normal_density(z_val, prior_mean, prior_std)

    amplitude = np.exp(grid[:, 0])
    decay_rate = np.exp(grid[:, 1])
    sigma = np.exp(grid[:, 2])
    phi = np.exp(-decay_rate[:, None] * t[None, :])
    phi2 = np.sum(phi * phi, axis=1)
    base = -0.5 * (x_val.shape[1] * LOG_2PI + 2.0 * x_val.shape[1] * np.log(sigma))
    grid_quadratic = amplitude * amplitude * phi2
    inv_sigma2 = 1.0 / np.square(sigma)

    nlls: list[float] = []
    log_marginals: list[float] = []
    start = time.perf_counter()
    for start_index in range(0, x_val.shape[0], int(args.chunk_size)):
        stop = min(start_index + int(args.chunk_size), x_val.shape[0])
        x = x_val[start_index:stop]
        x2 = np.sum(x * x, axis=1)
        x_phi = x @ phi.T
        sse = x2[:, None] - 2.0 * amplitude[None, :] * x_phi + grid_quadratic[None, :]
        log_like_grid = base[None, :] - 0.5 * sse * inv_sigma2[None, :]
        log_integrand = log_like_grid + log_prior_grid[None, :] + log_cell_volume
        max_log = np.max(log_integrand, axis=1)
        log_marginal = max_log + np.log(np.sum(np.exp(log_integrand - max_log[:, None]), axis=1))
        true_log_like = np.diag(log_likelihood(x, z_val[start_index:stop], t))
        posterior_log_prob = true_log_like + log_prior_true[start_index:stop] - log_marginal
        nlls.extend((-posterior_log_prob).tolist())
        log_marginals.extend(log_marginal.tolist())
    nll_array = np.asarray(nlls, dtype=np.float64)
    return {
        "examples": int(x_val.shape[0]),
        "grid_size_per_dim": int(args.grid_size),
        "grid_points": int(grid.shape[0]),
        "sigma_width": float(args.sigma_width),
        "bayes_nll_mean": float(np.mean(nll_array)),
        "bayes_nll_std_error": float(np.std(nll_array, ddof=1) / math.sqrt(max(1, nll_array.size))),
        "bayes_nll_quantiles": {
            "q01": float(np.quantile(nll_array, 0.01)),
            "q05": float(np.quantile(nll_array, 0.05)),
            "q50": float(np.quantile(nll_array, 0.50)),
            "q95": float(np.quantile(nll_array, 0.95)),
            "q99": float(np.quantile(nll_array, 0.99)),
        },
        "elapsed_seconds": float(time.perf_counter() - start),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Bayes-optimal decay posterior NLL by grid integration.")
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--sigma-width", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
