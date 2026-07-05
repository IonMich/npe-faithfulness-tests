from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp, pbdv, roots_hermitenorm


DEFAULT_OUTPUT = Path(
    "runs/00_shared_assets/readme_entropy/sign_bayes_entropy_hybrid_1m.json"
)
LOG_2PI = math.log(2.0 * math.pi)
LOG_2 = math.log(2.0)


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    return value


def normal_logpdf(value: np.ndarray, mean: np.ndarray | float, std: float) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized * standardized - math.log(std) - 0.5 * LOG_2PI


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


def make_standard_normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(int(order))
    log_weights = np.log(weights) - 0.5 * LOG_2PI
    return np.asarray(nodes, dtype=np.float64), np.asarray(log_weights, dtype=np.float64)


def sign_theta1_log_evidence(
    x1: np.ndarray,
    *,
    prior_std: float,
    noise_std: float,
    tail_switch: float,
    tail_order: int,
    chunk_size: int,
) -> np.ndarray:
    x1 = np.asarray(x1, dtype=np.float64)
    result = np.empty_like(x1)
    closed_mask = x1 <= tail_switch
    if np.any(closed_mask):
        # With y=theta1^2, p(x1) is a one-sided Gaussian/conjugate-prior
        # integral that reduces to a parabolic-cylinder function.
        x_closed = x1[closed_mask]
        nu = 0.5
        beta = 1.0 / (2.0 * noise_std * noise_std)
        gamma = 1.0 / (2.0 * prior_std * prior_std) - x_closed / (noise_std * noise_std)
        arg = gamma / math.sqrt(2.0 * beta)
        cylinder, _ = pbdv(-nu, arg)
        log_integral = (
            (-nu / 2.0) * math.log(2.0 * beta)
            + math.lgamma(nu)
            + gamma * gamma / (8.0 * beta)
            + np.log(cylinder)
        )
        result[closed_mask] = (
            -math.log(2.0 * math.pi * noise_std * prior_std)
            - x_closed * x_closed / (2.0 * noise_std * noise_std)
            + log_integral
        )

    tail_mask = ~closed_mask
    if np.any(tail_mask):
        # The closed form can overflow in the far positive tail. There the
        # equivalent expectation over y~N(x1, noise_std^2) is smooth.
        nodes, log_weights = make_standard_normal_quadrature(tail_order)
        x_tail = x1[tail_mask]
        tail_result = np.empty_like(x_tail)
        log_const = -math.log(prior_std) - 0.5 * LOG_2PI
        for start in range(0, x_tail.shape[0], chunk_size):
            stop = min(start + chunk_size, x_tail.shape[0])
            y = x_tail[start:stop, None] + noise_std * nodes[None, :]
            safe_y = np.where(y > 0.0, y, 1.0)
            terms = (
                log_weights[None, :]
                - 0.5 * np.log(safe_y)
                - safe_y / (2.0 * prior_std * prior_std)
            )
            terms = np.where(y > 0.0, terms, -np.inf)
            tail_result[start:stop] = log_const + logsumexp(terms, axis=1)
        result[tail_mask] = tail_result
    return result


def sign_theta1_nll(
    theta1: np.ndarray,
    x1: np.ndarray,
    *,
    prior_std: float,
    noise_std: float,
    tail_switch: float,
    tail_order: int,
    chunk_size: int,
) -> np.ndarray:
    log_evidence = sign_theta1_log_evidence(
        x1,
        prior_std=prior_std,
        noise_std=noise_std,
        tail_switch=tail_switch,
        tail_order=tail_order,
        chunk_size=chunk_size,
    )
    true_log_like = normal_logpdf(x1, theta1 * theta1, noise_std)
    true_log_prior = normal_logpdf(theta1, 0.0, prior_std)
    return -true_log_like - true_log_prior + log_evidence


def sign_theta2_nll(
    theta2: np.ndarray,
    x2: np.ndarray,
    *,
    prior_std: float,
    noise_std: float,
) -> tuple[np.ndarray, float]:
    prior_var = prior_std * prior_std
    noise_var = noise_std * noise_std
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_var)
    posterior_std = math.sqrt(posterior_var)
    posterior_mean = posterior_var * x2 / noise_var
    nll = -normal_logpdf(theta2, posterior_mean, posterior_std)
    entropy = 0.5 * math.log(2.0 * math.pi * math.e * posterior_var)
    return nll, entropy


def parse_orders(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    prior_std = 1.8
    theta1_noise_std = 0.22
    theta2_noise_std = 0.16
    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()

    theta1 = rng.normal(0.0, prior_std, size=args.examples)
    theta2 = rng.normal(0.0, prior_std, size=args.examples)
    x1 = theta1 * theta1 + rng.normal(0.0, theta1_noise_std, size=args.examples)
    x2 = theta2 + rng.normal(0.0, theta2_noise_std, size=args.examples)

    theta1_nll = sign_theta1_nll(
        theta1,
        x1,
        prior_std=prior_std,
        noise_std=theta1_noise_std,
        tail_switch=args.tail_switch,
        tail_order=args.tail_order,
        chunk_size=args.chunk_size,
    )
    theta2_nll, theta2_entropy = sign_theta2_nll(
        theta2,
        x2,
        prior_std=prior_std,
        noise_std=theta2_noise_std,
    )
    total_nll = theta1_nll + theta2_nll
    folded_nll = total_nll - LOG_2

    quadrature_checks: dict[str, Any] = {}
    check_count = min(args.check_examples, args.examples)
    if check_count > 0:
        for order in parse_orders(args.check_tail_orders):
            check = sign_theta1_nll(
                theta1[:check_count],
                x1[:check_count],
                prior_std=prior_std,
                noise_std=theta1_noise_std,
                tail_switch=args.tail_switch,
                tail_order=order,
                chunk_size=args.chunk_size,
            )
            quadrature_checks[str(order)] = {
                "examples": int(check_count),
                "theta1_nll_mean": float(np.mean(check)),
                "theta1_nll_delta_vs_main_tail_order": float(
                    np.mean(check - theta1_nll[:check_count])
                ),
                "theta1_nll_max_abs_delta_vs_main_tail_order": float(
                    np.max(np.abs(check - theta1_nll[:check_count]))
                ),
            }

    elapsed = time.perf_counter() - start_time
    return {
        "case": "sign",
        "model": {
            "theta_prior": "N(0, diag(1.8^2, 1.8^2))",
            "x1": "theta1^2 + Normal(0, 0.22^2)",
            "x2": "theta2 + Normal(0, 0.16^2)",
            "raw_coordinate_target": "(theta1, theta2)",
            "folded_coordinate_target": "(abs(theta1), theta2)",
        },
        "examples": int(args.examples),
        "seed": int(args.seed),
        "tail_switch": float(args.tail_switch),
        "tail_order": int(args.tail_order),
        "chunk_size": int(args.chunk_size),
        "elapsed_seconds": float(elapsed),
        "nll_raw_theta": summarize(total_nll),
        "nll_folded_abs_theta1_theta2": summarize(folded_nll),
        "nll_components": {
            "theta1_raw": summarize(theta1_nll),
            "theta2": summarize(theta2_nll),
            "theta2_analytic_entropy": float(theta2_entropy),
            "folded_minus_raw_shift": float(-LOG_2),
        },
        "quadrature_checks": quadrature_checks,
        "method": (
            "Monte Carlo over p(theta)p(x|theta). The theta1 evidence p(x1) "
            "uses the closed-form parabolic-cylinder integral for non-tail x1 "
            "and standard-normal Gauss-Hermite expectation around y=x1 for "
            "the positive tail; theta2 uses the closed-form conjugate Gaussian "
            "posterior."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the sign-symmetry model population NLL entropy floor."
    )
    parser.add_argument("--examples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--tail-switch", type=float, default=10.0)
    parser.add_argument("--tail-order", type=int, default=41)
    parser.add_argument("--chunk-size", type=int, default=16_384)
    parser.add_argument("--check-examples", type=int, default=50_000)
    parser.add_argument("--check-tail-orders", default="21,61,101")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
