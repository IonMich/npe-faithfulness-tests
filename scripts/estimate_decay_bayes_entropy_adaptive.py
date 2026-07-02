from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp

from mcmc_decay_inference import PRIOR_LOG_MEAN, PRIOR_LOG_STD


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/15_broad_scaling/202_bayes_entropy_high_precision"
)
LOG_2PI = math.log(2.0 * math.pi)


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


def log_normal_prior(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (z - mean[None, :]) / std[None, :]
    return -0.5 * np.sum(standardized * standardized, axis=1) - float(
        np.sum(np.log(std)) + 0.5 * z.shape[1] * LOG_2PI
    )


class DecayPosteriorIntegrator:
    def __init__(
        self,
        *,
        t: np.ndarray,
        gh_order: int,
        proposal_scale: float,
        hessian_eps: float,
    ) -> None:
        self.t = np.asarray(t, dtype=np.float64)
        self.n_obs = int(self.t.size)
        self.prior_mean = PRIOR_LOG_MEAN.detach().cpu().numpy().astype(np.float64)
        self.prior_std = PRIOR_LOG_STD.detach().cpu().numpy().astype(np.float64)
        self.prior_var = self.prior_std * self.prior_std
        self.prior_log_const = -float(
            np.sum(np.log(self.prior_std)) + 0.5 * 3 * LOG_2PI
        )
        self.proposal_scale = float(proposal_scale)
        self.hessian_eps = float(hessian_eps)
        nodes, weights = hermgauss(int(gh_order))
        self.gh_order = int(gh_order)
        logw = np.log(weights) - 0.5 * math.log(math.pi)
        mesh = np.meshgrid(
            math.sqrt(2.0) * nodes,
            math.sqrt(2.0) * nodes,
            math.sqrt(2.0) * nodes,
            indexing="ij",
        )
        self.standard_nodes = np.stack([item.ravel() for item in mesh], axis=1).astype(
            np.float64
        )
        self.log_weights = (
            logw[:, None, None] + logw[None, :, None] + logw[None, None, :]
        ).ravel()

    def log_prior_one(self, z: np.ndarray) -> float:
        standardized = (z - self.prior_mean) / self.prior_std
        return float(-0.5 * np.sum(standardized * standardized) + self.prior_log_const)

    def log_joint_and_grad(self, z: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        amplitude = math.exp(float(z[0]))
        decay_rate = math.exp(float(z[1]))
        sigma = math.exp(float(z[2]))
        inv_sigma2 = 1.0 / (sigma * sigma)
        phi = np.exp(-decay_rate * self.t)
        mean = amplitude * phi
        residual = x - mean
        sse = float(np.sum(residual * residual))
        log_like = -0.5 * (
            self.n_obs * LOG_2PI + 2.0 * self.n_obs * z[2] + sse * inv_sigma2
        )
        log_prior = self.log_prior_one(z)
        dmean_dlog_amplitude = mean
        dmean_dlog_decay = -decay_rate * self.t * mean
        grad_like = np.asarray(
            [
                inv_sigma2 * np.sum(residual * dmean_dlog_amplitude),
                inv_sigma2 * np.sum(residual * dmean_dlog_decay),
                -self.n_obs + sse * inv_sigma2,
            ],
            dtype=np.float64,
        )
        grad_prior = -(z - self.prior_mean) / self.prior_var
        return float(log_like + log_prior), grad_like + grad_prior

    def negative_log_joint_and_grad(
        self,
        z: np.ndarray,
        x: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        value, grad = self.log_joint_and_grad(z, x)
        return -value, -grad

    def negative_hessian(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        hessian = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            step = np.zeros(3, dtype=np.float64)
            step[axis] = self.hessian_eps
            _, grad_plus = self.log_joint_and_grad(z + step, x)
            _, grad_minus = self.log_joint_and_grad(z - step, x)
            hessian[:, axis] = -(grad_plus - grad_minus) / (2.0 * self.hessian_eps)
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def batch_log_joint(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        amplitude = np.exp(z[:, 0])
        decay_rate = np.exp(z[:, 1])
        sigma = np.exp(z[:, 2])
        phi = np.exp(-decay_rate[:, None] * self.t[None, :])
        mean = amplitude[:, None] * phi
        residual = x[None, :] - mean
        sse = np.sum(residual * residual, axis=1)
        log_like = -0.5 * (
            self.n_obs * LOG_2PI
            + 2.0 * self.n_obs * np.log(sigma)
            + sse / np.square(sigma)
        )
        return log_like + log_normal_prior(z, self.prior_mean, self.prior_std)

    def true_log_likelihood(self, z: np.ndarray, x: np.ndarray) -> float:
        return float(self.batch_log_joint(z[None, :], x)[0] - self.log_prior_one(z))

    def find_mode(self, x: np.ndarray, initial_z: np.ndarray) -> tuple[np.ndarray, bool, int]:
        result = minimize(
            lambda z: self.negative_log_joint_and_grad(z, x),
            np.asarray(initial_z, dtype=np.float64),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-6, "maxiter": 100},
        )
        return np.asarray(result.x, dtype=np.float64), bool(result.success), int(result.nit)

    def log_marginal_likelihood(self, x: np.ndarray, mode: np.ndarray, covariance: np.ndarray) -> float:
        scaled_covariance = covariance * (self.proposal_scale * self.proposal_scale)
        try:
            cholesky = np.linalg.cholesky(scaled_covariance)
        except np.linalg.LinAlgError:
            jitter = 1e-8 * np.eye(3, dtype=np.float64)
            cholesky = np.linalg.cholesky(scaled_covariance + jitter)
        z_nodes = mode[None, :] + self.standard_nodes @ cholesky.T
        log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        log_proposal = -0.5 * (
            3 * LOG_2PI + log_det + np.sum(self.standard_nodes * self.standard_nodes, axis=1)
        )
        log_terms = self.log_weights + self.batch_log_joint(z_nodes, x) - log_proposal
        return float(logsumexp(log_terms))

    def evaluate_one(self, x: np.ndarray, z_true: np.ndarray) -> dict[str, float | int | bool]:
        mode, optimizer_success, optimizer_iterations = self.find_mode(x, z_true)
        hessian = self.negative_hessian(mode, x)
        covariance = np.linalg.inv(hessian)
        log_marginal = self.log_marginal_likelihood(x, mode, covariance)
        true_log_like = self.true_log_likelihood(z_true, x)
        true_log_prior = self.log_prior_one(z_true)
        posterior_log_prob = true_log_like + true_log_prior - log_marginal
        nll = -posterior_log_prob
        return {
            "nll": float(nll),
            "log_marginal": float(log_marginal),
            "true_log_like": float(true_log_like),
            "true_log_prior": float(true_log_prior),
            "posterior_log_prob": float(posterior_log_prob),
            "mode_log_joint": float(self.log_joint_and_grad(mode, x)[0]),
            "optimizer_success": bool(optimizer_success),
            "optimizer_iterations": int(optimizer_iterations),
        }


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


def load_existing(output_npz: Path) -> dict[str, np.ndarray] | None:
    if not output_npz.exists():
        return None
    existing = np.load(output_npz, allow_pickle=False)
    return {key: np.asarray(existing[key]) for key in existing.files}


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    validation_cache = Path(args.validation_cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{args.name}.npz"
    output_json = output_dir / f"{args.name}.json"

    data = np.load(validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float64)
    z_val = np.asarray(data["z_val"], dtype=np.float64)
    t = np.asarray(data["t"], dtype=np.float64)
    total = int(x_val.shape[0])
    start = max(0, int(args.start_index))
    stop = total if args.stop_index <= 0 else min(total, int(args.stop_index))
    if args.examples > 0:
        stop = min(stop, start + int(args.examples))
    if stop <= start:
        raise ValueError(f"Empty range: start={start}, stop={stop}")
    requested_indices = np.arange(start, stop, dtype=np.int64)

    existing = load_existing(output_npz) if args.resume else None
    if existing is None:
        done_indices = np.asarray([], dtype=np.int64)
        rows: dict[str, list[float | int | bool]] = {
            "index": [],
            "nll": [],
            "log_marginal": [],
            "true_log_like": [],
            "true_log_prior": [],
            "posterior_log_prob": [],
            "mode_log_joint": [],
            "optimizer_success": [],
            "optimizer_iterations": [],
        }
    else:
        done_indices = np.asarray(existing["index"], dtype=np.int64)
        rows = {key: value.tolist() for key, value in existing.items()}
    done = set(int(index) for index in done_indices.tolist())
    pending = [int(index) for index in requested_indices.tolist() if int(index) not in done]

    integrator = DecayPosteriorIntegrator(
        t=t,
        gh_order=int(args.gh_order),
        proposal_scale=float(args.proposal_scale),
        hessian_eps=float(args.hessian_eps),
    )

    start_time = time.perf_counter()
    last_flush = time.perf_counter()
    for counter, index in enumerate(pending, start=1):
        result = integrator.evaluate_one(x_val[index], z_val[index])
        rows["index"].append(index)
        for key, value in result.items():
            rows[key].append(value)
        now = time.perf_counter()
        should_flush = counter == len(pending) or counter % int(args.flush_every) == 0
        should_report = counter == 1 or counter % int(args.report_every) == 0
        if should_flush:
            ordered = np.argsort(np.asarray(rows["index"], dtype=np.int64))
            arrays = {}
            for key, value in rows.items():
                array = np.asarray(value)
                arrays[key] = array[ordered]
            np.savez_compressed(output_npz, **arrays)
            nll = np.asarray(arrays["nll"], dtype=np.float64)
            summary = {
                "name": args.name,
                "validation_cache": str(validation_cache),
                "output_npz": str(output_npz),
                "output_json": str(output_json),
                "start_index": int(start),
                "stop_index": int(stop),
                "requested_examples": int(requested_indices.size),
                "completed_examples": int(nll.size),
                "pending_examples": int(requested_indices.size - nll.size),
                "gh_order": int(args.gh_order),
                "gh_points": int(integrator.standard_nodes.shape[0]),
                "proposal_scale": float(args.proposal_scale),
                "hessian_eps": float(args.hessian_eps),
                "used_true_z_as_optimizer_initialization": True,
                "optimizer_success_count": int(np.count_nonzero(arrays["optimizer_success"])),
                "optimizer_failure_count": int(nll.size - np.count_nonzero(arrays["optimizer_success"])),
                "nll": summarize(nll),
                "elapsed_seconds_this_run": float(now - start_time),
                "seconds_since_last_flush": float(now - last_flush),
            }
            output_json.write_text(
                json.dumps(json_ready(summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            last_flush = now
        if should_report:
            nll = np.asarray(rows["nll"], dtype=np.float64)
            print(
                json.dumps(
                    {
                        "processed_this_run": counter,
                        "pending_this_run": len(pending),
                        "total_completed": int(nll.size),
                        "latest_index": index,
                        "mean": float(np.mean(nll)),
                        "std_error": float(np.std(nll, ddof=1) / math.sqrt(nll.size))
                        if nll.size > 1
                        else None,
                        "elapsed_seconds": float(now - start_time),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not pending and output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))
    if output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))
    raise RuntimeError("No output was written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the single-decay Bayes entropy floor with posterior-centered "
            "Gauss-Hermite evidence integration."
        )
    )
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="adaptive_bayes_entropy")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=0)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--gh-order", type=int, default=11)
    parser.add_argument("--proposal-scale", type=float, default=1.0)
    parser.add_argument("--hessian-eps", type=float, default=1e-4)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate(args)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
