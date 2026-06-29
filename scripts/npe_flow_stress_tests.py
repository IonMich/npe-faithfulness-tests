from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import artifact_paths as ap
from typing import Callable

import arviz as az
import corner
import matplotlib
import numpy as np
import torch
import zuko
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corner_truth import overplot_true_values, true_theta_legend_handle


LOG_2PI = math.log(2.0 * math.pi)
LOG_2 = math.log(2.0)


TensorLogLike = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
NumpyArrayFn = Callable[[np.ndarray], np.ndarray]
SimulatorFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]
InitialFn = Callable[[int, np.ndarray, np.random.Generator], np.ndarray]
NpeInverseFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class StressCase:
    name: str
    z_dim: int
    prior_mean: np.ndarray
    prior_std: np.ndarray
    true_z: np.ndarray
    param_names: tuple[str, ...]
    diagnostic_names: tuple[str, ...]
    mcmc_proposal_scale: np.ndarray
    hmc_step_size: float
    hmc_leapfrog_steps: int
    simulate_x: SimulatorFn
    mean_x: NumpyArrayFn
    context: NumpyArrayFn
    log_likelihood: TensorLogLike
    display: NumpyArrayFn
    diagnostic_transform: NumpyArrayFn
    initial_z: InitialFn
    observed_axis: np.ndarray | None = None
    observed_kind: str = "vector"
    mode_metric: Callable[[np.ndarray], dict[str, float]] | None = None
    linear_adjustment_default: bool = True
    npe_transform: NumpyArrayFn | None = None
    npe_inverse: NpeInverseFn | None = None
    npe_center_from_context: NumpyArrayFn | None = None


class ConditionalSplineFlow(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        context_dim: int,
        transforms: int,
        hidden_features: tuple[int, ...],
        bins: int,
    ) -> None:
        super().__init__()
        self.flow = zuko.flows.NSF(
            z_dim,
            context=context_dim,
            transforms=transforms,
            hidden_features=hidden_features,
            bins=bins,
        )

    def log_prob(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.flow(context).log_prob(z)

    @torch.no_grad()
    def sample(self, n: int, context: torch.Tensor, chunk_size: int = 65_536) -> torch.Tensor:
        samples: list[torch.Tensor] = []
        for start in range(0, n, chunk_size):
            current = min(chunk_size, n - start)
            draw = self.flow(context).sample((current,))
            if draw.ndim == 3:
                draw = draw[:, 0, :]
            samples.append(draw)
        return torch.cat(samples, dim=0)


def parse_hidden_features(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def torch_dtype_for(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type in {"mps", "cuda"} else torch.float64


def log_normal_diag_torch(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    delta = (value - mean) / std
    return (-0.5 * delta.square() - torch.log(std) - 0.5 * LOG_2PI).sum(dim=1)


def sample_prior(case: StressCase, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(case.prior_mean[None, :], case.prior_std[None, :], size=(n, case.z_dim))


def prior_logpdf_np(case: StressCase, z: np.ndarray) -> np.ndarray:
    return (
        -0.5 * ((z - case.prior_mean[None, :]) / case.prior_std[None, :]) ** 2
        - np.log(case.prior_std[None, :])
        - 0.5 * LOG_2PI
    ).sum(axis=1)


def gaussian_logpdf_np(z: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    inv = np.linalg.inv(covariance)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    delta = z - mean[None, :]
    maha = np.sum((delta @ inv) * delta, axis=1)
    return -0.5 * (z.shape[1] * LOG_2PI + log_det + maha)


def fit_gaussian_proposal(
    samples: np.ndarray,
    *,
    inflation: float,
    prior_mixture: float,
) -> dict[str, object]:
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False) * inflation**2
    covariance = covariance + np.eye(samples.shape[1]) * 1e-5
    return {
        "kind": "gaussian_mixture",
        "mean": mean,
        "covariance": covariance,
        "inflation": float(inflation),
        "prior_mixture": float(prior_mixture),
    }


def sample_proposal(
    case: StressCase,
    n: int,
    rng: np.random.Generator,
    proposal: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if proposal is None:
        z = sample_prior(case, n, rng)
        return z, prior_logpdf_np(case, z)
    mean = np.asarray(proposal["mean"])
    covariance = np.asarray(proposal["covariance"])
    prior_mixture = float(proposal["prior_mixture"])
    use_prior = rng.random(n) < prior_mixture
    z = np.empty((n, case.z_dim), dtype=np.float64)
    if np.any(use_prior):
        z[use_prior] = sample_prior(case, int(use_prior.sum()), rng)
    if np.any(~use_prior):
        z[~use_prior] = rng.multivariate_normal(mean, covariance, size=int((~use_prior).sum()))
    log_prior = prior_logpdf_np(case, z)
    log_gaussian = gaussian_logpdf_np(z, mean, covariance)
    log_r = logsumexp(
        np.column_stack([
            math.log(max(prior_mixture, 1e-12)) + log_prior,
            math.log(max(1.0 - prior_mixture, 1e-12)) + log_gaussian,
        ]),
        axis=1,
    )
    return z, log_r


def log_posterior_factory(
    case: StressCase,
    x0: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    prior_mean = torch.as_tensor(case.prior_mean, device=device, dtype=dtype)
    prior_std = torch.as_tensor(case.prior_std, device=device, dtype=dtype)
    x0_t = torch.as_tensor(x0, device=device, dtype=dtype)

    def logp(z: torch.Tensor) -> torch.Tensor:
        return log_normal_diag_torch(z, prior_mean, prior_std) + case.log_likelihood(z, x0_t)

    return logp


def simulate_contexts(
    case: StressCase,
    z: np.ndarray,
    rng: np.random.Generator,
    *,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, z.shape[0], chunk_size):
        stop = min(start + chunk_size, z.shape[0])
        x = case.simulate_x(z[start:stop], rng)
        chunks.append(case.context(x))
    return np.concatenate(chunks, axis=0)


def context_distances(
    context: np.ndarray,
    observed_context: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (context - center[None, :]) / scale[None, :]
    observed = (observed_context - center) / scale
    return np.sqrt(np.mean((standardized - observed[None, :]) ** 2, axis=1))


def fit_local_region(
    case: StressCase,
    observed_context: np.ndarray,
    *,
    pilot_count: int,
    quantile: float,
    kernel_quantile: float,
    simulate_chunk_size: int,
    rng: np.random.Generator,
    proposal: dict[str, object] | None,
) -> dict[str, object]:
    z, _log_r = sample_proposal(case, pilot_count, rng, proposal)
    context = simulate_contexts(case, z, rng, chunk_size=simulate_chunk_size)
    center = context.mean(axis=0)
    scale = np.maximum(context.std(axis=0), 1e-6)
    distances = context_distances(context, observed_context, center, scale)
    radius = float(np.quantile(distances, quantile))
    kernel_bandwidth = None
    if kernel_quantile > 0.0:
        kernel_bandwidth = float(np.quantile(distances, kernel_quantile))
    return {
        "center": center,
        "scale": scale,
        "radius": radius,
        "quantile": quantile,
        "kernel_quantile": kernel_quantile,
        "kernel_bandwidth": kernel_bandwidth,
        "pilot_count": int(pilot_count),
        "pilot_distance_summary": summarize_vector(distances),
    }


def collect_local_training_data(
    case: StressCase,
    observed_context: np.ndarray,
    region: dict[str, object],
    *,
    target_count: int,
    max_candidates: int,
    simulate_chunk_size: int,
    rng: np.random.Generator,
    proposal: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    accepted_z: list[np.ndarray] = []
    accepted_context: list[np.ndarray] = []
    accepted_distance: list[np.ndarray] = []
    accepted_log_weight: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    center = np.asarray(region["center"])
    scale = np.asarray(region["scale"])
    radius = float(region["radius"])
    start = time.perf_counter()
    while accepted_count < target_count and candidate_count < max_candidates:
        current = min(simulate_chunk_size, max_candidates - candidate_count)
        z, log_r = sample_proposal(case, current, rng, proposal)
        x = case.simulate_x(z, rng)
        context = case.context(x)
        distances = context_distances(context, observed_context, center, scale)
        mask = distances <= radius
        if np.any(mask):
            accepted_z.append(z[mask])
            accepted_context.append(context[mask])
            accepted_distance.append(distances[mask])
            accepted_log_weight.append(prior_logpdf_np(case, z[mask]) - log_r[mask])
            accepted_count += int(mask.sum())
        candidate_count += current

    if accepted_count < target_count:
        raise RuntimeError(
            f"{case.name}: only accepted {accepted_count} local simulations after "
            f"{candidate_count} candidates; need {target_count}."
        )

    z_all = np.concatenate(accepted_z, axis=0)[:target_count]
    context_all = np.concatenate(accepted_context, axis=0)[:target_count]
    distance_all = np.concatenate(accepted_distance, axis=0)[:target_count]
    log_weight_all = np.concatenate(accepted_log_weight, axis=0)[:target_count]
    diagnostics = {
        "candidate_count": int(candidate_count),
        "raw_accepted_count": int(accepted_count),
        "target_count": int(target_count),
        "acceptance_rate": float(accepted_count / max(candidate_count, 1)),
        "collection_seconds": float(time.perf_counter() - start),
        "accepted_distance_summary": summarize_vector(distance_all),
    }
    return z_all, context_all, distance_all, log_weight_all, diagnostics


def fit_linear_target_adjustment(
    z: np.ndarray,
    context_std: np.ndarray,
    observed_context_std: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    delta = context_std - observed_context_std[None, :]
    design = np.column_stack([np.ones(z.shape[0]), delta])
    normalized = weights / np.maximum(np.mean(weights), 1e-12)
    sqrt_w = np.sqrt(normalized)
    xw = design * sqrt_w[:, None]
    zw = z * sqrt_w[:, None]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ zw)
    return beta[1:]


def make_kernel_weights(distances: np.ndarray, bandwidth: float | None) -> np.ndarray:
    if bandwidth is None or bandwidth <= 0.0:
        return np.ones_like(distances, dtype=np.float32)
    weights = np.exp(-0.5 * (distances / max(bandwidth, 1e-8)) ** 2)
    weights = weights / max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def train_flow(
    *,
    train_context: np.ndarray,
    train_z: np.ndarray,
    train_weights: np.ndarray,
    val_context: np.ndarray,
    val_z: np.ndarray,
    val_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalSplineFlow, dict[str, object]]:
    torch.manual_seed(args.seed + 1000)
    model = ConditionalSplineFlow(
        z_dim=train_z.shape[1],
        context_dim=train_context.shape[1],
        transforms=args.transforms,
        hidden_features=args.hidden_features,
        bins=args.bins,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1001)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_context.astype(np.float32)),
            torch.from_numpy(train_z.astype(np.float32)),
            torch.from_numpy(train_weights.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    val_context_t = torch.from_numpy(val_context.astype(np.float32)).to(device)
    val_z_t = torch.from_numpy(val_z.astype(np.float32)).to(device)
    val_w_t = torch.from_numpy(val_weights.astype(np.float32)).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    history = {"train_nll": [], "val_nll": [], "lr": []}
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for batch_context, batch_z, batch_w in loader:
            batch_context = batch_context.to(device)
            batch_z = batch_z.to(device)
            batch_w = batch_w.to(device)
            nll = -model.log_prob(batch_z, batch_context)
            loss = (nll * batch_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float((nll.detach() * batch_w).sum().cpu())
            train_weight_sum += float(batch_w.detach().sum().cpu())

        model.eval()
        with torch.no_grad():
            val_nll = -model.log_prob(val_z_t, val_context_t)
            val_loss = (val_nll * val_w_t).sum() / val_w_t.sum().clamp_min(1e-12)
            val_loss_float = float(val_loss.detach().cpu())
        train_loss = train_loss_sum / max(train_weight_sum, 1e-12)
        history["train_nll"].append(float(train_loss))
        history["val_nll"].append(val_loss_float)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()

        if val_loss_float < best_val:
            best_val = val_loss_float
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch {epoch:04d} train_nll={train_loss:.4f} "
                f"val_nll={val_loss_float:.4f} best={best_val:.4f}"
            )
        if epochs_since_best >= args.patience:
            break

    synchronize(device)
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "epochs_completed": len(history["train_nll"]),
        "best_epoch": int(best_epoch),
        "best_val_nll": float(best_val),
        "final_train_nll": float(history["train_nll"][-1]),
        "final_val_nll": float(history["val_nll"][-1]),
        "training_seconds": float(time.perf_counter() - start),
        "history": history,
    }


@torch.no_grad()
def sample_npe(
    model: ConditionalSplineFlow,
    observed_context_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    z_chol: np.ndarray | None,
    *,
    n: int,
    device: torch.device,
) -> np.ndarray:
    context_t = torch.from_numpy(observed_context_std[None, :].astype(np.float32)).to(device)
    z_std_samples = model.sample(n, context_t).detach().cpu().numpy()
    if z_chol is not None:
        return z_std_samples @ z_chol.T + z_mean[None, :]
    return z_std_samples * z_std[None, :] + z_mean[None, :]


def value_and_grad(
    logp: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    value = logp(z_req)
    (grad,) = torch.autograd.grad(value.sum(), z_req)
    return value.detach(), grad.detach()


def run_random_walk_mcmc(
    case: StressCase,
    x0: np.ndarray,
    *,
    chains: int,
    steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed + 200)
    torch.manual_seed(seed + 201)
    logp = log_posterior_factory(case, x0, device=device, dtype=dtype)
    z0 = case.initial_z(chains, x0, rng)
    z_current = torch.as_tensor(z0, device=device, dtype=dtype)
    logp_current = logp(z_current)
    proposal_scale = torch.as_tensor(case.mcmc_proposal_scale, device=device, dtype=dtype)
    z_samples = torch.empty(steps, chains, case.z_dim, device=device, dtype=dtype)
    accepted = torch.empty(steps, chains, device=device, dtype=torch.bool)

    synchronize(device)
    start = time.perf_counter()
    proposal_noise = torch.randn(steps, chains, case.z_dim, device=device, dtype=dtype)
    accept_uniform = torch.log(torch.rand(steps, chains, device=device, dtype=dtype))
    for step in range(steps):
        proposal = z_current + proposal_noise[step] * proposal_scale
        logp_proposal = logp(proposal)
        accept = accept_uniform[step] < (logp_proposal - logp_current)
        z_current = torch.where(accept[:, None], proposal, z_current)
        logp_current = torch.where(accept, logp_proposal, logp_current)
        z_samples[step] = z_current
        accepted[step] = accept
    synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        z_samples.detach().cpu().numpy().transpose(1, 0, 2),
        accepted.detach().cpu().numpy().transpose(1, 0),
        elapsed,
    )


def run_hmc(
    case: StressCase,
    x0: np.ndarray,
    *,
    chains: int,
    steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed + 300)
    torch.manual_seed(seed + 301)
    logp = log_posterior_factory(case, x0, device=device, dtype=dtype)
    z0 = case.initial_z(chains, x0, rng)
    z_current = torch.as_tensor(z0, device=device, dtype=dtype)
    logp_current, _ = value_and_grad(logp, z_current)
    z_samples = torch.empty(steps, chains, case.z_dim, device=device, dtype=dtype)
    accepted = torch.empty(steps, chains, device=device, dtype=torch.bool)
    energy_error = torch.empty(steps, chains, device=device, dtype=dtype)
    step_size = case.hmc_step_size

    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        momentum_current = torch.randn_like(z_current)
        current_kinetic = 0.5 * momentum_current.square().sum(dim=1)
        z_proposal = z_current.detach()
        momentum = momentum_current.detach()
        _, grad = value_and_grad(logp, z_proposal)
        momentum = momentum + 0.5 * step_size * grad
        logp_proposal = logp_current
        for leapfrog_index in range(case.hmc_leapfrog_steps):
            z_proposal = z_proposal + step_size * momentum
            logp_proposal, grad = value_and_grad(logp, z_proposal)
            if leapfrog_index != case.hmc_leapfrog_steps - 1:
                momentum = momentum + step_size * grad
        momentum = momentum + 0.5 * step_size * grad
        proposed_kinetic = 0.5 * momentum.square().sum(dim=1)
        energy_delta = (-logp_proposal + proposed_kinetic) - (-logp_current + current_kinetic)
        accept_log_prob = -energy_delta
        accept_uniform = torch.log(torch.rand(chains, device=device, dtype=dtype))
        accept = torch.isfinite(accept_log_prob) & (accept_uniform < accept_log_prob)
        z_current = torch.where(accept[:, None], z_proposal, z_current)
        logp_current = torch.where(accept, logp_proposal, logp_current)
        z_samples[step] = z_current
        accepted[step] = accept
        energy_error[step] = energy_delta
    synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        z_samples.detach().cpu().numpy().transpose(1, 0, 2),
        accepted.detach().cpu().numpy().transpose(1, 0),
        energy_error.detach().cpu().numpy().transpose(1, 0),
        elapsed,
    )


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def summarize_matrix(samples: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        values = samples[:, index]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return summary


def transform_chains(
    z_samples: np.ndarray,
    transform: NumpyArrayFn,
) -> np.ndarray:
    chains, steps, dim = z_samples.shape
    flat = z_samples.reshape(chains * steps, dim)
    transformed = transform(flat)
    return transformed.reshape(chains, steps, transformed.shape[1])


def arviz_diagnostics(
    z_samples: np.ndarray,
    burn_in: int,
    transform: NumpyArrayFn,
    names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    transformed = transform_chains(z_samples[:, burn_in:, :], transform)
    diagnostics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        draws = transformed[:, :, index]
        diagnostics[name] = {
            "rhat": float(az.rhat(draws, chain_axis=0, draw_axis=1)),
            "ess_bulk": float(az.ess(draws, method="bulk", chain_axis=0, draw_axis=1)),
            "ess_tail": float(az.ess(draws, method="tail", prob=0.05, chain_axis=0, draw_axis=1)),
        }
    return diagnostics


def convergence_flags(
    diagnostics: dict[str, dict[str, float]],
    *,
    rhat_threshold: float,
    ess_threshold: float,
) -> dict[str, object]:
    rhat_values = [values["rhat"] for values in diagnostics.values()]
    ess_values = [values["ess_bulk"] for values in diagnostics.values()]
    return {
        "max_rhat": float(np.nanmax(rhat_values)),
        "min_bulk_ess": float(np.nanmin(ess_values)),
        "rhat_ok": bool(np.nanmax(rhat_values) <= rhat_threshold),
        "ess_ok": bool(np.nanmin(ess_values) >= ess_threshold),
        "ok": bool(np.nanmax(rhat_values) <= rhat_threshold and np.nanmin(ess_values) >= ess_threshold),
    }


def flatten_post_burn(z_samples: np.ndarray, burn_in: int, thin: int = 1) -> np.ndarray:
    return z_samples[:, burn_in::thin, :].reshape(-1, z_samples.shape[2])


def normalized_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, object]:
    pooled = np.concatenate([a, b], axis=0)
    scale = np.maximum(np.std(pooled, axis=0), 1e-8)
    per_dim: dict[str, float] = {}
    values = []
    for index, name in enumerate(names):
        distance = wasserstein_distance(a[:, index], b[:, index]) / scale[index]
        per_dim[name] = float(distance)
        values.append(distance)
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "per_dim": per_dim,
    }


def pairwise_agreement(
    case: StressCase,
    mcmc_z: np.ndarray,
    hmc_z: np.ndarray,
    npe_z: np.ndarray,
) -> dict[str, object]:
    display = {
        "mcmc": case.display(mcmc_z),
        "hmc": case.display(hmc_z),
        "npe": case.display(npe_z),
    }
    diagnostic = {
        "mcmc": case.diagnostic_transform(mcmc_z),
        "hmc": case.diagnostic_transform(hmc_z),
        "npe": case.diagnostic_transform(npe_z),
    }
    return {
        "display": {
            "mcmc_hmc": normalized_wasserstein(display["mcmc"], display["hmc"], case.param_names),
            "mcmc_npe": normalized_wasserstein(display["mcmc"], display["npe"], case.param_names),
            "hmc_npe": normalized_wasserstein(display["hmc"], display["npe"], case.param_names),
        },
        "diagnostic": {
            "mcmc_hmc": normalized_wasserstein(diagnostic["mcmc"], diagnostic["hmc"], case.diagnostic_names),
            "mcmc_npe": normalized_wasserstein(diagnostic["mcmc"], diagnostic["npe"], case.diagnostic_names),
            "hmc_npe": normalized_wasserstein(diagnostic["hmc"], diagnostic["npe"], case.diagnostic_names),
        },
    }


def subsample_rows(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if values.shape[0] <= n:
        return values
    indices = rng.choice(values.shape[0], size=n, replace=False)
    return values[indices]


def plot_corner_overlay(
    case: StressCase,
    *,
    mcmc_z: np.ndarray,
    hmc_z: np.ndarray,
    npe_z: np.ndarray,
    outfile: Path,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed + 900)
    mcmc = case.display(subsample_rows(mcmc_z, max_points, rng))
    hmc = case.display(subsample_rows(hmc_z, max_points, rng))
    npe = case.display(subsample_rows(npe_z, max_points, rng))
    max_dim = min(mcmc.shape[1], 6)
    labels = list(case.param_names[:max_dim])
    truths = case.display(case.true_z[None, :])[0, :max_dim]
    figure = corner.corner(
        mcmc[:, :max_dim],
        labels=labels,
        color="#1f77b4",
        hist_kwargs={"density": True, "alpha": 0.35},
        plot_datapoints=False,
        fill_contours=False,
        show_titles=True,
        title_fmt=".3f",
    )
    corner.corner(
        hmc[:, :max_dim],
        labels=labels,
        color="#d95f02",
        hist_kwargs={"density": True, "alpha": 0.25},
        plot_datapoints=False,
        fill_contours=False,
        fig=figure,
    )
    corner.corner(
        npe[:, :max_dim],
        labels=labels,
        color="#2ca02c",
        hist_kwargs={"density": True, "alpha": 0.25},
        plot_datapoints=False,
        fill_contours=False,
        fig=figure,
    )
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="MCMC"),
        plt.Line2D([0], [0], color="#d95f02", lw=2, label="HMC"),
        plt.Line2D([0], [0], color="#2ca02c", lw=2, label="NPE flow"),
        true_theta_legend_handle(),
    ]
    overplot_true_values(figure, truths)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    figure.suptitle(f"{case.name}: MCMC/HMC/NPE posterior overlay", y=0.995, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trace_overlay(
    case: StressCase,
    *,
    mcmc_z: np.ndarray,
    hmc_z: np.ndarray,
    burn_in_mcmc: int,
    burn_in_hmc: int,
    outfile: Path,
) -> None:
    mcmc_diag = transform_chains(mcmc_z, case.diagnostic_transform)
    hmc_diag = transform_chains(hmc_z, case.diagnostic_transform)
    dims = min(mcmc_diag.shape[2], 6)
    figure, axes = plt.subplots(dims, 2, figsize=(12, 2.0 * dims), sharex="col")
    if dims == 1:
        axes = np.asarray([axes])
    for index in range(dims):
        ax_mcmc = axes[index, 0]
        ax_hmc = axes[index, 1]
        for chain in range(mcmc_diag.shape[0]):
            ax_mcmc.plot(mcmc_diag[chain, :, index], lw=0.6, alpha=0.55)
        for chain in range(hmc_diag.shape[0]):
            ax_hmc.plot(hmc_diag[chain, :, index], lw=0.6, alpha=0.55)
        ax_mcmc.axvline(burn_in_mcmc, color="black", lw=0.8, alpha=0.6)
        ax_hmc.axvline(burn_in_hmc, color="black", lw=0.8, alpha=0.6)
        ax_mcmc.set_ylabel(case.diagnostic_names[index])
        ax_mcmc.set_title("MCMC" if index == 0 else "")
        ax_hmc.set_title("HMC" if index == 0 else "")
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    figure.suptitle(f"{case.name}: diagnostic trace plots", y=0.995, fontsize=14)
    figure.tight_layout()
    figure.savefig(outfile, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_predictive_overlay(
    case: StressCase,
    x0: np.ndarray,
    *,
    mcmc_z: np.ndarray,
    hmc_z: np.ndarray,
    npe_z: np.ndarray,
    outfile: Path,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed + 950)
    mcmc_mean = case.mean_x(subsample_rows(mcmc_z, max_points, rng))
    hmc_mean = case.mean_x(subsample_rows(hmc_z, max_points, rng))
    npe_mean = case.mean_x(subsample_rows(npe_z, max_points, rng))
    if case.observed_kind == "curve":
        axis = case.observed_axis if case.observed_axis is not None else np.arange(x0.shape[0])
        figure, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.scatter(axis, x0, s=18, color="black", label="observed", zorder=4)
        for label, values, color in [
            ("MCMC", mcmc_mean, "#1f77b4"),
            ("HMC", hmc_mean, "#d95f02"),
            ("NPE flow", npe_mean, "#2ca02c"),
        ]:
            low, mid, high = np.quantile(values, [0.05, 0.50, 0.95], axis=0)
            ax.plot(axis, mid, color=color, lw=1.8, label=label)
            ax.fill_between(axis, low, high, color=color, alpha=0.14, linewidth=0)
        ax.set_xlabel("observation coordinate")
        ax.set_ylabel("x")
        ax.legend(frameon=False)
    elif case.name == "label_switch":
        figure, ax = plt.subplots(figsize=(8.0, 4.8))
        bins = np.linspace(float(np.min(x0)) - 0.7, float(np.max(x0)) + 0.7, 60)
        ax.hist(x0, bins=bins, density=True, histtype="step", color="black", lw=2.0, label="observed")
        grid = np.linspace(bins.min(), bins.max(), 300)
        for label, z, color in [
            ("MCMC", subsample_rows(mcmc_z, max_points, rng), "#1f77b4"),
            ("HMC", subsample_rows(hmc_z, max_points, rng), "#d95f02"),
            ("NPE flow", subsample_rows(npe_z, max_points, rng), "#2ca02c"),
        ]:
            mu1 = z[:, 0:1]
            mu2 = z[:, 1:2]
            sigma = np.exp(z[:, 2:3])
            density = (
                np.exp(-0.5 * ((grid[None, :] - mu1) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
                + np.exp(-0.5 * ((grid[None, :] - mu2) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
            ) * 0.5
            ax.plot(grid, np.mean(density, axis=0), color=color, lw=1.8, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend(frameon=False)
    else:
        figure, ax = plt.subplots(figsize=(6.2, 5.8))
        ax.scatter(mcmc_mean[:, 0], mcmc_mean[:, 1], s=8, alpha=0.18, color="#1f77b4", label="MCMC")
        ax.scatter(hmc_mean[:, 0], hmc_mean[:, 1], s=8, alpha=0.18, color="#d95f02", label="HMC")
        ax.scatter(npe_mean[:, 0], npe_mean[:, 1], s=8, alpha=0.18, color="#2ca02c", label="NPE flow")
        ax.scatter([x0[0]], [x0[1]], s=70, color="black", label="observed", zorder=5)
        ax.set_xlabel("mean x1")
        ax.set_ylabel("mean x2")
        ax.legend(frameon=False)
    figure.suptitle(f"{case.name}: posterior predictive means", y=0.995, fontsize=14)
    figure.tight_layout()
    figure.savefig(outfile, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_npe(
    case: StressCase,
    x0: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    proposal_reference_z: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(args.seed + 400)
    proposal = None
    if args.npe_proposal == "hmc_gaussian":
        if proposal_reference_z is None:
            raise ValueError("--npe-proposal hmc_gaussian requires HMC reference samples.")
        proposal = fit_gaussian_proposal(
            proposal_reference_z,
            inflation=args.proposal_inflation,
            prior_mixture=args.proposal_prior_mixture,
        )
    elif args.npe_proposal != "prior":
        raise ValueError(f"Unknown NPE proposal: {args.npe_proposal}")
    observed_context = case.context(x0[None, :])[0]
    region = fit_local_region(
        case,
        observed_context,
        pilot_count=args.local_pilot,
        quantile=args.local_quantile,
        kernel_quantile=args.kernel_quantile,
        simulate_chunk_size=args.simulate_chunk_size,
        rng=rng,
        proposal=proposal,
    )
    z_raw, context, distance, log_weight_base, collect_summary = collect_local_training_data(
        case,
        observed_context,
        region,
        target_count=args.npe_train_count + args.npe_val_count,
        max_candidates=args.local_max_candidates,
        simulate_chunk_size=args.simulate_chunk_size,
        rng=rng,
        proposal=proposal,
    )
    z = case.npe_transform(z_raw) if case.npe_transform is not None else z_raw
    observed_target_center = None
    if case.npe_center_from_context is not None:
        target_center = case.npe_center_from_context(context)
        observed_target_center = case.npe_center_from_context(observed_context[None, :])[0]
        z = z - target_center
    context_mean = context.mean(axis=0)
    context_std = np.maximum(context.std(axis=0), 1e-6)
    context_standardized = (context - context_mean[None, :]) / context_std[None, :]
    observed_context_std = (observed_context - context_mean) / context_std
    weights = make_kernel_weights(distance, region["kernel_bandwidth"])
    if proposal is not None:
        log_weight_base = log_weight_base - (logsumexp(log_weight_base) - math.log(len(log_weight_base)))
        weights = weights * np.exp(np.clip(log_weight_base, -20.0, 20.0)).astype(np.float32)
        weights = weights / max(float(weights.mean()), 1e-12)
    slope = None
    adjusted_z = z.copy()
    use_linear_adjustment = (
        case.linear_adjustment_default if args.linear_adjustment is None else bool(args.linear_adjustment)
    )
    if use_linear_adjustment:
        slope = fit_linear_target_adjustment(
            z,
            context_standardized,
            observed_context_std,
            weights,
            ridge=args.linear_ridge,
        )
        adjusted_z = z - (context_standardized - observed_context_std[None, :]) @ slope
    z_mean = adjusted_z.mean(axis=0)
    z_chol = None
    if args.full_target_whiten:
        covariance = np.cov(adjusted_z, rowvar=False) + np.eye(adjusted_z.shape[1]) * 1e-6
        z_chol = np.linalg.cholesky(covariance)
        z_standardized = np.linalg.solve(z_chol, (adjusted_z - z_mean[None, :]).T).T
        z_std = np.ones(adjusted_z.shape[1], dtype=np.float64)
    else:
        z_std = np.maximum(adjusted_z.std(axis=0), 1e-6)
        z_standardized = (adjusted_z - z_mean[None, :]) / z_std[None, :]

    indices = np.arange(z.shape[0])
    rng.shuffle(indices)
    train_idx = indices[: args.npe_train_count]
    val_idx = indices[args.npe_train_count : args.npe_train_count + args.npe_val_count]
    model, training_summary = train_flow(
        train_context=context_standardized[train_idx],
        train_z=z_standardized[train_idx],
        train_weights=weights[train_idx],
        val_context=context_standardized[val_idx],
        val_z=z_standardized[val_idx],
        val_weights=weights[val_idx],
        args=args,
        device=device,
    )
    npe_target = sample_npe(
        model,
        observed_context_std,
        z_mean,
        z_std,
        z_chol,
        n=args.npe_samples,
        device=device,
    )
    if observed_target_center is not None:
        npe_target = npe_target + observed_target_center[None, :]
    npe_z = case.npe_inverse(npe_target, rng) if case.npe_inverse is not None else npe_target
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "z_mean": z_mean,
            "z_std": z_std,
            "z_chol": z_chol,
            "linear_slope": slope,
            "observed_target_center": observed_target_center,
            "npe_parameterization": "case_transform" if case.npe_transform is not None else "raw_z",
            "args": vars(args),
        },
        output_dir / f"{case.name}_npe_model.pt",
    )
    return npe_z, {
        "observed_context": observed_context.tolist(),
        "region": json_ready(region),
        "collection": collect_summary,
        "training": training_summary,
        "context_dim": int(context.shape[1]),
        "z_dim": int(z.shape[1]),
        "local_weight_summary": summarize_vector(weights),
        "proposal": json_ready(proposal) if proposal is not None else {"kind": "prior"},
        "linear_adjustment": {
            "enabled": bool(use_linear_adjustment),
            "slope_frobenius_norm": None if slope is None else float(np.linalg.norm(slope)),
        },
        "full_target_whiten": bool(args.full_target_whiten),
        "npe_parameterization": "case_transform" if case.npe_transform is not None else "raw_z",
        "context_centering": bool(case.npe_center_from_context is not None),
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_case(case: StressCase, args: argparse.Namespace) -> dict[str, object]:
    if args.case_subdirs:
        case_output_dir = Path(args.output_root) / case.name / "results"
        case_figure_dir = Path(args.figure_root) / case.name / "figures"
    else:
        case_output_dir = Path(args.output_root) / "results"
        case_figure_dir = Path(args.figure_root) / "figures"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    case_figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x0 = case.simulate_x(case.true_z[None, :], rng)[0]
    mcmc_device = torch.device(args.mcmc_device)
    hmc_device = torch.device(args.hmc_device)
    npe_device = choose_device(args.npe_device)
    mcmc_dtype = torch_dtype_for(mcmc_device)
    hmc_dtype = torch_dtype_for(hmc_device)
    print(f"[{case.name}] observed x shape={x0.shape} npe_device={npe_device}")

    mcmc_z, mcmc_accept, mcmc_seconds = run_random_walk_mcmc(
        case,
        x0,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        seed=args.seed,
        device=mcmc_device,
        dtype=mcmc_dtype,
    )
    print(f"[{case.name}] MCMC seconds={mcmc_seconds:.2f} acceptance={mcmc_accept.mean():.3f}")

    hmc_z, hmc_accept, hmc_energy_error, hmc_seconds = run_hmc(
        case,
        x0,
        chains=args.hmc_chains,
        steps=args.hmc_steps,
        seed=args.seed,
        device=hmc_device,
        dtype=hmc_dtype,
    )
    print(f"[{case.name}] HMC seconds={hmc_seconds:.2f} acceptance={hmc_accept.mean():.3f}")

    hmc_reference_for_npe = flatten_post_burn(hmc_z, args.hmc_burn_in, thin=args.hmc_thin)
    npe_z, npe_summary = run_npe(
        case,
        x0,
        args=args,
        device=npe_device,
        output_dir=case_output_dir,
        proposal_reference_z=hmc_reference_for_npe,
    )
    print(f"[{case.name}] NPE samples={npe_z.shape[0]}")

    mcmc_post_z = flatten_post_burn(mcmc_z, args.mcmc_burn_in, thin=args.mcmc_thin)
    hmc_post_z = flatten_post_burn(hmc_z, args.hmc_burn_in, thin=args.hmc_thin)
    if mcmc_post_z.shape[0] > args.compare_sample_count:
        mcmc_post_z = subsample_rows(mcmc_post_z, args.compare_sample_count, rng)
    if hmc_post_z.shape[0] > args.compare_sample_count:
        hmc_post_z = subsample_rows(hmc_post_z, args.compare_sample_count, rng)
    if npe_z.shape[0] > args.compare_sample_count:
        npe_compare_z = subsample_rows(npe_z, args.compare_sample_count, rng)
    else:
        npe_compare_z = npe_z

    mcmc_raw_diagnostics = arviz_diagnostics(mcmc_z, args.mcmc_burn_in, case.display, case.param_names)
    hmc_raw_diagnostics = arviz_diagnostics(hmc_z, args.hmc_burn_in, case.display, case.param_names)
    mcmc_diag_diagnostics = arviz_diagnostics(
        mcmc_z,
        args.mcmc_burn_in,
        case.diagnostic_transform,
        case.diagnostic_names,
    )
    hmc_diag_diagnostics = arviz_diagnostics(
        hmc_z,
        args.hmc_burn_in,
        case.diagnostic_transform,
        case.diagnostic_names,
    )
    agreement = pairwise_agreement(case, mcmc_post_z, hmc_post_z, npe_compare_z)
    mode_metrics = None
    if case.mode_metric is not None:
        mode_metrics = {
            "mcmc": case.mode_metric(mcmc_post_z),
            "hmc": case.mode_metric(hmc_post_z),
            "npe": case.mode_metric(npe_compare_z),
        }

    np.savez_compressed(
        case_output_dir / f"{case.name}_samples.npz",
        x0=x0,
        true_z=case.true_z,
        mcmc_z=mcmc_z,
        hmc_z=hmc_z,
        npe_z=npe_z,
        mcmc_accept=mcmc_accept,
        hmc_accept=hmc_accept,
        hmc_energy_error=hmc_energy_error,
        param_names=np.array(case.param_names),
        diagnostic_names=np.array(case.diagnostic_names),
    )

    corner_path = case_figure_dir / f"{case.name}_mcmc_hmc_npe_corner.png"
    trace_path = case_figure_dir / f"{case.name}_trace.png"
    predictive_path = case_figure_dir / f"{case.name}_predictive.png"
    plot_corner_overlay(
        case,
        mcmc_z=mcmc_post_z,
        hmc_z=hmc_post_z,
        npe_z=npe_compare_z,
        outfile=corner_path,
        seed=args.seed,
        max_points=args.plot_sample_count,
    )
    plot_trace_overlay(
        case,
        mcmc_z=mcmc_z,
        hmc_z=hmc_z,
        burn_in_mcmc=args.mcmc_burn_in,
        burn_in_hmc=args.hmc_burn_in,
        outfile=trace_path,
    )
    plot_predictive_overlay(
        case,
        x0,
        mcmc_z=mcmc_post_z,
        hmc_z=hmc_post_z,
        npe_z=npe_compare_z,
        outfile=predictive_path,
        seed=args.seed,
        max_points=args.plot_sample_count,
    )

    display_summaries = {
        "mcmc": summarize_matrix(case.display(mcmc_post_z), case.param_names),
        "hmc": summarize_matrix(case.display(hmc_post_z), case.param_names),
        "npe": summarize_matrix(case.display(npe_compare_z), case.param_names),
    }
    max_diag_agreement = max(
        agreement["diagnostic"]["mcmc_hmc"]["mean"],
        agreement["diagnostic"]["mcmc_npe"]["mean"],
        agreement["diagnostic"]["hmc_npe"]["mean"],
    )
    diagnostic_target_met = (
        None if args.agreement_target is None else bool(max_diag_agreement <= args.agreement_target)
    )
    summary: dict[str, object] = {
        "case": case.name,
        "seed": args.seed,
        "x0": x0.tolist(),
        "true_z": case.true_z.tolist(),
        "true_display": case.display(case.true_z[None, :])[0].tolist(),
        "devices": {
            "mcmc": str(mcmc_device),
            "hmc": str(hmc_device),
            "npe": str(npe_device),
        },
        "runtime_seconds": {
            "mcmc": float(mcmc_seconds),
            "hmc": float(hmc_seconds),
            "npe_training": float(npe_summary["training"]["training_seconds"]),
            "npe_total_with_simulation": float(
                npe_summary["training"]["training_seconds"]
                + npe_summary["collection"]["collection_seconds"]
            ),
        },
        "acceptance": {
            "mcmc_mean": float(mcmc_accept.mean()),
            "mcmc_by_chain": mcmc_accept.mean(axis=1).tolist(),
            "hmc_mean": float(hmc_accept.mean()),
            "hmc_by_chain": hmc_accept.mean(axis=1).tolist(),
        },
        "hmc_energy_error_abs_summary": summarize_vector(np.abs(hmc_energy_error[:, args.hmc_burn_in :]).ravel()),
        "raw_diagnostics": {
            "mcmc": mcmc_raw_diagnostics,
            "hmc": hmc_raw_diagnostics,
        },
        "diagnostic_diagnostics": {
            "mcmc": mcmc_diag_diagnostics,
            "hmc": hmc_diag_diagnostics,
        },
        "convergence_flags": {
            "mcmc_raw": convergence_flags(
                mcmc_raw_diagnostics,
                rhat_threshold=args.rhat_threshold,
                ess_threshold=args.ess_threshold,
            ),
            "hmc_raw": convergence_flags(
                hmc_raw_diagnostics,
                rhat_threshold=args.rhat_threshold,
                ess_threshold=args.ess_threshold,
            ),
            "mcmc_diagnostic": convergence_flags(
                mcmc_diag_diagnostics,
                rhat_threshold=args.rhat_threshold,
                ess_threshold=args.ess_threshold,
            ),
            "hmc_diagnostic": convergence_flags(
                hmc_diag_diagnostics,
                rhat_threshold=args.rhat_threshold,
                ess_threshold=args.ess_threshold,
            ),
        },
        "agreement": agreement,
        "agreement_flags": {
            "target": None if args.agreement_target is None else float(args.agreement_target),
            "target_source": "not_set" if args.agreement_target is None else "explicit",
            "max_mean_diagnostic_wasserstein": float(max_diag_agreement),
            "diagnostic_target_met": diagnostic_target_met,
            "note": (
                "Pairwise MCMC/HMC/NPE agreement is reported for diagnostics. "
                "Use a model-specific grid/reference calibration for success claims."
            ),
        },
        "mode_metrics": mode_metrics,
        "display_summaries": display_summaries,
        "npe": npe_summary,
        "paths": {
            "samples": str(case_output_dir / f"{case.name}_samples.npz"),
            "summary": str(case_output_dir / f"{case.name}_summary.json"),
            "corner": str(corner_path),
            "trace": str(trace_path),
            "predictive": str(predictive_path),
        },
    }
    with (case_output_dir / f"{case.name}_summary.json").open("w") as handle:
        json.dump(json_ready(summary), handle, indent=2)
    print(
        f"[{case.name}] diagnostic Wasserstein means: "
        f"MCMC-HMC={agreement['diagnostic']['mcmc_hmc']['mean']:.4f}, "
        f"MCMC-NPE={agreement['diagnostic']['mcmc_npe']['mean']:.4f}, "
        f"HMC-NPE={agreement['diagnostic']['hmc_npe']['mean']:.4f}"
    )
    return summary


def make_default_initial_z(case: StressCase) -> InitialFn:
    def initial(chains: int, _x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(case.prior_mean[None, :], case.prior_std[None, :] * 0.8, size=(chains, case.z_dim))

    return initial


def sign_mode_metric(z: np.ndarray) -> dict[str, float]:
    positive = z[:, 0] > 0.0
    return {
        "positive_theta1_fraction": float(np.mean(positive)),
        "mode_mass_error_vs_half": float(abs(np.mean(positive) - 0.5)),
    }


def label_mode_metric(z: np.ndarray) -> dict[str, float]:
    first_low = z[:, 0] < z[:, 1]
    return {
        "mu1_less_than_mu2_fraction": float(np.mean(first_low)),
        "mode_mass_error_vs_half": float(abs(np.mean(first_low) - 0.5)),
    }


def mixture_em_summary(x: np.ndarray, *, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    lower = sorted_x[:, : n // 2]
    upper = sorted_x[:, n // 2 :]
    mu_low = lower.mean(axis=1)
    mu_high = upper.mean(axis=1)
    lower_var = lower.var(axis=1)
    upper_var = upper.var(axis=1)
    sigma = np.sqrt(np.maximum(0.5 * (lower_var + upper_var), 1e-5))
    for _ in range(iterations):
        inv_var = 1.0 / np.maximum(sigma[:, None] ** 2, 1e-8)
        log_low = -0.5 * (x - mu_low[:, None]) ** 2 * inv_var
        log_high = -0.5 * (x - mu_high[:, None]) ** 2 * inv_var
        responsibility_low = 1.0 / (1.0 + np.exp(np.clip(log_high - log_low, -50.0, 50.0)))
        responsibility_high = 1.0 - responsibility_low
        n_low = np.maximum(responsibility_low.sum(axis=1), 1e-6)
        n_high = np.maximum(responsibility_high.sum(axis=1), 1e-6)
        mu_low = (responsibility_low * x).sum(axis=1) / n_low
        mu_high = (responsibility_high * x).sum(axis=1) / n_high
        swap = mu_low > mu_high
        if np.any(swap):
            old_low = mu_low.copy()
            mu_low[swap] = mu_high[swap]
            mu_high[swap] = old_low[swap]
        variance = (
            responsibility_low * (x - mu_low[:, None]) ** 2
            + responsibility_high * (x - mu_high[:, None]) ** 2
        ).sum(axis=1) / x.shape[1]
        sigma = np.sqrt(np.maximum(variance, 1e-6))
    return mu_low, mu_high, np.log(np.maximum(sigma, 1e-6))


def two_exp_mode_metric(z: np.ndarray) -> dict[str, float]:
    k1 = np.exp(z[:, 1])
    k2 = np.exp(z[:, 3])
    first_slow = k1 < k2
    return {
        "k1_less_than_k2_fraction": float(np.mean(first_slow)),
        "mode_mass_error_vs_half": float(abs(np.mean(first_slow) - 0.5)),
    }


def make_sign_case() -> StressCase:
    sigma = np.array([0.22, 0.16])
    prior_mean = np.array([0.0, 0.0])
    prior_std = np.array([1.8, 1.8])
    true_z = np.array([0.85, -0.45])

    def mean_x(z: np.ndarray) -> np.ndarray:
        return np.column_stack([z[:, 0] ** 2, z[:, 1]])

    def simulate_x(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return mean_x(z) + rng.normal(0.0, sigma[None, :], size=(z.shape[0], 2))

    def context(x: np.ndarray) -> np.ndarray:
        return x

    def log_likelihood(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sigma_t = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)
        mean = torch.stack([z[:, 0].square(), z[:, 1]], dim=1)
        residual = (x0[None, :] - mean) / sigma_t[None, :]
        return (-0.5 * residual.square() - torch.log(sigma_t)[None, :] - 0.5 * LOG_2PI).sum(dim=1)

    def display(z: np.ndarray) -> np.ndarray:
        return z

    def diagnostic(z: np.ndarray) -> np.ndarray:
        return np.column_stack([np.abs(z[:, 0]), z[:, 1]])

    def npe_transform(z: np.ndarray) -> np.ndarray:
        return np.column_stack([np.abs(z[:, 0]), z[:, 1]])

    def npe_inverse(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sign = np.where(rng.random(u.shape[0]) < 0.5, -1.0, 1.0)
        return np.column_stack([sign * np.maximum(u[:, 0], 0.0), u[:, 1]])

    def initial(chains: int, x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        root = math.sqrt(max(float(x0[0]), 1e-4))
        z = np.empty((chains, 2))
        signs = np.where(np.arange(chains) % 2 == 0, 1.0, -1.0)
        z[:, 0] = signs * root + rng.normal(0.0, 0.08, size=chains)
        z[:, 1] = x0[1] + rng.normal(0.0, 0.08, size=chains)
        return z

    return StressCase(
        name="sign",
        z_dim=2,
        prior_mean=prior_mean,
        prior_std=prior_std,
        true_z=true_z,
        param_names=(r"$\theta_1$", r"$\theta_2$"),
        diagnostic_names=(r"$|\theta_1|$", r"$\theta_2$"),
        mcmc_proposal_scale=np.array([0.16, 0.13]),
        hmc_step_size=0.100,
        hmc_leapfrog_steps=14,
        simulate_x=simulate_x,
        mean_x=mean_x,
        context=context,
        log_likelihood=log_likelihood,
        display=display,
        diagnostic_transform=diagnostic,
        initial_z=initial,
        observed_kind="vector",
        mode_metric=sign_mode_metric,
        linear_adjustment_default=False,
        npe_transform=npe_transform,
        npe_inverse=npe_inverse,
    )


def make_banana_case() -> StressCase:
    sigma = np.array([0.20, 0.18])
    b = 0.65
    c = 0.70
    prior_mean = np.array([0.0, 0.0])
    prior_std = np.array([1.8, 1.8])
    true_z = np.array([0.90, -0.25])

    def mean_x(z: np.ndarray) -> np.ndarray:
        return np.column_stack([z[:, 0], z[:, 1] + b * (z[:, 0] ** 2 - c)])

    def simulate_x(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return mean_x(z) + rng.normal(0.0, sigma[None, :], size=(z.shape[0], 2))

    def context(x: np.ndarray) -> np.ndarray:
        return x

    def log_likelihood(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sigma_t = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)
        mean = torch.stack([z[:, 0], z[:, 1] + b * (z[:, 0].square() - c)], dim=1)
        residual = (x0[None, :] - mean) / sigma_t[None, :]
        return (-0.5 * residual.square() - torch.log(sigma_t)[None, :] - 0.5 * LOG_2PI).sum(dim=1)

    def display(z: np.ndarray) -> np.ndarray:
        return z

    def initial(chains: int, x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = np.empty((chains, 2))
        z[:, 0] = x0[0] + rng.normal(0.0, 0.20, size=chains)
        z[:, 1] = x0[1] - b * (z[:, 0] ** 2 - c) + rng.normal(0.0, 0.20, size=chains)
        return z

    return StressCase(
        name="banana",
        z_dim=2,
        prior_mean=prior_mean,
        prior_std=prior_std,
        true_z=true_z,
        param_names=(r"$\theta_1$", r"$\theta_2$"),
        diagnostic_names=(r"$\theta_1$", r"$\theta_2$"),
        mcmc_proposal_scale=np.array([0.20, 0.20]),
        hmc_step_size=0.070,
        hmc_leapfrog_steps=16,
        simulate_x=simulate_x,
        mean_x=mean_x,
        context=context,
        log_likelihood=log_likelihood,
        display=display,
        diagnostic_transform=display,
        initial_z=initial,
        observed_kind="vector",
    )


def make_label_switch_case() -> StressCase:
    n_obs = 80
    prior_mean = np.array([0.0, 0.0, math.log(0.45)])
    prior_std = np.array([2.2, 2.2, 0.55])
    true_z = np.array([-1.25, 1.15, math.log(0.34)])

    def mean_x(z: np.ndarray) -> np.ndarray:
        return np.column_stack([z[:, 0], z[:, 1], np.exp(z[:, 2])])

    def simulate_x(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        choose_first = rng.random((z.shape[0], n_obs)) < 0.5
        mean = np.where(choose_first, z[:, 0:1], z[:, 1:2])
        return mean + rng.normal(0.0, np.exp(z[:, 2:3]), size=(z.shape[0], n_obs))

    def context(x: np.ndarray) -> np.ndarray:
        quantiles = np.quantile(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=1).T
        mean = x.mean(axis=1)
        log_std = np.log(np.maximum(x.std(axis=1), 1e-6))
        centered = x - mean[:, None]
        skew = np.mean(centered**3, axis=1) / np.maximum(x.std(axis=1), 1e-6) ** 3
        sorted_x = np.sort(x, axis=1)
        lower = sorted_x[:, : n_obs // 2]
        upper = sorted_x[:, n_obs // 2 :]
        lower_mean = lower.mean(axis=1)
        upper_mean = upper.mean(axis=1)
        lower_std = np.maximum(lower.std(axis=1), 1e-6)
        upper_std = np.maximum(upper.std(axis=1), 1e-6)
        pooled_within = np.sqrt(0.5 * (lower_std**2 + upper_std**2))
        split_gap = upper_mean - lower_mean
        em_low, em_high, em_log_sigma = mixture_em_summary(x, iterations=14)
        return np.column_stack([
            mean,
            log_std,
            skew,
            quantiles,
            lower_mean,
            upper_mean,
            np.log(lower_std),
            np.log(upper_std),
            np.log(np.maximum(pooled_within, 1e-6)),
            split_gap,
            em_low,
            em_high,
            em_log_sigma,
        ])

    def log_likelihood(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mu1 = z[:, 0:1]
        mu2 = z[:, 1:2]
        log_sigma = z[:, 2:3]
        sigma = torch.exp(log_sigma)
        y = x0[None, :]
        log_a = -0.5 * ((y - mu1) / sigma).square() - log_sigma - 0.5 * LOG_2PI
        log_b = -0.5 * ((y - mu2) / sigma).square() - log_sigma - 0.5 * LOG_2PI
        return (torch.logaddexp(log_a, log_b) - LOG_2).sum(dim=1)

    def display(z: np.ndarray) -> np.ndarray:
        return np.column_stack([z[:, 0], z[:, 1], np.exp(z[:, 2])])

    def diagnostic(z: np.ndarray) -> np.ndarray:
        low = np.minimum(z[:, 0], z[:, 1])
        high = np.maximum(z[:, 0], z[:, 1])
        return np.column_stack([low, high, np.exp(z[:, 2])])

    def npe_transform(z: np.ndarray) -> np.ndarray:
        low = np.minimum(z[:, 0], z[:, 1])
        high = np.maximum(z[:, 0], z[:, 1])
        return np.column_stack([low, high, z[:, 2]])

    def npe_inverse(ordered_z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raw = ordered_z.copy()
        swap = rng.random(raw.shape[0]) < 0.5
        raw[swap, 0] = ordered_z[swap, 1]
        raw[swap, 1] = ordered_z[swap, 0]
        return raw

    def initial(chains: int, x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        low = float(np.quantile(x0, 0.25))
        high = float(np.quantile(x0, 0.75))
        sigma0 = max(float(np.std(x0)) * 0.35, 0.10)
        z = np.empty((chains, 3))
        for chain in range(chains):
            if chain % 2 == 0:
                z[chain, :2] = [low, high]
            else:
                z[chain, :2] = [high, low]
        z[:, :2] += rng.normal(0.0, 0.12, size=(chains, 2))
        z[:, 2] = math.log(sigma0) + rng.normal(0.0, 0.08, size=chains)
        return z

    return StressCase(
        name="label_switch",
        z_dim=3,
        prior_mean=prior_mean,
        prior_std=prior_std,
        true_z=true_z,
        param_names=(r"$\mu_1$", r"$\mu_2$", r"$\sigma$"),
        diagnostic_names=(r"$\mu_{low}$", r"$\mu_{high}$", r"$\sigma$"),
        mcmc_proposal_scale=np.array([0.055, 0.055, 0.035]),
        hmc_step_size=0.010,
        hmc_leapfrog_steps=28,
        simulate_x=simulate_x,
        mean_x=mean_x,
        context=context,
        log_likelihood=log_likelihood,
        display=display,
        diagnostic_transform=diagnostic,
        initial_z=initial,
        observed_kind="label_switch",
        mode_metric=label_mode_metric,
        npe_transform=npe_transform,
        npe_inverse=npe_inverse,
    )


def make_linear6_case() -> StressCase:
    n_obs = 32
    t = np.linspace(0.0, 1.0, n_obs)
    basis = np.column_stack([
        np.ones_like(t),
        t - 0.5,
        np.sin(2.0 * math.pi * t),
        np.cos(2.0 * math.pi * t),
        np.sin(4.0 * math.pi * t),
        np.cos(4.0 * math.pi * t),
    ])
    q, _ = np.linalg.qr(basis)
    basis = q * math.sqrt(n_obs)
    d_w = basis.shape[1]
    prior_mean = np.concatenate([np.zeros(d_w), [math.log(0.25)]])
    prior_std = np.concatenate([np.ones(d_w) * 1.25, [0.50]])
    true_z = np.array([0.70, -0.35, 0.80, -0.20, 0.35, 0.12, math.log(0.20)])
    pinv = np.linalg.pinv(basis)

    def mean_x(z: np.ndarray) -> np.ndarray:
        return z[:, :d_w] @ basis.T

    def simulate_x(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return mean_x(z) + rng.normal(0.0, np.exp(z[:, -1:]), size=(z.shape[0], n_obs))

    def context(x: np.ndarray) -> np.ndarray:
        coef = x @ pinv.T
        fitted = coef @ basis.T
        residual = x - fitted
        log_sigma_hat = np.log(np.maximum(np.sqrt(np.mean(residual**2, axis=1)), 1e-6))
        return np.column_stack([coef, log_sigma_hat])

    def log_likelihood(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        basis_t = torch.as_tensor(basis, device=z.device, dtype=z.dtype)
        mean = z[:, :d_w] @ basis_t.T
        log_sigma = z[:, -1:]
        residual = x0[None, :] - mean
        return (-0.5 * residual.square() * torch.exp(-2.0 * log_sigma) - log_sigma - 0.5 * LOG_2PI).sum(dim=1)

    def display(z: np.ndarray) -> np.ndarray:
        return np.column_stack([z[:, :d_w], np.exp(z[:, -1])])

    def initial(chains: int, x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        context0 = context(x0[None, :])[0]
        z = np.empty((chains, d_w + 1))
        z[:, :d_w] = context0[:d_w][None, :] + rng.normal(0.0, 0.08, size=(chains, d_w))
        z[:, -1] = context0[-1] + rng.normal(0.0, 0.05, size=chains)
        return z

    return StressCase(
        name="linear6",
        z_dim=d_w + 1,
        prior_mean=prior_mean,
        prior_std=prior_std,
        true_z=true_z,
        param_names=(r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$", r"$w_5$", r"$w_6$", r"$\sigma$"),
        diagnostic_names=(r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$", r"$w_5$", r"$w_6$", r"$\sigma$"),
        mcmc_proposal_scale=np.array([0.040] * d_w + [0.032]),
        hmc_step_size=0.018,
        hmc_leapfrog_steps=22,
        simulate_x=simulate_x,
        mean_x=mean_x,
        context=context,
        log_likelihood=log_likelihood,
        display=display,
        diagnostic_transform=display,
        initial_z=initial,
        observed_axis=t,
        observed_kind="curve",
    )


def two_exp_context(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = y.shape[1]
    sample_indices = np.linspace(0, n - 1, 14).round().astype(int)
    downsample = y[:, sample_indices]
    area = np.trapezoid(y, t, axis=1)
    early = np.maximum(y[:, 0] - y[:, max(1, n // 5)], 1e-6)
    middle = np.maximum(y[:, n // 3] - y[:, 2 * n // 3], 1e-6)
    late = np.maximum(y[:, -max(2, n // 5)] - y[:, -1], 1e-6)
    log_slopes = np.column_stack([
        np.log(early),
        np.log(middle),
        np.log(late),
    ])
    return np.column_stack([downsample, area, log_slopes])


def make_two_exp_profile_summary(t: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    rates = np.exp(np.linspace(math.log(0.12), math.log(2.40), 36))
    pairs = np.array([(rates[i], rates[j]) for i in range(len(rates)) for j in range(i + 1, len(rates))])
    basis = np.stack([
        np.exp(-pairs[:, 0, None] * t[None, :]),
        np.exp(-pairs[:, 1, None] * t[None, :]),
    ], axis=1)
    gram = np.einsum("pkt,plt->pkl", basis, basis)
    inv_gram = np.linalg.inv(gram + np.eye(2)[None, :, :] * 1e-8)

    def profile(y: np.ndarray) -> np.ndarray:
        bty = np.einsum("nt,pkt->npk", y, basis)
        amplitudes = np.einsum("pkl,npl->npk", inv_gram, bty)
        y2 = np.sum(y**2, axis=1)[:, None]
        sse = y2 - np.einsum("npk,npk->np", amplitudes, bty)
        sse = np.where(np.all(amplitudes > 1e-6, axis=2), sse, np.inf)
        fallback_sse = y2 - np.einsum("npk,npk->np", np.maximum(amplitudes, 1e-6), bty)
        bad = ~np.any(np.isfinite(sse), axis=1)
        if np.any(bad):
            sse[bad] = fallback_sse[bad]
        best = np.argmin(sse, axis=1)
        row = np.arange(y.shape[0])
        best_amplitudes = np.maximum(amplitudes[row, best], 1e-6)
        best_rates = pairs[best]
        best_sse = np.maximum(sse[row, best], 1e-8)
        sigma_hat = np.sqrt(best_sse / y.shape[1])
        return np.column_stack([
            np.log(best_amplitudes[:, 0]),
            np.log(best_rates[:, 0]),
            np.log(best_amplitudes[:, 1]),
            np.log(best_rates[:, 1]),
            np.log(np.maximum(sigma_hat, 1e-6)),
        ])

    return profile


def make_two_exp_case(*, ordered: bool) -> StressCase:
    n_obs = 45
    t = np.linspace(0.0, 6.0, n_obs)
    profile_summary = make_two_exp_profile_summary(t)
    sigma_true = 0.18
    if ordered:
        name = "two_exp_ordered"
        prior_mean = np.array([math.log(2.5), math.log(0.35), math.log(1.4), math.log(0.75), math.log(0.25)])
        prior_std = np.array([0.60, 0.55, 0.65, 0.60, 0.45])
        true_z = np.array([math.log(2.7), math.log(0.32), math.log(1.35), math.log(0.90), math.log(sigma_true)])
        param_names = (r"$A_1$", r"$k_1$", r"$A_2$", r"$k_2$", r"$\sigma$")
        diagnostic_names = param_names
        mode_metric = None
    else:
        name = "two_exp_unordered"
        prior_mean = np.array([math.log(2.2), math.log(0.55), math.log(2.2), math.log(0.55), math.log(0.25)])
        prior_std = np.array([0.70, 0.75, 0.70, 0.75, 0.45])
        true_z = np.array([math.log(2.7), math.log(0.32), math.log(1.35), math.log(1.22), math.log(sigma_true)])
        param_names = (r"$A_1$", r"$k_1$", r"$A_2$", r"$k_2$", r"$\sigma$")
        diagnostic_names = (r"$A_{slow}$", r"$k_{slow}$", r"$A_{fast}$", r"$k_{fast}$", r"$\sigma$")
        mode_metric = two_exp_mode_metric

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a1 = np.exp(z[:, 0])
        k1 = np.exp(z[:, 1])
        a2 = np.exp(z[:, 2])
        if ordered:
            k2 = k1 + np.exp(z[:, 3])
        else:
            k2 = np.exp(z[:, 3])
        sigma = np.exp(z[:, 4])
        return a1, k1, a2, k2, sigma

    def mean_x(z: np.ndarray) -> np.ndarray:
        a1, k1, a2, k2, _sigma = unpack(z)
        return a1[:, None] * np.exp(-k1[:, None] * t[None, :]) + a2[:, None] * np.exp(-k2[:, None] * t[None, :])

    def simulate_x(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        _a1, _k1, _a2, _k2, sigma = unpack(z)
        return mean_x(z) + rng.normal(0.0, sigma[:, None], size=(z.shape[0], n_obs))

    def context(x: np.ndarray) -> np.ndarray:
        return np.column_stack([profile_summary(x), x])

    def log_likelihood(z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        t_t = torch.as_tensor(t, device=z.device, dtype=z.dtype)
        a1 = torch.exp(z[:, 0:1])
        k1 = torch.exp(z[:, 1:2])
        a2 = torch.exp(z[:, 2:3])
        if ordered:
            k2 = k1 + torch.exp(z[:, 3:4])
        else:
            k2 = torch.exp(z[:, 3:4])
        log_sigma = z[:, 4:5]
        mean = a1 * torch.exp(-k1 * t_t[None, :]) + a2 * torch.exp(-k2 * t_t[None, :])
        residual = x0[None, :] - mean
        return (-0.5 * residual.square() * torch.exp(-2.0 * log_sigma) - log_sigma - 0.5 * LOG_2PI).sum(dim=1)

    def display(z: np.ndarray) -> np.ndarray:
        a1, k1, a2, k2, sigma = unpack(z)
        return np.column_stack([a1, k1, a2, k2, sigma])

    def diagnostic(z: np.ndarray) -> np.ndarray:
        values = display(z)
        if ordered:
            return values
        first_slow = values[:, 1] <= values[:, 3]
        slow_a = np.where(first_slow, values[:, 0], values[:, 2])
        slow_k = np.where(first_slow, values[:, 1], values[:, 3])
        fast_a = np.where(first_slow, values[:, 2], values[:, 0])
        fast_k = np.where(first_slow, values[:, 3], values[:, 1])
        return np.column_stack([slow_a, slow_k, fast_a, fast_k, values[:, 4]])

    def ridge_transform(z: np.ndarray) -> np.ndarray:
        values = display(z)
        a1 = values[:, 0]
        k1 = values[:, 1]
        a2 = values[:, 2]
        k2 = values[:, 3]
        sigma = values[:, 4]
        total = np.maximum(a1 + a2, 1e-8)
        ratio = np.log(np.maximum(a1, 1e-8)) - np.log(np.maximum(a2, 1e-8))
        delta = np.maximum(k2 - k1, 1e-8)
        return np.column_stack([
            np.log(total),
            ratio,
            np.log(np.maximum(k1, 1e-8)),
            np.log(delta),
            np.log(np.maximum(sigma, 1e-8)),
        ])

    def ridge_inverse(u: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
        total = np.exp(u[:, 0])
        fraction = 1.0 / (1.0 + np.exp(-np.clip(u[:, 1], -40.0, 40.0)))
        a1 = np.maximum(total * fraction, 1e-8)
        a2 = np.maximum(total * (1.0 - fraction), 1e-8)
        k1 = np.exp(u[:, 2])
        delta = np.exp(u[:, 3])
        sigma = np.exp(u[:, 4])
        return np.column_stack([
            np.log(a1),
            np.log(k1),
            np.log(a2),
            np.log(delta),
            np.log(sigma),
        ])

    def center_from_context(context_values: np.ndarray) -> np.ndarray:
        log_a1 = context_values[:, 0]
        log_k1 = context_values[:, 1]
        log_a2 = context_values[:, 2]
        log_k2 = context_values[:, 3]
        log_sigma = context_values[:, 4]
        if ordered:
            a1 = np.exp(log_a1)
            a2 = np.exp(log_a2)
            total = np.maximum(a1 + a2, 1e-8)
            ratio = log_a1 - log_a2
            delta = np.maximum(np.exp(log_k2) - np.exp(log_k1), 1e-8)
            return np.column_stack([np.log(total), ratio, log_k1, np.log(delta), log_sigma])
        return np.column_stack([log_a1, log_k1, log_a2, log_k2, log_sigma])

    def initial(chains: int, _x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = np.tile(true_z[None, :], (chains, 1))
        if not ordered:
            for chain in range(chains):
                if chain % 2 == 1:
                    z[chain, 0], z[chain, 2] = z[chain, 2], z[chain, 0]
                    z[chain, 1], z[chain, 3] = z[chain, 3], z[chain, 1]
        z += rng.normal(0.0, np.array([0.08, 0.06, 0.10, 0.08, 0.05]), size=z.shape)
        return z

    return StressCase(
        name=name,
        z_dim=5,
        prior_mean=prior_mean,
        prior_std=prior_std,
        true_z=true_z,
        param_names=param_names,
        diagnostic_names=diagnostic_names,
        mcmc_proposal_scale=np.array([0.040, 0.035, 0.050, 0.040, 0.030]),
        hmc_step_size=0.006,
        hmc_leapfrog_steps=34,
        simulate_x=simulate_x,
        mean_x=mean_x,
        context=context,
        log_likelihood=log_likelihood,
        display=display,
        diagnostic_transform=diagnostic,
        initial_z=initial,
        observed_axis=t,
        observed_kind="curve",
        mode_metric=mode_metric,
        npe_transform=ridge_transform if ordered else None,
        npe_inverse=ridge_inverse if ordered else None,
        npe_center_from_context=center_from_context if ordered else None,
    )


def build_cases() -> dict[str, StressCase]:
    cases = [
        make_sign_case(),
        make_banana_case(),
        make_label_switch_case(),
        make_linear6_case(),
        make_two_exp_case(ordered=True),
        make_two_exp_case(ordered=False),
    ]
    return {case.name: case for case in cases}


def write_report(summaries: list[dict[str, object]], outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NPE Flow Stress-Test Results",
        "",
        "This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.",
        "",
        "| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        agreement = summary["agreement"]["diagnostic"]
        flags = summary["convergence_flags"]
        target = summary["agreement_flags"]
        lines.append(
            "| {case} | {mcmc_ok} | {hmc_ok} | {max_w:.4f} | {target_met} | {mcmc_hmc:.4f} | {mcmc_npe:.4f} | {hmc_npe:.4f} |".format(
                case=summary["case"],
                mcmc_ok=flags["mcmc_diagnostic"]["ok"],
                hmc_ok=flags["hmc_diagnostic"]["ok"],
                max_w=target["max_mean_diagnostic_wasserstein"],
                target_met=target["diagnostic_target_met"],
                mcmc_hmc=agreement["mcmc_hmc"]["mean"],
                mcmc_npe=agreement["mcmc_npe"]["mean"],
                hmc_npe=agreement["hmc_npe"]["mean"],
            )
        )
    lines.append("")
    for summary in summaries:
        lines.extend([
            f"## {summary['case']}",
            "",
            f"- Summary JSON: `{summary['paths']['summary']}`",
            f"- Corner overlay: `{summary['paths']['corner']}`",
            f"- Trace plot: `{summary['paths']['trace']}`",
            f"- Predictive plot: `{summary['paths']['predictive']}`",
            f"- Runtime seconds: MCMC {summary['runtime_seconds']['mcmc']:.2f}, HMC {summary['runtime_seconds']['hmc']:.2f}, NPE train {summary['runtime_seconds']['npe_training']:.2f}",
            f"- MCMC acceptance: {summary['acceptance']['mcmc_mean']:.3f}",
            f"- HMC acceptance: {summary['acceptance']['hmc_mean']:.3f}",
            "",
        ])
        if summary["mode_metrics"] is not None:
            lines.append(f"- Mode metrics: `{json.dumps(summary['mode_metrics'])}`")
            lines.append("")
    outfile.write_text("\n".join(lines) + "\n")


def default_output_root_for_case(case_name: str) -> Path:
    roots = {
        "sign": ap.STRESS_SIGN_NPE_ROOT / "00_npe_flow_stress_tests",
        "banana": ap.STRESS_BANANA_NPE_ROOT / "00_npe_flow_stress_tests",
        "label_switch": ap.STRESS_LABEL_SWITCH_NPE_ROOT / "00_npe_flow_stress_tests",
        "linear6": ap.STRESS_LINEAR6_NPE_ROOT / "00_npe_flow_stress_tests",
        "two_exp_ordered": ap.TWO_EXP_NPE_ROOT / "00_npe_flow_stress_tests_two_exp_ordered",
    }
    return roots[case_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(build_cases().keys()), action="append")
    parser.add_argument("--all", action="store_true", help="Run every implemented stress-test case.")
    parser.add_argument(
        "--case-subdirs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write each case under its own subdirectory. Defaults to on for multi-case runs and off for one-case runs.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--figure-root", default=None)
    parser.add_argument("--report", default="notes/npe-flow-stress-test-results.md")
    parser.add_argument("--mcmc-device", default="cpu")
    parser.add_argument("--hmc-device", default="cpu")
    parser.add_argument("--npe-device", default="auto")
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=8000)
    parser.add_argument("--mcmc-burn-in", type=int, default=3000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--hmc-chains", type=int, default=8)
    parser.add_argument("--hmc-steps", type=int, default=3000)
    parser.add_argument("--hmc-burn-in", type=int, default=800)
    parser.add_argument("--hmc-thin", type=int, default=1)
    parser.add_argument("--local-pilot", type=int, default=80_000)
    parser.add_argument("--local-quantile", type=float, default=0.025)
    parser.add_argument("--kernel-quantile", type=float, default=0.025)
    parser.add_argument("--local-max-candidates", type=int, default=2_500_000)
    parser.add_argument("--simulate-chunk-size", type=int, default=20_000)
    parser.add_argument("--npe-train-count", type=int, default=45_000)
    parser.add_argument("--npe-val-count", type=int, default=8_000)
    parser.add_argument("--npe-samples", type=int, default=60_000)
    parser.add_argument("--npe-proposal", choices=("prior", "hmc_gaussian"), default="prior")
    parser.add_argument("--proposal-inflation", type=float, default=1.5)
    parser.add_argument("--proposal-prior-mixture", type=float, default=0.02)
    parser.add_argument("--linear-adjustment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--linear-ridge", type=float, default=1e-3)
    parser.add_argument("--full-target-whiten", action="store_true")
    parser.add_argument("--transforms", type=int, default=8)
    parser.add_argument("--hidden-features", type=parse_hidden_features, default=(128, 128))
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--compare-sample-count", type=int, default=50_000)
    parser.add_argument("--plot-sample-count", type=int, default=8_000)
    parser.add_argument(
        "--agreement-target",
        type=float,
        default=None,
        help=(
            "Optional pairwise diagnostic agreement threshold. Leave unset for unscored "
            "stress runs; calibrated success should come from MCMC/HMC-to-grid reference checks."
        ),
    )
    parser.add_argument("--rhat-threshold", type=float, default=1.05)
    parser.add_argument("--ess-threshold", type=float, default=400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases()
    if args.all:
        selected_names = list(cases.keys())
    elif args.case:
        selected_names = args.case
    else:
        raise SystemExit("Specify --case CASE or --all.")
    if args.output_root is None and args.figure_root is None:
        if len(selected_names) != 1:
            raise SystemExit("Specify --output-root and --figure-root for multi-case stress-test runs.")
        default_root = default_output_root_for_case(selected_names[0])
        args.output_root = default_root
        args.figure_root = default_root
    elif args.output_root is None:
        args.output_root = args.figure_root
    elif args.figure_root is None:
        args.figure_root = args.output_root
    if args.case_subdirs is None:
        args.case_subdirs = len(selected_names) > 1
    summaries = []
    for case_name in selected_names:
        summaries.append(run_case(cases[case_name], args))
    write_report(summaries, Path(args.report))


if __name__ == "__main__":
    main()
