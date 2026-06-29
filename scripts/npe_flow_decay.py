from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import artifact_paths as ap

import corner
import matplotlib
import numpy as np
import torch
import zuko
from scipy.special import logsumexp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abc_faithfulness_decay import (
    decay_fit_summaries,
    make_k_grid,
    sample_gaussian_proposal,
)
from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, subsample, summarize_samples
from corner_truth import overplot_true_values, true_theta_legend_handle
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_stage1_decay import posterior_predictive_band, sample_grid_reference

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOG_2PI = math.log(2.0 * math.pi)

PROPOSAL_SOURCES = {
    "snpe_diag": {
        "path": ap.SNPE_SEQUENTIAL_GAUSSIANS_RESULTS / "snpe_sequential_samples.npz",
        "key": "z_final_corrected_diag_gaussian",
        "label": "SNPE diagonal",
    },
    "snpe_mdn": {
        "path": ap.SNPE_SEQUENTIAL_MDN_RESULTS / "snpe_sequential_samples.npz",
        "key": "z_final_corrected_mdn",
        "label": "SNPE MDN",
    },
    "local_mdn": {
        "path": ap.NPE_LOCAL_REGION_Q0005_MDN_20K_RESULTS / "npe_local_region_samples.npz",
        "key": "z_samples_mdn",
        "label": "local MDN",
    },
    "stage1_mdn": {
        "path": ap.NPE_STAGE1_SCALED_RESULTS / "npe_stage1_samples.npz",
        "key": "z_samples_mdn",
        "label": "broad MDN",
    },
}


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


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


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


def gaussian_logpdf(
    z: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    inv = np.linalg.inv(covariance)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    delta = z - mean[None, :]
    maha = np.sum((delta @ inv) * delta, axis=1)
    return -0.5 * (z.shape[1] * LOG_2PI + log_det + maha)


def fit_gaussian_proposal(z_samples: np.ndarray, inflation: float) -> dict[str, np.ndarray | float]:
    mean = z_samples.mean(axis=0)
    covariance = np.cov(z_samples, rowvar=False) * inflation**2
    covariance = covariance + np.eye(z_samples.shape[1]) * 1e-5
    chol = np.linalg.cholesky(covariance)
    return {
        "mean": mean,
        "covariance": covariance,
        "cholesky": chol,
        "inflation": inflation,
    }


def load_proposal_source(source: str) -> np.ndarray:
    spec = PROPOSAL_SOURCES[source]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Proposal source is missing: {path}")
    data = np.load(path, allow_pickle=True)
    key = spec["key"]
    if key not in data.files:
        raise KeyError(f"{path} has no key {key}. Keys: {data.files}")
    return np.asarray(data[key], dtype=np.float64)


def mean_normalized_wasserstein_value(result: dict[str, object]) -> float:
    value = result["mean_normalized_wasserstein"]
    if isinstance(value, dict):
        return float(value["value"])
    return float(value)


def sample_proposal_z(
    *,
    n: int,
    source: str,
    inflation: float,
    prior_mixture: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if source == "prior":
        z = sample_prior_z(n, rng)
        log_r = prior_logpdf_z(z)
        return z, log_r, {
            "source": "prior",
            "kind": "prior",
            "inflation": 1.0,
        }

    source_z = load_proposal_source(source)
    proposal = fit_gaussian_proposal(source_z, inflation)
    mean = np.asarray(proposal["mean"])
    covariance = np.asarray(proposal["covariance"])
    prior_mixture = float(prior_mixture)
    if not 0.0 <= prior_mixture < 1.0:
        raise ValueError("--proposal-prior-mixture must be in [0, 1).")
    if prior_mixture > 0.0:
        use_prior = rng.random(n) < prior_mixture
        z = np.empty((n, 3), dtype=np.float64)
        z[use_prior] = sample_prior_z(int(use_prior.sum()), rng)
        z[~use_prior] = sample_gaussian_proposal(proposal, int((~use_prior).sum()), rng)
        log_prior = prior_logpdf_z(z)
        log_gaussian = gaussian_logpdf(z, mean, covariance)
        log_r = logsumexp(
            np.column_stack([
                math.log(prior_mixture) + log_prior,
                math.log1p(-prior_mixture) + log_gaussian,
            ]),
            axis=1,
        )
    else:
        z = sample_gaussian_proposal(proposal, n, rng)
        log_r = gaussian_logpdf(z, mean, covariance)
    metadata = {
        "source": source,
        "label": PROPOSAL_SOURCES[source]["label"],
        "path": str(PROPOSAL_SOURCES[source]["path"]),
        "key": PROPOSAL_SOURCES[source]["key"],
        "kind": "inflated_gaussian",
        "inflation": inflation,
        "prior_mixture": prior_mixture,
        "source_samples": int(source_z.shape[0]),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
    }
    return z, log_r, metadata


def simulate_context_from_z(
    *,
    z: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    k_grid: np.ndarray,
    simulate_chunk_size: int,
    summary_chunk_size: int,
    context_kind: str,
) -> np.ndarray:
    context_dim = 3 if context_kind == "indirect" else 8
    context = np.empty((z.shape[0], context_dim), dtype=np.float64)
    for start in range(0, z.shape[0], simulate_chunk_size):
        stop = min(start + simulate_chunk_size, z.shape[0])
        theta = np.exp(z[start:stop])
        mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
        x = mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)
        context[start:stop] = make_context_summaries(
            x,
            t,
            k_grid,
            kind=context_kind,
            chunk_size=summary_chunk_size,
        )
    return context


def profile_sse_at_k(x: np.ndarray, t: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.exp(-k[:, None] * t[None, :])
    denom = np.sum(basis**2, axis=1)
    amplitude = np.maximum(np.sum(x * basis, axis=1) / np.maximum(denom, 1e-12), 1e-8)
    residual = x - amplitude[:, None] * basis
    sse = np.maximum(np.sum(residual**2, axis=1), 1e-12)
    return amplitude, sse


def make_context_summaries(
    x: np.ndarray,
    t: np.ndarray,
    k_grid: np.ndarray,
    *,
    kind: str,
    chunk_size: int,
) -> np.ndarray:
    base = decay_fit_summaries(x, t, k_grid, chunk_size=chunk_size)
    if kind == "indirect":
        return base
    if kind != "enhanced":
        raise ValueError("context_kind must be indirect or enhanced")

    log_a = base[:, 0]
    log_k = base[:, 1]
    log_sigma = base[:, 2]
    a = np.exp(log_a)
    k = np.exp(log_k)
    sigma = np.maximum(np.exp(log_sigma), 1e-8)
    fitted = a[:, None] * np.exp(-k[:, None] * t[None, :])
    residual = x - fitted
    standardized = residual / sigma[:, None]
    residual_mean = standardized.mean(axis=1)
    residual_skew = np.mean(standardized**3, axis=1)
    residual_lag = np.mean(standardized[:, :-1] * standardized[:, 1:], axis=1)
    early = standardized[:, : max(1, x.shape[1] // 5)].mean(axis=1)
    late = standardized[:, -max(1, x.shape[1] // 5) :].mean(axis=1)
    _, sse0 = profile_sse_at_k(x, t, k)
    _, sse_minus = profile_sse_at_k(x, t, k * np.exp(-0.08))
    _, sse_plus = profile_sse_at_k(x, t, k * np.exp(0.08))
    curvature = np.maximum((sse_minus - 2.0 * sse0 + sse_plus) / np.maximum(sse0, 1e-12), 1e-8)
    log_curvature = np.log(curvature)
    return np.column_stack([
        log_a,
        log_k,
        log_sigma,
        residual_mean,
        residual_skew,
        residual_lag,
        early - late,
        log_curvature,
    ])


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
    *,
    observed_context: np.ndarray,
    t: np.ndarray,
    k_grid: np.ndarray,
    simulations: int,
    quantile: float,
    kernel_quantile: float,
    rng: np.random.Generator,
    simulate_chunk_size: int,
    summary_chunk_size: int,
    context_kind: str,
) -> dict[str, object]:
    z = sample_prior_z(simulations, rng)
    context = simulate_context_from_z(
        z=z,
        t=t,
        rng=rng,
        k_grid=k_grid,
        simulate_chunk_size=simulate_chunk_size,
        summary_chunk_size=summary_chunk_size,
        context_kind=context_kind,
    )
    center = context.mean(axis=0)
    scale = np.maximum(context.std(axis=0), 1e-6)
    distances = context_distances(context, observed_context, center, scale)
    radius = float(np.quantile(distances, quantile))
    kernel_bandwidth = None
    if kernel_quantile > 0.0:
        kernel_bandwidth = float(np.quantile(distances, kernel_quantile))
    return {
        "radius": radius,
        "quantile": quantile,
        "kernel_quantile": kernel_quantile,
        "kernel_bandwidth": kernel_bandwidth,
        "center": center,
        "scale": scale,
        "pilot_simulations": simulations,
        "pilot_distance_summary": {
            "min": float(distances.min()),
            "q001": float(np.quantile(distances, 0.001)),
            "q005": float(np.quantile(distances, 0.005)),
            "q01": float(np.quantile(distances, 0.01)),
            "q02": float(np.quantile(distances, 0.02)),
            "median": float(np.median(distances)),
        },
    }


def collect_local_prior_data(
    *,
    target_count: int,
    observed_context: np.ndarray,
    t: np.ndarray,
    k_grid: np.ndarray,
    region: dict[str, object],
    rng: np.random.Generator,
    simulate_chunk_size: int,
    summary_chunk_size: int,
    context_kind: str,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    accepted_z: list[np.ndarray] = []
    accepted_context: list[np.ndarray] = []
    accepted_distances: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    start = time.perf_counter()
    center = np.asarray(region["center"])
    scale = np.asarray(region["scale"])
    radius = float(region["radius"])
    while accepted_count < target_count and candidate_count < max_candidates:
        current = min(simulate_chunk_size, max_candidates - candidate_count)
        z = sample_prior_z(current, rng)
        context = simulate_context_from_z(
            z=z,
            t=t,
            rng=rng,
            k_grid=k_grid,
            simulate_chunk_size=simulate_chunk_size,
            summary_chunk_size=summary_chunk_size,
            context_kind=context_kind,
        )
        distances = context_distances(context, observed_context, center, scale)
        mask = distances <= radius
        if np.any(mask):
            accepted_z.append(z[mask])
            accepted_context.append(context[mask])
            accepted_distances.append(distances[mask])
            accepted_count += int(mask.sum())
        candidate_count += current

    if accepted_count < target_count:
        raise RuntimeError(
            f"Only accepted {accepted_count} local simulations after {candidate_count} candidates; "
            f"need {target_count}. Increase --local-max-candidates or --local-quantile."
        )

    z_all = np.concatenate(accepted_z, axis=0)[:target_count]
    context_all = np.concatenate(accepted_context, axis=0)[:target_count]
    distance_all = np.concatenate(accepted_distances, axis=0)[:target_count]
    diagnostics = {
        "candidate_count": int(candidate_count),
        "accepted_count": int(target_count),
        "raw_accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / max(candidate_count, 1)),
        "collection_seconds": float(time.perf_counter() - start),
        "accepted_distance_summary": {
            "min": float(distance_all.min()),
            "median": float(np.median(distance_all)),
            "max": float(distance_all.max()),
        },
    }
    return z_all, context_all, distance_all, diagnostics


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum((values - mean[None, :]) ** 2 * normalized[:, None], axis=0)
    return mean, np.sqrt(np.maximum(variance, 1e-6))


def log_weights_to_mean_one_weights(log_weights: np.ndarray, clip_quantile: float | None) -> np.ndarray:
    logw = log_weights.copy()
    if clip_quantile is not None and clip_quantile < 1.0:
        cap = np.quantile(logw[np.isfinite(logw)], clip_quantile)
        logw = np.minimum(logw, cap)
    log_mean = logsumexp(logw) - math.log(len(logw))
    weights = np.exp(logw - log_mean)
    return weights.astype(np.float32)


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    normalized = weights / weights.sum()
    ess = 1.0 / np.sum(normalized**2)
    return {
        "min": float(weights.min()),
        "q01": float(np.quantile(weights, 0.01)),
        "median": float(np.median(weights)),
        "q99": float(np.quantile(weights, 0.99)),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "ess": float(ess),
        "ess_fraction": float(ess / len(weights)),
    }


def fit_linear_target_adjustment(
    *,
    z: np.ndarray,
    context_std: np.ndarray,
    observed_context_std: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> dict[str, np.ndarray]:
    delta = context_std - observed_context_std[None, :]
    design = np.column_stack([np.ones(z.shape[0]), delta])
    normalized = weights.astype(np.float64)
    normalized = normalized / np.mean(normalized)
    sqrt_w = np.sqrt(normalized)
    xw = design * sqrt_w[:, None]
    yw = z * sqrt_w[:, None]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    return {
        "intercept": beta[0],
        "slope": beta[1:],
    }


def apply_linear_target_adjustment(
    z: np.ndarray,
    context_std: np.ndarray,
    observed_context_std: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    delta = context_std - observed_context_std[None, :]
    return z - delta @ slope


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
        samples = []
        for start in range(0, n, chunk_size):
            current = min(chunk_size, n - start)
            drawn = self.flow(context).sample((current,))
            if drawn.ndim == 3:
                drawn = drawn[:, 0, :]
            samples.append(drawn)
        return torch.cat(samples, dim=0)


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
    torch.manual_seed(args.seed + 100)
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
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
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
    patience = args.patience
    epochs_since_best = 0
    history = {"train_nll": [], "val_nll": [], "lr": []}

    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
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
            weighted_loss_sum += float((nll.detach() * batch_w).sum().cpu())
            weight_sum += float(batch_w.detach().sum().cpu())

        model.eval()
        with torch.no_grad():
            val_nll = -model.log_prob(val_z_t, val_context_t)
            val_loss = (val_nll * val_w_t).sum() / val_w_t.sum().clamp_min(1e-12)
            val_loss_float = float(val_loss.detach().cpu())
        train_loss = weighted_loss_sum / max(weight_sum, 1e-12)
        history["train_nll"].append(train_loss)
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

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:04d} train_nll={train_loss:.4f} "
                f"val_nll={val_loss_float:.4f} best={best_val:.4f}"
            )

        if epochs_since_best >= patience:
            break

    synchronize(device)
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "epochs_completed": len(history["train_nll"]),
        "best_epoch": best_epoch,
        "best_val_nll": best_val,
        "final_train_nll": history["train_nll"][-1],
        "final_val_nll": history["val_nll"][-1],
        "training_seconds": float(time.perf_counter() - start),
        "history": history,
    }


@torch.no_grad()
def sample_flow_posterior(
    *,
    model: ConditionalSplineFlow,
    observed_context: np.ndarray,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    linear_adjustment: dict[str, np.ndarray] | None,
    n: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    context_std_value = ((observed_context - context_mean) / context_std).astype(np.float32)
    context_t = torch.from_numpy(context_std_value[None, :]).to(device)
    z_standardized = model.sample(n, context_t).detach().cpu().numpy()
    z = z_standardized * z_std[None, :] + z_mean[None, :]
    if linear_adjustment is not None:
        observed_context_std = np.asarray(linear_adjustment["observed_context_std"])
        slope = np.asarray(linear_adjustment["slope"])
        delta = context_std_value.astype(np.float64) - observed_context_std
        z = z + delta[None, :] @ slope
    theta = np.exp(z)
    return z, theta


def plot_training(metrics: dict[str, object], outfile: Path) -> None:
    history = metrics["history"]
    epochs = np.arange(1, len(history["train_nll"]) + 1)
    figure, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(epochs, history["train_nll"], color="#2f6fbb", lw=1.4, label="train")
    ax.plot(epochs, history["val_nll"], color="#b85c38", lw=1.8, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted negative log likelihood")
    ax.set_title("Conditional spline-flow NPE training")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_corner_overlay(
    *,
    flow_samples: np.ndarray,
    reference_samples: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
    max_samples: int,
) -> None:
    labels = [r"$A$", r"$k$", r"$\sigma$"]
    figure = corner.corner(
        subsample(reference_samples, max_samples, seed=120),
        labels=labels,
        color="#172033",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.8},
        contour_kwargs={"linewidths": 1.5},
    )
    corner.corner(
        subsample(flow_samples, max_samples, seed=121),
        fig=figure,
        labels=labels,
        color="#2f6fbb",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.6},
        contour_kwargs={"linewidths": 1.4},
    )
    handles = [
        plt.Line2D([0], [0], color="#172033", lw=2, label="Grid reference"),
        plt.Line2D([0], [0], color="#2f6fbb", lw=2, label="Spline-flow NPE"),
        true_theta_legend_handle(),
    ]
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    figure.subplots_adjust(top=0.90)
    figure.suptitle("NPE conditional spline-flow posterior", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictive_overlay(
    *,
    flow_samples: np.ndarray,
    reference_samples: np.ndarray,
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
    for label, samples, color, seed in [
        ("Grid reference", reference_samples, "#172033", 140),
        ("Spline-flow NPE", flow_samples, "#2f6fbb", 141),
    ]:
        lower, median, upper = posterior_predictive_band(samples, t_grid, seed=seed, max_draws=900)
        ax.fill_between(t_grid, lower, upper, color=color, alpha=0.10)
        ax.plot(t_grid, median, color=color, linewidth=2.0, label=label)
    ax.set_xlabel("time t")
    ax.set_ylabel("replicated observation y")
    ax.set_title("Spline-flow NPE posterior predictive")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional spline-flow NPE for the decay posterior.")
    parser.add_argument("--training-mode", choices=["weighted_proposal", "local_prior"], default="weighted_proposal")
    parser.add_argument("--train-simulations", type=int, default=250_000)
    parser.add_argument("--val-simulations", type=int, default=60_000)
    parser.add_argument("--proposal-source", choices=["prior", *PROPOSAL_SOURCES.keys()], default="snpe_diag")
    parser.add_argument("--proposal-inflation", type=float, default=2.5)
    parser.add_argument("--proposal-prior-mixture", type=float, default=0.0)
    parser.add_argument("--importance-clip-quantile", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=20.0)
    parser.add_argument("--linear-target-adjustment", action="store_true")
    parser.add_argument("--linear-adjustment-ridge", type=float, default=1e-4)
    parser.add_argument("--transforms", type=int, default=12)
    parser.add_argument("--hidden-features", type=parse_int_list, default=(256, 256))
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--posterior-samples", type=int, default=100_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument(
        "--target-wasserstein",
        type=float,
        default=None,
        help=(
            "Optional override. By default the target is max(full MCMC-to-grid, "
            "full HMC-to-grid) for the loaded reference samples."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--context-kind", choices=["indirect", "enhanced"], default="indirect")
    parser.add_argument("--k-grid-points", type=int, default=260)
    parser.add_argument("--k-min", type=float, default=0.04)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--simulate-chunk-size", type=int, default=80_000)
    parser.add_argument("--summary-chunk-size", type=int, default=40_000)
    parser.add_argument("--local-pilot-simulations", type=int, default=200_000)
    parser.add_argument("--local-quantile", type=float, default=0.005)
    parser.add_argument("--local-kernel-quantile", type=float, default=0.0)
    parser.add_argument("--local-max-candidates", type=int, default=30_000_000)
    parser.add_argument(
        "--local-region-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON containing local_training.region. When set "
            "with --training-mode local_prior, reuse that declared local region "
            "instead of fitting a fresh pilot region."
        ),
    )
    parser.add_argument("--max-corner-samples", type=int, default=25_000)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_FLOW_DECAY_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_FLOW_DECAY_FIGURES)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 2)
    device = choose_device(args.device)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=combined_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    mcmc_to_grid = compare_to_reference(mcmc["posterior_theta"], reference)
    hmc_to_grid = compare_to_reference(hmc["posterior_theta"], reference)
    recommended_target = max(
        mean_normalized_wasserstein_value(mcmc_to_grid),
        mean_normalized_wasserstein_value(hmc_to_grid),
    )
    target_wasserstein = float(args.target_wasserstein) if args.target_wasserstein is not None else recommended_target
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.posterior_samples, 100_000),
        seed=args.seed + 20,
    )
    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    t = t_obs.numpy()
    observed_x = y_obs.numpy()
    observed_context = make_context_summaries(
        observed_x[None, :],
        t,
        make_k_grid(args.k_grid_points, args.k_min, args.k_max),
        kind=args.context_kind,
        chunk_size=1,
    )[0]
    k_grid = make_k_grid(args.k_grid_points, args.k_min, args.k_max)

    data_start = time.perf_counter()
    local_metadata: dict[str, object] | None = None
    if args.training_mode == "weighted_proposal":
        train_z, train_log_r, train_proposal_metadata = sample_proposal_z(
            n=args.train_simulations,
            source=args.proposal_source,
            inflation=args.proposal_inflation,
            prior_mixture=args.proposal_prior_mixture,
            rng=rng,
        )
        val_z, val_log_r, _ = sample_proposal_z(
            n=args.val_simulations,
            source=args.proposal_source,
            inflation=args.proposal_inflation,
            prior_mixture=args.proposal_prior_mixture,
            rng=rng,
        )
        train_context = simulate_context_from_z(
            z=train_z,
            t=t,
            rng=rng,
            k_grid=k_grid,
            simulate_chunk_size=args.simulate_chunk_size,
            summary_chunk_size=args.summary_chunk_size,
            context_kind=args.context_kind,
        )
        val_context = simulate_context_from_z(
            z=val_z,
            t=t,
            rng=rng,
            k_grid=k_grid,
            simulate_chunk_size=args.simulate_chunk_size,
            summary_chunk_size=args.summary_chunk_size,
            context_kind=args.context_kind,
        )
        train_logw = prior_logpdf_z(train_z) - train_log_r
        val_logw = prior_logpdf_z(val_z) - val_log_r
        clip = None if args.importance_clip_quantile >= 1.0 else args.importance_clip_quantile
        train_weights = log_weights_to_mean_one_weights(train_logw, clip)
        val_weights = log_weights_to_mean_one_weights(val_logw, clip)
    else:
        train_proposal_metadata = {
            "source": "prior",
            "kind": "local_prior_filter",
            "local_quantile": args.local_quantile,
            "local_region_source": "summary" if args.local_region_summary is not None else "pilot",
            "local_region_summary": None if args.local_region_summary is None else str(args.local_region_summary),
        }
        if args.local_region_summary is None:
            region = fit_local_region(
                observed_context=observed_context,
                t=t,
                k_grid=k_grid,
                simulations=args.local_pilot_simulations,
                quantile=args.local_quantile,
                kernel_quantile=args.local_kernel_quantile,
                rng=rng,
                simulate_chunk_size=args.simulate_chunk_size,
                summary_chunk_size=args.summary_chunk_size,
                context_kind=args.context_kind,
            )
        else:
            local_summary = json.loads(args.local_region_summary.read_text(encoding="utf-8"))
            region = local_summary["local_training"]["region"]
            if args.local_kernel_quantile > 0.0 and region.get("kernel_bandwidth") is None:
                raise ValueError(
                    "--local-kernel-quantile requires a saved region with kernel_bandwidth "
                    "when --local-region-summary is used"
                )
        local_z, local_context, local_distances, collection = collect_local_prior_data(
            target_count=args.train_simulations + args.val_simulations,
            observed_context=observed_context,
            t=t,
            k_grid=k_grid,
            region=region,
            rng=rng,
            simulate_chunk_size=args.simulate_chunk_size,
            summary_chunk_size=args.summary_chunk_size,
            context_kind=args.context_kind,
            max_candidates=args.local_max_candidates,
        )
        order = rng.permutation(local_z.shape[0])
        train_index = order[: args.train_simulations]
        val_index = order[args.train_simulations :]
        train_z = local_z[train_index]
        val_z = local_z[val_index]
        train_context = local_context[train_index]
        val_context = local_context[val_index]
        if args.local_kernel_quantile > 0.0:
            bandwidth = float(region["kernel_bandwidth"])
            local_weights = np.exp(-0.5 * (local_distances / bandwidth) ** 2)
            local_weights = (local_weights / np.mean(local_weights)).astype(np.float32)
            train_weights = local_weights[train_index]
            val_weights = local_weights[val_index]
        else:
            train_weights = np.ones(args.train_simulations, dtype=np.float32)
            val_weights = np.ones(args.val_simulations, dtype=np.float32)
        local_metadata = {
            "region": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in region.items()
            },
            "collection": collection,
            "train_distance_summary": {
                "min": float(local_distances[train_index].min()),
                "median": float(np.median(local_distances[train_index])),
                "max": float(local_distances[train_index].max()),
            },
            "validation_distance_summary": {
                "min": float(local_distances[val_index].min()),
                "median": float(np.median(local_distances[val_index])),
                "max": float(local_distances[val_index].max()),
            },
        }
    data_seconds = time.perf_counter() - data_start

    train_weight_for_moments = train_weights.astype(np.float64)
    context_mean, context_std = weighted_moments(train_context, train_weight_for_moments)
    train_context_std = ((train_context - context_mean[None, :]) / context_std[None, :]).astype(np.float32)
    val_context_std = ((val_context - context_mean[None, :]) / context_std[None, :]).astype(np.float32)
    observed_context_std = ((observed_context - context_mean) / context_std).astype(np.float64)
    linear_adjustment = None
    train_target_z = train_z
    val_target_z = val_z
    if args.linear_target_adjustment:
        fitted_adjustment = fit_linear_target_adjustment(
            z=train_z,
            context_std=train_context_std.astype(np.float64),
            observed_context_std=observed_context_std,
            weights=train_weights,
            ridge=args.linear_adjustment_ridge,
        )
        slope = np.asarray(fitted_adjustment["slope"])
        train_target_z = apply_linear_target_adjustment(
            train_z,
            train_context_std.astype(np.float64),
            observed_context_std,
            slope,
        )
        val_target_z = apply_linear_target_adjustment(
            val_z,
            val_context_std.astype(np.float64),
            observed_context_std,
            slope,
        )
        linear_adjustment = {
            "slope": slope,
            "intercept": np.asarray(fitted_adjustment["intercept"]),
            "observed_context_std": observed_context_std,
            "ridge": np.asarray(args.linear_adjustment_ridge),
        }
    z_mean, z_std = weighted_moments(train_target_z, train_weight_for_moments)
    train_z_std = ((train_target_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    val_z_std = ((val_target_z - z_mean[None, :]) / z_std[None, :]).astype(np.float32)

    print(
        f"training spline-flow NPE on {device}; "
        f"mode={args.training_mode}, proposal={args.proposal_source}, "
        f"train={args.train_simulations}, val={args.val_simulations}"
    )
    print(f"train weight ESS fraction: {weight_diagnostics(train_weights)['ess_fraction']:.3f}")
    model, train_metrics = train_flow(
        train_context=train_context_std,
        train_z=train_z_std,
        train_weights=train_weights,
        val_context=val_context_std,
        val_z=val_z_std,
        val_weights=val_weights,
        args=args,
        device=device,
    )

    z_samples, theta_samples = sample_flow_posterior(
        model=model,
        observed_context=observed_context,
        context_mean=context_mean,
        context_std=context_std,
        z_mean=z_mean,
        z_std=z_std,
        linear_adjustment=linear_adjustment,
        n=args.posterior_samples,
        device=device,
    )
    faithfulness = compare_to_reference(theta_samples, reference)
    mean_w = faithfulness["mean_normalized_wasserstein"]["value"]
    target_pass = bool(mean_w <= target_wasserstein)

    samples_path = args.output_dir / "npe_flow_decay_samples.npz"
    model_path = args.output_dir / "npe_flow_decay_model.pt"
    np.savez_compressed(
        samples_path,
        t=t,
        y=observed_x,
        true_theta=true_theta.numpy(),
        observed_context=observed_context,
        context_mean=context_mean,
        context_std=context_std,
        z_mean=z_mean,
        z_std=z_std,
        linear_adjustment_slope=None if linear_adjustment is None else np.asarray(linear_adjustment["slope"]),
        linear_adjustment_intercept=None if linear_adjustment is None else np.asarray(linear_adjustment["intercept"]),
        linear_adjustment_observed_context_std=None
        if linear_adjustment is None
        else np.asarray(linear_adjustment["observed_context_std"]),
        z_samples=z_samples,
        theta_samples=theta_samples,
        reference_theta_samples=reference_samples,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "context_mean": context_mean,
            "context_std": context_std,
            "z_mean": z_mean,
            "z_std": z_std,
            "linear_adjustment": linear_adjustment,
        },
        model_path,
    )

    training_png = args.figure_dir / "npe_flow_decay_training.png"
    corner_png = args.figure_dir / "npe_flow_decay_corner_overlay.png"
    predictive_png = args.figure_dir / "npe_flow_decay_predictive_overlay.png"
    plot_training(train_metrics, training_png)
    plot_corner_overlay(
        flow_samples=theta_samples,
        reference_samples=reference_samples,
        true_theta=true_theta.numpy(),
        outfile=corner_png,
        max_samples=args.max_corner_samples,
    )
    plot_predictive_overlay(
        flow_samples=theta_samples,
        reference_samples=reference_samples,
        t=t,
        y=observed_x,
        true_theta=true_theta.numpy(),
        outfile=predictive_png,
    )

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "device": str(device),
        "proposal": train_proposal_metadata,
        "local_training": local_metadata,
        "target_wasserstein": target_wasserstein,
        "target_source": "explicit" if args.target_wasserstein is not None else "mcmc_hmc_to_grid",
        "recommended_targets": {
            "mean_normalized_wasserstein": recommended_target,
            "rule": "max(full MCMC-to-grid, full HMC-to-grid)",
        },
        "target_pass": target_pass,
        "target_ratio": float(mean_w / target_wasserstein),
        "observed_context": observed_context.tolist(),
        "context_kind": args.context_kind,
        "standardization": {
            "context_mean": context_mean.tolist(),
            "context_std": context_std.tolist(),
            "z_mean": z_mean.tolist(),
            "z_std": z_std.tolist(),
        },
        "linear_target_adjustment": None if linear_adjustment is None else {
            "slope": np.asarray(linear_adjustment["slope"]).tolist(),
            "intercept": np.asarray(linear_adjustment["intercept"]).tolist(),
            "observed_context_std": np.asarray(linear_adjustment["observed_context_std"]).tolist(),
            "ridge": float(np.asarray(linear_adjustment["ridge"])),
        },
        "weight_diagnostics": {
            "train": weight_diagnostics(train_weights),
            "validation": weight_diagnostics(val_weights),
        },
        "training": train_metrics,
        "posterior_summary": summarize_samples(theta_samples),
        "faithfulness_to_grid_reference": faithfulness,
        "reference_sampler_diagnostics": {
            "mcmc_to_grid": mcmc_to_grid,
            "hmc_to_grid": hmc_to_grid,
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "outputs": {
            "summary_json": str(args.output_dir / "npe_flow_decay_summary.json"),
            "samples_npz": str(samples_path),
            "model_pt": str(model_path),
            "training_png": str(training_png),
            "corner_overlay": str(corner_png),
            "predictive_overlay": str(predictive_png),
        },
        "timing_seconds": {
            "data_generation": data_seconds,
            "total": float(time.perf_counter() - total_start),
        },
    }
    summary_json = args.output_dir / "npe_flow_decay_summary.json"
    summary_json.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_path}")
    print(f"model: {model_path}")
    print(f"corner_overlay: {corner_png}")
    print(f"predictive_overlay: {predictive_png}")
    print(f"mean normalized Wasserstein: {mean_w:.5f}")
    print(f"target_wasserstein: {target_wasserstein:.5f} ({summary['target_source']})")
    print(f"target pass: {target_pass}")


if __name__ == "__main__":
    main()
