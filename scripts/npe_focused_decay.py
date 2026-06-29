from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import artifact_paths as ap

import corner
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import (
    build_grid_reference,
    compare_to_reference,
    load_samples,
    log_posterior_z_numpy,
    summarize_samples,
)
from corner_truth import overplot_true_values, true_theta_legend_handle
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    make_model,
    sample_posterior_for_observation,
    standardize,
    synchronize_device,
    train_one_model,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class FocusedConfig:
    train_simulations: int
    val_simulations: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    hidden_layers: int
    mdn_components: int
    flow_layers: int
    flow_context_dim: int
    seed: int
    observed_seed: int
    requested_device: str
    families: list[str]
    posterior_samples: int
    resampled_samples: int
    reference_grid_size: int
    proposal_inflation: float


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def simulator_from_z(z: np.ndarray, seed: int, n_observations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z_torch = torch.as_tensor(z, dtype=torch.float64)
    theta = torch.exp(z_torch)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    mean = theta[:, 0:1] * torch.exp(-theta[:, 1:2] * t[None, :])
    x = mean + torch.randn(z.shape[0], n_observations, generator=generator, dtype=torch.float64) * theta[:, 2:3]
    return x.numpy(), t.numpy()


def make_proposal(reference_z: np.ndarray, inflation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = reference_z.mean(axis=0)
    cov = np.cov(reference_z.T)
    cov = cov * inflation**2
    cov = cov + np.eye(cov.shape[0]) * 1e-5
    chol = np.linalg.cholesky(cov)
    return mean, cov, chol


def sample_proposal(
    *,
    n: int,
    mean: np.ndarray,
    chol: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(n, mean.shape[0]))
    return mean[None, :] + eps @ chol.T


def log_mvn(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(cov)
    diff = z - mean[None, :]
    solved = np.linalg.solve(chol, diff.T).T
    maha = np.sum(solved**2, axis=1)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    dim = mean.shape[0]
    return -0.5 * (dim * math.log(2.0 * math.pi) + log_det + maha)


def log_prior_z(z: np.ndarray) -> np.ndarray:
    mean = PRIOR_LOG_MEAN.numpy()
    std = PRIOR_LOG_STD.numpy()
    return (
        -0.5 * ((z - mean[None, :]) / std[None, :]) ** 2
        - np.log(std[None, :])
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=1)


def systematic_resample(weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right")


def correct_samples_to_prior(
    *,
    z_samples: np.ndarray,
    theta_samples: np.ndarray,
    proposal_mean: np.ndarray,
    proposal_cov: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    log_weights = log_prior_z(z_samples) - log_mvn(z_samples, proposal_mean, proposal_cov)
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    ess = 1.0 / np.sum(weights**2)
    index = systematic_resample(weights, n=n, seed=seed)
    diagnostics = {
        "importance_ess": float(ess),
        "importance_ess_fraction": float(ess / len(weights)),
        "max_normalized_weight": float(weights.max()),
        "min_normalized_weight": float(weights.min()),
    }
    return z_samples[index], theta_samples[index], diagnostics


@torch.no_grad()
def model_log_prob_on_z(
    *,
    model: torch.nn.Module,
    z_samples: np.ndarray,
    observed_x: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int = 16_384,
) -> np.ndarray:
    x_standardized = standardize(observed_x[None, :], x_mean, x_std).astype(np.float32)
    x_tensor = torch.from_numpy(x_standardized).to(device)
    z_standardized = ((z_samples - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    log_jacobian = float(np.log(z_std).sum())
    chunks = []
    for start in range(0, len(z_samples), batch_size):
        stop = min(start + batch_size, len(z_samples))
        z_tensor = torch.from_numpy(z_standardized[start:stop]).to(device)
        x_batch = x_tensor.expand(stop - start, -1)
        log_prob_std = model.log_prob(z_tensor, x_batch)
        chunks.append((log_prob_std.detach().cpu().numpy() - log_jacobian).astype(np.float64))
    return np.concatenate(chunks)


def exact_target_correct_samples(
    *,
    model: torch.nn.Module,
    z_samples: np.ndarray,
    theta_samples: np.ndarray,
    observed_x: np.ndarray,
    t: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    log_target = log_posterior_z_numpy(z_samples, t=t, y=observed_x)
    log_proposal = model_log_prob_on_z(
        model=model,
        z_samples=z_samples,
        observed_x=observed_x,
        x_mean=x_mean,
        x_std=x_std,
        z_mean=z_mean,
        z_std=z_std,
        device=device,
    )
    log_weights = log_target - log_proposal
    finite = np.isfinite(log_weights)
    if not np.all(finite):
        log_weights = np.where(finite, log_weights, -np.inf)
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    ess = 1.0 / np.sum(weights**2)
    index = systematic_resample(weights, n=n, seed=seed)
    diagnostics = {
        "importance_ess": float(ess),
        "importance_ess_fraction": float(ess / len(weights)),
        "max_normalized_weight": float(weights.max()),
        "min_normalized_weight": float(weights.min()),
    }
    return z_samples[index], theta_samples[index], diagnostics


def plot_focused_corner(
    samples_by_family: dict[str, np.ndarray],
    reference_samples: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
    title: str,
    max_samples: int = 25_000,
) -> None:
    labels = [r"$A$", r"$k$", r"$\sigma$"]
    figure = corner.corner(
        reference_samples[:max_samples],
        labels=labels,
        color="#172033",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.8},
        contour_kwargs={"linewidths": 1.5},
    )
    for family, samples in samples_by_family.items():
        selected = samples[:max_samples] if len(samples) <= max_samples else samples[np.random.default_rng(3).choice(len(samples), max_samples, replace=False)]
        corner.corner(
            selected,
            fig=figure,
            labels=labels,
            color=FAMILY_COLORS[family],
            plot_datapoints=False,
            fill_contours=False,
            levels=(0.50, 0.90),
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
            contour_kwargs={"linewidths": 1.3},
        )
    handles = [
        plt.Line2D([0], [0], color="#172033", lw=2, label="Grid reference"),
        true_theta_legend_handle(),
        *[
            plt.Line2D([0], [0], color=FAMILY_COLORS[family], lw=2, label=FAMILY_LABELS[family])
            for family in samples_by_family
        ],
    ]
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    figure.subplots_adjust(top=0.90)
    figure.suptitle(title, y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictive_overlay(
    *,
    samples_by_family: dict[str, np.ndarray],
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
    for index, (family, samples) in enumerate(samples_by_family.items(), start=1):
        rng = np.random.default_rng(300 + index)
        selected = samples[rng.choice(len(samples), min(900, len(samples)), replace=False)]
        mean = selected[:, 0:1] * np.exp(-selected[:, 1:2] * t_grid[None, :])
        predictive = mean + rng.normal(0.0, selected[:, 2:3], size=mean.shape)
        lower, median, upper = np.quantile(predictive, [0.05, 0.50, 0.95], axis=0)
        color = FAMILY_COLORS[family]
        ax.fill_between(t_grid, lower, upper, color=color, alpha=0.11)
        ax.plot(t_grid, median, color=color, linewidth=2.0, label=FAMILY_LABELS[family])
    ax.set_xlabel("time t")
    ax.set_ylabel("replicated observation y")
    ax.set_title("Focused NPE posterior predictive overlay")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train focused proposal-corrected NPE models.")
    parser.add_argument("--train-simulations", type=int, default=20_000)
    parser.add_argument("--val-simulations", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--flow-context-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--families", type=parse_families, default=list(FAMILIES))
    parser.add_argument("--posterior-samples", type=int, default=120_000)
    parser.add_argument("--resampled-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--proposal-inflation", type=float, default=2.5)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_FOCUSED_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_FOCUSED_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = choose_training_device(args.device)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    reference_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    proposal_mean, proposal_cov, proposal_chol = make_proposal(reference_z, args.proposal_inflation)

    config = FocusedConfig(
        train_simulations=args.train_simulations,
        val_simulations=args.val_simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        mdn_components=args.mdn_components,
        flow_layers=args.flow_layers,
        flow_context_dim=args.flow_context_dim,
        seed=args.seed,
        observed_seed=args.observed_seed,
        requested_device=args.device,
        families=args.families,
        posterior_samples=args.posterior_samples,
        resampled_samples=args.resampled_samples,
        reference_grid_size=args.reference_grid_size,
        proposal_inflation=args.proposal_inflation,
    )

    train_z = sample_proposal(n=args.train_simulations, mean=proposal_mean, chol=proposal_chol, seed=args.seed + 10)
    val_z = sample_proposal(n=args.val_simulations, mean=proposal_mean, chol=proposal_chol, seed=args.seed + 11)
    train_x, t = simulator_from_z(train_z, seed=args.seed + 20)
    val_x, _ = simulator_from_z(val_z, seed=args.seed + 21)
    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    observed_x = y_obs.numpy()
    true_theta_np = true_theta.numpy()

    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 1e-6)
    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)
    train_x_std = standardize(train_x, x_mean, x_std).astype(np.float32)
    val_x_std = standardize(val_x, x_mean, x_std).astype(np.float32)
    train_z_std = standardize(train_z, z_mean, z_std).astype(np.float32)
    val_z_std = standardize(val_z, z_mean, z_std).astype(np.float32)

    stage1_config = Stage1Config(
        train_simulations=args.train_simulations,
        val_simulations=args.val_simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        mdn_components=args.mdn_components,
        flow_layers=args.flow_layers,
        flow_context_dim=args.flow_context_dim,
        seed=args.seed,
        observed_seed=args.observed_seed,
        requested_device=args.device,
        families=args.families,
        posterior_samples=args.posterior_samples,
        reference_grid_size=args.reference_grid_size,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 30)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_x_tensor = torch.from_numpy(val_x_std)
    val_z_tensor = torch.from_numpy(val_z_std)

    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=reference_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=120_000,
    )
    rng = np.random.default_rng(args.seed + 99)
    grid_index = rng.choice(len(reference["weights"]), size=min(args.resampled_samples, 80_000), replace=True, p=reference["weights"])
    grid_reference_samples = reference["theta_grid"][grid_index]

    results: dict[str, dict[str, object]] = {}
    corrected_theta_by_family: dict[str, np.ndarray] = {}
    exact_corrected_theta_by_family: dict[str, np.ndarray] = {}
    uncorrected_theta_by_family: dict[str, np.ndarray] = {}
    model_paths: dict[str, str] = {}

    for family in args.families:
        print(f"focused training {family} on {device}")
        model, metrics = train_one_model(
            family=family,
            config=stage1_config,
            train_loader=train_loader,
            val_x=val_x_tensor,
            val_z=val_z_tensor,
            device=device,
            x_dim=train_x_std.shape[1],
            z_dim=train_z_std.shape[1],
        )
        z_uncorrected, theta_uncorrected = sample_posterior_for_observation(
            model=model,
            observed_x=observed_x,
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            n=args.posterior_samples,
            device=device,
        )
        z_corrected, theta_corrected, weight_diag = correct_samples_to_prior(
            z_samples=z_uncorrected,
            theta_samples=theta_uncorrected,
            proposal_mean=proposal_mean,
            proposal_cov=proposal_cov,
            n=args.resampled_samples,
            seed=args.seed + 200 + FAMILIES.index(family),
        )
        z_exact_corrected, theta_exact_corrected, exact_weight_diag = exact_target_correct_samples(
            model=model,
            z_samples=z_uncorrected,
            theta_samples=theta_uncorrected,
            observed_x=observed_x,
            t=t_obs.numpy(),
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            device=device,
            n=args.resampled_samples,
            seed=args.seed + 300 + FAMILIES.index(family),
        )
        uncorrected_theta_by_family[family] = theta_uncorrected
        corrected_theta_by_family[family] = theta_corrected
        exact_corrected_theta_by_family[family] = theta_exact_corrected
        metrics["uncorrected_posterior_summary"] = summarize_samples(theta_uncorrected)
        metrics["proposal_corrected_posterior_summary"] = summarize_samples(theta_corrected)
        metrics["exact_target_corrected_posterior_summary"] = summarize_samples(theta_exact_corrected)
        metrics["proposal_importance_correction"] = weight_diag
        metrics["exact_target_importance_correction"] = exact_weight_diag
        metrics["uncorrected_faithfulness_to_grid_reference"] = compare_to_reference(theta_uncorrected, reference)
        metrics["proposal_corrected_faithfulness_to_grid_reference"] = compare_to_reference(theta_corrected, reference)
        metrics["exact_target_corrected_faithfulness_to_grid_reference"] = compare_to_reference(theta_exact_corrected, reference)
        results[family] = metrics

        model_path = args.output_dir / f"{family}_focused_model.pt"
        torch.save(
            {
                "family": family,
                "state_dict": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "z_mean": z_mean,
                "z_std": z_std,
                "proposal_mean": proposal_mean,
                "proposal_cov": proposal_cov,
                "config": asdict(config),
            },
            model_path,
        )
        model_paths[family] = str(model_path)

    samples_npz = args.output_dir / "npe_focused_samples.npz"
    np.savez_compressed(
        samples_npz,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        proposal_mean=proposal_mean,
        proposal_cov=proposal_cov,
        x_mean=x_mean,
        x_std=x_std,
        z_mean=z_mean,
        z_std=z_std,
        **{f"theta_uncorrected_{family}": samples for family, samples in uncorrected_theta_by_family.items()},
        **{f"theta_proposal_corrected_{family}": samples for family, samples in corrected_theta_by_family.items()},
        **{f"theta_exact_target_corrected_{family}": samples for family, samples in exact_corrected_theta_by_family.items()},
    )

    corner_png = args.figure_dir / "npe_focused_proposal_corrected_corner_overlay.png"
    predictive_png = args.figure_dir / "npe_focused_proposal_corrected_predictive_overlay.png"
    exact_corner_png = args.figure_dir / "npe_focused_exact_corrected_corner_overlay.png"
    exact_predictive_png = args.figure_dir / "npe_focused_exact_corrected_predictive_overlay.png"
    plot_focused_corner(
        corrected_theta_by_family,
        grid_reference_samples,
        true_theta_np,
        corner_png,
        title="Focused NPE posterior overlay after prior/proposal correction",
    )
    plot_predictive_overlay(
        samples_by_family=corrected_theta_by_family,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        outfile=predictive_png,
    )
    plot_focused_corner(
        exact_corrected_theta_by_family,
        grid_reference_samples,
        true_theta_np,
        exact_corner_png,
        title="Focused NPE posterior overlay after exact target correction",
    )
    plot_predictive_overlay(
        samples_by_family=exact_corrected_theta_by_family,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        outfile=exact_predictive_png,
    )

    summary = {
        "config": asdict(config),
        "device": str(device),
        "proposal": {
            "source": "inflated covariance of combined MCMC/HMC posterior samples",
            "mean": proposal_mean.tolist(),
            "cov": proposal_cov.tolist(),
        },
        "model_paths": model_paths,
        "samples_npz": str(samples_npz),
        "figures": {
            "proposal_corrected_corner_overlay": str(corner_png),
            "proposal_corrected_predictive_overlay": str(predictive_png),
            "exact_target_corrected_corner_overlay": str(exact_corner_png),
            "exact_target_corrected_predictive_overlay": str(exact_predictive_png),
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "results": results,
    }
    summary_json = args.output_dir / "npe_focused_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"proposal_corrected_corner_overlay: {corner_png}")
    print(f"exact_target_corrected_corner_overlay: {exact_corner_png}")
    print("proposal-corrected mean normalized Wasserstein to grid reference:")
    for family in args.families:
        value = results[family]["proposal_corrected_faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        ess_frac = results[family]["proposal_importance_correction"]["importance_ess_fraction"]
        print(f"  {family}: {value:.5f} (importance ESS frac {ess_frac:.3f})")
    print("exact-target-corrected mean normalized Wasserstein to grid reference:")
    for family in args.families:
        value = results[family]["exact_target_corrected_faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        ess_frac = results[family]["exact_target_importance_correction"]["importance_ess_fraction"]
        print(f"  {family}: {value:.5f} (importance ESS frac {ess_frac:.3f})")


if __name__ == "__main__":
    main()
