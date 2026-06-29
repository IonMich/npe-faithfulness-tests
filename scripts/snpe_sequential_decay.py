from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, summarize_samples
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_focused_decay import log_mvn, plot_focused_corner, plot_predictive_overlay, systematic_resample
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    sample_grid_reference,
    sample_posterior_for_observation,
    standardize,
    train_one_model,
)
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def log_prior_z(z: np.ndarray) -> np.ndarray:
    mean = PRIOR_LOG_MEAN.numpy()
    std = PRIOR_LOG_STD.numpy()
    return (
        -0.5 * ((z - mean[None, :]) / std[None, :]) ** 2
        - np.log(std[None, :])
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=1)


def simulator_from_z(z: np.ndarray, seed: int, n_observations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z_torch = torch.as_tensor(z, dtype=torch.float64)
    theta = torch.exp(z_torch)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    mean = theta[:, 0:1] * torch.exp(-theta[:, 1:2] * t[None, :])
    x = mean + torch.randn(z.shape[0], n_observations, generator=generator, dtype=torch.float64) * theta[:, 2:3]
    return x.numpy(), t.numpy()


def prior_proposal() -> dict[str, object]:
    mean = PRIOR_LOG_MEAN.numpy()
    std = PRIOR_LOG_STD.numpy()
    cov = np.diag(std**2)
    return {
        "kind": "prior",
        "mean": mean,
        "cov": cov,
        "chol": np.linalg.cholesky(cov),
        "inflation": 1.0,
        "source": "prior",
    }


def gaussian_proposal_from_samples(
    *,
    z_samples: np.ndarray,
    inflation: float,
    source: str,
) -> dict[str, object]:
    mean = z_samples.mean(axis=0)
    cov = np.cov(z_samples.T)
    cov = cov * inflation**2
    cov = cov + np.eye(cov.shape[0]) * 1e-5
    chol = np.linalg.cholesky(cov)
    return {
        "kind": "gaussian",
        "mean": mean,
        "cov": cov,
        "chol": chol,
        "inflation": inflation,
        "source": source,
    }


def sample_proposal(proposal: dict[str, object], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.asarray(proposal["mean"])
    chol = np.asarray(proposal["chol"])
    eps = rng.normal(size=(n, mean.shape[0]))
    return mean[None, :] + eps @ chol.T


def log_proposal_z(z: np.ndarray, proposal: dict[str, object]) -> np.ndarray:
    if proposal["kind"] == "prior":
        return log_prior_z(z)
    return log_mvn(z, np.asarray(proposal["mean"]), np.asarray(proposal["cov"]))


def correct_proposal_posterior_to_prior(
    *,
    z_samples: np.ndarray,
    theta_samples: np.ndarray,
    proposal: dict[str, object],
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    log_weights = log_prior_z(z_samples) - log_proposal_z(z_samples, proposal)
    finite = np.isfinite(log_weights)
    if not np.all(finite):
        log_weights = np.where(finite, log_weights, -np.inf)
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights_sum = weights.sum()
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        raise RuntimeError("proposal correction produced invalid weights")
    weights = weights / weights_sum
    ess = 1.0 / np.sum(weights**2)
    indices = systematic_resample(weights, n=n, seed=seed)
    diagnostics = {
        "importance_ess": float(ess),
        "importance_ess_fraction": float(ess / len(weights)),
        "max_normalized_weight": float(weights.max()),
        "min_normalized_weight": float(weights.min()),
    }
    return z_samples[indices], theta_samples[indices], diagnostics


def train_round_model(
    *,
    family: str,
    round_index: int,
    proposal: dict[str, object],
    args: argparse.Namespace,
    observed_x: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_z = sample_proposal(proposal, args.train_simulations, seed=args.seed + round_index * 100 + 10)
    val_z = sample_proposal(proposal, args.val_simulations, seed=args.seed + round_index * 100 + 11)
    train_x, _ = simulator_from_z(train_z, seed=args.seed + round_index * 100 + 20)
    val_x, _ = simulator_from_z(val_z, seed=args.seed + round_index * 100 + 21)

    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 1e-6)
    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)
    train_x_std = standardize(train_x, x_mean, x_std).astype(np.float32)
    val_x_std = standardize(val_x, x_mean, x_std).astype(np.float32)
    train_z_std = standardize(train_z, z_mean, z_std).astype(np.float32)
    val_z_std = standardize(val_z, z_mean, z_std).astype(np.float32)

    config = Stage1Config(
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
        seed=args.seed + round_index * 1000,
        observed_seed=args.observed_seed,
        requested_device=args.device,
        families=[family],
        posterior_samples=args.posterior_samples,
        reference_grid_size=args.reference_grid_size,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed + round_index * 100 + 30)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    model, metrics = train_one_model(
        family=family,
        config=config,
        train_loader=train_loader,
        val_x=torch.from_numpy(val_x_std),
        val_z=torch.from_numpy(val_z_std),
        device=device,
        x_dim=train_x.shape[1],
        z_dim=train_z.shape[1],
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
    z_corrected, theta_corrected, correction = correct_proposal_posterior_to_prior(
        z_samples=z_uncorrected,
        theta_samples=theta_uncorrected,
        proposal=proposal,
        n=args.resampled_samples,
        seed=args.seed + round_index * 100 + 40,
    )
    metrics["round"] = round_index
    metrics["proposal"] = serialize_proposal(proposal)
    metrics["correction"] = correction
    metrics["uncorrected_posterior_summary"] = summarize_samples(theta_uncorrected)
    metrics["proposal_corrected_posterior_summary"] = summarize_samples(theta_corrected)
    arrays = {
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
    }
    return metrics, z_uncorrected, theta_uncorrected, z_corrected, theta_corrected, arrays


def serialize_proposal(proposal: dict[str, object]) -> dict[str, object]:
    return {
        "kind": proposal["kind"],
        "source": proposal["source"],
        "inflation": float(proposal["inflation"]),
        "mean": np.asarray(proposal["mean"]).tolist(),
        "cov": np.asarray(proposal["cov"]).tolist(),
    }


def plot_round_distances(
    *,
    summary: dict[str, object],
    outfile: Path,
) -> None:
    figure, ax = plt.subplots(figsize=(9.5, 5.8))
    target = float(summary["target_wasserstein"])
    for family in summary["families"]:
        values = [
            round_result["proposal_corrected_faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
            for round_result in summary["results"][family]["rounds"]
        ]
        rounds = np.arange(1, len(values) + 1)
        ax.plot(
            rounds,
            values,
            marker="o",
            linewidth=2.0,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
        )
    ax.axhline(target, color="#111827", linestyle="--", linewidth=1.6, label=f"target = {target:.3f}")
    ax.set_xlabel("SNPE round")
    ax.set_ylabel("mean normalized Wasserstein to grid posterior")
    ax.set_title("Sequential SNPE proposal-corrected faithfulness")
    ax.set_xticks(np.arange(1, int(summary["rounds"]) + 1))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential proposal-corrected SNPE for the decay simulator.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--train-simulations", type=int, default=25_000)
    parser.add_argument("--val-simulations", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--mdn-components", type=int, default=8)
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-context-dim", type=int, default=96)
    parser.add_argument("--proposal-inflation", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--families", type=parse_families, default=["mdn"])
    parser.add_argument("--posterior-samples", type=int, default=120_000)
    parser.add_argument("--resampled-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.SNPE_SEQUENTIAL_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.SNPE_SEQUENTIAL_FIGURES)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    args.target_wasserstein, args.target_source, args.recommended_targets = resolve_target_wasserstein(
        args.target_wasserstein,
        summary_path=args.target_summary,
    )
    if args.target_summary is not None:
        args.target_summary = str(args.target_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = choose_training_device(args.device)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)

    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    observed_x = y_obs.numpy()
    true_theta_np = true_theta.numpy()

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    reference_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=reference_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.resampled_samples, 80_000),
        seed=args.seed + 90,
    )

    all_results: dict[str, dict[str, object]] = {}
    final_corrected_theta_by_family: dict[str, np.ndarray] = {}
    final_uncorrected_theta_by_family: dict[str, np.ndarray] = {}
    final_corrected_z_by_family: dict[str, np.ndarray] = {}

    for family in args.families:
        print(f"sequential SNPE family={family} device={device}")
        proposal = prior_proposal()
        family_rounds: list[dict[str, object]] = []
        family_start = time.perf_counter()
        corrected_z = None
        corrected_theta = None
        uncorrected_theta = None
        for round_index in range(1, args.rounds + 1):
            print(f"  round {round_index}/{args.rounds}, proposal={proposal['kind']} from {proposal['source']}")
            round_metrics, z_uncorrected, theta_uncorrected, z_corrected, theta_corrected, _ = train_round_model(
                family=family,
                round_index=round_index,
                proposal=proposal,
                args=args,
                observed_x=observed_x,
                device=device,
            )
            uncorrected_metrics = compare_to_reference(theta_uncorrected, reference)
            corrected_metrics = compare_to_reference(theta_corrected, reference)
            corrected_value = corrected_metrics["mean_normalized_wasserstein"]["value"]
            round_metrics["uncorrected_faithfulness_to_grid_reference"] = uncorrected_metrics
            round_metrics["proposal_corrected_faithfulness_to_grid_reference"] = corrected_metrics
            round_metrics["target_wasserstein"] = args.target_wasserstein
            round_metrics["target_ratio"] = float(corrected_value / args.target_wasserstein)
            round_metrics["target_pass"] = bool(corrected_value <= args.target_wasserstein)
            family_rounds.append(round_metrics)
            corrected_z = z_corrected
            corrected_theta = theta_corrected
            uncorrected_theta = theta_uncorrected
            proposal = gaussian_proposal_from_samples(
                z_samples=z_corrected,
                inflation=args.proposal_inflation,
                source=f"round_{round_index}_proposal_corrected_posterior",
            )
            print(
                f"    corrected W={corrected_value:.5f}, "
                f"target_ratio={round_metrics['target_ratio']:.2f}x, "
                f"ESS_frac={round_metrics['correction']['importance_ess_fraction']:.3f}"
            )

        assert corrected_z is not None and corrected_theta is not None and uncorrected_theta is not None
        final_corrected_z_by_family[family] = corrected_z
        final_corrected_theta_by_family[family] = corrected_theta
        final_uncorrected_theta_by_family[family] = uncorrected_theta
        all_results[family] = {
            "family_seconds": float(time.perf_counter() - family_start),
            "rounds": family_rounds,
            "final_proposal_for_next_round": serialize_proposal(proposal),
        }

    samples_npz = args.output_dir / "snpe_sequential_samples.npz"
    np.savez_compressed(
        samples_npz,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        **{f"z_final_corrected_{family}": samples for family, samples in final_corrected_z_by_family.items()},
        **{f"theta_final_corrected_{family}": samples for family, samples in final_corrected_theta_by_family.items()},
        **{f"theta_final_uncorrected_{family}": samples for family, samples in final_uncorrected_theta_by_family.items()},
    )

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "inputs": {
            "mcmc_samples": str(args.mcmc_samples),
            "hmc_samples": str(args.hmc_samples),
        },
        "device": str(device),
        "families": args.families,
        "rounds": args.rounds,
        "target_wasserstein": args.target_wasserstein,
        "samples_npz": str(samples_npz),
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "results": all_results,
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }

    distance_png = args.figure_dir / "snpe_sequential_round_distances.png"
    corner_png = args.figure_dir / "snpe_sequential_final_corner_overlay.png"
    predictive_png = args.figure_dir / "snpe_sequential_final_predictive_overlay.png"
    summary["figures"] = {
        "round_distances": str(distance_png),
        "final_corner_overlay": str(corner_png),
        "final_predictive_overlay": str(predictive_png),
    }
    plot_round_distances(summary=summary, outfile=distance_png)
    plot_focused_corner(
        final_corrected_theta_by_family,
        reference_samples,
        true_theta_np,
        corner_png,
        title="Sequential SNPE final proposal-corrected posterior",
    )
    plot_predictive_overlay(
        samples_by_family=final_corrected_theta_by_family,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        outfile=predictive_png,
    )

    summary_json = args.output_dir / "snpe_sequential_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"round_distances: {distance_png}")
    print(f"final_corner_overlay: {corner_png}")
    print("final proposal-corrected mean normalized Wasserstein:")
    for family in args.families:
        final_round = all_results[family]["rounds"][-1]
        value = final_round["proposal_corrected_faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(
            f"  {family}: {value:.5f} "
            f"(target_ratio={final_round['target_ratio']:.2f}x, "
            f"pass={final_round['target_pass']})"
        )


if __name__ == "__main__":
    main()
