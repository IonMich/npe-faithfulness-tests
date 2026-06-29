from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from sbi.inference import SNPE
from torch.distributions import MultivariateNormal

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, summarize_samples
from mcmc_decay_inference import PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_focused_decay import plot_focused_corner, plot_predictive_overlay
from npe_stage1_decay import FAMILY_COLORS, FAMILY_LABELS, sample_grid_reference
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SBI_COLORS = {
    "mdn": "#c06f2d",
    "maf": "#2f6fbb",
    "nsf": "#7a5cc2",
}


def simulator_from_z_tensor(z: torch.Tensor, seed: int, n_observations: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = z.to(dtype=torch.float64, device="cpu")
    theta = torch.exp(z)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    mean = theta[:, 0:1] * torch.exp(-theta[:, 1:2] * t[None, :])
    x = mean + torch.randn(z.shape[0], n_observations, generator=generator, dtype=torch.float64) * theta[:, 2:3]
    return x.to(dtype=torch.float32), t


def plot_round_distances(summary: dict[str, object], outfile: Path) -> None:
    rounds = np.array([item["round"] for item in summary["round_results"]])
    corrected = np.array([
        item["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        for item in summary["round_results"]
    ])
    target = float(summary["target_wasserstein"])
    estimator = summary["density_estimator"]
    figure, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(
        rounds,
        corrected,
        marker="o",
        linewidth=2.0,
        color=SBI_COLORS.get(estimator, "#c06f2d"),
        label=f"sbi SNPE-C {estimator}",
    )
    ax.axhline(target, color="#111827", linestyle="--", linewidth=1.6, label=f"target = {target:.3f}")
    ax.set_xlabel("SNPE round")
    ax.set_ylabel("mean normalized Wasserstein to grid posterior")
    ax.set_title("sbi sequential SNPE-C faithfulness")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sbi SNPE-C on the decay simulator.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--simulations-per-round", type=int, default=25_000)
    parser.add_argument("--density-estimator", choices=["mdn", "maf", "nsf"], default="mdn")
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-num-epochs", type=int, default=180)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--num-atoms", type=int, default=10)
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.SNPE_SBI_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.SNPE_SBI_FIGURES)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    prior_mean = PRIOR_LOG_MEAN.to(dtype=torch.float32)
    prior_cov = torch.diag(PRIOR_LOG_STD.to(dtype=torch.float32) ** 2)
    prior = MultivariateNormal(prior_mean, prior_cov)

    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    x_o = y_obs.to(dtype=torch.float32)
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
        n=min(args.posterior_samples, 80_000),
        seed=args.seed + 88,
    )

    inference = SNPE(
        prior=prior,
        density_estimator=args.density_estimator,
        device=args.device,
        show_progress_bars=False,
    )
    proposal = prior
    posterior = None
    round_results: list[dict[str, object]] = []
    final_theta_samples = None
    final_z_samples = None

    for round_index in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        print(f"sbi SNPE-C round {round_index}/{args.rounds}, proposal={type(proposal).__name__}")
        theta = proposal.sample((args.simulations_per_round,))
        x, _ = simulator_from_z_tensor(theta, seed=args.seed + round_index * 100)
        append_kwargs = {} if round_index == 1 else {"proposal": proposal}
        density_estimator = inference.append_simulations(theta.to(torch.float32), x, **append_kwargs).train(
            num_atoms=args.num_atoms,
            training_batch_size=args.training_batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=0.1,
            stop_after_epochs=args.stop_after_epochs,
            max_num_epochs=args.max_num_epochs,
            show_train_summary=False,
            retrain_from_scratch=False,
        )
        posterior = inference.build_posterior(density_estimator, sample_with="direct").set_default_x(x_o)
        z_samples = posterior.sample((args.posterior_samples,), show_progress_bars=False).detach().cpu().numpy()
        theta_samples = np.exp(z_samples)
        metrics = compare_to_reference(theta_samples, reference)
        value = metrics["mean_normalized_wasserstein"]["value"]
        round_result = {
            "round": round_index,
            "round_seconds": float(time.perf_counter() - round_start),
            "faithfulness_to_grid_reference": metrics,
            "posterior_summary": summarize_samples(theta_samples),
            "target_wasserstein": args.target_wasserstein,
            "target_ratio": float(value / args.target_wasserstein),
            "target_pass": bool(value <= args.target_wasserstein),
        }
        round_results.append(round_result)
        print(f"  W={value:.5f}, target_ratio={round_result['target_ratio']:.2f}x, pass={round_result['target_pass']}")
        proposal = posterior
        final_theta_samples = theta_samples
        final_z_samples = z_samples

    assert posterior is not None and final_theta_samples is not None and final_z_samples is not None

    samples_npz = args.output_dir / "snpe_sbi_samples.npz"
    np.savez_compressed(
        samples_npz,
        t=t_obs.numpy(),
        y=y_obs.numpy(),
        true_theta=true_theta_np,
        z_final=final_z_samples,
        theta_final=final_theta_samples,
    )

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "density_estimator": args.density_estimator,
        "rounds": args.rounds,
        "target_wasserstein": args.target_wasserstein,
        "samples_npz": str(samples_npz),
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "round_results": round_results,
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }

    round_distances_png = args.figure_dir / "snpe_sbi_round_distances.png"
    corner_png = args.figure_dir / "snpe_sbi_final_corner_overlay.png"
    predictive_png = args.figure_dir / "snpe_sbi_final_predictive_overlay.png"
    plot_round_distances(summary, round_distances_png)
    label = f"sbi_{args.density_estimator}"
    FAMILY_COLORS[label] = SBI_COLORS.get(args.density_estimator, "#c06f2d")
    FAMILY_LABELS[label] = f"sbi SNPE-C {args.density_estimator}"
    final_samples_by_label = {label: final_theta_samples}
    plot_focused_corner(
        final_samples_by_label,
        reference_samples,
        true_theta_np,
        corner_png,
        title=f"sbi sequential SNPE-C final posterior ({args.density_estimator})",
    )
    plot_predictive_overlay(
        samples_by_family=final_samples_by_label,
        t=t_obs.numpy(),
        y=y_obs.numpy(),
        true_theta=true_theta_np,
        outfile=predictive_png,
    )
    summary["figures"] = {
        "round_distances": str(round_distances_png),
        "final_corner_overlay": str(corner_png),
        "final_predictive_overlay": str(predictive_png),
    }
    summary_json = args.output_dir / "snpe_sbi_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final = round_results[-1]
    value = final["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"round_distances: {round_distances_png}")
    print(f"final_corner_overlay: {corner_png}")
    print(
        f"final W={value:.5f}, "
        f"target_ratio={final['target_ratio']:.2f}x, "
        f"pass={final['target_pass']}"
    )


if __name__ == "__main__":
    main()
