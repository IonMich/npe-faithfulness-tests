from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from sbi.inference import SNLE, SNRE
from torch.distributions import MultivariateNormal

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples, summarize_samples
from mcmc_decay_inference import PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data
from npe_focused_decay import plot_focused_corner, plot_predictive_overlay
from npe_stage1_decay import FAMILY_COLORS, FAMILY_LABELS, sample_grid_reference
from snpe_sbi_decay import simulator_from_z_tensor
from target_calibration import resolve_target_wasserstein

matplotlib.use("Agg")


def make_inference(args: argparse.Namespace, prior: MultivariateNormal):
    if args.method == "snle":
        return SNLE(
            prior=prior,
            density_estimator=args.density_estimator,
            device=args.device,
            show_progress_bars=False,
        )
    return SNRE(
        prior=prior,
        classifier=args.classifier,
        device=args.device,
        show_progress_bars=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alternate sbi methods SNLE/SNRE on the decay simulator.")
    parser.add_argument("--method", choices=["snle", "snre"], default="snle")
    parser.add_argument("--simulations", type=int, default=25_000)
    parser.add_argument("--density-estimator", choices=["maf", "nsf", "mdn"], default="maf")
    parser.add_argument("--classifier", default="resnet")
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-num-epochs", type=int, default=120)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--num-atoms", type=int, default=10)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--target-wasserstein", type=float, default=None)
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ap.SBI_ALTERNATE_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.SBI_ALTERNATE_FIGURES)
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
        seed=args.seed + 2,
    )

    inference = make_inference(args, prior)
    theta = prior.sample((args.simulations,))
    x, _ = simulator_from_z_tensor(theta, seed=args.seed + 10)
    print(f"training {args.method} on {args.simulations} simulations")
    if args.method == "snle":
        estimator = inference.append_simulations(theta, x).train(
            training_batch_size=args.training_batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=0.1,
            stop_after_epochs=args.stop_after_epochs,
            max_num_epochs=args.max_num_epochs,
            show_train_summary=False,
        )
    else:
        estimator = inference.append_simulations(theta, x).train(
            num_atoms=args.num_atoms,
            training_batch_size=args.training_batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=0.1,
            stop_after_epochs=args.stop_after_epochs,
            max_num_epochs=args.max_num_epochs,
            show_train_summary=False,
        )
    posterior = inference.build_posterior(
        estimator,
        sample_with="mcmc",
        mcmc_method="slice_np_vectorized",
    )
    print(f"sampling {args.posterior_samples} posterior samples")
    z_samples = posterior.sample((args.posterior_samples,), x=x_o, show_progress_bars=False).detach().cpu().numpy()
    theta_samples = np.exp(z_samples)
    metrics = compare_to_reference(theta_samples, reference)
    value = metrics["mean_normalized_wasserstein"]["value"]
    result = {
        "posterior_summary": summarize_samples(theta_samples),
        "faithfulness_to_grid_reference": metrics,
        "target_wasserstein": args.target_wasserstein,
        "target_ratio": float(value / args.target_wasserstein),
        "target_pass": bool(value <= args.target_wasserstein),
    }

    samples_npz = args.output_dir / "sbi_alternate_samples.npz"
    np.savez_compressed(
        samples_npz,
        t=t_obs.numpy(),
        y=y_obs.numpy(),
        true_theta=true_theta_np,
        z_samples=z_samples,
        theta_samples=theta_samples,
    )

    label = f"sbi_{args.method}"
    FAMILY_COLORS[label] = "#7a5cc2" if args.method == "snle" else "#3f8f5f"
    FAMILY_LABELS[label] = f"sbi {args.method.upper()}"
    corner_png = args.figure_dir / "sbi_alternate_corner_overlay.png"
    predictive_png = args.figure_dir / "sbi_alternate_predictive_overlay.png"
    plot_focused_corner(
        {label: theta_samples},
        reference_samples,
        true_theta_np,
        corner_png,
        title=f"sbi {args.method.upper()} posterior",
    )
    plot_predictive_overlay(
        samples_by_family={label: theta_samples},
        t=t_obs.numpy(),
        y=y_obs.numpy(),
        true_theta=true_theta_np,
        outfile=predictive_png,
    )

    summary = {
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir", "mcmc_samples", "hmc_samples"}
        },
        "samples_npz": str(samples_npz),
        "figures": {
            "corner_overlay": str(corner_png),
            "predictive_overlay": str(predictive_png),
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "result": result,
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }
    summary_json = args.output_dir / "sbi_alternate_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"corner_overlay: {corner_png}")
    print(
        f"mean normalized W={value:.5f} "
        f"(target_ratio={result['target_ratio']:.2f}x, pass={result['target_pass']})"
    )


if __name__ == "__main__":
    main()
