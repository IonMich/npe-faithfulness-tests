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
from sbi.neural_nets import posterior_nn
from torch.distributions import MultivariateNormal

from npe_flow_stress_tests import (
    build_cases,
    flatten_post_burn,
    pairwise_agreement,
    plot_corner_overlay,
    plot_predictive_overlay,
    plot_trace_overlay,
    summarize_matrix,
)

matplotlib.use("Agg")


def simulate_sbi_context(
    *,
    case_name: str,
    z: torch.Tensor,
    seed: int,
    context_kind: str,
) -> torch.Tensor:
    case = build_cases()[case_name]
    rng = np.random.default_rng(seed)
    z_np = z.detach().cpu().numpy().astype(np.float64)
    x = case.simulate_x(z_np, rng)
    if context_kind == "raw":
        context = x
    elif context_kind == "profile":
        context = case.context(x)
    else:
        raise ValueError("context_kind must be raw or profile")
    return torch.from_numpy(context.astype(np.float32))


def load_reference(path: Path, *, mcmc_burn_in: int, hmc_burn_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    x0 = np.asarray(data["x0"], dtype=np.float64)
    mcmc_z = flatten_post_burn(np.asarray(data["mcmc_z"]), mcmc_burn_in)
    hmc_z = flatten_post_burn(np.asarray(data["hmc_z"]), hmc_burn_in)
    return x0, mcmc_z, hmc_z


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sbi SNPE-C for the ordered two-exponential stress case.")
    parser.add_argument("--case", default="two_exp_ordered")
    parser.add_argument("--reference-samples", type=Path, default=ap.TWO_EXP_RIDGECOORDS_RESULTS / "two_exp_ordered" / "two_exp_ordered_samples.npz")
    parser.add_argument("--mcmc-burn-in", type=int, default=40_000)
    parser.add_argument("--hmc-burn-in", type=int, default=1_500)
    parser.add_argument("--context-kind", choices=["raw", "profile"], default="raw")
    parser.add_argument("--initial-proposal", choices=["prior", "hmc_gaussian"], default="prior")
    parser.add_argument("--proposal-inflation", type=float, default=1.5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--simulations-per-round", type=int, default=25_000)
    parser.add_argument("--density-estimator", choices=["mdn", "maf", "nsf"], default="nsf")
    parser.add_argument("--z-score-x", choices=["independent", "structured", "none"], default="independent")
    parser.add_argument("--z-score-theta", choices=["independent", "structured", "none"], default="independent")
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-num-epochs", type=int, default=160)
    parser.add_argument("--stop-after-epochs", type=int, default=25)
    parser.add_argument("--num-atoms", type=int, default=10)
    parser.add_argument("--posterior-samples", type=int, default=70_000)
    parser.add_argument("--compare-sample-count", type=int, default=50_000)
    parser.add_argument("--plot-sample-count", type=int, default=8_000)
    parser.add_argument(
        "--agreement-target",
        type=float,
        default=None,
        help=(
            "Optional pairwise diagnostic agreement threshold. Leave unset for unscored "
            "runs; calibrated success should come from MCMC/HMC-to-grid reference checks."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ap.SBI_TWO_EXP_ORDERED_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.SBI_TWO_EXP_ORDERED_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    case = build_cases()[args.case]
    x0, mcmc_z, hmc_z = load_reference(
        args.reference_samples,
        mcmc_burn_in=args.mcmc_burn_in,
        hmc_burn_in=args.hmc_burn_in,
    )
    if args.context_kind == "raw":
        x_o = torch.from_numpy(x0.astype(np.float32))
    else:
        x_o = torch.from_numpy(case.context(x0[None, :]).astype(np.float32)[0])

    prior_mean = torch.from_numpy(case.prior_mean.astype(np.float32))
    prior_cov = torch.diag(torch.from_numpy((case.prior_std.astype(np.float32)) ** 2))
    prior = MultivariateNormal(prior_mean, prior_cov)
    density_estimator = posterior_nn(
        model=args.density_estimator,
        z_score_theta=args.z_score_theta,
        z_score_x=args.z_score_x,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        num_bins=args.num_bins,
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        device=args.device,
        show_progress_bars=False,
    )

    if args.initial_proposal == "hmc_gaussian":
        proposal_mean = torch.from_numpy(hmc_z.mean(axis=0).astype(np.float32))
        proposal_cov = torch.from_numpy((np.cov(hmc_z, rowvar=False) * args.proposal_inflation**2 + np.eye(case.z_dim) * 1e-5).astype(np.float32))
        proposal = MultivariateNormal(proposal_mean, proposal_cov)
    else:
        proposal = prior
    posterior = None
    final_z = None
    round_results: list[dict[str, object]] = []
    for round_index in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        print(f"SNPE-C round {round_index}/{args.rounds}, proposal={type(proposal).__name__}")
        z = proposal.sample((args.simulations_per_round,))
        x = simulate_sbi_context(
            case_name=args.case,
            z=z,
            seed=args.seed + 100 * round_index,
            context_kind=args.context_kind,
        )
        append_kwargs = (
            {}
            if round_index == 1 and args.initial_proposal == "prior"
            else {"proposal": proposal}
        )
        density_estimator = inference.append_simulations(z.to(torch.float32), x, **append_kwargs).train(
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
        if z_samples.shape[0] > args.compare_sample_count:
            rng = np.random.default_rng(args.seed + round_index)
            z_compare = z_samples[rng.choice(z_samples.shape[0], args.compare_sample_count, replace=False)]
        else:
            z_compare = z_samples
        mcmc_compare = mcmc_z
        hmc_compare = hmc_z
        if mcmc_compare.shape[0] > args.compare_sample_count:
            rng = np.random.default_rng(args.seed + 10 + round_index)
            mcmc_compare = mcmc_compare[rng.choice(mcmc_compare.shape[0], args.compare_sample_count, replace=False)]
        if hmc_compare.shape[0] > args.compare_sample_count:
            rng = np.random.default_rng(args.seed + 20 + round_index)
            hmc_compare = hmc_compare[rng.choice(hmc_compare.shape[0], args.compare_sample_count, replace=False)]
        agreement = pairwise_agreement(case, mcmc_compare, hmc_compare, z_compare)
        max_mean = max(
            agreement["diagnostic"]["mcmc_hmc"]["mean"],
            agreement["diagnostic"]["mcmc_npe"]["mean"],
            agreement["diagnostic"]["hmc_npe"]["mean"],
        )
        result = {
            "round": round_index,
            "round_seconds": float(time.perf_counter() - round_start),
            "agreement": agreement,
            "max_mean_diagnostic_wasserstein": float(max_mean),
            "target": None if args.agreement_target is None else float(args.agreement_target),
            "target_source": "not_set" if args.agreement_target is None else "explicit",
            "target_met": None if args.agreement_target is None else bool(max_mean <= args.agreement_target),
        }
        round_results.append(result)
        final_z = z_samples
        print(
            f"  MCMC-HMC={agreement['diagnostic']['mcmc_hmc']['mean']:.4f} "
            f"MCMC-NPE={agreement['diagnostic']['mcmc_npe']['mean']:.4f} "
            f"HMC-NPE={agreement['diagnostic']['hmc_npe']['mean']:.4f} "
            f"target={result['target_met']}"
        )
        proposal = posterior

    assert posterior is not None and final_z is not None
    samples_npz = args.output_dir / "sbi_two_exp_ordered_samples.npz"
    np.savez_compressed(
        samples_npz,
        x0=x0,
        mcmc_z=mcmc_z,
        hmc_z=hmc_z,
        npe_z=final_z,
        parameter_names=np.array(case.param_names),
        diagnostic_names=np.array(case.diagnostic_names),
    )

    rng = np.random.default_rng(args.seed + 999)
    mcmc_plot = mcmc_z[rng.choice(mcmc_z.shape[0], min(args.compare_sample_count, mcmc_z.shape[0]), replace=False)]
    hmc_plot = hmc_z[rng.choice(hmc_z.shape[0], min(args.compare_sample_count, hmc_z.shape[0]), replace=False)]
    npe_plot = final_z[rng.choice(final_z.shape[0], min(args.compare_sample_count, final_z.shape[0]), replace=False)]
    corner_path = args.figure_dir / "sbi_two_exp_ordered_mcmc_hmc_npe_corner.png"
    predictive_path = args.figure_dir / "sbi_two_exp_ordered_predictive.png"
    trace_path = args.figure_dir / "sbi_two_exp_ordered_reference_trace.png"
    plot_corner_overlay(
        case,
        mcmc_z=mcmc_plot,
        hmc_z=hmc_plot,
        npe_z=npe_plot,
        outfile=corner_path,
        seed=args.seed,
        max_points=args.plot_sample_count,
    )
    plot_predictive_overlay(
        case,
        x0,
        mcmc_z=mcmc_plot,
        hmc_z=hmc_plot,
        npe_z=npe_plot,
        outfile=predictive_path,
        seed=args.seed,
        max_points=args.plot_sample_count,
    )
    ref = np.load(args.reference_samples)
    plot_trace_overlay(
        case,
        mcmc_z=np.asarray(ref["mcmc_z"]),
        hmc_z=np.asarray(ref["hmc_z"]),
        burn_in_mcmc=args.mcmc_burn_in,
        burn_in_hmc=args.hmc_burn_in,
        outfile=trace_path,
    )

    final_agreement = round_results[-1]["agreement"]
    summary = {
        "case": args.case,
        "method": "sbi_SNPE_C",
        "context_kind": args.context_kind,
        "density_estimator": args.density_estimator,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir"}
        },
        "round_results": round_results,
        "final_agreement": final_agreement,
        "final_agreement_flags": {
            "target": None if args.agreement_target is None else float(args.agreement_target),
            "target_source": "not_set" if args.agreement_target is None else "explicit",
            "max_mean_diagnostic_wasserstein": round_results[-1]["max_mean_diagnostic_wasserstein"],
            "diagnostic_target_met": round_results[-1]["target_met"],
            "note": (
                "Pairwise MCMC/HMC/NPE agreement is reported for diagnostics. "
                "Use a model-specific grid/reference calibration for success claims."
            ),
        },
        "display_summaries": {
            "mcmc": summarize_matrix(case.display(mcmc_plot), case.param_names),
            "hmc": summarize_matrix(case.display(hmc_plot), case.param_names),
            "npe": summarize_matrix(case.display(npe_plot), case.param_names),
        },
        "paths": {
            "samples": str(samples_npz),
            "summary": str(args.output_dir / "sbi_two_exp_ordered_summary.json"),
            "corner": str(corner_path),
            "predictive": str(predictive_path),
            "trace": str(trace_path),
        },
        "timing_seconds": {
            "total": float(time.perf_counter() - total_start),
        },
    }
    summary_path = args.output_dir / "sbi_two_exp_ordered_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"samples: {samples_npz}")
    print(f"corner: {corner_path}")
    print(f"final target met: {round_results[-1]['target_met']}")


if __name__ == "__main__":
    main()
