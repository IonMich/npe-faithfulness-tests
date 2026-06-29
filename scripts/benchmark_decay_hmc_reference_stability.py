from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from evaluate_decay_amortization_panel import sample_prior_predictive_observations
from hmc_decay_inference import HMCConfig, run_hmc
from mcmc_decay_inference import PARAMETER_NAMES, arviz_diagnostics, choose_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/11_convergence_benchmarks/06_hmc_two_reference_stability"
)


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


def parse_int_list(value: str) -> list[int]:
    values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one positive integer is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Values must be positive.")
    return sorted(set(values))


def parse_index_list(value: str) -> list[int]:
    values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one observation index is required.")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Observation indices must be non-negative.")
    return sorted(set(values))


def posterior_draws(theta_samples: np.ndarray, burn_in: int) -> np.ndarray:
    return theta_samples[:, burn_in:, :].reshape(-1, theta_samples.shape[-1])


def maybe_subsample(
    samples: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_samples <= 0 or samples.shape[0] <= max_samples:
        return samples
    indices = rng.choice(samples.shape[0], size=max_samples, replace=False)
    return samples[indices]


def normalized_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    reference_sd: np.ndarray,
) -> float:
    values = []
    for index in range(len(PARAMETER_NAMES)):
        sd = max(float(reference_sd[index]), 1e-12)
        values.append(wasserstein_distance(left[:, index], right[:, index]) / sd)
    return float(np.mean(values))


def diagnostic_summary(
    *,
    diagnostics: dict[str, dict[str, float]],
    energy_error: np.ndarray,
    burn_in: int,
    divergence_threshold: float,
) -> dict[str, float | int]:
    rhat = [values["rhat"] for values in diagnostics.values()]
    bulk = [values["ess_bulk"] for values in diagnostics.values()]
    tail = [values["ess_tail"] for values in diagnostics.values()]
    post_energy_error = energy_error[:, burn_in:]
    divergent = ~np.isfinite(post_energy_error) | (np.abs(post_energy_error) > divergence_threshold)
    return {
        "max_rhat": float(np.max(rhat)),
        "min_bulk_ess": float(np.min(bulk)),
        "min_tail_ess": float(np.min(tail)),
        "max_abs_energy_error": float(np.nanmax(np.abs(post_energy_error))),
        "mean_abs_energy_error": float(np.nanmean(np.abs(post_energy_error))),
        "divergence_count": int(np.sum(divergent)),
    }


def diagnostics_pass(
    *,
    summary: dict[str, float | int],
    rhat_threshold: float,
    min_ess: float,
) -> bool:
    return (
        float(summary["max_rhat"]) < rhat_threshold
        and float(summary["min_bulk_ess"]) >= min_ess
        and float(summary["min_tail_ess"]) >= min_ess
        and int(summary["divergence_count"]) == 0
    )


def summarize(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "min": None,
            "q16": None,
            "median": None,
            "q84": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "q16": float(np.quantile(arr, 0.16)),
        "median": float(np.median(arr)),
        "q84": float(np.quantile(arr, 0.84)),
        "max": float(arr.max()),
    }


def summarize_by_steps(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for steps in sorted({int(row["steps"]) for row in rows}):
        step_rows = [row for row in rows if int(row["steps"]) == steps]
        output.append({
            "steps": steps,
            "n": len(step_rows),
            "pair_seconds": summarize([float(row["pair_seconds"]) for row in step_rows]),
            "run_seconds": summarize([float(row["mean_run_seconds"]) for row in step_rows]),
            "pair_w": summarize([float(row["pair_w"]) for row in step_rows]),
            "pair_w_self_scaled": summarize(
                [float(row["pair_w_self_scaled"]) for row in step_rows]
            ),
            "pooled_w_to_long_pair": summarize(
                [float(row["pooled_w_to_long_pair"]) for row in step_rows]
            ),
            "worst_max_rhat": summarize([float(row["worst_max_rhat"]) for row in step_rows]),
            "worst_min_bulk_ess": summarize(
                [float(row["worst_min_bulk_ess"]) for row in step_rows]
            ),
            "worst_min_tail_ess": summarize(
                [float(row["worst_min_tail_ess"]) for row in step_rows]
            ),
            "diagnostics_pair_pass_fraction": float(
                np.mean([bool(row["diagnostics_pair_pass"]) for row in step_rows])
            ),
            "reference_pair_pass_fraction": float(
                np.mean([bool(row["reference_pair_pass"]) for row in step_rows])
            ),
        })
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one_hmc(
    *,
    t: np.ndarray,
    y: np.ndarray,
    chains: int,
    steps: int,
    burn_in: int,
    seed: int,
    step_size: float,
    leapfrog_steps: int,
    requested_device: str,
    device: torch.device,
    dtype: torch.dtype,
    divergence_threshold: float,
    rhat_threshold: float,
    min_ess: float,
) -> dict[str, object]:
    config = HMCConfig(
        chains=chains,
        steps=steps,
        burn_in=burn_in,
        seed=seed,
        step_size=step_size,
        leapfrog_steps=leapfrog_steps,
        requested_device=requested_device,
    )
    z_samples, theta_samples, accepted, energy_error, seconds = run_hmc(
        t=torch.as_tensor(t, dtype=torch.float64),
        y=torch.as_tensor(y, dtype=torch.float64),
        config=config,
        device=device,
        dtype=dtype,
    )
    diagnostics = arviz_diagnostics(theta_samples, burn_in)
    summary = diagnostic_summary(
        diagnostics=diagnostics,
        energy_error=energy_error,
        burn_in=burn_in,
        divergence_threshold=divergence_threshold,
    )
    return {
        "config": config,
        "z_samples": z_samples,
        "theta_samples": theta_samples,
        "accepted": accepted,
        "energy_error": energy_error,
        "seconds": float(seconds),
        "diagnostics": diagnostics,
        "diagnostic_summary": summary,
        "diagnostics_pass": diagnostics_pass(
            summary=summary,
            rhat_threshold=rhat_threshold,
            min_ess=min_ess,
        ),
        "acceptance_rate": float(accepted.mean()),
    }


def run_hmc_pair(
    *,
    t: np.ndarray,
    y: np.ndarray,
    chains: int,
    steps: int,
    burn_in: int,
    seed_a: int,
    seed_b: int,
    step_size: float,
    leapfrog_steps: int,
    requested_device: str,
    device: torch.device,
    dtype: torch.dtype,
    divergence_threshold: float,
    rhat_threshold: float,
    min_ess: float,
) -> tuple[dict[str, object], dict[str, object]]:
    first = run_one_hmc(
        t=t,
        y=y,
        chains=chains,
        steps=steps,
        burn_in=burn_in,
        seed=seed_a,
        step_size=step_size,
        leapfrog_steps=leapfrog_steps,
        requested_device=requested_device,
        device=device,
        dtype=dtype,
        divergence_threshold=divergence_threshold,
        rhat_threshold=rhat_threshold,
        min_ess=min_ess,
    )
    second = run_one_hmc(
        t=t,
        y=y,
        chains=chains,
        steps=steps,
        burn_in=burn_in,
        seed=seed_b,
        step_size=step_size,
        leapfrog_steps=leapfrog_steps,
        requested_device=requested_device,
        device=device,
        dtype=dtype,
        divergence_threshold=divergence_threshold,
        rhat_threshold=rhat_threshold,
        min_ess=min_ess,
    )
    return first, second


def plot_convergence(
    *,
    summary_rows: list[dict[str, object]],
    outfile: Path,
) -> None:
    seconds = np.asarray([row["pair_seconds"]["median"] for row in summary_rows], dtype=np.float64)
    x_labels = [str(row["steps"]) for row in summary_rows]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    ax = axes[0, 0]
    pair_w = np.asarray([row["pair_w"]["median"] for row in summary_rows], dtype=np.float64)
    pair_w_q16 = np.asarray([row["pair_w"]["q16"] for row in summary_rows], dtype=np.float64)
    pair_w_q84 = np.asarray([row["pair_w"]["q84"] for row in summary_rows], dtype=np.float64)
    pooled_w = np.asarray(
        [row["pooled_w_to_long_pair"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    ax.plot(seconds, pair_w, marker="o", color="#2f6fbb", linewidth=2.0, label="ref 1 to ref 2")
    ax.fill_between(seconds, pair_w_q16, pair_w_q84, color="#2f6fbb", alpha=0.16)
    ax.plot(
        seconds,
        pooled_w,
        marker="s",
        color="#2f855a",
        linewidth=1.8,
        label="pooled pair to long pair",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median wall time for two-reference protocol (seconds)")
    ax.set_ylabel("mean normalized Wasserstein")
    ax.set_title("Reference agreement")
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[0, 1]
    rhat = np.asarray([row["worst_max_rhat"]["median"] for row in summary_rows], dtype=np.float64)
    rhat_q16 = np.asarray([row["worst_max_rhat"]["q16"] for row in summary_rows], dtype=np.float64)
    rhat_q84 = np.asarray([row["worst_max_rhat"]["q84"] for row in summary_rows], dtype=np.float64)
    ax.plot(seconds, rhat, marker="o", color="#8a5a9c", linewidth=2.0)
    ax.fill_between(seconds, rhat_q16, rhat_q84, color="#8a5a9c", alpha=0.16)
    ax.axhline(1.01, color="#172033", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("median wall time for two-reference protocol (seconds)")
    ax.set_ylabel("worst max R-hat across two runs")
    ax.set_title("R-hat")
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    bulk = np.asarray(
        [row["worst_min_bulk_ess"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    tail = np.asarray(
        [row["worst_min_tail_ess"]["median"] for row in summary_rows],
        dtype=np.float64,
    )
    ax.plot(seconds, bulk, marker="o", color="#b85c38", linewidth=2.0, label="bulk ESS")
    ax.plot(seconds, tail, marker="s", color="#5f6f87", linewidth=1.8, label="tail ESS")
    ax.axhline(1000.0, color="#172033", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median wall time for two-reference protocol (seconds)")
    ax.set_ylabel("worst min ESS across two runs")
    ax.set_title("Effective sample size")
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[1, 1]
    diag_fraction = np.asarray(
        [row["diagnostics_pair_pass_fraction"] for row in summary_rows],
        dtype=np.float64,
    )
    reference_fraction = np.asarray(
        [row["reference_pair_pass_fraction"] for row in summary_rows],
        dtype=np.float64,
    )
    ax.plot(seconds, diag_fraction, marker="o", color="#404b5a", linewidth=2.0, label="diagnostics")
    ax.plot(
        seconds,
        reference_fraction,
        marker="s",
        color="#bf4d5a",
        linewidth=1.8,
        label="diagnostics + W",
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("median wall time for two-reference protocol (seconds)")
    ax.set_ylabel("fraction of signals passing")
    ax.set_title("Two-reference screen")
    ax.grid(alpha=0.22)
    ax.legend()

    for ax in axes.ravel():
        ymin = ax.get_ylim()[0]
        for x, label in zip(seconds, x_labels, strict=True):
            ax.annotate(
                label,
                (x, ymin),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark two-independent-HMC-reference stability for decay signals.",
    )
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument(
        "--observation-indices",
        type=parse_index_list,
        default=None,
        help=(
            "Optional comma-separated zero-based observation indices to run after drawing "
            "--num-observations prior-predictive signals."
        ),
    )
    parser.add_argument(
        "--steps",
        type=parse_int_list,
        default=parse_int_list("1000,2000,5000,10000"),
        help="Comma-separated per-run HMC step budgets to test.",
    )
    parser.add_argument("--baseline-steps", type=int, default=20_000)
    parser.add_argument("--burn-in-fraction", type=float, default=0.20)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--n-observations", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--step-size", type=float, default=0.009)
    parser.add_argument("--leapfrog-steps", type=int, default=10)
    parser.add_argument("--divergence-threshold", type=float, default=100.0)
    parser.add_argument("--rhat-threshold", type=float, default=1.01)
    parser.add_argument("--min-ess", type=float, default=1000.0)
    parser.add_argument("--max-w-samples", type=int, default=250_000)
    parser.add_argument("--absolute-w-tolerance", type=float, default=0.02)
    parser.add_argument("--pair-w-tolerance", type=float, default=0.03)
    parser.add_argument("--floor-multiplier", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_observations <= 0:
        parser.error("--num-observations must be positive")
    if args.n_observations <= 0:
        parser.error("--n-observations must be positive")
    if args.chains < 2:
        parser.error("--chains must be at least 2")
    if not (0.0 <= args.burn_in_fraction < 1.0):
        parser.error("--burn-in-fraction must be in [0, 1)")
    if args.baseline_steps <= max(args.steps):
        parser.error("--baseline-steps must be larger than the largest tested --steps")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if args.leapfrog_steps < 1:
        parser.error("--leapfrog-steps must be at least 1")
    if args.min_ess <= 0.0:
        parser.error("--min-ess must be positive")
    return args


def burn_in_for_steps(steps: int, burn_in_fraction: float) -> int:
    burn_in = int(round(steps * burn_in_fraction))
    return min(max(burn_in, 0), steps - 1)


def main() -> None:
    args = parse_args()
    results_dir = args.output_dir / "results"
    figure_dir = args.output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = choose_device(args.device)
    selected_indices = (
        list(range(args.num_observations))
        if args.observation_indices is None
        else list(args.observation_indices)
    )
    sample_count = max(args.num_observations, max(selected_indices) + 1)
    t, x_panel, z_true, _, panel_metadata = sample_prior_predictive_observations(
        n=sample_count,
        seed=args.seed,
        n_observations=args.n_observations,
    )

    baseline_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []

    print(f"device: {device} ({str(dtype).replace('torch.', '')})", flush=True)
    for run_position, obs_index in enumerate(selected_indices):
        y = x_panel[obs_index]
        true_theta = np.exp(z_true[obs_index])
        baseline_burn_in = burn_in_for_steps(args.baseline_steps, args.burn_in_fraction)
        baseline_seed_a = args.seed + 1_000_000 + 10_000 * (obs_index + 1) + 1
        baseline_seed_b = args.seed + 1_000_000 + 10_000 * (obs_index + 1) + 2
        print(
            f"observation {obs_index} ({run_position + 1}/{len(selected_indices)}): "
            f"long pair steps={args.baseline_steps}",
            flush=True,
        )
        baseline_a, baseline_b = run_hmc_pair(
            t=t,
            y=y,
            chains=args.chains,
            steps=args.baseline_steps,
            burn_in=baseline_burn_in,
            seed_a=baseline_seed_a,
            seed_b=baseline_seed_b,
            step_size=args.step_size,
            leapfrog_steps=args.leapfrog_steps,
            requested_device=args.device,
            device=device,
            dtype=dtype,
            divergence_threshold=args.divergence_threshold,
            rhat_threshold=args.rhat_threshold,
            min_ess=args.min_ess,
        )
        baseline_theta_a = posterior_draws(np.asarray(baseline_a["theta_samples"]), baseline_burn_in)
        baseline_theta_b = posterior_draws(np.asarray(baseline_b["theta_samples"]), baseline_burn_in)
        baseline_pooled = np.vstack([baseline_theta_a, baseline_theta_b])
        baseline_sd = np.std(baseline_pooled, axis=0, ddof=1)
        baseline_pair_sd = np.std(baseline_pooled, axis=0, ddof=1)
        rng = np.random.default_rng(args.seed + 3_000_000 + obs_index)
        baseline_pair_w = normalized_wasserstein(
            maybe_subsample(baseline_theta_a, max_samples=args.max_w_samples, rng=rng),
            maybe_subsample(baseline_theta_b, max_samples=args.max_w_samples, rng=rng),
            baseline_sd,
        )
        baseline_pair_w_self_scaled = normalized_wasserstein(
            maybe_subsample(baseline_theta_a, max_samples=args.max_w_samples, rng=rng),
            maybe_subsample(baseline_theta_b, max_samples=args.max_w_samples, rng=rng),
            baseline_pair_sd,
        )
        baseline_diag_a = baseline_a["diagnostic_summary"]
        baseline_diag_b = baseline_b["diagnostic_summary"]
        baseline_pair_pass = bool(baseline_a["diagnostics_pass"]) and bool(
            baseline_b["diagnostics_pass"]
        )
        baseline_pair_seconds = float(baseline_a["seconds"]) + float(baseline_b["seconds"])
        baseline_rows.append({
            "observation_index": obs_index,
            "steps": args.baseline_steps,
            "burn_in": baseline_burn_in,
            "pair_seconds": baseline_pair_seconds,
            "seconds_a": baseline_a["seconds"],
            "seconds_b": baseline_b["seconds"],
            "acceptance_rate_a": baseline_a["acceptance_rate"],
            "acceptance_rate_b": baseline_b["acceptance_rate"],
            "pair_w": baseline_pair_w,
            "pair_w_self_scaled": baseline_pair_w_self_scaled,
            "diagnostics_pair_pass": baseline_pair_pass,
            "worst_max_rhat": max(
                float(baseline_diag_a["max_rhat"]),
                float(baseline_diag_b["max_rhat"]),
            ),
            "worst_min_bulk_ess": min(
                float(baseline_diag_a["min_bulk_ess"]),
                float(baseline_diag_b["min_bulk_ess"]),
            ),
            "worst_min_tail_ess": min(
                float(baseline_diag_a["min_tail_ess"]),
                float(baseline_diag_b["min_tail_ess"]),
            ),
            "max_divergence_count": max(
                int(baseline_diag_a["divergence_count"]),
                int(baseline_diag_b["divergence_count"]),
            ),
            "true_A": float(true_theta[0]),
            "true_k": float(true_theta[1]),
            "true_sigma": float(true_theta[2]),
        })
        pooled_threshold = max(
            float(args.absolute_w_tolerance),
            float(args.floor_multiplier * baseline_pair_w),
        )
        pair_threshold = max(
            float(args.pair_w_tolerance),
            float(args.floor_multiplier * baseline_pair_w),
        )
        print(
            f"  long pair seconds={baseline_pair_seconds:.3f} "
            f"pair_w={baseline_pair_w:.4f} "
            f"diagnostics_pass={baseline_pair_pass}",
            flush=True,
        )

        first_stable: dict[str, object] | None = None
        for steps in args.steps:
            burn_in = burn_in_for_steps(steps, args.burn_in_fraction)
            seed_a = args.seed + 10_000 * (obs_index + 1) + 10 * steps + 1
            seed_b = args.seed + 10_000 * (obs_index + 1) + 10 * steps + 2
            run_a, run_b = run_hmc_pair(
                t=t,
                y=y,
                chains=args.chains,
                steps=steps,
                burn_in=burn_in,
                seed_a=seed_a,
                seed_b=seed_b,
                step_size=args.step_size,
                leapfrog_steps=args.leapfrog_steps,
                requested_device=args.device,
                device=device,
                dtype=dtype,
                divergence_threshold=args.divergence_threshold,
                rhat_threshold=args.rhat_threshold,
                min_ess=args.min_ess,
            )
            theta_a = posterior_draws(np.asarray(run_a["theta_samples"]), burn_in)
            theta_b = posterior_draws(np.asarray(run_b["theta_samples"]), burn_in)
            pooled = np.vstack([theta_a, theta_b])
            pair_sd = np.std(pooled, axis=0, ddof=1)
            rng = np.random.default_rng(args.seed + 4_000_000 + obs_index * 1000 + steps)
            theta_a_w = maybe_subsample(theta_a, max_samples=args.max_w_samples, rng=rng)
            theta_b_w = maybe_subsample(theta_b, max_samples=args.max_w_samples, rng=rng)
            pooled_w = maybe_subsample(pooled, max_samples=args.max_w_samples, rng=rng)
            baseline_pooled_w = maybe_subsample(
                baseline_pooled,
                max_samples=args.max_w_samples,
                rng=rng,
            )
            pair_w = normalized_wasserstein(theta_a_w, theta_b_w, baseline_sd)
            pair_w_self_scaled = normalized_wasserstein(theta_a_w, theta_b_w, pair_sd)
            pooled_w_to_long_pair = normalized_wasserstein(
                pooled_w,
                baseline_pooled_w,
                baseline_sd,
            )
            diag_a = run_a["diagnostic_summary"]
            diag_b = run_b["diagnostic_summary"]
            diagnostics_pair_pass = bool(run_a["diagnostics_pass"]) and bool(
                run_b["diagnostics_pass"]
            )
            reference_pair_pass = (
                diagnostics_pair_pass
                and pair_w <= pair_threshold
                and pooled_w_to_long_pair <= pooled_threshold
            )
            row = {
                "observation_index": obs_index,
                "steps": steps,
                "burn_in": burn_in,
                "pair_seconds": float(run_a["seconds"]) + float(run_b["seconds"]),
                "seconds_a": run_a["seconds"],
                "seconds_b": run_b["seconds"],
                "mean_run_seconds": 0.5 * (float(run_a["seconds"]) + float(run_b["seconds"])),
                "acceptance_rate_a": run_a["acceptance_rate"],
                "acceptance_rate_b": run_b["acceptance_rate"],
                "pair_w": pair_w,
                "pair_w_self_scaled": pair_w_self_scaled,
                "pooled_w_to_long_pair": pooled_w_to_long_pair,
                "baseline_pair_w": baseline_pair_w,
                "baseline_pair_w_self_scaled": baseline_pair_w_self_scaled,
                "pair_threshold": pair_threshold,
                "pooled_threshold": pooled_threshold,
                "diagnostics_pair_pass": diagnostics_pair_pass,
                "reference_pair_pass": reference_pair_pass,
                "worst_max_rhat": max(float(diag_a["max_rhat"]), float(diag_b["max_rhat"])),
                "worst_min_bulk_ess": min(
                    float(diag_a["min_bulk_ess"]),
                    float(diag_b["min_bulk_ess"]),
                ),
                "worst_min_tail_ess": min(
                    float(diag_a["min_tail_ess"]),
                    float(diag_b["min_tail_ess"]),
                ),
                "max_divergence_count": max(
                    int(diag_a["divergence_count"]),
                    int(diag_b["divergence_count"]),
                ),
                "true_A": float(true_theta[0]),
                "true_k": float(true_theta[1]),
                "true_sigma": float(true_theta[2]),
            }
            rows.append(row)
            if first_stable is None and reference_pair_pass:
                first_stable = row
            print(
                f"  steps={steps} pair_seconds={row['pair_seconds']:.3f} "
                f"pair_w={pair_w:.4f} pooled_w={pooled_w_to_long_pair:.4f} "
                f"Rhat={row['worst_max_rhat']:.4f} "
                f"ESS={row['worst_min_bulk_ess']:.0f} pass={reference_pair_pass}",
                flush=True,
            )

        observation_rows.append({
            "observation_index": obs_index,
            "true_theta": {
                name: float(true_theta[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "baseline_pair_seconds": baseline_pair_seconds,
            "baseline_pair_w": baseline_pair_w,
            "baseline_pair_w_self_scaled": baseline_pair_w_self_scaled,
            "baseline_diagnostics_pair_pass": baseline_pair_pass,
            "pair_threshold": pair_threshold,
            "pooled_threshold": pooled_threshold,
            "first_stable_steps": None if first_stable is None else int(first_stable["steps"]),
            "first_stable_pair_seconds": (
                None if first_stable is None else float(first_stable["pair_seconds"])
            ),
        })

    summary_rows = summarize_by_steps(rows)
    baseline_summary = {
        "pair_seconds": summarize([float(row["pair_seconds"]) for row in baseline_rows]),
        "pair_w": summarize([float(row["pair_w"]) for row in baseline_rows]),
        "pair_w_self_scaled": summarize(
            [float(row["pair_w_self_scaled"]) for row in baseline_rows]
        ),
        "worst_max_rhat": summarize([float(row["worst_max_rhat"]) for row in baseline_rows]),
        "worst_min_bulk_ess": summarize(
            [float(row["worst_min_bulk_ess"]) for row in baseline_rows]
        ),
        "worst_min_tail_ess": summarize(
            [float(row["worst_min_tail_ess"]) for row in baseline_rows]
        ),
        "diagnostics_pair_pass_fraction": float(
            np.mean([bool(row["diagnostics_pair_pass"]) for row in baseline_rows])
        ),
    }
    first_stable_seconds = [
        float(row["first_stable_pair_seconds"])
        for row in observation_rows
        if row["first_stable_pair_seconds"] is not None
    ]
    first_stable_steps = [
        float(row["first_stable_steps"])
        for row in observation_rows
        if row["first_stable_steps"] is not None
    ]

    figure_path = figure_dir / "decay_hmc_two_reference_stability.png"
    rows_csv = results_dir / "hmc_two_reference_rows.csv"
    baseline_csv = results_dir / "hmc_long_pair_rows.csv"
    summary_json = results_dir / "decay_hmc_two_reference_stability_summary.json"
    plot_convergence(summary_rows=summary_rows, outfile=figure_path)
    write_csv(rows, rows_csv)
    write_csv(baseline_rows, baseline_csv)

    output = {
        "config": json_ready(vars(args)),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "panel_metadata": panel_metadata,
        "selected_observation_indices": selected_indices,
        "baseline_summary": baseline_summary,
        "budget_summary": summary_rows,
        "observation_rows": observation_rows,
        "first_stable_summary": {
            "pair_seconds": summarize(first_stable_seconds),
            "steps": summarize(first_stable_steps),
            "observations_passing": len(first_stable_seconds),
            "observations_total": len(selected_indices),
        },
        "reference_pair_screen_definition": {
            "per_run_diagnostics": {
                "max_rhat_below": args.rhat_threshold,
                "min_bulk_ess_above": args.min_ess,
                "min_tail_ess_above": args.min_ess,
                "divergence_count": 0,
            },
            "pair_w_threshold": (
                "pair_w <= max(pair_w_tolerance, "
                "floor_multiplier * long_pair_w)"
            ),
            "pooled_w_threshold": (
                "pooled_w_to_long_pair <= max(absolute_w_tolerance, "
                "floor_multiplier * long_pair_w)"
            ),
            "absolute_w_tolerance": args.absolute_w_tolerance,
            "pair_w_tolerance": args.pair_w_tolerance,
            "floor_multiplier": args.floor_multiplier,
        },
        "outputs": {
            "figure": str(figure_path),
            "rows_csv": str(rows_csv),
            "baseline_csv": str(baseline_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")

    print(f"figure: {figure_path}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print("budget medians:", flush=True)
    for row in summary_rows:
        print(
            f"  steps={row['steps']}: pair_seconds={row['pair_seconds']['median']:.3f}, "
            f"pair_w={row['pair_w']['median']:.4f}, "
            f"pooled_w={row['pooled_w_to_long_pair']['median']:.4f}, "
            f"max_Rhat={row['worst_max_rhat']['median']:.4f}, "
            f"min_bulk_ESS={row['worst_min_bulk_ess']['median']:.0f}, "
            f"pass_fraction={row['reference_pair_pass_fraction']:.2f}",
            flush=True,
        )
    print(
        "first stable two-reference seconds median: "
        f"{output['first_stable_summary']['pair_seconds']['median']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
