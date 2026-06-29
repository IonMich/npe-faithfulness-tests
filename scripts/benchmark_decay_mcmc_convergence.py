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
from mcmc_decay_inference import (
    MCMCConfig,
    PARAMETER_NAMES,
    arviz_diagnostics,
    choose_device,
    convergence_flags,
    run_random_walk_metropolis,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/11_convergence_benchmarks/04_mcmc_prior_predictive_stability"
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


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("proposal scale must contain three comma-separated floats")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("proposal scales must be positive")
    return pieces[0], pieces[1], pieces[2]


def posterior_draws(theta_samples: np.ndarray, burn_in: int) -> np.ndarray:
    return theta_samples[:, burn_in:, :].reshape(-1, theta_samples.shape[-1])


def split_half_draws(theta_samples: np.ndarray, burn_in: int) -> tuple[np.ndarray, np.ndarray]:
    kept = theta_samples[:, burn_in:, :]
    midpoint = kept.shape[1] // 2
    if midpoint <= 0:
        raise ValueError("not enough post-burn-in samples to split")
    return (
        kept[:, :midpoint, :].reshape(-1, kept.shape[-1]),
        kept[:, midpoint:, :].reshape(-1, kept.shape[-1]),
    )


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


def diagnostic_summary(diagnostics: dict[str, dict[str, float]]) -> dict[str, float]:
    rhat = [values["rhat"] for values in diagnostics.values()]
    bulk = [values["ess_bulk"] for values in diagnostics.values()]
    tail = [values["ess_tail"] for values in diagnostics.values()]
    return {
        "max_rhat": float(np.max(rhat)),
        "min_bulk_ess": float(np.min(bulk)),
        "min_tail_ess": float(np.min(tail)),
    }


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
        item: dict[str, object] = {
            "steps": steps,
            "n": len(step_rows),
            "seconds": summarize([float(row["seconds"]) for row in step_rows]),
            "w_to_baseline": summarize([float(row["w_to_baseline"]) for row in step_rows]),
            "split_half_w": summarize([float(row["split_half_w"]) for row in step_rows]),
            "max_rhat": summarize([float(row["max_rhat"]) for row in step_rows]),
            "min_bulk_ess": summarize([float(row["min_bulk_ess"]) for row in step_rows]),
            "min_tail_ess": summarize([float(row["min_tail_ess"]) for row in step_rows]),
            "acceptance_rate": summarize([float(row["acceptance_rate"]) for row in step_rows]),
            "diagnostics_pass_fraction": float(
                np.mean([bool(row["diagnostics_pass"]) for row in step_rows])
            ),
            "stable_screen_fraction": float(
                np.mean([bool(row["stable_screen"]) for row in step_rows])
            ),
        }
        output.append(item)
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one_mcmc(
    *,
    t: np.ndarray,
    y: np.ndarray,
    chains: int,
    steps: int,
    burn_in: int,
    seed: int,
    proposal_scale: tuple[float, float, float],
    requested_device: str,
    sampler_variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    config = MCMCConfig(
        chains=chains,
        steps=steps,
        burn_in=burn_in,
        seed=seed,
        proposal_scale=proposal_scale,
        requested_device=requested_device,
        sampler_variant=sampler_variant,
    )
    z_samples, theta_samples, accepted, seconds = run_random_walk_metropolis(
        t=torch.as_tensor(t, dtype=torch.float64),
        y=torch.as_tensor(y, dtype=torch.float64),
        config=config,
        device=device,
        dtype=dtype,
    )
    diagnostics = arviz_diagnostics(theta_samples, burn_in)
    flags = convergence_flags(diagnostics)
    summary = diagnostic_summary(diagnostics)
    return {
        "config": config,
        "z_samples": z_samples,
        "theta_samples": theta_samples,
        "accepted": accepted,
        "seconds": float(seconds),
        "diagnostics": diagnostics,
        "convergence_flags": flags,
        "diagnostic_summary": summary,
        "acceptance_rate": float(accepted.mean()),
    }


def plot_convergence(
    *,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    outfile: Path,
) -> None:
    seconds = np.asarray([row["seconds"]["median"] for row in summary_rows], dtype=np.float64)
    x_labels = [str(row["steps"]) for row in summary_rows]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    ax = axes[0, 0]
    w = np.asarray([row["w_to_baseline"]["median"] for row in summary_rows], dtype=np.float64)
    w_q16 = np.asarray([row["w_to_baseline"]["q16"] for row in summary_rows], dtype=np.float64)
    w_q84 = np.asarray([row["w_to_baseline"]["q84"] for row in summary_rows], dtype=np.float64)
    split = np.asarray([row["split_half_w"]["median"] for row in summary_rows], dtype=np.float64)
    ax.plot(seconds, w, marker="o", color="#2f6fbb", linewidth=2.0, label="run to long baseline")
    ax.fill_between(seconds, w_q16, w_q84, color="#2f6fbb", alpha=0.16)
    ax.plot(seconds, split, marker="s", color="#2f855a", linewidth=1.8, label="within-run split")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median wall time per signal (seconds)")
    ax.set_ylabel("mean normalized Wasserstein")
    ax.set_title("Posterior distance")
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[0, 1]
    rhat = np.asarray([row["max_rhat"]["median"] for row in summary_rows], dtype=np.float64)
    rhat_q16 = np.asarray([row["max_rhat"]["q16"] for row in summary_rows], dtype=np.float64)
    rhat_q84 = np.asarray([row["max_rhat"]["q84"] for row in summary_rows], dtype=np.float64)
    ax.plot(seconds, rhat, marker="o", color="#8a5a9c", linewidth=2.0)
    ax.fill_between(seconds, rhat_q16, rhat_q84, color="#8a5a9c", alpha=0.16)
    ax.axhline(1.01, color="#172033", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("median wall time per signal (seconds)")
    ax.set_ylabel("median max R-hat across parameters")
    ax.set_title("R-hat")
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    bulk = np.asarray([row["min_bulk_ess"]["median"] for row in summary_rows], dtype=np.float64)
    tail = np.asarray([row["min_tail_ess"]["median"] for row in summary_rows], dtype=np.float64)
    ax.plot(seconds, bulk, marker="o", color="#b85c38", linewidth=2.0, label="bulk ESS")
    ax.plot(seconds, tail, marker="s", color="#5f6f87", linewidth=1.8, label="tail ESS")
    ax.axhline(400.0, color="#172033", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median wall time per signal (seconds)")
    ax.set_ylabel("median min ESS across parameters")
    ax.set_title("Effective sample size")
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[1, 1]
    diag_fraction = np.asarray(
        [row["diagnostics_pass_fraction"] for row in summary_rows],
        dtype=np.float64,
    )
    stable_fraction = np.asarray(
        [row["stable_screen_fraction"] for row in summary_rows],
        dtype=np.float64,
    )
    ax.plot(seconds, diag_fraction, marker="o", color="#404b5a", linewidth=2.0, label="diagnostics")
    ax.plot(seconds, stable_fraction, marker="s", color="#bf4d5a", linewidth=1.8, label="diagnostics + W")
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("median wall time per signal (seconds)")
    ax.set_ylabel("fraction of signals passing")
    ax.set_title("Stability screen")
    ax.grid(alpha=0.22)
    ax.legend()

    for ax in axes.ravel():
        for x, label in zip(seconds, x_labels, strict=True):
            ax.annotate(label, (x, ax.get_ylim()[0]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark random-walk MCMC stability for random exponential-decay signals.",
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
        default=parse_int_list("3000,6000,12000,24000,48000,96000"),
        help="Comma-separated MCMC step budgets to test.",
    )
    parser.add_argument("--baseline-steps", type=int, default=192_000)
    parser.add_argument("--burn-in-fraction", type=float, default=0.25)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--n-observations", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "--sampler-variant",
        choices=["baseline", "pregenerated", "low-overhead", "optimized"],
        default="low-overhead",
    )
    parser.add_argument("--proposal-scale", type=parse_proposal_scale, default=(0.030, 0.030, 0.040))
    parser.add_argument("--max-w-samples", type=int, default=250_000)
    parser.add_argument("--absolute-w-tolerance", type=float, default=0.02)
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
        baseline_seed = args.seed + 10_000 * (obs_index + 1)
        print(
            f"observation {obs_index} ({run_position + 1}/{len(selected_indices)}): "
            f"baseline steps={args.baseline_steps}",
            flush=True,
        )
        baseline = run_one_mcmc(
            t=t,
            y=y,
            chains=args.chains,
            steps=args.baseline_steps,
            burn_in=baseline_burn_in,
            seed=baseline_seed,
            proposal_scale=args.proposal_scale,
            requested_device=args.device,
            sampler_variant=args.sampler_variant,
            device=device,
            dtype=dtype,
        )
        baseline_theta = posterior_draws(
            np.asarray(baseline["theta_samples"]),
            baseline_burn_in,
        )
        reference_sd = np.std(baseline_theta, axis=0, ddof=1)
        rng = np.random.default_rng(args.seed + 1_000_000 + obs_index)
        baseline_left, baseline_right = split_half_draws(
            np.asarray(baseline["theta_samples"]),
            baseline_burn_in,
        )
        baseline_split_w = normalized_wasserstein(
            maybe_subsample(baseline_left, max_samples=args.max_w_samples, rng=rng),
            maybe_subsample(baseline_right, max_samples=args.max_w_samples, rng=rng),
            reference_sd,
        )
        baseline_diag = baseline["diagnostic_summary"]
        baseline_flags = baseline["convergence_flags"]
        baseline_rows.append({
            "observation_index": obs_index,
            "steps": args.baseline_steps,
            "burn_in": baseline_burn_in,
            "seconds": baseline["seconds"],
            "acceptance_rate": baseline["acceptance_rate"],
            "split_half_w": baseline_split_w,
            "max_rhat": baseline_diag["max_rhat"],
            "min_bulk_ess": baseline_diag["min_bulk_ess"],
            "min_tail_ess": baseline_diag["min_tail_ess"],
            "diagnostics_pass": all(bool(value) for value in baseline_flags.values()),
            "true_A": float(true_theta[0]),
            "true_k": float(true_theta[1]),
            "true_sigma": float(true_theta[2]),
        })
        stability_threshold = max(
            float(args.absolute_w_tolerance),
            float(args.floor_multiplier * baseline_split_w),
        )
        print(
            f"  baseline seconds={baseline['seconds']:.3f} "
            f"split_w={baseline_split_w:.4f} threshold={stability_threshold:.4f}",
            flush=True,
        )

        first_stable: dict[str, object] | None = None
        for steps in args.steps:
            burn_in = burn_in_for_steps(steps, args.burn_in_fraction)
            run_seed = args.seed + 10_000 * (obs_index + 1) + steps
            run = run_one_mcmc(
                t=t,
                y=y,
                chains=args.chains,
                steps=steps,
                burn_in=burn_in,
                seed=run_seed,
                proposal_scale=args.proposal_scale,
                requested_device=args.device,
                sampler_variant=args.sampler_variant,
                device=device,
                dtype=dtype,
            )
            theta = posterior_draws(np.asarray(run["theta_samples"]), burn_in)
            rng = np.random.default_rng(args.seed + 2_000_000 + obs_index * 1000 + steps)
            w_to_baseline = normalized_wasserstein(
                maybe_subsample(theta, max_samples=args.max_w_samples, rng=rng),
                maybe_subsample(baseline_theta, max_samples=args.max_w_samples, rng=rng),
                reference_sd,
            )
            left, right = split_half_draws(np.asarray(run["theta_samples"]), burn_in)
            split_w = normalized_wasserstein(
                maybe_subsample(left, max_samples=args.max_w_samples, rng=rng),
                maybe_subsample(right, max_samples=args.max_w_samples, rng=rng),
                reference_sd,
            )
            diag = run["diagnostic_summary"]
            flags = run["convergence_flags"]
            diagnostics_pass = all(bool(value) for value in flags.values())
            stable_screen = diagnostics_pass and w_to_baseline <= stability_threshold
            row = {
                "observation_index": obs_index,
                "steps": steps,
                "burn_in": burn_in,
                "seconds": run["seconds"],
                "acceptance_rate": run["acceptance_rate"],
                "w_to_baseline": w_to_baseline,
                "split_half_w": split_w,
                "baseline_split_half_w": baseline_split_w,
                "stability_threshold": stability_threshold,
                "max_rhat": diag["max_rhat"],
                "min_bulk_ess": diag["min_bulk_ess"],
                "min_tail_ess": diag["min_tail_ess"],
                "diagnostics_pass": diagnostics_pass,
                "stable_screen": stable_screen,
                "true_A": float(true_theta[0]),
                "true_k": float(true_theta[1]),
                "true_sigma": float(true_theta[2]),
            }
            rows.append(row)
            if first_stable is None and stable_screen:
                first_stable = row
            print(
                f"  steps={steps} seconds={run['seconds']:.3f} "
                f"W={w_to_baseline:.4f} Rhat={diag['max_rhat']:.4f} "
                f"bulkESS={diag['min_bulk_ess']:.0f} stable={stable_screen}",
                flush=True,
            )

        observation_rows.append({
            "observation_index": obs_index,
            "true_theta": {
                name: float(true_theta[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "baseline_seconds": baseline["seconds"],
            "baseline_split_half_w": baseline_split_w,
            "stability_threshold": stability_threshold,
            "first_stable_steps": None if first_stable is None else int(first_stable["steps"]),
            "first_stable_seconds": None if first_stable is None else float(first_stable["seconds"]),
        })

    summary_rows = summarize_by_steps(rows)
    baseline_summary = {
        "seconds": summarize([float(row["seconds"]) for row in baseline_rows]),
        "split_half_w": summarize([float(row["split_half_w"]) for row in baseline_rows]),
        "max_rhat": summarize([float(row["max_rhat"]) for row in baseline_rows]),
        "min_bulk_ess": summarize([float(row["min_bulk_ess"]) for row in baseline_rows]),
        "min_tail_ess": summarize([float(row["min_tail_ess"]) for row in baseline_rows]),
        "diagnostics_pass_fraction": float(
            np.mean([bool(row["diagnostics_pass"]) for row in baseline_rows])
        ),
    }
    first_stable_seconds = [
        float(row["first_stable_seconds"])
        for row in observation_rows
        if row["first_stable_seconds"] is not None
    ]
    first_stable_steps = [
        float(row["first_stable_steps"])
        for row in observation_rows
        if row["first_stable_steps"] is not None
    ]

    figure_path = figure_dir / "decay_mcmc_convergence.png"
    rows_csv = results_dir / "mcmc_convergence_rows.csv"
    baseline_csv = results_dir / "mcmc_baseline_rows.csv"
    summary_json = results_dir / "decay_mcmc_convergence_summary.json"
    plot_convergence(rows=rows, summary_rows=summary_rows, outfile=figure_path)
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
            "seconds": summarize(first_stable_seconds),
            "steps": summarize(first_stable_steps),
            "observations_passing": len(first_stable_seconds),
            "observations_total": len(selected_indices),
        },
        "stability_screen_definition": {
            "diagnostics": {
                "all_rhat_below": 1.01,
                "all_bulk_ess_above": 400,
                "all_tail_ess_above": 400,
            },
            "w_to_baseline_threshold": (
                "w_to_baseline <= max(absolute_w_tolerance, "
                "floor_multiplier * baseline_split_half_w)"
            ),
            "absolute_w_tolerance": args.absolute_w_tolerance,
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
            f"  steps={row['steps']}: seconds={row['seconds']['median']:.3f}, "
            f"W={row['w_to_baseline']['median']:.4f}, "
            f"max_Rhat={row['max_rhat']['median']:.4f}, "
            f"min_bulk_ESS={row['min_bulk_ess']['median']:.0f}, "
            f"stable_fraction={row['stable_screen_fraction']:.2f}",
            flush=True,
        )
    print(
        "first stable seconds median: "
        f"{output['first_stable_summary']['seconds']['median']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
