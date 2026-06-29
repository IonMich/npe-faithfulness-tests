from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from npe_flow_stress_tests import make_sign_case


RAW_NAMES = (r"$\theta_1$", r"$\theta_2$")
DIAGNOSTIC_NAMES = (r"$|\theta_1|$", r"$\theta_2$")


def log_normal(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (
        -0.5 * ((value - mean) / std) ** 2
        - math.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    )


def weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    mean = float(np.sum(values * weights))
    variance = float(np.sum((values - mean) ** 2 * weights))
    return {
        "mean": mean,
        "sd": math.sqrt(max(variance, 0.0)),
        "q05": float(np.interp(0.05, cumulative, sorted_values)),
        "q16": float(np.interp(0.16, cumulative, sorted_values)),
        "median": float(np.interp(0.50, cumulative, sorted_values)),
        "q84": float(np.interp(0.84, cumulative, sorted_values)),
        "q95": float(np.interp(0.95, cumulative, sorted_values)),
    }


def sample_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q16": float(np.quantile(values, 0.16)),
        "median": float(np.quantile(values, 0.50)),
        "q84": float(np.quantile(values, 0.84)),
        "q95": float(np.quantile(values, 0.95)),
    }


def build_grid_reference(
    *,
    x0: np.ndarray,
    grid_size: int,
    grid_limit: float,
) -> dict[str, object]:
    case = make_sign_case()
    theta1 = np.linspace(-grid_limit, grid_limit, grid_size)
    theta2 = np.linspace(-grid_limit, grid_limit, grid_size)
    prior_mean = case.prior_mean
    prior_std = case.prior_std
    sigma = np.array([0.22, 0.16])

    logp_theta1 = (
        log_normal(theta1, float(prior_mean[0]), float(prior_std[0]))
        + log_normal(float(x0[0]), theta1**2, float(sigma[0]))
    )
    logp_theta2 = (
        log_normal(theta2, float(prior_mean[1]), float(prior_std[1]))
        + log_normal(float(x0[1]), theta2, float(sigma[1]))
    )

    logw = logp_theta1[:, None] + logp_theta2[None, :]
    weights = np.exp(logw - logsumexp(logw))
    marginal_theta1 = weights.sum(axis=1)
    marginal_theta2 = weights.sum(axis=0)

    abs_theta1 = np.abs(theta1)
    positive_mass = float(marginal_theta1[theta1 > 0.0].sum())
    return {
        "theta1_grid": theta1,
        "theta2_grid": theta2,
        "weights": weights,
        "marginal_theta1": marginal_theta1,
        "marginal_theta2": marginal_theta2,
        "raw": {
            RAW_NAMES[0]: {
                "values": theta1,
                "weights": marginal_theta1,
                "summary": weighted_summary(theta1, marginal_theta1),
            },
            RAW_NAMES[1]: {
                "values": theta2,
                "weights": marginal_theta2,
                "summary": weighted_summary(theta2, marginal_theta2),
            },
        },
        "diagnostic": {
            DIAGNOSTIC_NAMES[0]: {
                "values": abs_theta1,
                "weights": marginal_theta1,
                "summary": weighted_summary(abs_theta1, marginal_theta1),
            },
            DIAGNOSTIC_NAMES[1]: {
                "values": theta2,
                "weights": marginal_theta2,
                "summary": weighted_summary(theta2, marginal_theta2),
            },
        },
        "edge_mass": {
            RAW_NAMES[0]: {
                "lower_grid_edge": float(marginal_theta1[0]),
                "upper_grid_edge": float(marginal_theta1[-1]),
            },
            RAW_NAMES[1]: {
                "lower_grid_edge": float(marginal_theta2[0]),
                "upper_grid_edge": float(marginal_theta2[-1]),
            },
        },
        "mode_mass": {
            "positive_theta1": positive_mass,
            "negative_theta1": float(1.0 - positive_mass),
            "positive_mass_error_vs_half": abs(positive_mass - 0.5),
        },
    }


def compare_samples_to_reference(
    samples: np.ndarray,
    reference: dict[str, object],
    *,
    diagnostic: bool,
) -> dict[str, object]:
    family = "diagnostic" if diagnostic else "raw"
    names = DIAGNOSTIC_NAMES if diagnostic else RAW_NAMES
    transformed = np.column_stack([np.abs(samples[:, 0]), samples[:, 1]]) if diagnostic else samples
    per_dim = {}
    values = []
    for index, name in enumerate(names):
        ref = reference[family][name]
        ref_values = np.asarray(ref["values"])
        ref_weights = np.asarray(ref["weights"])
        ref_sd = max(float(ref["summary"]["sd"]), 1e-12)
        distance = wasserstein_distance(transformed[:, index], ref_values, v_weights=ref_weights)
        normalized = distance / ref_sd
        per_dim[name] = {
            "wasserstein": float(distance),
            "normalized_wasserstein": float(normalized),
        }
        values.append(normalized)
    return {
        "mean_normalized_wasserstein": float(np.mean(values)),
        "max_normalized_wasserstein": float(np.max(values)),
        "per_dim": per_dim,
    }


def compare_samples_pairwise(
    left: np.ndarray,
    right: np.ndarray,
    *,
    diagnostic: bool,
) -> dict[str, object]:
    names = DIAGNOSTIC_NAMES if diagnostic else RAW_NAMES
    left_t = np.column_stack([np.abs(left[:, 0]), left[:, 1]]) if diagnostic else left
    right_t = np.column_stack([np.abs(right[:, 0]), right[:, 1]]) if diagnostic else right
    pooled = np.concatenate([left_t, right_t], axis=0)
    scale = np.maximum(np.std(pooled, axis=0), 1e-12)
    per_dim = {}
    values = []
    for index, name in enumerate(names):
        distance = wasserstein_distance(left_t[:, index], right_t[:, index])
        normalized = distance / scale[index]
        per_dim[name] = float(normalized)
        values.append(normalized)
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "per_dim": per_dim,
    }


def sample_grid(
    reference: dict[str, object],
    *,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    theta1 = np.asarray(reference["theta1_grid"])
    theta2 = np.asarray(reference["theta2_grid"])
    flat_weights = np.asarray(reference["weights"]).reshape(-1)
    flat_index = rng.choice(flat_weights.size, size=n, replace=True, p=flat_weights)
    i, j = np.unravel_index(flat_index, (theta1.size, theta2.size))
    return np.column_stack([theta1[i], theta2[j]])


def split_chains(samples: np.ndarray, burn_in: int, split: int) -> tuple[np.ndarray, np.ndarray]:
    posterior = samples[:, burn_in:, :]
    return (
        posterior[:split].reshape(-1, posterior.shape[2]),
        posterior[split:].reshape(-1, posterior.shape[2]),
    )


def flatten_post_burn(samples: np.ndarray, burn_in: int) -> np.ndarray:
    return samples[:, burn_in:, :].reshape(-1, samples.shape[2])


def mode_summary(samples: np.ndarray) -> dict[str, float]:
    positive = float(np.mean(samples[:, 0] > 0.0))
    return {
        "positive_theta1": positive,
        "negative_theta1": float(1.0 - positive),
        "positive_mass_error_vs_half": abs(positive - 0.5),
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_calibration(summary: dict[str, object], outfile: Path) -> None:
    diagnostics = summary["diagnostics"]
    old_target = summary.get("historical_target")
    target = float(summary["recommended_targets"]["diagnostic_mean_normalized_wasserstein"])
    labels = [
        "MCMC to grid",
        "HMC to grid",
        "NPE to grid",
        "MCMC-HMC",
        "MCMC-NPE",
        "HMC-NPE",
        "grid sample median",
    ]
    values = [
        diagnostics["mcmc_to_grid"]["diagnostic"]["mean_normalized_wasserstein"],
        diagnostics["hmc_to_grid"]["diagnostic"]["mean_normalized_wasserstein"],
        diagnostics["npe_to_grid"]["diagnostic"]["mean_normalized_wasserstein"],
        diagnostics["pairwise"]["diagnostic"]["mcmc_hmc"]["mean"],
        diagnostics["pairwise"]["diagnostic"]["mcmc_npe"]["mean"],
        diagnostics["pairwise"]["diagnostic"]["hmc_npe"]["mean"],
        summary["grid_replicates"]["to_grid"]["diagnostic"]["median"],
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(values))
    ax.bar(x, values, color="#4f6f7d", alpha=0.82)
    ax.axhline(target, color="#111827", linestyle="-", linewidth=1.6, label=f"sign calibrated = {target:.4f}")
    if old_target is not None:
        old_target_float = float(old_target)
        ax.axhline(
            old_target_float,
            color="#9b2f2f",
            linestyle="--",
            linewidth=1.5,
            label=f"historical = {old_target_float:.3f}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("mean normalized Wasserstein")
    ax.set_title("Sign model: diagnostic target calibration")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a model-specific target for the sign-symmetry stress case.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("runs/02_stress_sign/01_npe_flow/01_npe_flow_stress_tests_sign/results/sign_samples.npz"),
    )
    parser.add_argument("--mcmc-burn-in", type=int, default=3000)
    parser.add_argument("--hmc-burn-in", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--grid-limit", type=float, default=4.0)
    parser.add_argument("--grid-sample-count", type=int, default=60_000)
    parser.add_argument("--grid-replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--historical-target", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/02_stress_sign/02_reference_calibration/01_sign_grid_reference/results"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("runs/02_stress_sign/02_reference_calibration/01_sign_grid_reference/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.samples, allow_pickle=True)
    x0 = np.asarray(data["x0"], dtype=np.float64)
    mcmc = np.asarray(data["mcmc_z"], dtype=np.float64)
    hmc = np.asarray(data["hmc_z"], dtype=np.float64)
    npe = np.asarray(data["npe_z"], dtype=np.float64)
    reference = build_grid_reference(x0=x0, grid_size=args.grid_size, grid_limit=args.grid_limit)

    mcmc_post = flatten_post_burn(mcmc, args.mcmc_burn_in)
    hmc_post = flatten_post_burn(hmc, args.hmc_burn_in)
    mcmc_a, mcmc_b = split_chains(mcmc, args.mcmc_burn_in, split=mcmc.shape[0] // 2)
    hmc_a, hmc_b = split_chains(hmc, args.hmc_burn_in, split=hmc.shape[0] // 2)

    rng = np.random.default_rng(args.seed)
    grid_to_grid_raw = []
    grid_to_grid_diagnostic = []
    for index in range(args.grid_replicates):
        grid_sample = sample_grid(reference, n=args.grid_sample_count, rng=rng)
        grid_to_grid_raw.append(compare_samples_to_reference(grid_sample, reference, diagnostic=False))
        grid_to_grid_diagnostic.append(compare_samples_to_reference(grid_sample, reference, diagnostic=True))

    diagnostics = {
        "mcmc_to_grid": {
            "raw": compare_samples_to_reference(mcmc_post, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(mcmc_post, reference, diagnostic=True),
        },
        "hmc_to_grid": {
            "raw": compare_samples_to_reference(hmc_post, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(hmc_post, reference, diagnostic=True),
        },
        "npe_to_grid": {
            "raw": compare_samples_to_reference(npe, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(npe, reference, diagnostic=True),
        },
        "mcmc_chain_half_1_to_grid": {
            "raw": compare_samples_to_reference(mcmc_a, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(mcmc_a, reference, diagnostic=True),
        },
        "mcmc_chain_half_2_to_grid": {
            "raw": compare_samples_to_reference(mcmc_b, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(mcmc_b, reference, diagnostic=True),
        },
        "hmc_chain_half_1_to_grid": {
            "raw": compare_samples_to_reference(hmc_a, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(hmc_a, reference, diagnostic=True),
        },
        "hmc_chain_half_2_to_grid": {
            "raw": compare_samples_to_reference(hmc_b, reference, diagnostic=False),
            "diagnostic": compare_samples_to_reference(hmc_b, reference, diagnostic=True),
        },
        "pairwise": {
            "raw": {
                "mcmc_hmc": compare_samples_pairwise(mcmc_post, hmc_post, diagnostic=False),
                "mcmc_npe": compare_samples_pairwise(mcmc_post, npe, diagnostic=False),
                "hmc_npe": compare_samples_pairwise(hmc_post, npe, diagnostic=False),
            },
            "diagnostic": {
                "mcmc_hmc": compare_samples_pairwise(mcmc_post, hmc_post, diagnostic=True),
                "mcmc_npe": compare_samples_pairwise(mcmc_post, npe, diagnostic=True),
                "hmc_npe": compare_samples_pairwise(hmc_post, npe, diagnostic=True),
            },
        },
    }

    diagnostic_target = max(
        diagnostics["mcmc_to_grid"]["diagnostic"]["mean_normalized_wasserstein"],
        diagnostics["hmc_to_grid"]["diagnostic"]["mean_normalized_wasserstein"],
    )
    raw_target = max(
        diagnostics["mcmc_to_grid"]["raw"]["mean_normalized_wasserstein"],
        diagnostics["hmc_to_grid"]["raw"]["mean_normalized_wasserstein"],
    )
    mode_target = max(
        mode_summary(mcmc_post)["positive_mass_error_vs_half"],
        mode_summary(hmc_post)["positive_mass_error_vs_half"],
    )
    summary = {
        "case": "sign",
        "samples": str(args.samples),
        "x0": x0,
        "historical_target": args.historical_target,
        "metric_note": (
            "Diagnostic distances use (abs(theta_1), theta_2), which factors out the sign symmetry. "
            "Raw distances and mode mass are also reported because the true posterior is bimodal."
        ),
        "grid_reference": {
            "grid_size": args.grid_size,
            "grid_limit": args.grid_limit,
            "edge_mass": reference["edge_mass"],
            "raw_summary": {name: reference["raw"][name]["summary"] for name in RAW_NAMES},
            "diagnostic_summary": {
                name: reference["diagnostic"][name]["summary"] for name in DIAGNOSTIC_NAMES
            },
            "mode_mass": reference["mode_mass"],
        },
        "sample_summaries": {
            "mcmc": {
                "raw": {name: sample_summary(mcmc_post[:, i]) for i, name in enumerate(RAW_NAMES)},
                "diagnostic": {
                    DIAGNOSTIC_NAMES[0]: sample_summary(np.abs(mcmc_post[:, 0])),
                    DIAGNOSTIC_NAMES[1]: sample_summary(mcmc_post[:, 1]),
                },
                "mode_mass": mode_summary(mcmc_post),
            },
            "hmc": {
                "raw": {name: sample_summary(hmc_post[:, i]) for i, name in enumerate(RAW_NAMES)},
                "diagnostic": {
                    DIAGNOSTIC_NAMES[0]: sample_summary(np.abs(hmc_post[:, 0])),
                    DIAGNOSTIC_NAMES[1]: sample_summary(hmc_post[:, 1]),
                },
                "mode_mass": mode_summary(hmc_post),
            },
            "npe": {
                "raw": {name: sample_summary(npe[:, i]) for i, name in enumerate(RAW_NAMES)},
                "diagnostic": {
                    DIAGNOSTIC_NAMES[0]: sample_summary(np.abs(npe[:, 0])),
                    DIAGNOSTIC_NAMES[1]: sample_summary(npe[:, 1]),
                },
                "mode_mass": mode_summary(npe),
            },
        },
        "diagnostics": diagnostics,
        "grid_replicates": {
            "sample_count": args.grid_sample_count,
            "replicates": args.grid_replicates,
            "to_grid": {
                "raw": {
                    "values": [item["mean_normalized_wasserstein"] for item in grid_to_grid_raw],
                    "median": float(np.median([item["mean_normalized_wasserstein"] for item in grid_to_grid_raw])),
                    "max": float(np.max([item["mean_normalized_wasserstein"] for item in grid_to_grid_raw])),
                },
                "diagnostic": {
                    "values": [item["mean_normalized_wasserstein"] for item in grid_to_grid_diagnostic],
                    "median": float(
                        np.median([item["mean_normalized_wasserstein"] for item in grid_to_grid_diagnostic])
                    ),
                    "max": float(np.max([item["mean_normalized_wasserstein"] for item in grid_to_grid_diagnostic])),
                },
            },
        },
        "recommended_targets": {
            "diagnostic_mean_normalized_wasserstein": float(diagnostic_target),
            "raw_mean_normalized_wasserstein": float(raw_target),
            "positive_mode_mass_error": float(mode_target),
            "rule": "max(full MCMC-to-grid, full HMC-to-grid) for the selected coordinate family",
        },
        "target_checks": {
            "npe_passes_diagnostic_target": bool(
                diagnostics["npe_to_grid"]["diagnostic"]["mean_normalized_wasserstein"] <= diagnostic_target
            ),
            "npe_passes_raw_target": bool(
                diagnostics["npe_to_grid"]["raw"]["mean_normalized_wasserstein"] <= raw_target
            ),
            "npe_passes_mode_mass_target": bool(
                mode_summary(npe)["positive_mass_error_vs_half"] <= mode_target
            ),
            "npe_passes_historical_diagnostic_target": None
            if args.historical_target is None
            else bool(
                diagnostics["npe_to_grid"]["diagnostic"]["mean_normalized_wasserstein"]
                <= args.historical_target
            ),
        },
        "outputs": {
            "summary_json": str(args.output_dir / "sign_target_calibration_summary.json"),
            "calibration_png": str(args.figure_dir / "sign_target_calibration.png"),
        },
    }

    summary_json = args.output_dir / "sign_target_calibration_summary.json"
    figure_path = args.figure_dir / "sign_target_calibration.png"
    summary_json.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    plot_calibration(json_ready(summary), figure_path)
    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")
    print(
        "diagnostic target:",
        f"{diagnostic_target:.6f}",
        "NPE diagnostic:",
        f"{diagnostics['npe_to_grid']['diagnostic']['mean_normalized_wasserstein']:.6f}",
    )
    print(
        "raw target:",
        f"{raw_target:.6f}",
        "NPE raw:",
        f"{diagnostics['npe_to_grid']['raw']['mean_normalized_wasserstein']:.6f}",
    )
    print(
        "mode target:",
        f"{mode_target:.6f}",
        "NPE mode error:",
        f"{mode_summary(npe)['positive_mass_error_vs_half']:.6f}",
    )


if __name__ == "__main__":
    main()
