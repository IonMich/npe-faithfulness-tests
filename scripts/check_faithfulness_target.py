from __future__ import annotations

import argparse
import json
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
from scipy.stats import wasserstein_distance

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples
from mcmc_decay_inference import PARAMETER_NAMES
from npe_stage1_decay import sample_grid_reference

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def chain_split_samples(samples: dict[str, object], split: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(samples["theta_samples"])
    burn_in = int(samples["burn_in"])
    posterior = theta[:, burn_in:, :]
    return (
        posterior[:split].reshape(-1, 3),
        posterior[split:].reshape(-1, 3),
    )


def step_split_samples(samples: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(samples["theta_samples"])
    burn_in = int(samples["burn_in"])
    posterior = theta[:, burn_in:, :]
    return (
        posterior[:, 0::2, :].reshape(-1, 3),
        posterior[:, 1::2, :].reshape(-1, 3),
    )


def sample_to_sample_normalized_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    reference: dict[str, object],
) -> dict[str, object]:
    ref_summary = reference["summary"]
    per_parameter = {}
    values = []
    for index, name in enumerate(PARAMETER_NAMES):
        ref_sd = max(float(ref_summary[name]["sd"]), 1e-12)
        distance = wasserstein_distance(left[:, index], right[:, index])
        normalized = distance / ref_sd
        per_parameter[name] = {
            "wasserstein": float(distance),
            "normalized_wasserstein": float(normalized),
        }
        values.append(normalized)
    return {
        "mean_normalized_wasserstein": float(np.mean(values)),
        "per_parameter": per_parameter,
    }


def grid_replicate_diagnostics(
    *,
    reference: dict[str, object],
    sample_count: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    samples = [
        sample_grid_reference(reference, n=sample_count, seed=seed + index)
        for index in range(replicates)
    ]
    to_grid = [
        compare_to_reference(sample, reference)["mean_normalized_wasserstein"]["value"]
        for sample in samples
    ]
    pairwise = []
    for i in range(replicates):
        for j in range(i + 1, replicates):
            pairwise.append(
                sample_to_sample_normalized_wasserstein(samples[i], samples[j], reference)[
                    "mean_normalized_wasserstein"
                ]
            )
    return {
        "sample_count": sample_count,
        "replicates": replicates,
        "to_grid": {
            "values": to_grid,
            "mean": float(np.mean(to_grid)),
            "median": float(np.median(to_grid)),
            "max": float(np.max(to_grid)),
        },
        "pairwise": {
            "values": pairwise,
            "mean": float(np.mean(pairwise)),
            "median": float(np.median(pairwise)),
            "max": float(np.max(pairwise)),
        },
    }


def plot_target_diagnostics(summary: dict[str, object], outfile: Path) -> None:
    target = float(summary["target_wasserstein"])
    labels = []
    values = []
    for key, label in [
        ("mcmc_chain_half_1_to_grid", "MCMC chain half 1"),
        ("mcmc_chain_half_2_to_grid", "MCMC chain half 2"),
        ("hmc_chain_half_1_to_grid", "HMC chain half 1"),
        ("hmc_chain_half_2_to_grid", "HMC chain half 2"),
        ("mcmc_step_even_to_grid", "MCMC even steps"),
        ("hmc_step_even_to_grid", "HMC even steps"),
    ]:
        labels.append(label)
        values.append(summary["diagnostics"][key]["mean_normalized_wasserstein"]["value"])
    labels.extend(["Grid sample to grid", "Grid sample pairwise"])
    values.extend([
        summary["grid_replicates"]["to_grid"]["median"],
        summary["grid_replicates"]["pairwise"]["median"],
    ])

    figure, ax = plt.subplots(figsize=(11, 6.2))
    positions = np.arange(len(values))
    ax.bar(positions, values, color="#4f6f7d", alpha=0.82)
    ax.axhline(target, color="#111827", linestyle="--", linewidth=1.8, label=f"target = {target:.3f}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean normalized Wasserstein")
    ax.set_title("Reference split diagnostics for MC-level target")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def mean_normalized_wasserstein_value(result: dict[str, object]) -> float:
    value = result["mean_normalized_wasserstein"]
    if isinstance(value, dict):
        return float(value["value"])
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the strict NPE faithfulness target is numerically fair.")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--grid-sample-count", type=int, default=60_000)
    parser.add_argument("--grid-replicates", type=int, default=8)
    parser.add_argument(
        "--target-wasserstein",
        type=float,
        default=None,
        help=(
            "Optional override. By default the target is max(full MCMC-to-grid, "
            "full HMC-to-grid)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--output-dir", type=Path, default=ap.FAITHFULNESS_TARGET_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.FAITHFULNESS_TARGET_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

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

    mcmc_chain_a, mcmc_chain_b = chain_split_samples(mcmc, split=np.asarray(mcmc["theta_samples"]).shape[0] // 2)
    hmc_chain_a, hmc_chain_b = chain_split_samples(hmc, split=np.asarray(hmc["theta_samples"]).shape[0] // 2)
    mcmc_step_a, mcmc_step_b = step_split_samples(mcmc)
    hmc_step_a, hmc_step_b = step_split_samples(hmc)

    diagnostics = {
        "mcmc_full_to_grid": compare_to_reference(mcmc["posterior_theta"], reference),
        "hmc_full_to_grid": compare_to_reference(hmc["posterior_theta"], reference),
        "mcmc_chain_half_1_to_grid": compare_to_reference(mcmc_chain_a, reference),
        "mcmc_chain_half_2_to_grid": compare_to_reference(mcmc_chain_b, reference),
        "hmc_chain_half_1_to_grid": compare_to_reference(hmc_chain_a, reference),
        "hmc_chain_half_2_to_grid": compare_to_reference(hmc_chain_b, reference),
        "mcmc_step_even_to_grid": compare_to_reference(mcmc_step_a, reference),
        "mcmc_step_odd_to_grid": compare_to_reference(mcmc_step_b, reference),
        "hmc_step_even_to_grid": compare_to_reference(hmc_step_a, reference),
        "hmc_step_odd_to_grid": compare_to_reference(hmc_step_b, reference),
        "mcmc_chain_halves_pairwise": sample_to_sample_normalized_wasserstein(mcmc_chain_a, mcmc_chain_b, reference),
        "hmc_chain_halves_pairwise": sample_to_sample_normalized_wasserstein(hmc_chain_a, hmc_chain_b, reference),
        "mcmc_hmc_full_pairwise": sample_to_sample_normalized_wasserstein(
            mcmc["posterior_theta"],
            hmc["posterior_theta"],
            reference,
        ),
    }
    grid_replicates = grid_replicate_diagnostics(
        reference=reference,
        sample_count=args.grid_sample_count,
        replicates=args.grid_replicates,
        seed=args.seed,
    )
    recommended_target = max(
        mean_normalized_wasserstein_value(diagnostics["mcmc_full_to_grid"]),
        mean_normalized_wasserstein_value(diagnostics["hmc_full_to_grid"]),
    )
    target_wasserstein = float(args.target_wasserstein) if args.target_wasserstein is not None else recommended_target
    summary = {
        "target_wasserstein": target_wasserstein,
        "target_source": "explicit" if args.target_wasserstein is not None else "mcmc_hmc_to_grid",
        "recommended_targets": {
            "mean_normalized_wasserstein": recommended_target,
            "rule": "max(full MCMC-to-grid, full HMC-to-grid)",
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "diagnostics": diagnostics,
        "grid_replicates": grid_replicates,
    }
    summary_json = args.output_dir / "faithfulness_target_check_summary.json"
    figure_path = args.figure_dir / "faithfulness_target_check.png"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_target_diagnostics(summary, figure_path)

    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")
    print(f"target_wasserstein: {target_wasserstein:.5f} ({summary['target_source']})")
    print("mean normalized Wasserstein diagnostics:")
    for key in [
        "mcmc_full_to_grid",
        "hmc_full_to_grid",
        "mcmc_chain_half_1_to_grid",
        "mcmc_chain_half_2_to_grid",
        "hmc_chain_half_1_to_grid",
        "hmc_chain_half_2_to_grid",
        "mcmc_chain_halves_pairwise",
        "hmc_chain_halves_pairwise",
        "mcmc_hmc_full_pairwise",
    ]:
        value = diagnostics[key]["mean_normalized_wasserstein"]
        if isinstance(value, dict):
            value = value["value"]
        print(f"  {key}: {value:.5f}")
    print(
        "  grid_sample_to_grid_median: "
        f"{grid_replicates['to_grid']['median']:.5f}"
    )
    print(
        "  grid_sample_pairwise_median: "
        f"{grid_replicates['pairwise']['median']:.5f}"
    )


if __name__ == "__main__":
    main()
