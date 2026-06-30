from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import artifact_paths as ap

import matplotlib
import numpy as np
import torch
from scipy.special import logsumexp

from compare_decay_samplers import compare_to_reference, log_posterior_z_numpy, summarize_samples, weighted_quantile
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    make_model,
    sample_posterior_for_observation,
    standardize,
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


def simulate_test_observations(
    *,
    n: int,
    seed: int,
    n_observations: int = 40,
) -> list[dict[str, np.ndarray]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    z = PRIOR_LOG_MEAN.to(dtype=torch.float64)[None, :] + torch.randn(
        n,
        3,
        generator=generator,
        dtype=torch.float64,
    ) * PRIOR_LOG_STD.to(dtype=torch.float64)[None, :]
    theta = torch.exp(z)
    mean = theta[:, 0:1] * torch.exp(-theta[:, 1:2] * t[None, :])
    x = mean + torch.randn(n, n_observations, generator=generator, dtype=torch.float64) * theta[:, 2:3]
    return [
        {
            "index": np.array(index),
            "t": t.numpy(),
            "x": x[index].numpy(),
            "z_true": z[index].numpy(),
            "theta_true": theta[index].numpy(),
        }
        for index in range(n)
    ]


def reference_ranges_from_npe_samples(
    *,
    z_samples_by_family: dict[str, np.ndarray],
    true_theta: np.ndarray,
) -> np.ndarray:
    combined = np.vstack(list(z_samples_by_family.values()))
    true_z = np.log(true_theta)
    ranges = []
    for dim in range(3):
        low, high = np.quantile(combined[:, dim], [0.0005, 0.9995])
        width = high - low
        low -= max(0.45 * width, 0.16)
        high += max(0.45 * width, 0.16)
        low = min(low, true_z[dim] - 0.25)
        high = max(high, true_z[dim] + 0.25)
        ranges.append((low, high))
    return np.asarray(ranges)


def build_grid_reference_from_ranges(
    *,
    t: np.ndarray,
    y: np.ndarray,
    true_theta: np.ndarray,
    z_ranges: np.ndarray,
    grid_size: int,
    chunk_size: int,
) -> dict[str, object]:
    axes = [np.linspace(low, high, grid_size) for low, high in z_ranges]
    mesh = np.meshgrid(*axes, indexing="ij")
    z_grid = np.column_stack([axis.reshape(-1) for axis in mesh])
    logp = np.empty(z_grid.shape[0], dtype=np.float64)
    for start in range(0, len(z_grid), chunk_size):
        stop = min(start + chunk_size, len(z_grid))
        logp[start:stop] = log_posterior_z_numpy(z_grid[start:stop], t=t, y=y)

    weights = np.exp(logp - logsumexp(logp))
    theta_grid = np.exp(z_grid)
    summary = {}
    for index, name in enumerate(PARAMETER_NAMES):
        values = theta_grid[:, index]
        q05, q16, q50, q84, q95 = weighted_quantile(values, weights, [0.05, 0.16, 0.50, 0.84, 0.95])
        mean = np.sum(values * weights)
        variance = np.sum((values - mean) ** 2 * weights)
        summary[name] = {
            "mean": float(mean),
            "sd": float(math.sqrt(max(variance, 0.0))),
            "q05": float(q05),
            "q16": float(q16),
            "median": float(q50),
            "q84": float(q84),
            "q95": float(q95),
        }
    weight_cube = weights.reshape(grid_size, grid_size, grid_size)
    edge_mass = {}
    for index, name in enumerate(PARAMETER_NAMES):
        edge_mass[name] = {
            "lower": float(np.take(weight_cube, indices=0, axis=index).sum()),
            "upper": float(np.take(weight_cube, indices=grid_size - 1, axis=index).sum()),
        }
    return {
        "grid_size": grid_size,
        "grid_points": int(len(z_grid)),
        "z_ranges": {
            name: [float(z_ranges[index, 0]), float(z_ranges[index, 1])]
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "edge_mass": edge_mass,
        "theta_grid": theta_grid,
        "weights": weights,
        "summary": summary,
    }


def load_stage1_model(
    *,
    family: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, np.ndarray], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    config = Stage1Config(
        train_simulations=cfg["train_simulations"],
        val_simulations=cfg["val_simulations"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        hidden_dim=cfg["hidden_dim"],
        hidden_layers=cfg["hidden_layers"],
        mdn_components=cfg["mdn_components"],
        flow_layers=cfg["flow_layers"],
        flow_context_dim=cfg["flow_context_dim"],
        seed=cfg["seed"],
        observed_seed=cfg["observed_seed"],
        requested_device=cfg["requested_device"],
        families=cfg["families"],
        posterior_samples=cfg["posterior_samples"],
        reference_grid_size=cfg["reference_grid_size"],
        spline_bins=int(cfg.get("spline_bins", 12)),
    )
    model = make_model(family, config, x_dim=40, z_dim=3).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = {
        "x_mean": np.asarray(checkpoint["x_mean"]),
        "x_std": np.asarray(checkpoint["x_std"]),
        "z_mean": np.asarray(checkpoint["z_mean"]),
        "z_std": np.asarray(checkpoint["z_std"]),
    }
    return model, stats, cfg


def plot_multi_x_summary(summary: dict[str, object], outfile: Path) -> None:
    families = summary["families"]
    target_wasserstein = summary.get("target_wasserstein")
    values = [
        [
            obs["families"][family]["mean_normalized_wasserstein"]
            for obs in summary["observations"]
        ]
        for family in families
    ]
    figure, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(families))
    box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
    for patch, family in zip(box["boxes"], families, strict=True):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.35)
        patch.set_edgecolor(FAMILY_COLORS[family])
    for median in box["medians"]:
        median.set_color("#172033")
        median.set_linewidth(2.0)
    for index, family in enumerate(families):
        jitter = np.linspace(-0.08, 0.08, len(values[index]))
        ax.scatter(
            positions[index] + jitter,
            values[index],
            color=FAMILY_COLORS[family],
            s=30,
            zorder=4,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([FAMILY_LABELS[family] for family in families], rotation=15, ha="right")
    ax.set_ylabel("mean normalized Wasserstein to grid posterior")
    ax.set_title("Broad NPE faithfulness across multiple observations")
    if target_wasserstein is not None:
        ax.axhline(
            float(target_wasserstein),
            color="#111827",
            linewidth=1.6,
            linestyle="--",
            label=f"target = {float(target_wasserstein):.3f}",
        )
        ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained broad NPE models across multiple x values.")
    parser.add_argument("--stage1-dir", type=Path, default=ap.NPE_STAGE1_RESULTS)
    parser.add_argument("--families", type=parse_families, default=list(FAMILIES))
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--posterior-samples", type=int, default=40_000)
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--grid-chunk-size", type=int, default=120_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "--target-wasserstein",
        type=float,
        default=None,
        help="Optional override for mean normalized Wasserstein distance.",
    )
    parser.add_argument("--target-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_MULTI_X_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_MULTI_X_FIGURES)
    return parser.parse_args()


def main() -> None:
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
    np.random.seed(args.seed + 17)
    torch.manual_seed(args.seed + 19)
    observations = simulate_test_observations(n=args.num_observations, seed=args.seed)
    models = {}
    stats_by_family = {}
    for family in args.families:
        model, stats, _ = load_stage1_model(
            family=family,
            checkpoint_path=args.stage1_dir / f"{family}_model.pt",
            device=device,
        )
        models[family] = model
        stats_by_family[family] = stats

    output_observations = []
    for obs in observations:
        print(f"evaluating observation {int(obs['index'])}")
        z_samples_by_family = {}
        theta_samples_by_family = {}
        for family, model in models.items():
            stats = stats_by_family[family]
            z_samples, theta_samples = sample_posterior_for_observation(
                model=model,
                observed_x=obs["x"],
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
                z_mean=stats["z_mean"],
                z_std=stats["z_std"],
                n=args.posterior_samples,
                device=device,
            )
            z_samples_by_family[family] = z_samples
            theta_samples_by_family[family] = theta_samples

        z_ranges = reference_ranges_from_npe_samples(
            z_samples_by_family=z_samples_by_family,
            true_theta=obs["theta_true"],
        )
        reference = build_grid_reference_from_ranges(
            t=obs["t"],
            y=obs["x"],
            true_theta=obs["theta_true"],
            z_ranges=z_ranges,
            grid_size=args.grid_size,
            chunk_size=args.grid_chunk_size,
        )
        family_results = {}
        for family, theta_samples in theta_samples_by_family.items():
            metrics = compare_to_reference(theta_samples, reference)
            mean_normalized_wasserstein = metrics["mean_normalized_wasserstein"]["value"]
            family_results[family] = {
                "mean_normalized_wasserstein": mean_normalized_wasserstein,
                "target_pass": bool(mean_normalized_wasserstein <= args.target_wasserstein),
                "target_ratio": float(mean_normalized_wasserstein / args.target_wasserstein),
                "metrics": metrics,
                "posterior_summary": summarize_samples(theta_samples),
            }
        output_observations.append(
            {
                "index": int(obs["index"]),
                "theta_true": {
                    name: float(obs["theta_true"][idx])
                    for idx, name in enumerate(PARAMETER_NAMES)
                },
                "grid_reference": {
                    "grid_size": reference["grid_size"],
                    "grid_points": reference["grid_points"],
                    "edge_mass": reference["edge_mass"],
                    "posterior_summary": reference["summary"],
                },
                "families": family_results,
            }
        )

    aggregate = {}
    for family in args.families:
        values = np.array([
            obs["families"][family]["mean_normalized_wasserstein"]
            for obs in output_observations
        ])
        aggregate[family] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
            "target": float(args.target_wasserstein),
            "target_pass_count": int((values <= args.target_wasserstein).sum()),
            "target_pass_fraction": float((values <= args.target_wasserstein).mean()),
            "all_observations_pass": bool(np.all(values <= args.target_wasserstein)),
            "median_target_ratio": float(np.median(values / args.target_wasserstein)),
            "max_target_ratio": float(values.max() / args.target_wasserstein),
        }

    summary = {
        "device": str(device),
        "families": args.families,
        "num_observations": args.num_observations,
        "posterior_samples": args.posterior_samples,
        "grid_size": args.grid_size,
        "target_wasserstein": args.target_wasserstein,
        "aggregate": aggregate,
        "observations": output_observations,
    }
    summary_json = args.output_dir / "npe_multi_x_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = args.figure_dir / "npe_multi_x_wasserstein.png"
    plot_multi_x_summary(summary, figure_path)
    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")
    print("aggregate mean normalized Wasserstein:")
    for family in args.families:
        values = aggregate[family]
        print(
            f"  {family}: median={values['median']:.4f}, "
            f"mean={values['mean']:.4f}, max={values['max']:.4f}, "
            f"target_passes={values['target_pass_count']}/{args.num_observations}, "
            f"max_target_ratio={values['max_target_ratio']:.1f}x"
        )


if __name__ == "__main__":
    main()
