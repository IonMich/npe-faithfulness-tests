from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import artifact_paths as ap

import corner
import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import ks_2samp, wasserstein_distance

from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corner_truth import overplot_true_values, true_theta_legend_handle


COLORS = {
    "MCMC": "#2f6fbb",
    "HMC": "#b85c38",
}


def load_samples(path: Path, label: str) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    burn_in = int(np.asarray(data["burn_in"]).item())
    theta_samples = np.asarray(data["theta_samples"])
    z_samples = np.asarray(data["z_samples"])
    return {
        "label": label,
        "path": str(path),
        "burn_in": burn_in,
        "theta_samples": theta_samples,
        "z_samples": z_samples,
        "posterior_theta": theta_samples[:, burn_in:, :].reshape(-1, 3),
        "posterior_z": z_samples[:, burn_in:, :].reshape(-1, 3),
        "t": np.asarray(data["t"]),
        "y": np.asarray(data["y"]),
        "true_theta": np.asarray(data["true_theta"]),
    }


def subsample(values: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if values.shape[0] <= max_samples:
        return values
    rng = np.random.default_rng(seed)
    index = rng.choice(values.shape[0], size=max_samples, replace=False)
    return values[index]


def summarize_samples(samples: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        values = samples[:, index]
        q05, q16, q50, q84, q95 = np.quantile(values, [0.05, 0.16, 0.50, 0.84, 0.95])
        summary[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "q05": float(q05),
            "q16": float(q16),
            "median": float(q50),
            "q84": float(q84),
            "q95": float(q95),
        }
    return summary


def plot_overlay_corner(
    mcmc_samples: np.ndarray,
    hmc_samples: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
    max_plot_samples: int,
) -> None:
    mcmc_plot = subsample(mcmc_samples, max_plot_samples, seed=101)
    hmc_plot = subsample(hmc_samples, max_plot_samples, seed=202)
    labels = [r"$A$", r"$k$", r"$\sigma$"]

    figure = corner.corner(
        mcmc_plot,
        labels=labels,
        color=COLORS["MCMC"],
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.7},
        contour_kwargs={"linewidths": 1.5},
    )
    corner.corner(
        hmc_plot,
        fig=figure,
        labels=labels,
        color=COLORS["HMC"],
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.7},
        contour_kwargs={"linewidths": 1.5},
    )
    handles = [
        plt.Line2D([0], [0], color=COLORS["MCMC"], lw=2, label="Random-walk MCMC"),
        plt.Line2D([0], [0], color=COLORS["HMC"], lw=2, label="HMC"),
        true_theta_legend_handle(),
    ]
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    figure.subplots_adjust(top=0.90)
    figure.suptitle("Posterior overlay: random-walk MCMC vs HMC", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def posterior_predictive_draws(
    samples: np.ndarray,
    t_grid: np.ndarray,
    seed: int,
    max_draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = subsample(samples, max_draws, seed=seed)
    mean_curves = selected[:, 0, None] * np.exp(-selected[:, 1, None] * t_grid[None, :])
    rng = np.random.default_rng(seed + 1)
    predictive = mean_curves + rng.normal(
        loc=0.0,
        scale=selected[:, 2, None],
        size=mean_curves.shape,
    )
    return mean_curves, predictive


def plot_overlay_predictive(
    *,
    t: np.ndarray,
    y: np.ndarray,
    true_theta: np.ndarray,
    mcmc_samples: np.ndarray,
    hmc_samples: np.ndarray,
    outfile: Path,
) -> None:
    t_grid = np.linspace(float(t.min()), float(t.max()), 220)
    figure, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(t, y, color="#172033", s=28, zorder=5, label="observed data")

    true_mean = true_theta[0] * np.exp(-true_theta[1] * t_grid)
    ax.plot(t_grid, true_mean, color="#172033", lw=1.7, linestyle="--", label="true mean")

    for label, samples, seed in [
        ("MCMC", mcmc_samples, 303),
        ("HMC", hmc_samples, 404),
    ]:
        mean_curves, predictive = posterior_predictive_draws(
            samples,
            t_grid,
            seed=seed,
            max_draws=700,
        )
        pred_lower, pred_median, pred_upper = np.quantile(predictive, [0.05, 0.50, 0.95], axis=0)
        mean_subset = subsample(mean_curves, 70, seed=seed + 10)
        color = COLORS[label]
        for curve in mean_subset:
            ax.plot(t_grid, curve, color=color, alpha=0.035, lw=0.9)
        ax.fill_between(
            t_grid,
            pred_lower,
            pred_upper,
            color=color,
            alpha=0.13,
            label=f"{label} 90% posterior predictive",
        )
        ax.plot(t_grid, pred_median, color=color, lw=2.0, label=f"{label} predictive median")

    ax.set_title("Posterior predictive overlay with posterior mean samples")
    ax.set_xlabel("time t")
    ax.set_ylabel("replicated observation y")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def log_posterior_z_numpy(z: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    log_amplitude = z[:, 0:1]
    log_decay_rate = z[:, 1:2]
    log_noise = z[:, 2:3]
    mean = np.exp(log_amplitude - np.exp(log_decay_rate) * t[None, :])
    residual = y[None, :] - mean
    inv_noise_var = np.exp(-2.0 * log_noise)
    log_likelihood = (
        -0.5 * residual**2 * inv_noise_var
        - log_noise
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=1)

    prior_mean = PRIOR_LOG_MEAN.numpy()
    prior_std = PRIOR_LOG_STD.numpy()
    delta = z - prior_mean[None, :]
    log_prior = (
        -0.5 * (delta / prior_std[None, :]) ** 2
        - np.log(prior_std[None, :])
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=1)
    return log_likelihood + log_prior


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, sorted_values)


def build_grid_reference(
    *,
    t: np.ndarray,
    y: np.ndarray,
    combined_z_samples: np.ndarray,
    true_theta: np.ndarray,
    grid_size: int,
    chunk_size: int,
) -> dict[str, object]:
    true_z = np.log(true_theta)
    axes = []
    ranges = []
    for index in range(3):
        low, high = np.quantile(combined_z_samples[:, index], [0.0005, 0.9995])
        width = high - low
        low -= max(0.30 * width, 0.08)
        high += max(0.30 * width, 0.08)
        low = min(low, true_z[index] - 0.08)
        high = max(high, true_z[index] + 0.08)
        axes.append(np.linspace(low, high, grid_size))
        ranges.append([float(low), float(high)])

    mesh = np.meshgrid(*axes, indexing="ij")
    z_grid = np.column_stack([axis.reshape(-1) for axis in mesh])
    logp = np.empty(z_grid.shape[0], dtype=np.float64)
    for start in range(0, z_grid.shape[0], chunk_size):
        stop = min(start + chunk_size, z_grid.shape[0])
        logp[start:stop] = log_posterior_z_numpy(z_grid[start:stop], t=t, y=y)

    log_norm = logsumexp(logp)
    weights = np.exp(logp - log_norm)
    theta_grid = np.exp(z_grid)

    summary: dict[str, dict[str, float]] = {}
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

    edge_mass = {}
    shape = (grid_size, grid_size, grid_size)
    weight_cube = weights.reshape(shape)
    for index, name in enumerate(PARAMETER_NAMES):
        lower = np.take(weight_cube, indices=0, axis=index).sum()
        upper = np.take(weight_cube, indices=grid_size - 1, axis=index).sum()
        edge_mass[name] = {"lower": float(lower), "upper": float(upper)}

    return {
        "grid_size": grid_size,
        "grid_points": int(z_grid.shape[0]),
        "z_ranges": {
            name: ranges[index]
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "edge_mass": edge_mass,
        "theta_grid": theta_grid,
        "weights": weights,
        "summary": summary,
    }


def compare_to_reference(
    samples: np.ndarray,
    reference: dict[str, object],
) -> dict[str, dict[str, float]]:
    theta_grid = reference["theta_grid"]
    weights = reference["weights"]
    ref_summary = reference["summary"]
    sample_summary = summarize_samples(samples)
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        sample_values = samples[:, index]
        ref_values = theta_grid[:, index]
        ref_sd = max(ref_summary[name]["sd"], 1e-12)
        wasserstein = wasserstein_distance(
            sample_values,
            ref_values,
            v_weights=weights,
        )
        metrics[name] = {
            "wasserstein_to_grid": float(wasserstein),
            "wasserstein_to_grid_in_ref_sd": float(wasserstein / ref_sd),
            "median_error": float(sample_summary[name]["median"] - ref_summary[name]["median"]),
            "q05_error": float(sample_summary[name]["q05"] - ref_summary[name]["q05"]),
            "q95_error": float(sample_summary[name]["q95"] - ref_summary[name]["q95"]),
        }
    metrics["mean_normalized_wasserstein"] = {
        "value": float(
            np.mean([
                metrics[name]["wasserstein_to_grid_in_ref_sd"]
                for name in PARAMETER_NAMES
            ])
        )
    }
    return metrics


def pairwise_sample_metrics(mcmc_samples: np.ndarray, hmc_samples: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        metrics[name] = {
            "ks_statistic": float(ks_2samp(mcmc_samples[:, index], hmc_samples[:, index]).statistic),
            "wasserstein": float(wasserstein_distance(mcmc_samples[:, index], hmc_samples[:, index])),
            "median_difference_hmc_minus_mcmc": float(
                np.median(hmc_samples[:, index]) - np.median(mcmc_samples[:, index])
            ),
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random-walk MCMC and HMC posteriors for the decay model.",
    )
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--max-corner-samples", type=int, default=30_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    t = mcmc["t"]
    y = mcmc["y"]
    true_theta = mcmc["true_theta"]

    overlay_corner = args.figure_dir / "sampler_overlay_corner.png"
    overlay_predictive = args.figure_dir / "sampler_overlay_predictive.png"
    summary_json = args.output_dir / "sampler_comparison_summary.json"

    plot_start = time.perf_counter()
    plot_overlay_corner(
        mcmc["posterior_theta"],
        hmc["posterior_theta"],
        true_theta,
        overlay_corner,
        args.max_corner_samples,
    )
    plot_overlay_predictive(
        t=t,
        y=y,
        true_theta=true_theta,
        mcmc_samples=mcmc["posterior_theta"],
        hmc_samples=hmc["posterior_theta"],
        outfile=overlay_predictive,
    )
    plotting_seconds = time.perf_counter() - plot_start

    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    grid_start = time.perf_counter()
    reference = build_grid_reference(
        t=t,
        y=y,
        combined_z_samples=combined_z,
        true_theta=true_theta,
        grid_size=args.reference_grid_size,
        chunk_size=args.reference_chunk_size,
    )
    grid_seconds = time.perf_counter() - grid_start

    metric_start = time.perf_counter()
    mcmc_vs_hmc = pairwise_sample_metrics(mcmc["posterior_theta"], hmc["posterior_theta"])
    mcmc_grid_metrics = compare_to_reference(mcmc["posterior_theta"], reference)
    hmc_grid_metrics = compare_to_reference(hmc["posterior_theta"], reference)
    metric_seconds = time.perf_counter() - metric_start

    output = {
        "inputs": {
            "mcmc_samples": str(args.mcmc_samples),
            "hmc_samples": str(args.hmc_samples),
        },
        "outputs": {
            "overlay_corner": str(overlay_corner),
            "overlay_predictive": str(overlay_predictive),
            "summary_json": str(summary_json),
        },
        "timing_seconds": {
            "plotting": plotting_seconds,
            "grid_reference": grid_seconds,
            "grid_points_per_second": reference["grid_points"] / grid_seconds,
            "metrics": metric_seconds,
            "total": time.perf_counter() - total_start,
        },
        "posterior_summary": {
            "mcmc": summarize_samples(mcmc["posterior_theta"]),
            "hmc": summarize_samples(hmc["posterior_theta"]),
            "grid_reference": reference["summary"],
        },
        "mcmc_vs_hmc": mcmc_vs_hmc,
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "z_ranges": reference["z_ranges"],
            "edge_mass": reference["edge_mass"],
        },
        "faithfulness_to_grid_reference": {
            "mcmc": mcmc_grid_metrics,
            "hmc": hmc_grid_metrics,
        },
    }
    summary_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"overlay_corner: {overlay_corner}")
    print(f"overlay_predictive: {overlay_predictive}")
    print(f"summary_json: {summary_json}")
    print(f"grid_reference_seconds: {grid_seconds:.3f}")
    print(f"grid_points_per_second: {reference['grid_points'] / grid_seconds:.0f}")
    print("mean normalized Wasserstein to grid reference:")
    print(
        "  MCMC: "
        f"{output['faithfulness_to_grid_reference']['mcmc']['mean_normalized_wasserstein']['value']:.5f}"
    )
    print(
        "  HMC:  "
        f"{output['faithfulness_to_grid_reference']['hmc']['mean_normalized_wasserstein']['value']:.5f}"
    )


if __name__ == "__main__":
    main()
