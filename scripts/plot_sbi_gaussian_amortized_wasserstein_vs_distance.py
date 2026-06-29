from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from scipy.special import ndtr, ndtri
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/09_sbi_docs/01_getting_started_wasserstein_vs_distance")


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


def parse_float_edges(value: str) -> list[float]:
    edges = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("At least two comma-separated bin edges are required.")
    if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise argparse.ArgumentTypeError("Bin edges must be strictly increasing.")
    if edges[0] < 0.0:
        raise argparse.ArgumentTypeError("Bin edges must be non-negative.")
    return edges


def simulator(theta: torch.Tensor, noise_scale: float) -> torch.Tensor:
    return theta + 1.0 + torch.randn_like(theta) * noise_scale


def sample_truncated_normal_reference(
    *,
    x: np.ndarray,
    n: int,
    noise_scale: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # p(theta | x) factorizes into Normal(x - 1, noise_scale^2), truncated by the uniform prior.
    loc = x - 1.0
    a = (low - loc) / noise_scale
    b = (high - loc) / noise_scale
    cdf_low = ndtr(a)
    cdf_high = ndtr(b)
    u = rng.uniform(cdf_low[None, :], cdf_high[None, :], size=(n, x.shape[0]))
    return loc[None, :] + noise_scale * ndtri(u)


def reference_sd(
    *,
    x: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    sd = samples.std(axis=0, ddof=1)
    return np.maximum(sd, 1e-12)


def mean_normalized_wasserstein(samples: np.ndarray, reference: np.ndarray) -> tuple[float, list[float]]:
    sd = reference_sd(x=np.zeros(reference.shape[1]), samples=reference)
    values = [
        wasserstein_distance(samples[:, index], reference[:, index]) / sd[index]
        for index in range(reference.shape[1])
    ]
    return float(np.mean(values)), [float(value) for value in values]


def sample_observations_by_distance(
    *,
    edges: list[float],
    observations_per_bin: int,
    num_dim: int,
    center: np.ndarray,
    half_width: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        for _ in range(observations_per_bin):
            direction = rng.normal(0.0, 1.0, size=num_dim)
            direction /= np.max(np.abs(direction))
            distance_over_half_width = rng.uniform(lower, upper)
            x = center + half_width * distance_over_half_width * direction
            observations.append({
                "bin_index": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "distance_over_half_width": float(distance_over_half_width),
                "x": x.astype(np.float32),
            })
    return observations


def posterior_sample(
    posterior: object,
    x: np.ndarray,
    n: int,
    *,
    max_sampling_time: float | None,
) -> np.ndarray:
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    kwargs = {"show_progress_bars": False}
    if max_sampling_time is not None:
        kwargs["max_sampling_time"] = max_sampling_time
        kwargs["return_partial_on_timeout"] = True
    try:
        samples = posterior.sample((n,), x=x_tensor, **kwargs)
    except TypeError:
        samples = posterior.sample((n,), x=x_tensor)
    samples_np = samples.detach().cpu().numpy()
    finite = np.isfinite(samples_np).all(axis=1)
    return samples_np[finite]


def summarize_by_bin(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for bin_index in sorted({int(row["bin_index"]) for row in rows}):
        group = [row for row in rows if int(row["bin_index"]) == bin_index]
        w = np.asarray([row["mean_normalized_wasserstein"] for row in group], dtype=np.float64)
        baseline_w = np.asarray(
            [row["baseline_mean_normalized_wasserstein"] for row in group],
            dtype=np.float64,
        )
        d = np.asarray([row["distance_over_half_width"] for row in group], dtype=np.float64)
        output.append({
            "bin_index": bin_index,
            "bin_lower": float(group[0]["bin_lower"]),
            "bin_upper": float(group[0]["bin_upper"]),
            "n": len(group),
            "distance_over_half_width_mean": float(d.mean()),
            "w_mean": float(w.mean()),
            "w_sd": float(w.std(ddof=1)) if len(w) > 1 else 0.0,
            "w_q25": float(np.quantile(w, 0.25)),
            "w_median": float(np.median(w)),
            "w_q75": float(np.quantile(w, 0.75)),
            "w_min": float(w.min()),
            "w_max": float(w.max()),
            "baseline_w_mean": float(baseline_w.mean()),
            "baseline_w_q25": float(np.quantile(baseline_w, 0.25)),
            "baseline_w_median": float(np.median(baseline_w)),
            "baseline_w_q75": float(np.quantile(baseline_w, 0.75)),
        })
    return output


def plot_distance_sweep(rows: list[dict[str, object]], bin_summary: list[dict[str, object]], outfile: Path) -> None:
    distance = np.asarray([row["distance_over_half_width"] for row in rows], dtype=np.float64)
    wasserstein = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    baseline_wasserstein = np.asarray(
        [row["baseline_mean_normalized_wasserstein"] for row in rows],
        dtype=np.float64,
    )
    bin_x = np.asarray([item["distance_over_half_width_mean"] for item in bin_summary], dtype=np.float64)
    bin_median = np.asarray([item["w_median"] for item in bin_summary], dtype=np.float64)
    bin_q25 = np.asarray([item["w_q25"] for item in bin_summary], dtype=np.float64)
    bin_q75 = np.asarray([item["w_q75"] for item in bin_summary], dtype=np.float64)
    baseline_bin_median = np.asarray([item["baseline_w_median"] for item in bin_summary], dtype=np.float64)

    figure, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.axvspan(0.0, 1.0, color="#2f6fbb", alpha=0.08, label="noiseless prior-predictive cube")
    ax.axvline(1.0, color="#172033", linestyle="--", linewidth=1.5, label="noiseless cube boundary")
    ax.scatter(distance, wasserstein, s=36, color="#2f6fbb", alpha=0.72, label="test observations")
    ax.scatter(
        distance,
        baseline_wasserstein,
        s=24,
        color="#6b7280",
        alpha=0.5,
        label="exact posterior MC baseline",
    )
    ax.plot(bin_x, bin_median, color="#b85c38", linewidth=2.1, marker="o", label="bin median")
    ax.fill_between(bin_x, bin_q25, bin_q75, color="#b85c38", alpha=0.18, label="bin IQR")
    ax.plot(
        bin_x,
        baseline_bin_median,
        color="#4b5563",
        linewidth=1.7,
        marker="s",
        linestyle=":",
        label="baseline bin median",
    )
    if np.nanmax(wasserstein) / max(np.nanmin(wasserstein), 1e-12) > 25:
        ax.set_yscale("log")
    ax.set_xlabel("max-coordinate distance from prior-predictive center / prior half-width")
    ax.set_ylabel("SBI NPE to analytic posterior mean normalized Wasserstein")
    ax.set_title("SBI getting-started NPE degradation with observation distance")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "index",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "distance_over_half_width",
        "x0",
        "x1",
        "x2",
        "mean_normalized_wasserstein",
        "w_theta0",
        "w_theta1",
        "w_theta2",
        "baseline_mean_normalized_wasserstein",
        "baseline_w_theta0",
        "baseline_w_theta1",
        "baseline_w_theta2",
        "posterior_sampling_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            x = np.asarray(row["x"], dtype=np.float64)
            per_dim = row["wasserstein_per_dim"]
            baseline_per_dim = row["baseline_wasserstein_per_dim"]
            writer.writerow({
                "index": row["index"],
                "bin_index": row["bin_index"],
                "bin_lower": row["bin_lower"],
                "bin_upper": row["bin_upper"],
                "distance_over_half_width": row["distance_over_half_width"],
                "x0": x[0],
                "x1": x[1],
                "x2": x[2],
                "mean_normalized_wasserstein": row["mean_normalized_wasserstein"],
                "w_theta0": per_dim[0],
                "w_theta1": per_dim[1],
                "w_theta2": per_dim[2],
                "baseline_mean_normalized_wasserstein": row["baseline_mean_normalized_wasserstein"],
                "baseline_w_theta0": baseline_per_dim[0],
                "baseline_w_theta1": baseline_per_dim[1],
                "baseline_w_theta2": baseline_per_dim[2],
                "posterior_sampling_seconds": row["posterior_sampling_seconds"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distance sweep for the SBI getting-started amortized Gaussian example.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-simulations", type=int, default=2_000)
    parser.add_argument("--num-dim", type=int, default=3)
    parser.add_argument("--prior-low", type=float, default=-2.0)
    parser.add_argument("--prior-high", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--bins", type=parse_float_edges, default=parse_float_edges("0,0.25,0.5,0.75,1,1.125,1.25"))
    parser.add_argument("--observations-per-bin", type=int, default=5)
    parser.add_argument("--posterior-samples", type=int, default=5_000)
    parser.add_argument("--reference-samples", type=int, default=20_000)
    parser.add_argument("--max-sampling-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)

    low = args.prior_low
    high = args.prior_high
    half_width = 0.5 * (high - low)
    prior = BoxUniform(
        low=low * torch.ones(args.num_dim),
        high=high * torch.ones(args.num_dim),
    )

    train_start = time.perf_counter()
    theta = prior.sample((args.num_simulations,))
    x = simulator(theta, args.noise_scale)
    inference = NPE(prior=prior)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    train_seconds = time.perf_counter() - train_start

    center = np.ones(args.num_dim, dtype=np.float64)
    observations = sample_observations_by_distance(
        edges=args.bins,
        observations_per_bin=args.observations_per_bin,
        num_dim=args.num_dim,
        center=center,
        half_width=half_width,
        rng=rng,
    )

    rows = []
    for index, observation in enumerate(observations):
        x_obs = np.asarray(observation["x"], dtype=np.float32)
        print(
            f"evaluating {index + 1}/{len(observations)} "
            f"d={observation['distance_over_half_width']:.3f}",
            flush=True,
        )
        sample_start = time.perf_counter()
        samples = posterior_sample(
            posterior,
            x_obs,
            args.posterior_samples,
            max_sampling_time=args.max_sampling_time,
        )
        if samples.shape[0] < 100:
            raise RuntimeError(
                f"Only {samples.shape[0]} finite posterior samples for observation {index}; "
                "reduce the distance range or switch to an MCMC posterior sampler."
            )
        sample_seconds = time.perf_counter() - sample_start
        reference = sample_truncated_normal_reference(
            x=x_obs.astype(np.float64),
            n=args.reference_samples,
            noise_scale=args.noise_scale,
            low=low,
            high=high,
            rng=rng,
        )
        exact_sampler_like_mcmc = sample_truncated_normal_reference(
            x=x_obs.astype(np.float64),
            n=args.posterior_samples,
            noise_scale=args.noise_scale,
            low=low,
            high=high,
            rng=rng,
        )
        mean_w, per_dim = mean_normalized_wasserstein(samples, reference)
        baseline_mean_w, baseline_per_dim = mean_normalized_wasserstein(exact_sampler_like_mcmc, reference)
        rows.append({
            "index": index,
            **json_ready(observation),
            "mean_normalized_wasserstein": mean_w,
            "wasserstein_per_dim": per_dim,
            "baseline_mean_normalized_wasserstein": baseline_mean_w,
            "baseline_wasserstein_per_dim": baseline_per_dim,
            "posterior_sampling_seconds": sample_seconds,
        })
        print(f"  W={mean_w:.4f}; exact-sampler baseline={baseline_mean_w:.4f}", flush=True)

    bin_summary = summarize_by_bin(rows)
    figure_path = figure_dir / "sbi_gaussian_amortized_wasserstein_vs_distance.png"
    json_path = results_dir / "sbi_gaussian_amortized_wasserstein_vs_distance_summary.json"
    csv_path = results_dir / "sbi_gaussian_amortized_wasserstein_vs_distance_observations.csv"
    plot_distance_sweep(rows, bin_summary, figure_path)
    write_csv(rows, csv_path)

    output = {
        "source_docs": {
            "name": "sbi getting-started tutorial",
            "url": "https://sbi.readthedocs.io/en/stable/tutorials/00_getting_started.html",
            "simulator": "x = theta + 1 + Normal(0, 0.1^2)",
            "prior": "BoxUniform([-2, -2, -2], [2, 2, 2])",
        },
        "config": json_ready(vars(args)),
        "training": {
            "num_simulations": args.num_simulations,
            "training_seconds": train_seconds,
        },
        "distance": {
            "center": center.tolist(),
            "normalizer": half_width,
            "definition": "max_j |x_j - center_j| / prior_half_width",
            "boundary": "1.0 is the noiseless prior-predictive cube boundary; Gaussian simulator noise makes x support technically unbounded",
        },
        "metric": {
            "npe_w": "mean over dimensions of 1D Wasserstein(NPE samples, analytic posterior samples), normalized by analytic posterior sample sd",
            "baseline_w": "same metric for independent analytic posterior samples with posterior_samples draws; approximates an ideal converged sampler with the same effective sample size",
        },
        "bin_summary": bin_summary,
        "observations": rows,
        "outputs": {
            "figure": str(figure_path),
            "summary_json": str(json_path),
            "observations_csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {json_path}")
    print(f"observations_csv: {csv_path}")
    print("bin medians:")
    for item in bin_summary:
        print(
            f"  [{item['bin_lower']:g}, {item['bin_upper']:g}): "
            f"median W={item['w_median']:.4f}, "
            f"baseline={item['baseline_w_median']:.4f}, "
            f"IQR=[{item['w_q25']:.4f}, {item['w_q75']:.4f}], n={item['n']}"
        )


if __name__ == "__main__":
    main()
