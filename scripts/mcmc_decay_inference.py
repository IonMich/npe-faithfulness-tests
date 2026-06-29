from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arviz as az
import corner
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corner_truth import overplot_true_values


PARAMETER_NAMES = ("A", "k", "sigma")
TRUE_THETA = torch.tensor([5.0, 0.55, 0.35])
PRIOR_LOG_MEAN = torch.log(torch.tensor([4.0, 0.50, 0.40]))
PRIOR_LOG_STD = torch.tensor([0.80, 0.80, 0.80])
SAMPLER_VARIANTS = {"baseline", "pregenerated", "low-overhead"}


@dataclass(frozen=True)
class MCMCConfig:
    chains: int
    steps: int
    burn_in: int
    seed: int
    proposal_scale: tuple[float, float, float]
    requested_device: str
    sampler_variant: str = "low-overhead"


@dataclass(frozen=True)
class RunPaths:
    summary_json: str
    samples_npz: str
    corner_png: str
    trace_png: str
    fit_png: str


def choose_device(requested: str) -> tuple[torch.device, torch.dtype]:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # For this small sequential random-walk MCMC workload, CPU is faster than MPS.
        # Use --device mps explicitly when investigating the Metal backend.
        device = torch.device("cpu")

    # Apple's MPS backend does not support float64 broadly; use float32 there.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return device, dtype


def simulate_decay_data(
    *,
    seed: int,
    n_observations: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    theta = TRUE_THETA.to(dtype=torch.float64)
    amplitude, decay_rate, noise = theta
    mean = amplitude * torch.exp(-decay_rate * t)
    y = mean + torch.randn(n_observations, generator=generator, dtype=torch.float64) * noise
    return t, y, theta


def log_normal_logpdf(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    z = (value - mean) / std
    return -0.5 * z.square() - torch.log(std) - 0.5 * math.log(2.0 * math.pi)


def log_posterior_z(
    z: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    prior_log_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    theta = torch.exp(z)
    amplitude = theta[:, 0:1]
    decay_rate = theta[:, 1:2]
    noise = theta[:, 2:3]

    mean = amplitude * torch.exp(-decay_rate * t[None, :])
    residual = y[None, :] - mean
    log_likelihood = (
        -0.5 * (residual / noise).square()
        - torch.log(noise)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=1)

    # The prior is defined directly on z = log(theta).
    log_prior = log_normal_logpdf(z, prior_log_mean, prior_log_std).sum(dim=1)
    return log_likelihood + log_prior


def log_posterior_z_low_overhead(
    z: torch.Tensor,
    t_row: torch.Tensor,
    y_row: torch.Tensor,
    prior_log_mean: torch.Tensor,
    prior_inv_var: torch.Tensor,
    prior_log_norm: torch.Tensor,
) -> torch.Tensor:
    log_amplitude = z[:, 0:1]
    log_decay_rate = z[:, 1:2]
    log_noise = z[:, 2:3]

    decay_rate = torch.exp(log_decay_rate)
    mean = torch.exp(log_amplitude - decay_rate * t_row)
    residual = y_row - mean
    inv_noise_var = torch.exp(-2.0 * log_noise)
    log_likelihood = (
        -0.5 * residual.square() * inv_noise_var
        - log_noise
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=1)

    delta = z - prior_log_mean
    log_prior = (-0.5 * delta.square() * prior_inv_var + prior_log_norm).sum(dim=1)
    return log_likelihood + log_prior


def normalize_sampler_variant(variant: str) -> str:
    if variant == "optimized":
        return "low-overhead"
    return variant


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run_random_walk_metropolis(
    *,
    t: torch.Tensor,
    y: torch.Tensor,
    config: MCMCConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sampler_variant = normalize_sampler_variant(config.sampler_variant)
    if sampler_variant not in SAMPLER_VARIANTS:
        raise ValueError(
            "sampler_variant must be one of: baseline, pregenerated, low-overhead"
        )

    torch.manual_seed(config.seed + 1)
    t_device = t.to(device=device, dtype=dtype)
    y_device = y.to(device=device, dtype=dtype)
    prior_log_mean = PRIOR_LOG_MEAN.to(device=device, dtype=dtype)
    prior_log_std = PRIOR_LOG_STD.to(device=device, dtype=dtype)
    proposal_scale = torch.tensor(
        config.proposal_scale,
        device=device,
        dtype=dtype,
    )

    initial_jitter = torch.randn(config.chains, 3, device=device, dtype=dtype)
    z_current = prior_log_mean + initial_jitter * prior_log_std * 0.85
    use_low_overhead_logp = sampler_variant == "low-overhead"
    if use_low_overhead_logp:
        t_row = t_device[None, :]
        y_row = y_device[None, :]
        prior_inv_var = prior_log_std.square().reciprocal()
        prior_log_norm = -torch.log(prior_log_std) - 0.5 * math.log(2.0 * math.pi)

        def logp(z: torch.Tensor) -> torch.Tensor:
            return log_posterior_z_low_overhead(
                z,
                t_row,
                y_row,
                prior_log_mean,
                prior_inv_var,
                prior_log_norm,
            )

    else:

        def logp(z: torch.Tensor) -> torch.Tensor:
            return log_posterior_z(z, t_device, y_device, prior_log_mean, prior_log_std)

    logp_current = logp(z_current)

    z_samples = torch.empty(config.steps, config.chains, 3, device=device, dtype=dtype)
    accepted = torch.empty(config.steps, config.chains, device=device, dtype=torch.bool)

    synchronize_device(device)
    start = time.perf_counter()
    if sampler_variant in {"pregenerated", "low-overhead"}:
        proposal_noises = torch.randn(
            config.steps,
            config.chains,
            3,
            device=device,
            dtype=dtype,
        )
        log_accept_uniforms = torch.log(
            torch.rand(config.steps, config.chains, device=device, dtype=dtype)
        )
    else:
        proposal_noises = None
        log_accept_uniforms = None

    for step in range(config.steps):
        if proposal_noises is None:
            proposal_noise = torch.randn_like(z_current)
            log_accept_uniform = torch.log(torch.rand(config.chains, device=device, dtype=dtype))
        else:
            proposal_noise = proposal_noises[step]
            log_accept_uniform = log_accept_uniforms[step]

        proposal = z_current + proposal_noise * proposal_scale
        logp_proposal = logp(proposal)
        log_accept_ratio = logp_proposal - logp_current
        accept = log_accept_uniform < log_accept_ratio

        z_current = torch.where(accept[:, None], proposal, z_current)
        logp_current = torch.where(accept, logp_proposal, logp_current)
        z_samples[step] = z_current
        accepted[step] = accept

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start

    z_samples_np = z_samples.detach().cpu().numpy().transpose(1, 0, 2)
    theta_samples_np = np.exp(z_samples_np)
    accepted_np = accepted.detach().cpu().numpy().transpose(1, 0)
    return z_samples_np, theta_samples_np, accepted_np, elapsed_seconds


def arviz_diagnostics(theta_samples: np.ndarray, burn_in: int) -> dict[str, dict[str, float]]:
    posterior = theta_samples[:, burn_in:, :]
    diagnostics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        draws = posterior[:, :, index]
        diagnostics[name] = {
            "rhat": float(az.rhat(draws, chain_axis=0, draw_axis=1)),
            "ess_bulk": float(az.ess(draws, method="bulk", chain_axis=0, draw_axis=1)),
            "ess_tail": float(az.ess(draws, method="tail", prob=0.05, chain_axis=0, draw_axis=1)),
        }
    return diagnostics


def posterior_summary(theta_samples: np.ndarray, burn_in: int) -> dict[str, dict[str, float]]:
    posterior = theta_samples[:, burn_in:, :].reshape(-1, 3)
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        values = posterior[:, index]
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


def convergence_flags(diagnostics: dict[str, dict[str, float]]) -> dict[str, bool]:
    return {
        "all_rhat_below_1_01": all(v["rhat"] < 1.01 for v in diagnostics.values()),
        "all_bulk_ess_above_400": all(v["ess_bulk"] > 400 for v in diagnostics.values()),
        "all_tail_ess_above_400": all(v["ess_tail"] > 400 for v in diagnostics.values()),
    }


def ess_per_second(
    diagnostics: dict[str, dict[str, float]],
    runtime_seconds: float,
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "bulk_ess_per_second": values["ess_bulk"] / runtime_seconds,
            "tail_ess_per_second": values["ess_tail"] / runtime_seconds,
        }
        for name, values in diagnostics.items()
    }


def plot_corner(theta_samples: np.ndarray, burn_in: int, true_theta: np.ndarray, outfile: Path) -> None:
    posterior = theta_samples[:, burn_in:, :].reshape(-1, 3)
    figure = corner.corner(
        posterior,
        labels=[r"$A$", r"$k$", r"$\sigma$"],
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 11},
        color="#2f6fbb",
        hist_kwargs={"density": True},
    )
    overplot_true_values(figure, true_theta)
    figure.subplots_adjust(top=0.90)
    figure.suptitle("Random-walk Metropolis posterior samples", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_traces(theta_samples: np.ndarray, burn_in: int, outfile: Path) -> None:
    steps = np.arange(theta_samples.shape[1])
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, theta_samples.shape[0]))
    for parameter_index, name in enumerate(PARAMETER_NAMES):
        ax = axes[parameter_index]
        for chain_index, color in enumerate(colors):
            ax.plot(
                steps,
                theta_samples[chain_index, :, parameter_index],
                color=color,
                alpha=0.48,
                linewidth=0.7,
            )
        ax.axvline(burn_in, color="#bf4d5a", linewidth=1.2, linestyle="--", label="burn-in")
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    axes[0].legend(loc="upper right")
    figure.suptitle("Trace plots by chain", fontsize=15)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_fit(
    *,
    t: np.ndarray,
    y: np.ndarray,
    theta_samples: np.ndarray,
    burn_in: int,
    true_theta: np.ndarray,
    outfile: Path,
) -> None:
    posterior = theta_samples[:, burn_in:, :].reshape(-1, 3)
    rng = np.random.default_rng(123)
    sample_count = min(400, posterior.shape[0])
    subset = posterior[rng.choice(posterior.shape[0], size=sample_count, replace=False)]

    t_grid = np.linspace(float(t.min()), float(t.max()), 200)
    curves = subset[:, 0, None] * np.exp(-subset[:, 1, None] * t_grid[None, :])
    lower, median, upper = np.quantile(curves, [0.05, 0.50, 0.95], axis=0)
    true_curve = true_theta[0] * np.exp(-true_theta[1] * t_grid)

    figure, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t, y, color="#172033", s=28, label="observed data")
    ax.fill_between(t_grid, lower, upper, color="#2f6fbb", alpha=0.18, label="90% posterior band")
    ax.plot(t_grid, median, color="#2f6fbb", linewidth=2.2, label="posterior median mean")
    ax.plot(t_grid, true_curve, color="#bf4d5a", linewidth=1.8, linestyle="--", label="true mean")
    ax.set_xlabel("time t")
    ax.set_ylabel("observed y")
    ax.set_title("Posterior fit for noisy exponential decay")
    ax.grid(alpha=0.2)
    ax.legend()
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    *,
    config: MCMCConfig,
    device: torch.device,
    dtype: torch.dtype,
    t: torch.Tensor,
    y: torch.Tensor,
    true_theta: torch.Tensor,
    z_samples: np.ndarray,
    theta_samples: np.ndarray,
    accepted: np.ndarray,
    elapsed_seconds: float,
    output_dir: Path,
    figure_dir: Path,
) -> tuple[dict[str, object], RunPaths]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "mcmc_decay_summary.json"
    samples_npz = output_dir / "mcmc_decay_samples.npz"
    corner_png = figure_dir / "mcmc_decay_corner.png"
    trace_png = figure_dir / "mcmc_decay_trace.png"
    fit_png = figure_dir / "mcmc_decay_fit.png"

    diagnostics = arviz_diagnostics(theta_samples, config.burn_in)
    flags = convergence_flags(diagnostics)
    posterior = posterior_summary(theta_samples, config.burn_in)
    acceptance_by_chain = accepted.mean(axis=1)

    np.savez_compressed(
        samples_npz,
        z_samples=z_samples,
        theta_samples=theta_samples,
        accepted=accepted,
        t=t.numpy(),
        y=y.numpy(),
        true_theta=true_theta.numpy(),
        burn_in=np.array(config.burn_in),
        parameter_names=np.array(PARAMETER_NAMES),
    )

    true_theta_np = true_theta.numpy()
    plot_corner(theta_samples, config.burn_in, true_theta_np, corner_png)
    plot_traces(theta_samples, config.burn_in, trace_png)
    plot_fit(
        t=t.numpy(),
        y=y.numpy(),
        theta_samples=theta_samples,
        burn_in=config.burn_in,
        true_theta=true_theta_np,
        outfile=fit_png,
    )

    paths = RunPaths(
        summary_json=str(summary_json),
        samples_npz=str(samples_npz),
        corner_png=str(corner_png),
        trace_png=str(trace_png),
        fit_png=str(fit_png),
    )
    summary: dict[str, object] = {
        "model": "y_i = A * exp(-k * t_i) + Normal(0, sigma)",
        "config": asdict(config),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "runtime_seconds": elapsed_seconds,
        "draws_after_burn_in_per_chain": config.steps - config.burn_in,
        "total_draws_after_burn_in": config.chains * (config.steps - config.burn_in),
        "acceptance_rate": {
            "mean": float(accepted.mean()),
            "per_chain": [float(x) for x in acceptance_by_chain],
        },
        "true_theta": {
            name: float(true_theta_np[index])
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "prior_on_log_parameters": {
            name: {
                "mean": float(PRIOR_LOG_MEAN[index]),
                "std": float(PRIOR_LOG_STD[index]),
            }
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "posterior_summary": posterior,
        "diagnostics": diagnostics,
        "ess_per_second": ess_per_second(diagnostics, elapsed_seconds),
        "convergence_flags": flags,
        "outputs": asdict(paths),
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, paths


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",")]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("proposal scale must be three comma-separated floats")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("proposal scales must be positive")
    return pieces[0], pieces[1], pieces[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plain random-walk Metropolis inference for noisy exponential decay.",
    )
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24_000)
    parser.add_argument("--burn-in", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--sampler-variant",
        choices=["baseline", "pregenerated", "low-overhead", "optimized"],
        default="low-overhead",
        help=(
            "baseline draws random numbers inside the loop; pregenerated draws all random "
            "numbers up front; low-overhead also uses a lower-overhead log-posterior expression. "
            "optimized is accepted as a deprecated alias for low-overhead."
        ),
    )
    parser.add_argument(
        "--proposal-scale",
        type=parse_proposal_scale,
        default=(0.030, 0.030, 0.040),
        help="Comma-separated random-walk stds for log(A),log(k),log(sigma).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.burn_in < 0 or args.burn_in >= args.steps:
        parser.error("--burn-in must be non-negative and smaller than --steps")
    if args.chains < 2:
        parser.error("--chains must be at least 2 for R-hat diagnostics")
    return args


def main() -> None:
    args = parse_args()
    config = MCMCConfig(
        chains=args.chains,
        steps=args.steps,
        burn_in=args.burn_in,
        seed=args.seed,
        proposal_scale=args.proposal_scale,
        requested_device=args.device,
        sampler_variant=normalize_sampler_variant(args.sampler_variant),
    )
    device, dtype = choose_device(args.device)
    t, y, true_theta = simulate_decay_data(seed=args.seed)

    z_samples, theta_samples, accepted, elapsed_seconds = run_random_walk_metropolis(
        t=t,
        y=y,
        config=config,
        device=device,
        dtype=dtype,
    )
    summary, paths = write_outputs(
        config=config,
        device=device,
        dtype=dtype,
        t=t,
        y=y,
        true_theta=true_theta,
        z_samples=z_samples,
        theta_samples=theta_samples,
        accepted=accepted,
        elapsed_seconds=elapsed_seconds,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )

    print(f"device: {summary['device']} ({summary['dtype']})")
    print(f"runtime_seconds: {summary['runtime_seconds']:.3f}")
    print(f"mean_acceptance_rate: {summary['acceptance_rate']['mean']:.3f}")
    print("diagnostics:")
    for name in PARAMETER_NAMES:
        values = summary["diagnostics"][name]
        print(
            f"  {name}: R-hat={values['rhat']:.4f}, "
            f"bulk_ESS={values['ess_bulk']:.1f}, tail_ESS={values['ess_tail']:.1f}"
        )
    print("posterior medians with 68% intervals:")
    for name in PARAMETER_NAMES:
        values = summary["posterior_summary"][name]
        print(
            f"  {name}: {values['median']:.4f} "
            f"[{values['q16']:.4f}, {values['q84']:.4f}]"
        )
    print(f"summary_json: {paths.summary_json}")
    print(f"samples_npz: {paths.samples_npz}")
    print(f"corner_png: {paths.corner_png}")
    print(f"trace_png: {paths.trace_png}")
    print(f"fit_png: {paths.fit_png}")


if __name__ == "__main__":
    main()
