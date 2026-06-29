from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import corner
import matplotlib
import numpy as np
import torch

from mcmc_decay_inference import (
    PARAMETER_NAMES,
    PRIOR_LOG_MEAN,
    PRIOR_LOG_STD,
    arviz_diagnostics,
    choose_device,
    convergence_flags,
    log_posterior_z_low_overhead,
    plot_fit,
    plot_traces,
    posterior_summary,
    simulate_decay_data,
    synchronize_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from corner_truth import overplot_true_values


@dataclass(frozen=True)
class HMCConfig:
    chains: int
    steps: int
    burn_in: int
    seed: int
    step_size: float
    leapfrog_steps: int
    requested_device: str


@dataclass(frozen=True)
class HMCRunPaths:
    summary_json: str
    samples_npz: str
    corner_png: str
    trace_png: str
    fit_png: str


def make_logp(
    *,
    t: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    t_device = t.to(device=device, dtype=dtype)
    y_device = y.to(device=device, dtype=dtype)
    prior_log_mean = PRIOR_LOG_MEAN.to(device=device, dtype=dtype)
    prior_log_std = PRIOR_LOG_STD.to(device=device, dtype=dtype)
    prior_inv_var = prior_log_std.square().reciprocal()
    prior_log_norm = -torch.log(prior_log_std) - 0.5 * math.log(2.0 * math.pi)
    t_row = t_device[None, :]
    y_row = y_device[None, :]

    def logp(z: torch.Tensor) -> torch.Tensor:
        return log_posterior_z_low_overhead(
            z,
            t_row,
            y_row,
            prior_log_mean,
            prior_inv_var,
            prior_log_norm,
        )

    return logp, prior_log_mean, prior_log_std


def value_and_grad(logp, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    value = logp(z_req)
    (grad,) = torch.autograd.grad(value.sum(), z_req)
    return value.detach(), grad.detach()


def run_hmc(
    *,
    t: torch.Tensor,
    y: torch.Tensor,
    config: HMCConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    torch.manual_seed(config.seed + 11)
    logp, prior_log_mean, prior_log_std = make_logp(
        t=t,
        y=y,
        device=device,
        dtype=dtype,
    )

    initial_jitter = torch.randn(config.chains, 3, device=device, dtype=dtype)
    z_current = prior_log_mean + initial_jitter * prior_log_std * 0.85
    logp_current, _ = value_and_grad(logp, z_current)

    z_samples = torch.empty(config.steps, config.chains, 3, device=device, dtype=dtype)
    accepted = torch.empty(config.steps, config.chains, device=device, dtype=torch.bool)
    energy_error = torch.empty(config.steps, config.chains, device=device, dtype=dtype)

    synchronize_device(device)
    start = time.perf_counter()
    for step in range(config.steps):
        momentum_current = torch.randn_like(z_current)
        current_kinetic = 0.5 * momentum_current.square().sum(dim=1)

        z_proposal = z_current.detach()
        momentum = momentum_current.detach()
        _, grad = value_and_grad(logp, z_proposal)
        momentum = momentum + 0.5 * config.step_size * grad

        logp_proposal = logp_current
        for leapfrog_index in range(config.leapfrog_steps):
            z_proposal = z_proposal + config.step_size * momentum
            logp_proposal, grad = value_and_grad(logp, z_proposal)
            if leapfrog_index != config.leapfrog_steps - 1:
                momentum = momentum + config.step_size * grad

        momentum = momentum + 0.5 * config.step_size * grad
        proposed_kinetic = 0.5 * momentum.square().sum(dim=1)
        current_hamiltonian = -logp_current + current_kinetic
        proposed_hamiltonian = -logp_proposal + proposed_kinetic
        step_energy_error = proposed_hamiltonian - current_hamiltonian
        log_accept_ratio = -step_energy_error
        log_accept_uniform = torch.log(torch.rand(config.chains, device=device, dtype=dtype))
        accept = torch.isfinite(log_accept_ratio) & (log_accept_uniform < log_accept_ratio)

        z_current = torch.where(accept[:, None], z_proposal, z_current)
        logp_current = torch.where(accept, logp_proposal, logp_current)
        z_samples[step] = z_current
        accepted[step] = accept
        energy_error[step] = step_energy_error

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start

    z_samples_np = z_samples.detach().cpu().numpy().transpose(1, 0, 2)
    theta_samples_np = np.exp(z_samples_np)
    accepted_np = accepted.detach().cpu().numpy().transpose(1, 0)
    energy_error_np = energy_error.detach().cpu().numpy().transpose(1, 0)
    return z_samples_np, theta_samples_np, accepted_np, energy_error_np, elapsed_seconds


def plot_hmc_corner(
    theta_samples: np.ndarray,
    burn_in: int,
    true_theta: np.ndarray,
    outfile: Path,
) -> None:
    posterior = theta_samples[:, burn_in:, :].reshape(-1, 3)
    figure = corner.corner(
        posterior,
        labels=[r"$A$", r"$k$", r"$\sigma$"],
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 11},
        color="#b85c38",
        hist_kwargs={"density": True},
    )
    overplot_true_values(figure, true_theta)
    figure.subplots_adjust(top=0.90)
    figure.suptitle("Hamiltonian Monte Carlo posterior samples", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


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


def write_outputs(
    *,
    config: HMCConfig,
    device: torch.device,
    dtype: torch.dtype,
    t: torch.Tensor,
    y: torch.Tensor,
    true_theta: torch.Tensor,
    z_samples: np.ndarray,
    theta_samples: np.ndarray,
    accepted: np.ndarray,
    energy_error: np.ndarray,
    elapsed_seconds: float,
    output_dir: Path,
    figure_dir: Path,
) -> tuple[dict[str, object], HMCRunPaths]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "hmc_decay_summary.json"
    samples_npz = output_dir / "hmc_decay_samples.npz"
    corner_png = figure_dir / "hmc_decay_corner.png"
    trace_png = figure_dir / "hmc_decay_trace.png"
    fit_png = figure_dir / "hmc_decay_fit.png"

    diagnostics = arviz_diagnostics(theta_samples, config.burn_in)
    flags = convergence_flags(diagnostics)
    posterior = posterior_summary(theta_samples, config.burn_in)
    acceptance_by_chain = accepted.mean(axis=1)
    abs_energy_error = np.abs(energy_error[:, config.burn_in:])
    divergent = ~np.isfinite(energy_error[:, config.burn_in:]) | (abs_energy_error > 100.0)

    np.savez_compressed(
        samples_npz,
        z_samples=z_samples,
        theta_samples=theta_samples,
        accepted=accepted,
        energy_error=energy_error,
        t=t.numpy(),
        y=y.numpy(),
        true_theta=true_theta.numpy(),
        burn_in=np.array(config.burn_in),
        parameter_names=np.array(PARAMETER_NAMES),
    )

    true_theta_np = true_theta.numpy()
    plot_hmc_corner(theta_samples, config.burn_in, true_theta_np, corner_png)
    plot_traces(theta_samples, config.burn_in, trace_png)
    plot_fit(
        t=t.numpy(),
        y=y.numpy(),
        theta_samples=theta_samples,
        burn_in=config.burn_in,
        true_theta=true_theta_np,
        outfile=fit_png,
    )

    paths = HMCRunPaths(
        summary_json=str(summary_json),
        samples_npz=str(samples_npz),
        corner_png=str(corner_png),
        trace_png=str(trace_png),
        fit_png=str(fit_png),
    )
    summary: dict[str, object] = {
        "model": "y_i = A * exp(-k * t_i) + Normal(0, sigma)",
        "sampler": "hmc",
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
        "energy_error": {
            "mean_abs_after_burn_in": float(np.nanmean(abs_energy_error)),
            "max_abs_after_burn_in": float(np.nanmax(abs_energy_error)),
            "divergence_threshold": 100.0,
            "divergence_count_after_burn_in": int(np.sum(divergent)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hamiltonian Monte Carlo inference for noisy exponential decay.",
    )
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--step-size", type=float, default=0.009)
    parser.add_argument("--leapfrog-steps", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.burn_in < 0 or args.burn_in >= args.steps:
        parser.error("--burn-in must be non-negative and smaller than --steps")
    if args.chains < 2:
        parser.error("--chains must be at least 2 for R-hat diagnostics")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if args.leapfrog_steps < 1:
        parser.error("--leapfrog-steps must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    config = HMCConfig(
        chains=args.chains,
        steps=args.steps,
        burn_in=args.burn_in,
        seed=args.seed,
        step_size=args.step_size,
        leapfrog_steps=args.leapfrog_steps,
        requested_device=args.device,
    )
    device, dtype = choose_device(args.device)
    t, y, true_theta = simulate_decay_data(seed=args.seed)

    z_samples, theta_samples, accepted, energy_error, elapsed_seconds = run_hmc(
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
        energy_error=energy_error,
        elapsed_seconds=elapsed_seconds,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )

    print(f"device: {summary['device']} ({summary['dtype']})")
    print(f"runtime_seconds: {summary['runtime_seconds']:.3f}")
    print(f"mean_acceptance_rate: {summary['acceptance_rate']['mean']:.3f}")
    print(
        "energy_error: "
        f"mean_abs={summary['energy_error']['mean_abs_after_burn_in']:.4f}, "
        f"max_abs={summary['energy_error']['max_abs_after_burn_in']:.4f}, "
        f"divergences={summary['energy_error']['divergence_count_after_burn_in']}"
    )
    print("diagnostics:")
    for name in PARAMETER_NAMES:
        values = summary["diagnostics"][name]
        print(
            f"  {name}: R-hat={values['rhat']:.4f}, "
            f"bulk_ESS={values['ess_bulk']:.1f}, tail_ESS={values['ess_tail']:.1f}, "
            f"bulk_ESS/sec={summary['ess_per_second'][name]['bulk_ess_per_second']:.1f}"
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
