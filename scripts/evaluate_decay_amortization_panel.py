from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

import artifact_paths as ap
from abc_faithfulness_decay import make_k_grid
from compare_decay_samplers import (
    build_grid_reference,
    compare_to_reference,
    load_samples,
    log_posterior_z_numpy,
    summarize_samples,
    weighted_quantile,
)
from hmc_decay_inference import HMCConfig, run_hmc
from mcmc_decay_inference import (
    MCMCConfig,
    PARAMETER_NAMES,
    PRIOR_LOG_MEAN,
    PRIOR_LOG_STD,
    arviz_diagnostics,
    choose_device as choose_sampler_device,
    convergence_flags,
    run_random_walk_metropolis,
    simulate_decay_data,
)
from npe_flow_decay import (
    ConditionalSplineFlow,
    context_distances,
    fit_local_region,
    make_context_summaries,
    sample_flow_posterior,
    sample_prior_z,
)
from npe_stage1_decay import (
    FAMILIES,
    FAMILY_COLORS,
    FAMILY_LABELS,
    Stage1Config,
    choose_training_device,
    make_model,
    sample_grid_reference,
    sample_posterior_for_observation,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("proposal scale must contain three comma-separated floats")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("proposal scales must be positive")
    return pieces[0], pieces[1], pieces[2]


def parse_float_triple(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("value must contain three comma-separated floats")
    return pieces[0], pieces[1], pieces[2]


def parse_path_list(value: str) -> list[Path]:
    paths = [Path(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("value must contain at least one path")
    return paths


def parse_int_list(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("value must contain at least one integer index")
    indices = [int(piece) for piece in pieces]
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("observation indices must be non-negative")
    return indices


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def simulate_x_from_z(z: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.exp(z)
    mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
    return mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)


def sample_prior_predictive_observations(
    *,
    n: int,
    seed: int,
    n_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, n_observations)
    z = sample_prior_z(n, rng)
    x = simulate_x_from_z(z, t, rng)
    return t, x, z, None, {
        "kind": "prior_predictive",
        "seed": seed,
        "n_observations": n_observations,
    }


def sample_x0_observation(
    *,
    observed_seed: int,
    n_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    t_obs, y_obs, true_theta = simulate_decay_data(
        seed=observed_seed,
        n_observations=n_observations,
    )
    t = t_obs.numpy()
    x = y_obs.numpy()[None, :]
    z = np.log(true_theta.numpy())[None, :]
    return t, x, z, np.array([0.0]), {
        "kind": "x0",
        "observed_seed": observed_seed,
        "n_observations": n_observations,
    }


def sample_local_x_observations(
    *,
    n: int,
    seed: int,
    observed_seed: int,
    n_observations: int,
    local_quantile: float,
    local_pilot_simulations: int,
    local_max_candidates: int,
    simulate_chunk_size: int,
    summary_chunk_size: int,
    context_kind: str,
    k_grid: np.ndarray,
    region_override: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    t_obs, y_obs, _ = simulate_decay_data(seed=observed_seed, n_observations=n_observations)
    t = t_obs.numpy()
    observed_x = y_obs.numpy()
    observed_context = make_context_summaries(
        observed_x[None, :],
        t,
        k_grid,
        kind=context_kind,
        chunk_size=1,
    )[0]
    if region_override is None:
        region = fit_local_region(
            observed_context=observed_context,
            t=t,
            k_grid=k_grid,
            simulations=local_pilot_simulations,
            quantile=local_quantile,
            kernel_quantile=0.0,
            rng=rng,
            simulate_chunk_size=simulate_chunk_size,
            summary_chunk_size=summary_chunk_size,
            context_kind=context_kind,
        )
        region_source = "pilot"
    else:
        region = region_override
        region_source = "summary"

    accepted_z: list[np.ndarray] = []
    accepted_x: list[np.ndarray] = []
    accepted_distances: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    center = np.asarray(region["center"])
    scale = np.asarray(region["scale"])
    radius = float(region["radius"])
    while accepted_count < n and candidate_count < local_max_candidates:
        current = min(simulate_chunk_size, local_max_candidates - candidate_count)
        z = sample_prior_z(current, rng)
        x = simulate_x_from_z(z, t, rng)
        context = make_context_summaries(
            x,
            t,
            k_grid,
            kind=context_kind,
            chunk_size=summary_chunk_size,
        )
        distances = context_distances(context, observed_context, center, scale)
        mask = distances <= radius
        if np.any(mask):
            accepted_z.append(z[mask])
            accepted_x.append(x[mask])
            accepted_distances.append(distances[mask])
            accepted_count += int(mask.sum())
        candidate_count += current

    if accepted_count < n:
        raise RuntimeError(
            f"Only accepted {accepted_count} local observations after {candidate_count} candidates; "
            f"need {n}. Increase --local-max-candidates or --local-quantile."
        )

    z_all = np.concatenate(accepted_z, axis=0)[:n]
    x_all = np.concatenate(accepted_x, axis=0)[:n]
    distances_all = np.concatenate(accepted_distances, axis=0)[:n]
    return t, x_all, z_all, distances_all, {
        "kind": "local_x",
        "seed": seed,
        "observed_seed": observed_seed,
        "region_source": region_source,
        "local_quantile": local_quantile,
        "local_pilot_simulations": local_pilot_simulations,
        "local_max_candidates": local_max_candidates,
        "candidate_count": int(candidate_count),
        "raw_accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / max(candidate_count, 1)),
        "context_kind": context_kind,
        "radius": radius,
        "pilot_distance_summary": region.get("pilot_distance_summary"),
        "accepted_distance_summary": {
            "min": float(distances_all.min()),
            "median": float(np.median(distances_all)),
            "max": float(distances_all.max()),
        },
    }


def prior_mahalanobis_distance(z: np.ndarray, center_z: np.ndarray) -> np.ndarray:
    scale = PRIOR_LOG_STD.numpy()
    return np.sqrt(np.sum(((z - center_z[None, :]) / scale[None, :]) ** 2, axis=1))


def build_observation_distance_metrics(
    *,
    panel_kind: str,
    true_z: np.ndarray,
    panel_distance: float | None,
    panel_metadata: dict[str, object],
    parameter_radius: float,
) -> dict[str, float | None]:
    prior_z_radius = float(prior_mahalanobis_distance(true_z[None, :], PRIOR_LOG_MEAN.numpy())[0])
    metrics: dict[str, float | None] = {
        "prior_z_radius": prior_z_radius,
        "distance_to_x0": panel_distance,
        "local_x0_distance": None,
        "local_x0_distance_over_radius": None,
        "parameter_region_distance": None,
        "parameter_region_distance_over_radius": None,
    }
    if panel_kind == "local_x" and panel_distance is not None:
        radius = float(panel_metadata["radius"])
        metrics["local_x0_distance"] = float(panel_distance)
        metrics["local_x0_distance_over_radius"] = float(panel_distance / max(radius, 1e-12))
    if panel_kind == "parameter_region" and panel_distance is not None:
        metrics["parameter_region_distance"] = float(panel_distance)
        metrics["parameter_region_distance_over_radius"] = float(panel_distance / max(parameter_radius, 1e-12))
    if panel_kind == "x0":
        metrics["distance_to_x0"] = 0.0
    return metrics


def resolve_parameter_center(args: argparse.Namespace) -> np.ndarray:
    if args.parameter_center == "prior_mean":
        return PRIOR_LOG_MEAN.numpy()
    if args.parameter_center == "true_theta":
        _, _, true_theta = simulate_decay_data(
            seed=args.observed_seed,
            n_observations=args.n_observations_per_curve,
        )
        return np.log(true_theta.numpy())
    if args.parameter_center == "custom":
        if args.parameter_center_z is None:
            raise ValueError("--parameter-center-z is required when --parameter-center custom")
        return np.asarray(args.parameter_center_z, dtype=np.float64)
    raise ValueError(f"Unknown parameter center: {args.parameter_center}")


def sample_parameter_region_observations(
    *,
    n: int,
    seed: int,
    n_observations: int,
    center_z: np.ndarray,
    radius: float,
    max_candidates: int,
    simulate_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, n_observations)
    accepted_z: list[np.ndarray] = []
    accepted_distances: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    while accepted_count < n and candidate_count < max_candidates:
        current = min(simulate_chunk_size, max_candidates - candidate_count)
        z = sample_prior_z(current, rng)
        distances = prior_mahalanobis_distance(z, center_z)
        mask = distances <= radius
        if np.any(mask):
            accepted_z.append(z[mask])
            accepted_distances.append(distances[mask])
            accepted_count += int(mask.sum())
        candidate_count += current
    if accepted_count < n:
        raise RuntimeError(
            f"Only accepted {accepted_count} parameter-region observations after "
            f"{candidate_count} candidates; need {n}. Increase --parameter-max-candidates "
            "or --parameter-radius."
        )
    z_all = np.concatenate(accepted_z, axis=0)[:n]
    distances_all = np.concatenate(accepted_distances, axis=0)[:n]
    x_all = simulate_x_from_z(z_all, t, rng)
    return t, x_all, z_all, distances_all, {
        "kind": "parameter_region",
        "seed": seed,
        "center_z": center_z.tolist(),
        "radius": radius,
        "max_candidates": max_candidates,
        "candidate_count": int(candidate_count),
        "raw_accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / max(candidate_count, 1)),
        "distance_summary": {
            "min": float(distances_all.min()),
            "median": float(np.median(distances_all)),
            "max": float(distances_all.max()),
        },
    }


def initial_z_ranges(
    *,
    z_samples_by_model: dict[str, np.ndarray],
    true_z: np.ndarray,
    padding_fraction: float,
    min_padding: float,
) -> np.ndarray:
    combined = np.vstack(list(z_samples_by_model.values()) + [true_z[None, :]])
    ranges = []
    for dim in range(3):
        low, high = np.quantile(combined[:, dim], [0.0005, 0.9995])
        width = max(high - low, 1e-6)
        padding = max(padding_fraction * width, min_padding)
        low -= padding
        high += padding
        low = min(low, true_z[dim] - min_padding)
        high = max(high, true_z[dim] + min_padding)
        ranges.append((low, high))
    return np.asarray(ranges, dtype=np.float64)


def include_restricted_region_in_ranges(
    z_ranges: np.ndarray,
    restricted_region: dict[str, object] | None,
) -> np.ndarray:
    if restricted_region is None:
        return z_ranges
    center_z = np.asarray(restricted_region["center_z"], dtype=np.float64)
    radius = float(restricted_region["radius"])
    scale = PRIOR_LOG_STD.numpy()
    region_low = center_z - radius * scale
    region_high = center_z + radius * scale
    ranges = z_ranges.copy()
    ranges[:, 0] = np.minimum(ranges[:, 0], region_low)
    ranges[:, 1] = np.maximum(ranges[:, 1], region_high)
    return ranges


def build_grid_reference_from_ranges(
    *,
    t: np.ndarray,
    y: np.ndarray,
    z_ranges: np.ndarray,
    grid_size: int,
    chunk_size: int,
    restricted_region: dict[str, object] | None,
) -> dict[str, object]:
    axes = [np.linspace(low, high, grid_size) for low, high in z_ranges]
    mesh = np.meshgrid(*axes, indexing="ij")
    z_grid = np.column_stack([axis.reshape(-1) for axis in mesh])
    logp = np.empty(z_grid.shape[0], dtype=np.float64)
    for start in range(0, len(z_grid), chunk_size):
        stop = min(start + chunk_size, len(z_grid))
        logp[start:stop] = log_posterior_z_numpy(z_grid[start:stop], t=t, y=y)
    if restricted_region is not None:
        center_z = np.asarray(restricted_region["center_z"], dtype=np.float64)
        radius = float(restricted_region["radius"])
        distances = prior_mahalanobis_distance(z_grid, center_z)
        logp = np.where(distances <= radius, logp, -np.inf)
        if not np.any(np.isfinite(logp)):
            raise RuntimeError("Restricted-prior grid has no finite posterior points. Widen z_ranges or grid.")
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


def max_edge_mass(reference: dict[str, object]) -> float:
    return max(
        max(float(values["lower"]), float(values["upper"]))
        for values in reference["edge_mass"].values()
    )


def build_adaptive_grid_reference(
    *,
    t: np.ndarray,
    y: np.ndarray,
    z_ranges: np.ndarray,
    grid_size: int,
    chunk_size: int,
    edge_mass_tolerance: float,
    max_expansions: int,
    restricted_region: dict[str, object] | None,
) -> tuple[dict[str, object], int]:
    ranges = z_ranges.copy()
    reference = build_grid_reference_from_ranges(
        t=t,
        y=y,
        z_ranges=ranges,
        grid_size=grid_size,
        chunk_size=chunk_size,
        restricted_region=restricted_region,
    )
    expansions = 0
    while max_edge_mass(reference) > edge_mass_tolerance and expansions < max_expansions:
        center = ranges.mean(axis=1)
        half_width = (ranges[:, 1] - ranges[:, 0]) * 0.75
        ranges = np.column_stack([center - half_width, center + half_width])
        reference = build_grid_reference_from_ranges(
            t=t,
            y=y,
            z_ranges=ranges,
            grid_size=grid_size,
            chunk_size=chunk_size,
            restricted_region=restricted_region,
        )
        expansions += 1
    return reference, expansions


def build_x0_mcmc_hmc_grid_reference(
    *,
    mcmc_samples: Path,
    hmc_samples: Path,
    grid_size: int,
    chunk_size: int,
) -> dict[str, object]:
    mcmc = load_samples(mcmc_samples, "MCMC")
    hmc = load_samples(hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
    return build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=combined_z,
        true_theta=mcmc["true_theta"],
        grid_size=grid_size,
        chunk_size=chunk_size,
    )


def sample_to_sample_normalized_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    reference: dict[str, object],
) -> float:
    values = []
    for index, name in enumerate(PARAMETER_NAMES):
        ref_sd = max(float(reference["summary"][name]["sd"]), 1e-12)
        values.append(wasserstein_distance(left[:, index], right[:, index]) / ref_sd)
    return float(np.mean(values))


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
            pairwise.append(sample_to_sample_normalized_wasserstein(samples[i], samples[j], reference))
    return {
        "sample_count": sample_count,
        "replicates": replicates,
        "to_grid": {
            "values": to_grid,
            "median": float(np.median(to_grid)),
            "max": float(np.max(to_grid)),
        },
        "pairwise": {
            "values": pairwise,
            "median": float(np.median(pairwise)),
            "max": float(np.max(pairwise)),
        },
    }


def grid_tolerance(diagnostics: dict[str, object], statistic: str) -> float:
    if statistic not in {"median", "max"}:
        raise ValueError("statistic must be median or max")
    return float(max(
        diagnostics["to_grid"][statistic],
        diagnostics["pairwise"][statistic],
    ))


def chain_split_samples(theta_samples: np.ndarray, burn_in: int) -> tuple[np.ndarray, np.ndarray]:
    posterior = theta_samples[:, burn_in:, :]
    split = max(1, posterior.shape[0] // 2)
    return (
        posterior[:split].reshape(-1, 3),
        posterior[split:].reshape(-1, 3),
    )


def step_split_samples(theta_samples: np.ndarray, burn_in: int) -> tuple[np.ndarray, np.ndarray]:
    posterior = theta_samples[:, burn_in:, :]
    return (
        posterior[:, 0::2, :].reshape(-1, 3),
        posterior[:, 1::2, :].reshape(-1, 3),
    )


def mean_normalized_wasserstein_value(result: dict[str, object]) -> float:
    value = result["mean_normalized_wasserstein"]
    if isinstance(value, dict):
        return float(value["value"])
    return float(value)


def sampler_reference_diagnostics(
    *,
    t: np.ndarray,
    y: np.ndarray,
    reference: dict[str, object],
    args: argparse.Namespace,
    obs_index: int,
) -> dict[str, object]:
    device, dtype = choose_sampler_device(args.reference_device)
    t_tensor = torch.as_tensor(t, dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.float64)

    mcmc_config = MCMCConfig(
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        burn_in=args.mcmc_burn_in,
        seed=args.seed + 20_000 + obs_index,
        proposal_scale=args.mcmc_proposal_scale,
        requested_device=args.reference_device,
        sampler_variant="low-overhead",
    )
    mcmc_z, mcmc_theta, mcmc_accepted, mcmc_seconds = run_random_walk_metropolis(
        t=t_tensor,
        y=y_tensor,
        config=mcmc_config,
        device=device,
        dtype=dtype,
    )
    mcmc_posterior = mcmc_theta[:, mcmc_config.burn_in :, :].reshape(-1, 3)
    mcmc_chain_a, mcmc_chain_b = chain_split_samples(mcmc_theta, mcmc_config.burn_in)
    mcmc_step_a, mcmc_step_b = step_split_samples(mcmc_theta, mcmc_config.burn_in)

    hmc_config = HMCConfig(
        chains=args.hmc_chains,
        steps=args.hmc_steps,
        burn_in=args.hmc_burn_in,
        seed=args.seed + 30_000 + obs_index,
        step_size=args.hmc_step_size,
        leapfrog_steps=args.hmc_leapfrog_steps,
        requested_device=args.reference_device,
    )
    hmc_z, hmc_theta, hmc_accepted, hmc_energy_error, hmc_seconds = run_hmc(
        t=t_tensor,
        y=y_tensor,
        config=hmc_config,
        device=device,
        dtype=dtype,
    )
    hmc_posterior = hmc_theta[:, hmc_config.burn_in :, :].reshape(-1, 3)
    hmc_chain_a, hmc_chain_b = chain_split_samples(hmc_theta, hmc_config.burn_in)
    hmc_step_a, hmc_step_b = step_split_samples(hmc_theta, hmc_config.burn_in)

    diagnostics = {
        "mcmc_to_grid": compare_to_reference(mcmc_posterior, reference),
        "hmc_to_grid": compare_to_reference(hmc_posterior, reference),
        "mcmc_chain_half_1_to_grid": compare_to_reference(mcmc_chain_a, reference),
        "mcmc_chain_half_2_to_grid": compare_to_reference(mcmc_chain_b, reference),
        "hmc_chain_half_1_to_grid": compare_to_reference(hmc_chain_a, reference),
        "hmc_chain_half_2_to_grid": compare_to_reference(hmc_chain_b, reference),
        "mcmc_step_even_to_grid": compare_to_reference(mcmc_step_a, reference),
        "mcmc_step_odd_to_grid": compare_to_reference(mcmc_step_b, reference),
        "hmc_step_even_to_grid": compare_to_reference(hmc_step_a, reference),
        "hmc_step_odd_to_grid": compare_to_reference(hmc_step_b, reference),
        "mcmc_chain_halves_pairwise": {
            "mean_normalized_wasserstein": sample_to_sample_normalized_wasserstein(
                mcmc_chain_a,
                mcmc_chain_b,
                reference,
            )
        },
        "hmc_chain_halves_pairwise": {
            "mean_normalized_wasserstein": sample_to_sample_normalized_wasserstein(
                hmc_chain_a,
                hmc_chain_b,
                reference,
            )
        },
        "mcmc_hmc_pairwise": {
            "mean_normalized_wasserstein": sample_to_sample_normalized_wasserstein(
                mcmc_posterior,
                hmc_posterior,
                reference,
            )
        },
    }
    tolerance_keys = [
        "mcmc_to_grid",
        "hmc_to_grid",
        "mcmc_chain_half_1_to_grid",
        "mcmc_chain_half_2_to_grid",
        "hmc_chain_half_1_to_grid",
        "hmc_chain_half_2_to_grid",
        "mcmc_step_even_to_grid",
        "mcmc_step_odd_to_grid",
        "hmc_step_even_to_grid",
        "hmc_step_odd_to_grid",
        "mcmc_chain_halves_pairwise",
        "hmc_chain_halves_pairwise",
        "mcmc_hmc_pairwise",
    ]
    tolerance_values = {
        key: mean_normalized_wasserstein_value(diagnostics[key])
        if key not in {"mcmc_chain_halves_pairwise", "hmc_chain_halves_pairwise", "mcmc_hmc_pairwise"}
        else float(diagnostics[key]["mean_normalized_wasserstein"])
        for key in tolerance_keys
    }
    tau_samplers = float(max(tolerance_values.values()))
    abs_energy_error = np.abs(hmc_energy_error[:, hmc_config.burn_in :])
    divergent = ~np.isfinite(hmc_energy_error[:, hmc_config.burn_in :]) | (abs_energy_error > 100.0)
    mcmc_arviz = arviz_diagnostics(mcmc_theta, mcmc_config.burn_in)
    hmc_arviz = arviz_diagnostics(hmc_theta, hmc_config.burn_in)
    mcmc_flags = convergence_flags(mcmc_arviz)
    hmc_flags = convergence_flags(hmc_arviz)
    hmc_divergence_count = int(np.sum(divergent))
    convergence_ok = bool(
        all(mcmc_flags.values())
        and all(hmc_flags.values())
        and hmc_divergence_count == 0
    )
    return {
        "mode": "mcmc_hmc",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "tau_samplers": tau_samplers,
        "convergence_ok": convergence_ok,
        "tolerance_values": tolerance_values,
        "diagnostics": diagnostics,
        "mcmc": {
            "config": {
                "chains": mcmc_config.chains,
                "steps": mcmc_config.steps,
                "burn_in": mcmc_config.burn_in,
                "proposal_scale": list(mcmc_config.proposal_scale),
                "seed": mcmc_config.seed,
            },
            "runtime_seconds": float(mcmc_seconds),
            "acceptance_rate": float(mcmc_accepted.mean()),
            "arviz": mcmc_arviz,
            "convergence_flags": mcmc_flags,
        },
        "hmc": {
            "config": {
                "chains": hmc_config.chains,
                "steps": hmc_config.steps,
                "burn_in": hmc_config.burn_in,
                "step_size": hmc_config.step_size,
                "leapfrog_steps": hmc_config.leapfrog_steps,
                "seed": hmc_config.seed,
            },
            "runtime_seconds": float(hmc_seconds),
            "acceptance_rate": float(hmc_accepted.mean()),
            "energy_error": {
                "mean_abs_after_burn_in": float(np.nanmean(abs_energy_error)),
                "max_abs_after_burn_in": float(np.nanmax(abs_energy_error)),
                "divergence_count_after_burn_in": hmc_divergence_count,
            },
            "arviz": hmc_arviz,
            "convergence_flags": hmc_flags,
        },
    }


def load_stage1_model(
    *,
    family: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
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
    return model, stats


def load_flow_checkpoint(
    *,
    path: Path,
    device: torch.device,
) -> tuple[ConditionalSplineFlow, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConditionalSplineFlow(
        z_dim=3,
        context_dim=len(np.asarray(checkpoint["context_mean"])),
        transforms=int(config["transforms"]),
        hidden_features=tuple(config["hidden_features"]),
        bins=int(config["bins"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state = {
        "config": config,
        "context_mean": np.asarray(checkpoint["context_mean"]),
        "context_std": np.asarray(checkpoint["context_std"]),
        "z_mean": np.asarray(checkpoint["z_mean"]),
        "z_std": np.asarray(checkpoint["z_std"]),
        "linear_adjustment": checkpoint.get("linear_adjustment"),
        "checkpoint_path": path,
    }
    return model, state


def ensemble_sample_counts(total: int, members: int) -> list[int]:
    if members <= 0:
        raise ValueError("ensemble must contain at least one member")
    if total < members:
        raise ValueError("--posterior-samples must be at least the number of ensemble members")
    base, remainder = divmod(total, members)
    return [base + (1 if index < remainder else 0) for index in range(members)]


def plot_panel_summary(summary: dict[str, object], outfile: Path) -> None:
    models = summary["models"]
    observations = summary["observations"]
    ratios = [
        [obs["models"][model]["tolerance_ratio"] for obs in observations]
        for model in models
    ]
    discrepancies = [
        [obs["models"][model]["mean_normalized_wasserstein"] for obs in observations]
        for model in models
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=False)
    for ax, values, ylabel, title in [
        (axes[0], discrepancies, "D_i", "NPE to grid reference"),
        (axes[1], ratios, "D_i / tau_i", "Grid-tolerance ratio"),
    ]:
        positions = np.arange(len(models))
        box = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
        for patch, model in zip(box["boxes"], models, strict=True):
            patch.set_facecolor(FAMILY_COLORS.get(model, "#6b7280"))
            patch.set_alpha(0.35)
        for index, series in enumerate(values):
            jitter = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else np.array([0.0])
            ax.scatter(
                positions[index] + jitter,
                series,
                color=FAMILY_COLORS.get(models[index], "#374151"),
                s=28,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([summary["model_labels"][model] for model in models], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.5, label="ratio = 1")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decay NPE amortization over held-out observation panels.")
    parser.add_argument("--model-kind", choices=["stage1", "flow_decay"], default="stage1")
    parser.add_argument("--stage1-dir", type=Path, default=Path("runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/results"))
    parser.add_argument("--families", type=parse_families, default=["mdn"])
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=Path("runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_model.pt"),
    )
    parser.add_argument(
        "--flow-checkpoints",
        type=parse_path_list,
        default=None,
        help=(
            "Optional comma-separated flow checkpoints. When provided with "
            "--model-kind flow_decay, evaluate an equal-weight posterior sample "
            "ensemble whose total sample count is --posterior-samples."
        ),
    )
    parser.add_argument(
        "--panel-distribution",
        choices=["x0", "prior_predictive", "local_x", "parameter_region"],
        default="prior_predictive",
    )
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument(
        "--observation-indices",
        type=parse_int_list,
        default=None,
        help=(
            "Optional comma-separated original panel indices to evaluate after "
            "generating the full panel. Useful for expensive MCMC/HMC tolerance "
            "calibration on selected observations."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--n-observations-per-curve", type=int, default=40)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--grid-chunk-size", type=int, default=120_000)
    parser.add_argument("--grid-range-padding", type=float, default=0.45)
    parser.add_argument("--grid-min-padding", type=float, default=0.16)
    parser.add_argument("--edge-mass-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-grid-expansions", type=int, default=2)
    parser.add_argument("--grid-sample-count", type=int, default=20_000)
    parser.add_argument("--grid-replicates", type=int, default=4)
    parser.add_argument("--grid-tolerance-stat", choices=["median", "max"], default="median")
    parser.add_argument("--tolerance-mode", choices=["grid_only", "mcmc_hmc"], default="grid_only")
    parser.add_argument(
        "--reference-source",
        choices=["adaptive", "x0_mcmc_hmc"],
        default="adaptive",
        help=(
            "Grid reference construction. x0_mcmc_hmc reproduces the legacy "
            "single-x_o reference from saved MCMC/HMC samples and is valid only "
            "with --panel-distribution x0."
        ),
    )
    parser.add_argument("--reference-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=24_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=6_000)
    parser.add_argument("--mcmc-proposal-scale", type=parse_proposal_scale, default=(0.030, 0.030, 0.040))
    parser.add_argument("--hmc-chains", type=int, default=8)
    parser.add_argument("--hmc-steps", type=int, default=5_000)
    parser.add_argument("--hmc-burn-in", type=int, default=1_000)
    parser.add_argument("--hmc-step-size", type=float, default=0.009)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=10)
    parser.add_argument("--context-kind", choices=["indirect", "enhanced"], default="indirect")
    parser.add_argument("--k-grid-points", type=int, default=260)
    parser.add_argument("--k-min", type=float, default=0.04)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--local-quantile", type=float, default=0.005)
    parser.add_argument("--local-pilot-simulations", type=int, default=100_000)
    parser.add_argument("--local-max-candidates", type=int, default=20_000_000)
    parser.add_argument(
        "--local-region-summary",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON containing local_training.region. When set, "
            "local_x panels reuse that declared region instead of fitting a new "
            "pilot region."
        ),
    )
    parser.add_argument("--parameter-center", choices=["true_theta", "prior_mean", "custom"], default="true_theta")
    parser.add_argument("--parameter-center-z", type=parse_float_triple, default=None)
    parser.add_argument("--parameter-radius", type=float, default=1.0)
    parser.add_argument("--parameter-max-candidates", type=int, default=5_000_000)
    parser.add_argument(
        "--reference-prior",
        choices=["auto", "full", "restricted"],
        default="auto",
        help=(
            "Reference prior for grid posteriors. auto uses restricted only for "
            "parameter_region panels."
        ),
    )
    parser.add_argument("--simulate-chunk-size", type=int, default=50_000)
    parser.add_argument("--summary-chunk-size", type=int, default=25_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/01_exponential_decay/07_amortization_panels/01_panel_smoke/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("runs/01_exponential_decay/07_amortization_panels/01_panel_smoke/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_source == "x0_mcmc_hmc" and args.panel_distribution != "x0":
        raise ValueError("--reference-source x0_mcmc_hmc requires --panel-distribution x0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = choose_training_device(args.device)
    torch.manual_seed(args.seed + 100)
    np.random.seed(args.seed + 101)
    k_grid = make_k_grid(args.k_grid_points, args.k_min, args.k_max)

    restricted_region = None
    if args.panel_distribution == "x0":
        if args.num_observations != 1:
            raise ValueError("--panel-distribution x0 requires --num-observations 1")
        t, x_panel, z_true, distances, panel_metadata = sample_x0_observation(
            observed_seed=args.observed_seed,
            n_observations=args.n_observations_per_curve,
        )
    elif args.panel_distribution == "prior_predictive":
        t, x_panel, z_true, distances, panel_metadata = sample_prior_predictive_observations(
            n=args.num_observations,
            seed=args.seed,
            n_observations=args.n_observations_per_curve,
        )
    elif args.panel_distribution == "local_x":
        local_region_override = None
        if args.local_region_summary is not None:
            local_summary = json.loads(args.local_region_summary.read_text(encoding="utf-8"))
            local_region_override = local_summary["local_training"]["region"]
        t, x_panel, z_true, distances, panel_metadata = sample_local_x_observations(
            n=args.num_observations,
            seed=args.seed,
            observed_seed=args.observed_seed,
            n_observations=args.n_observations_per_curve,
            local_quantile=args.local_quantile,
            local_pilot_simulations=args.local_pilot_simulations,
            local_max_candidates=args.local_max_candidates,
            simulate_chunk_size=args.simulate_chunk_size,
            summary_chunk_size=args.summary_chunk_size,
            context_kind=args.context_kind,
            k_grid=k_grid,
            region_override=local_region_override,
        )
    else:
        center_z = resolve_parameter_center(args)
        t, x_panel, z_true, distances, panel_metadata = sample_parameter_region_observations(
            n=args.num_observations,
            seed=args.seed,
            n_observations=args.n_observations_per_curve,
            center_z=center_z,
            radius=args.parameter_radius,
            max_candidates=args.parameter_max_candidates,
            simulate_chunk_size=args.simulate_chunk_size,
        )
        if args.reference_prior in {"auto", "restricted"}:
            restricted_region = {
                "center_z": center_z,
                "radius": args.parameter_radius,
            }
    if args.reference_prior == "restricted" and restricted_region is None:
        center_z = resolve_parameter_center(args)
        restricted_region = {
            "center_z": center_z,
            "radius": args.parameter_radius,
        }

    models: dict[str, object] = {}
    model_state: dict[str, object] = {}
    model_labels: dict[str, str] = {}
    if args.model_kind == "stage1":
        for family in args.families:
            model, stats = load_stage1_model(
                family=family,
                checkpoint_path=args.stage1_dir / f"{family}_model.pt",
                device=device,
            )
            models[family] = model
            model_state[family] = stats
            model_labels[family] = FAMILY_LABELS[family]
    else:
        flow_paths = args.flow_checkpoints if args.flow_checkpoints is not None else [args.flow_checkpoint]
        flow_members = [
            load_flow_checkpoint(path=path, device=device)
            for path in flow_paths
        ]
        models["flow_decay"] = flow_members[0][0] if len(flow_members) == 1 else flow_members
        model_state["flow_decay"] = flow_members[0][1] if len(flow_members) == 1 else flow_members
        model_labels["flow_decay"] = "Spline flow" if len(flow_members) == 1 else f"Spline flow ensemble ({len(flow_members)})"

    evaluation_indices = (
        list(range(args.num_observations))
        if args.observation_indices is None
        else args.observation_indices
    )
    invalid_indices = [index for index in evaluation_indices if index >= args.num_observations]
    if invalid_indices:
        raise ValueError(
            f"--observation-indices contains out-of-range indices {invalid_indices}; "
            f"--num-observations is {args.num_observations}"
        )

    output_observations = []
    for position, obs_index in enumerate(evaluation_indices, start=1):
        print(
            f"evaluating panel observation {obs_index} "
            f"({position}/{len(evaluation_indices)} selected)",
            flush=True,
        )
        observed_x = x_panel[obs_index]
        true_z = z_true[obs_index]
        true_theta = np.exp(true_z)
        z_samples_by_model: dict[str, np.ndarray] = {}
        theta_samples_by_model: dict[str, np.ndarray] = {}
        model_sampling_seconds: dict[str, float] = {}
        for model_name, model in models.items():
            sample_start = time.perf_counter()
            if args.model_kind == "stage1":
                stats = model_state[model_name]
                z_samples, theta_samples = sample_posterior_for_observation(
                    model=model,
                    observed_x=observed_x,
                    x_mean=stats["x_mean"],
                    x_std=stats["x_std"],
                    z_mean=stats["z_mean"],
                    z_std=stats["z_std"],
                    n=args.posterior_samples,
                    device=device,
                )
            else:
                state = model_state[model_name]
                if isinstance(state, list):
                    z_parts = []
                    theta_parts = []
                    counts = ensemble_sample_counts(args.posterior_samples, len(state))
                    for (member_model, member_state), count in zip(state, counts, strict=True):
                        context_kind = member_state["config"].get("context_kind", args.context_kind)
                        context = make_context_summaries(
                            observed_x[None, :],
                            t,
                            k_grid,
                            kind=context_kind,
                            chunk_size=1,
                        )[0]
                        z_member, theta_member = sample_flow_posterior(
                            model=member_model,
                            observed_context=context,
                            context_mean=member_state["context_mean"],
                            context_std=member_state["context_std"],
                            z_mean=member_state["z_mean"],
                            z_std=member_state["z_std"],
                            linear_adjustment=member_state["linear_adjustment"],
                            n=count,
                            device=device,
                        )
                        z_parts.append(z_member)
                        theta_parts.append(theta_member)
                    z_samples = np.vstack(z_parts)
                    theta_samples = np.vstack(theta_parts)
                else:
                    context_kind = state["config"].get("context_kind", args.context_kind)
                    context = make_context_summaries(
                        observed_x[None, :],
                        t,
                        k_grid,
                        kind=context_kind,
                        chunk_size=1,
                    )[0]
                    z_samples, theta_samples = sample_flow_posterior(
                        model=model,
                        observed_context=context,
                        context_mean=state["context_mean"],
                        context_std=state["context_std"],
                        z_mean=state["z_mean"],
                        z_std=state["z_std"],
                        linear_adjustment=state["linear_adjustment"],
                        n=args.posterior_samples,
                        device=device,
                    )
            z_samples_by_model[model_name] = z_samples
            theta_samples_by_model[model_name] = theta_samples
            model_sampling_seconds[model_name] = time.perf_counter() - sample_start

        grid_start = time.perf_counter()
        if args.reference_source == "x0_mcmc_hmc":
            reference = build_x0_mcmc_hmc_grid_reference(
                mcmc_samples=args.mcmc_samples,
                hmc_samples=args.hmc_samples,
                grid_size=args.grid_size,
                chunk_size=args.grid_chunk_size,
            )
            grid_expansions = 0
        else:
            z_ranges = initial_z_ranges(
                z_samples_by_model=z_samples_by_model,
                true_z=true_z,
                padding_fraction=args.grid_range_padding,
                min_padding=args.grid_min_padding,
            )
            z_ranges = include_restricted_region_in_ranges(z_ranges, restricted_region)
            reference, grid_expansions = build_adaptive_grid_reference(
                t=t,
                y=observed_x,
                z_ranges=z_ranges,
                grid_size=args.grid_size,
                chunk_size=args.grid_chunk_size,
                edge_mass_tolerance=args.edge_mass_tolerance,
                max_expansions=args.max_grid_expansions,
                restricted_region=restricted_region,
            )
        grid_reference_seconds = time.perf_counter() - grid_start
        grid_noise = grid_replicate_diagnostics(
            reference=reference,
            sample_count=args.grid_sample_count,
            replicates=args.grid_replicates,
            seed=args.seed + 10_000 + obs_index * 100,
        )
        tau_grid = grid_tolerance(grid_noise, args.grid_tolerance_stat)
        sampler_tolerance = None
        tau = tau_grid
        tolerance_valid = True
        if args.tolerance_mode == "mcmc_hmc":
            sampler_tolerance = sampler_reference_diagnostics(
                t=t,
                y=observed_x,
                reference=reference,
                args=args,
                obs_index=obs_index,
            )
            tau = float(max(tau_grid, float(sampler_tolerance["tau_samplers"])))
            tolerance_valid = bool(sampler_tolerance["convergence_ok"])
        model_results = {}
        for model_name, theta_samples in theta_samples_by_model.items():
            metrics = compare_to_reference(theta_samples, reference)
            discrepancy = float(metrics["mean_normalized_wasserstein"]["value"])
            ratio = float(discrepancy / max(tau, 1e-12))
            model_results[model_name] = {
                "mean_normalized_wasserstein": discrepancy,
                "tau_grid": tau_grid,
                "tau": tau,
                "grid_tolerance_ratio": float(discrepancy / max(tau_grid, 1e-12)),
                "tolerance_ratio": ratio,
                "tolerance_valid": tolerance_valid,
                "ratio_pass": bool(tolerance_valid and ratio <= 1.0),
                "metrics": metrics,
                "posterior_summary": summarize_samples(theta_samples),
            }
        panel_distance = None if distances is None else float(distances[obs_index])
        distance_metrics = build_observation_distance_metrics(
            panel_kind=str(panel_metadata["kind"]),
            true_z=true_z,
            panel_distance=panel_distance,
            panel_metadata=panel_metadata,
            parameter_radius=args.parameter_radius,
        )
        output_observations.append({
            "index": obs_index,
            "x": observed_x.tolist(),
            "z_true": true_z.tolist(),
            "theta_true": {
                name: float(true_theta[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "distance_to_x0": panel_distance,
            "distance_metrics": distance_metrics,
            "grid_reference": {
                "grid_size": reference["grid_size"],
                "grid_points": reference["grid_points"],
                "z_ranges": reference["z_ranges"],
                "edge_mass": reference["edge_mass"],
                "max_edge_mass": max_edge_mass(reference),
                "grid_expansions": grid_expansions,
                "posterior_summary": reference["summary"],
            },
            "grid_tolerance": {
                "mode": "grid_only",
                "statistic": args.grid_tolerance_stat,
                "tau_grid": tau_grid,
                "diagnostics": grid_noise,
            },
            "sampler_tolerance": sampler_tolerance,
            "tolerance": {
                "mode": args.tolerance_mode,
                "tau": tau,
                "valid": tolerance_valid,
            },
            "timing_seconds": {
                "model_sampling": model_sampling_seconds,
                "grid_reference": grid_reference_seconds,
            },
            "models": model_results,
        })

    aggregate = {}
    for model_name in models:
        discrepancies = np.array([
            obs["models"][model_name]["mean_normalized_wasserstein"]
            for obs in output_observations
        ])
        ratios = np.array([
            obs["models"][model_name]["tolerance_ratio"]
            for obs in output_observations
        ])
        grid_ratios = np.array([
            obs["models"][model_name]["grid_tolerance_ratio"]
            for obs in output_observations
        ])
        ratio_passes = np.array([
            obs["models"][model_name]["ratio_pass"]
            for obs in output_observations
        ], dtype=bool)
        valid_tolerances = np.array([
            obs["models"][model_name]["tolerance_valid"]
            for obs in output_observations
        ], dtype=bool)
        valid_ratios = ratios[valid_tolerances]
        valid_ratio_passes = ratio_passes[valid_tolerances]
        if valid_ratios.size:
            valid_ratio_summary = {
                "valid_mean_ratio": float(valid_ratios.mean()),
                "valid_median_ratio": float(np.median(valid_ratios)),
                "valid_q90_ratio": float(np.quantile(valid_ratios, 0.90)),
                "valid_max_ratio": float(valid_ratios.max()),
                "valid_pass_count_ratio_le_1": int(valid_ratio_passes.sum()),
                "valid_pass_fraction_ratio_le_1": float(valid_ratio_passes.mean()),
                "valid_ratio_values": valid_ratios.tolist(),
            }
        else:
            valid_ratio_summary = {
                "valid_mean_ratio": None,
                "valid_median_ratio": None,
                "valid_q90_ratio": None,
                "valid_max_ratio": None,
                "valid_pass_count_ratio_le_1": 0,
                "valid_pass_fraction_ratio_le_1": None,
                "valid_ratio_values": [],
            }
        aggregate[model_name] = {
            "mean_discrepancy": float(discrepancies.mean()),
            "median_discrepancy": float(np.median(discrepancies)),
            "q90_discrepancy": float(np.quantile(discrepancies, 0.90)),
            "max_discrepancy": float(discrepancies.max()),
            "mean_ratio": float(ratios.mean()),
            "median_ratio": float(np.median(ratios)),
            "q90_ratio": float(np.quantile(ratios, 0.90)),
            "max_ratio": float(ratios.max()),
            "mean_grid_only_ratio": float(grid_ratios.mean()),
            "median_grid_only_ratio": float(np.median(grid_ratios)),
            "q90_grid_only_ratio": float(np.quantile(grid_ratios, 0.90)),
            "max_grid_only_ratio": float(grid_ratios.max()),
            "valid_tolerance_count": int(valid_tolerances.sum()),
            "valid_tolerance_fraction": float(valid_tolerances.mean()),
            "invalid_tolerance_count": int((~valid_tolerances).sum()),
            "pass_count_ratio_le_1": int(ratio_passes.sum()),
            "pass_fraction_ratio_le_1": float(ratio_passes.mean()),
            "amortized_pass": bool(
                np.all(valid_tolerances)
                and valid_ratios.size > 0
                and np.quantile(valid_ratios, 0.90) <= 1.0
                and valid_ratio_passes.mean() >= 0.90
            ),
            "discrepancy_values": discrepancies.tolist(),
            "ratio_values": ratios.tolist(),
            "grid_only_ratio_values": grid_ratios.tolist(),
            **valid_ratio_summary,
        }

    summary = {
        "script": "scripts/evaluate_decay_amortization_panel.py",
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "figure_dir"}
        },
        "device": str(device),
        "panel_distribution": panel_metadata,
        "reference_prior": (
            "restricted"
            if restricted_region is not None
            else "full"
        ),
        "reference_source": args.reference_source,
        "restricted_region": restricted_region,
        "tolerance_mode": args.tolerance_mode,
        "evaluation_indices": evaluation_indices,
        "models": list(models.keys()),
        "model_labels": model_labels,
        "aggregate": aggregate,
        "observations": output_observations,
    }
    summary_path = args.output_dir / "decay_amortization_panel_summary.json"
    figure_path = args.figure_dir / "decay_amortization_panel_summary.png"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    plot_panel_summary(summary, figure_path)

    print(f"summary_json: {summary_path}")
    print(f"figure: {figure_path}")
    print(f"aggregate {args.tolerance_mode} tolerance ratios:")
    for model_name, values in aggregate.items():
        print(
            f"  {model_name}: median={values['median_ratio']:.3f}, "
            f"q90={values['q90_ratio']:.3f}, "
            f"passes={values['pass_count_ratio_le_1']}/{len(evaluation_indices)}"
        )


if __name__ == "__main__":
    main()
