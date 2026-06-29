from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

import npe_stage1_decay as stage1
from compare_decay_samplers import log_posterior_z_numpy, weighted_quantile
from mcmc_decay_inference import PARAMETER_NAMES, simulate_decay_data


DEFAULT_OUTPUT = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
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


def summarize_marginal(theta_axis: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights = weights / np.sum(weights)
    q05, q16, q50, q84, q95 = weighted_quantile(theta_axis, weights, [0.05, 0.16, 0.50, 0.84, 0.95])
    mean = float(np.sum(theta_axis * weights))
    variance = float(np.sum((theta_axis - mean) ** 2 * weights))
    return {
        "mean": mean,
        "sd": float(math.sqrt(max(variance, 0.0))),
        "q05": float(q05),
        "q16": float(q16),
        "median": float(q50),
        "q84": float(q84),
        "q95": float(q95),
    }


def flat_z_chunk(axes_z: list[np.ndarray], start: int, stop: int) -> np.ndarray:
    grid_size = axes_z[0].shape[0]
    index = np.arange(start, stop, dtype=np.int64)
    i0 = index // (grid_size * grid_size)
    i1 = (index // grid_size) % grid_size
    i2 = index % grid_size
    return np.column_stack((axes_z[0][i0], axes_z[1][i1], axes_z[2][i2]))


def edge_mass_from_weights(weights: np.ndarray, grid_size: int) -> dict[str, dict[str, float]]:
    cube = weights.reshape((grid_size, grid_size, grid_size))
    output = {}
    for axis, name in enumerate(PARAMETER_NAMES):
        lower = float(np.take(cube, indices=0, axis=axis).sum())
        upper = float(np.take(cube, indices=grid_size - 1, axis=axis).sum())
        output[name] = {"lower": lower, "upper": upper}
    return output


def max_edge_mass(edge_mass: dict[str, dict[str, float]]) -> float:
    return max(value for item in edge_mass.values() for value in item.values())


def build_signal_marginals(
    *,
    x: np.ndarray,
    t: np.ndarray,
    true_z: np.ndarray,
    grid_size: int,
    chunk_size: int,
    initial_half_width: float,
    edge_mass_tolerance: float,
    max_expand: int,
    target_sample_count: int,
    target_repeats: int,
    seed: int,
) -> dict[str, object]:
    half_width = float(initial_half_width)
    last: dict[str, object] | None = None
    for expand_index in range(max_expand + 1):
        axes_z = [
            np.linspace(true_z[index] - half_width, true_z[index] + half_width, grid_size)
            for index in range(3)
        ]
        grid_points = grid_size**3
        logp = np.empty(grid_points, dtype=np.float64)
        for start in range(0, grid_points, chunk_size):
            stop = min(start + chunk_size, grid_points)
            z_chunk = flat_z_chunk(axes_z, start, stop)
            logp[start:stop] = log_posterior_z_numpy(z_chunk, t=t, y=x)
        weights = np.exp(logp - logsumexp(logp))
        edge_mass = edge_mass_from_weights(weights, grid_size)
        last = {
            "axes_z": axes_z,
            "weights": weights,
            "edge_mass": edge_mass,
            "half_width": half_width,
            "expand_index": expand_index,
        }
        if max_edge_mass(edge_mass) <= edge_mass_tolerance:
            break
        half_width *= 1.5
    if last is None:
        raise RuntimeError("marginal reference construction failed")

    axes_z = last["axes_z"]
    weights = np.asarray(last["weights"], dtype=np.float64)
    cube = weights.reshape((grid_size, grid_size, grid_size))
    theta_axes = np.stack([np.exp(axis) for axis in axes_z])
    marginal_weights = np.empty((3, grid_size), dtype=np.float64)
    summaries: dict[str, dict[str, float]] = {}
    for axis, name in enumerate(PARAMETER_NAMES):
        sum_axes = tuple(item for item in range(3) if item != axis)
        marginal = cube.sum(axis=sum_axes)
        marginal = marginal / marginal.sum()
        marginal_weights[axis] = marginal
        summaries[name] = summarize_marginal(theta_axes[axis], marginal)

    rng = np.random.default_rng(seed)
    target_rows = []
    flat_count = grid_size**3
    for _ in range(target_repeats):
        flat_index = rng.choice(flat_count, size=target_sample_count, replace=True, p=weights)
        i0 = flat_index // (grid_size * grid_size)
        i1 = (flat_index // grid_size) % grid_size
        i2 = flat_index % grid_size
        sample = np.column_stack((theta_axes[0][i0], theta_axes[1][i1], theta_axes[2][i2]))
        values = []
        for axis, name in enumerate(PARAMETER_NAMES):
            w = wasserstein_distance(sample[:, axis], theta_axes[axis], v_weights=marginal_weights[axis])
            values.append(float(w / max(summaries[name]["sd"], 1e-12)))
        target_rows.append(float(np.mean(values)))
    target_array = np.asarray(target_rows, dtype=np.float64)
    target = {
        "sample_count": int(target_sample_count),
        "repeats": int(target_repeats),
        "mean": float(np.mean(target_array)),
        "q50": float(np.quantile(target_array, 0.50)),
        "q84": float(np.quantile(target_array, 0.84)),
        "max": float(np.max(target_array)),
    }
    return {
        "theta_axes": theta_axes,
        "marginal_weights": marginal_weights,
        "summary": summaries,
        "target": target,
        "edge_mass": last["edge_mass"],
        "max_edge_mass": max_edge_mass(last["edge_mass"]),
        "half_width": float(last["half_width"]),
        "expand_index": int(last["expand_index"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache 1D marginal grid references for a panel of decay observations.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--panel-seed", type=int, default=20261001)
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument("--initial-half-width", type=float, default=1.2)
    parser.add_argument("--edge-mass-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-expand", type=int, default=3)
    parser.add_argument("--target-sample-count", type=int, default=20_000)
    parser.add_argument("--target-repeats", type=int, default=5)
    parser.add_argument("--include-x0", action="store_true")
    parser.add_argument("--x0-seed", type=int, default=20260622)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    panel_x, panel_z, t = stage1.sample_decay_pairs(n=args.panel_size, seed=args.panel_seed)
    panel_theta = np.exp(panel_z)
    labels = [f"prior_panel_{index:04d}" for index in range(args.panel_size)]
    if args.include_x0:
        t_x0, y_x0, true_theta_x0 = simulate_decay_data(seed=args.x0_seed)
        panel_x = np.vstack([y_x0.detach().cpu().numpy()[None, :], panel_x])
        panel_z = np.vstack([np.log(true_theta_x0.detach().cpu().numpy())[None, :], panel_z])
        panel_theta = np.vstack([true_theta_x0.detach().cpu().numpy()[None, :], panel_theta])
        t = t_x0.detach().cpu().numpy()
        labels = ["x0"] + labels

    theta_axes = []
    marginal_weights = []
    target_wasserstein = []
    metadata_rows = []
    for index, label in enumerate(labels):
        signal_start = time.perf_counter()
        built = build_signal_marginals(
            x=panel_x[index],
            t=t,
            true_z=panel_z[index],
            grid_size=args.grid_size,
            chunk_size=args.chunk_size,
            initial_half_width=args.initial_half_width,
            edge_mass_tolerance=args.edge_mass_tolerance,
            max_expand=args.max_expand,
            target_sample_count=args.target_sample_count,
            target_repeats=args.target_repeats,
            seed=args.panel_seed + 10_000 + index,
        )
        theta_axes.append(np.asarray(built["theta_axes"], dtype=np.float32))
        marginal_weights.append(np.asarray(built["marginal_weights"], dtype=np.float32))
        target_wasserstein.append(float(built["target"]["q84"]))
        metadata_rows.append({
            "index": index,
            "label": label,
            "target_wasserstein_q84": float(built["target"]["q84"]),
            "target": built["target"],
            "summary": built["summary"],
            "edge_mass": built["edge_mass"],
            "max_edge_mass": built["max_edge_mass"],
            "half_width": built["half_width"],
            "expand_index": built["expand_index"],
            "seconds": time.perf_counter() - signal_start,
        })
        print(
            f"[{index + 1}/{len(labels)}] {label} "
            f"target_q84={built['target']['q84']:.5f} "
            f"edge={built['max_edge_mass']:.2e} "
            f"seconds={metadata_rows[-1]['seconds']:.1f}",
            flush=True,
        )

    theta_axes_array = np.stack(theta_axes)
    marginal_weights_array = np.stack(marginal_weights)
    target_array = np.asarray(target_wasserstein, dtype=np.float32)
    np.savez_compressed(
        args.output,
        x_panel=panel_x.astype(np.float32),
        z_panel=panel_z.astype(np.float32),
        theta_panel=panel_theta.astype(np.float32),
        t=np.asarray(t, dtype=np.float32),
        labels=np.asarray(labels),
        theta_axes=theta_axes_array,
        marginal_weights=marginal_weights_array,
        target_wasserstein=target_array,
        grid_size=np.asarray(args.grid_size, dtype=np.int64),
        target_sample_count=np.asarray(args.target_sample_count, dtype=np.int64),
    )
    output_bytes = args.output.stat().st_size
    metadata = {
        "output": args.output,
        "panel_size": len(labels),
        "prior_panel_size": int(args.panel_size),
        "include_x0": bool(args.include_x0),
        "panel_seed": int(args.panel_seed),
        "grid_size": int(args.grid_size),
        "grid_points_per_signal": int(args.grid_size**3),
        "target_definition": (
            "q84 over exact-grid posterior sample-to-marginal W repeats, using "
            "--target-sample-count samples; this is a numerical/evaluation floor, "
            "not the x0 MCMC/HMC target."
        ),
        "target_sample_count": int(args.target_sample_count),
        "target_repeats": int(args.target_repeats),
        "edge_mass_tolerance": float(args.edge_mass_tolerance),
        "signals": metadata_rows,
        "compressed_bytes": int(output_bytes),
        "compressed_mib": output_bytes / (1024**2),
        "total_seconds": time.perf_counter() - start,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    print(f"cache_npz: {args.output}")
    print(f"metadata_json: {metadata_path}")
    print(f"compressed_mib: {metadata['compressed_mib']:.3f}")
    print(f"total_seconds: {metadata['total_seconds']:.1f}")


if __name__ == "__main__":
    main()
