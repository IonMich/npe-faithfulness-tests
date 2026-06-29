from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import artifact_paths as ap

import numpy as np

from compare_decay_samplers import build_grid_reference, compare_to_reference, load_samples
from npe_flow_decay import mean_normalized_wasserstein_value


DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/13_reference_cache/01_x0_grid300")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache the single-decay x0 grid posterior reference.")
    parser.add_argument("--grid-size", type=int, default=300)
    parser.add_argument("--reference-chunk-size", type=int, default=120_000)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    reference_npz = results_dir / f"decay_x0_grid{args.grid_size}_reference.npz"
    metadata_json = results_dir / f"decay_x0_grid{args.grid_size}_reference_metadata.json"
    if reference_npz.exists() and metadata_json.exists() and not args.overwrite:
        print(f"reference_npz: {reference_npz}")
        print(f"metadata_json: {metadata_json}")
        print("status: exists")
        return

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])

    build_start = time.perf_counter()
    reference = build_grid_reference(
        t=mcmc["t"],
        y=mcmc["y"],
        combined_z_samples=combined_z,
        true_theta=mcmc["true_theta"],
        grid_size=args.grid_size,
        chunk_size=args.reference_chunk_size,
    )
    build_seconds = time.perf_counter() - build_start

    metric_start = time.perf_counter()
    mcmc_to_grid = compare_to_reference(mcmc["posterior_theta"], reference)
    hmc_to_grid = compare_to_reference(hmc["posterior_theta"], reference)
    metric_seconds = time.perf_counter() - metric_start

    save_start = time.perf_counter()
    np.savez_compressed(
        reference_npz,
        theta_grid=np.asarray(reference["theta_grid"], dtype=np.float64),
        weights=np.asarray(reference["weights"], dtype=np.float64),
    )
    save_seconds = time.perf_counter() - save_start

    file_size_bytes = reference_npz.stat().st_size
    raw_core_bytes = (
        np.asarray(reference["theta_grid"]).nbytes
        + np.asarray(reference["weights"]).nbytes
    )
    metadata = {
        "grid_size": int(reference["grid_size"]),
        "grid_points": int(reference["grid_points"]),
        "reference_npz": reference_npz,
        "metadata_json": metadata_json,
        "raw_core_bytes": int(raw_core_bytes),
        "compressed_bytes": int(file_size_bytes),
        "raw_core_mib": raw_core_bytes / (1024**2),
        "compressed_mib": file_size_bytes / (1024**2),
        "z_ranges": reference["z_ranges"],
        "edge_mass": reference["edge_mass"],
        "summary": reference["summary"],
        "mcmc_to_grid": mcmc_to_grid,
        "hmc_to_grid": hmc_to_grid,
        "recommended_target": max(
            mean_normalized_wasserstein_value(mcmc_to_grid),
            mean_normalized_wasserstein_value(hmc_to_grid),
        ),
        "timing_seconds": {
            "build_grid": build_seconds,
            "mcmc_hmc_metrics": metric_seconds,
            "save_compressed": save_seconds,
            "total": time.perf_counter() - total_start,
        },
        "inputs": {
            "mcmc_samples": args.mcmc_samples,
            "hmc_samples": args.hmc_samples,
        },
    }
    metadata_json.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")

    print(f"reference_npz: {reference_npz}")
    print(f"metadata_json: {metadata_json}")
    print(f"grid_points: {reference['grid_points']}")
    print(f"raw_core_mib: {metadata['raw_core_mib']:.1f}")
    print(f"compressed_mib: {metadata['compressed_mib']:.1f}")
    print(f"build_seconds: {build_seconds:.2f}")
    print(f"save_seconds: {save_seconds:.2f}")
    print(f"total_seconds: {metadata['timing_seconds']['total']:.2f}")


if __name__ == "__main__":
    main()
