from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import npe_stage1_decay as stage1


DEFAULT_OUTPUT = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache broad prior-predictive validation pairs for decay NPE NLL evaluation.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260990)
    parser.add_argument("--n-observations", type=int, default=40)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    x, z, t = stage1.sample_decay_pairs(
        n=args.simulations,
        seed=args.seed,
        n_observations=args.n_observations,
    )
    generate_seconds = time.perf_counter() - start

    dtype = np.float32 if args.dtype == "float32" else np.float64
    x = x.astype(dtype, copy=False)
    z = z.astype(dtype, copy=False)
    t = t.astype(dtype, copy=False)

    raw_bytes = int(x.nbytes + z.nbytes + t.nbytes)
    save_start = time.perf_counter()
    np.savez_compressed(
        args.output,
        x_val=x,
        z_val=z,
        t=t,
        seed=np.asarray(args.seed, dtype=np.int64),
        simulations=np.asarray(args.simulations, dtype=np.int64),
        n_observations=np.asarray(args.n_observations, dtype=np.int64),
        dtype=np.asarray(args.dtype),
    )
    save_seconds = time.perf_counter() - save_start
    file_bytes = args.output.stat().st_size

    metadata = {
        "output": args.output,
        "simulations": int(args.simulations),
        "seed": int(args.seed),
        "n_observations": int(args.n_observations),
        "dtype": args.dtype,
        "raw_bytes": raw_bytes,
        "raw_mib": raw_bytes / (1024**2),
        "compressed_bytes": int(file_bytes),
        "compressed_mib": file_bytes / (1024**2),
        "generate_seconds": generate_seconds,
        "save_seconds": save_seconds,
        "total_seconds": time.perf_counter() - start,
        "arrays": {
            "x_val": list(x.shape),
            "z_val": list(z.shape),
            "t": list(t.shape),
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    print(f"cache_npz: {args.output}")
    print(f"metadata_json: {metadata_path}")
    print(f"raw_mib: {metadata['raw_mib']:.1f}")
    print(f"compressed_mib: {metadata['compressed_mib']:.1f}")
    print(f"total_seconds: {metadata['total_seconds']:.2f}")


if __name__ == "__main__":
    main()
