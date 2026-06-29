from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import artifact_paths as ap

import torch

from mcmc_decay_inference import (
    MCMCConfig,
    SAMPLER_VARIANTS,
    normalize_sampler_variant,
    run_random_walk_metropolis,
    simulate_decay_data,
)


def benchmark_case(
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    config: MCMCConfig,
    repeats: int,
) -> dict[str, object]:
    t, y, _ = simulate_decay_data(seed=config.seed)

    # Short warmup so one-time backend setup is not the whole story.
    warmup_config = MCMCConfig(
        chains=config.chains,
        steps=min(2_000, config.steps),
        burn_in=min(500, config.burn_in),
        seed=config.seed,
        proposal_scale=config.proposal_scale,
        requested_device=config.requested_device,
    )
    run_random_walk_metropolis(
        t=t,
        y=y,
        config=warmup_config,
        device=device,
        dtype=dtype,
    )

    runtimes = []
    acceptance_rates = []
    for _ in range(repeats):
        _, _, accepted, elapsed_seconds = run_random_walk_metropolis(
            t=t,
            y=y,
            config=config,
            device=device,
            dtype=dtype,
        )
        runtimes.append(elapsed_seconds)
        acceptance_rates.append(float(accepted.mean()))

    return {
        "label": label,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "runtimes_seconds": runtimes,
        "median_seconds": statistics.median(runtimes),
        "mean_seconds": statistics.mean(runtimes),
        "acceptance_rate_mean": statistics.mean(acceptance_rates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MCMC sampler backends.")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24_000)
    parser.add_argument("--burn-in", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="baseline,pregenerated,low-overhead",
        help="Comma-separated sampler variants to benchmark.",
    )
    parser.add_argument("--output", type=Path, default=ap.MCMC_DEVICE_BENCHMARK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [
        normalize_sampler_variant(variant.strip())
        for variant in args.variants.split(",")
        if variant.strip()
    ]
    invalid_variants = sorted(set(variants) - SAMPLER_VARIANTS)
    if invalid_variants:
        raise ValueError(f"invalid sampler variants: {invalid_variants}")
    device_cases = [
        ("cpu_float64", torch.device("cpu"), torch.float64),
        ("cpu_float32", torch.device("cpu"), torch.float32),
    ]
    if torch.backends.mps.is_available():
        device_cases.insert(0, ("mps_float32", torch.device("mps"), torch.float32))

    results = []
    for variant in variants:
        config = MCMCConfig(
            chains=args.chains,
            steps=args.steps,
            burn_in=args.burn_in,
            seed=args.seed,
            proposal_scale=(0.030, 0.030, 0.040),
            requested_device="benchmark",
            sampler_variant=variant,
        )
        for label, device, dtype in device_cases:
            results.append(
                benchmark_case(
                    label=f"{label}_{variant}",
                    device=device,
                    dtype=dtype,
                    config=config,
                    repeats=args.repeats,
                )
            )

    reference_label = (
        "cpu_float64_baseline" if "baseline" in variants else f"cpu_float64_{variants[0]}"
    )
    baseline = next(item for item in results if item["label"] == reference_label)
    for item in results:
        item[f"speedup_vs_{reference_label}"] = (
            baseline["median_seconds"] / item["median_seconds"]
        )

    output = {
        "config": {
            "chains": args.chains,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "seed": args.seed,
            "repeats": args.repeats,
        },
        "mps_available": torch.backends.mps.is_available(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for item in results:
        print(
            f"{item['label']}: median={item['median_seconds']:.3f}s, "
            f"mean={item['mean_seconds']:.3f}s, "
            f"speedup_vs_{reference_label}={item[f'speedup_vs_{reference_label}']:.3f}x"
        )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
