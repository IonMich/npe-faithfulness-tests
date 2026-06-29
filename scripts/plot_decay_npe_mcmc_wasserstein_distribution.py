from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from compare_decay_samplers import compare_to_reference
from evaluate_decay_amortization_panel import build_grid_reference_from_ranges
from mcmc_decay_inference import (
    MCMCConfig,
    PARAMETER_NAMES,
    arviz_diagnostics,
    choose_device,
    convergence_flags,
    run_random_walk_metropolis,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "runs/01_exponential_decay/10_wasserstein_distributions/10_grid60_npe_mcmc"
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


def parse_model_list(value: str) -> list[str]:
    models = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not models:
        raise argparse.ArgumentTypeError("At least one model is required")
    return models


def parse_int_list(value: str) -> list[int]:
    values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one observation index is required")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Observation indices must be non-negative")
    return sorted(set(values))


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("proposal scale must contain three comma-separated floats")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("proposal scales must be positive")
    return pieces[0], pieces[1], pieces[2]


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q16": float(np.quantile(finite, 0.16)),
        "median": float(np.median(finite)),
        "q84": float(np.quantile(finite, 0.84)),
        "q90": float(np.quantile(finite, 0.90)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    return finite, np.arange(1, finite.size + 1) / finite.size


def mean_normalized_wasserstein(result: dict[str, object]) -> float:
    value = result["mean_normalized_wasserstein"]
    if isinstance(value, dict):
        return float(value["value"])
    return float(value)


def load_cached_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    cached: dict[int, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            obs_index = int(row["observation_index"])
            cached[obs_index] = {
                "observation_index": obs_index,
                "sampler": row["sampler"],
                "mean_normalized_wasserstein": float(row["mean_normalized_wasserstein"]),
                "runtime_seconds": float(row["runtime_seconds"]),
                "acceptance_rate": float(row["acceptance_rate"]),
                "max_rhat": float(row["max_rhat"]),
                "min_bulk_ess": float(row["min_bulk_ess"]),
                "min_tail_ess": float(row["min_tail_ess"]),
                "convergence_ok": row["convergence_ok"].lower() == "true",
            }
    return cached


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "observation_index",
        "sampler",
        "mean_normalized_wasserstein",
        "runtime_seconds",
        "acceptance_rate",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "convergence_ok",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["observation_index"])):
            writer.writerow({field: row.get(field) for field in fields})


def observation_indices(panel: dict[str, object], requested: list[int] | None) -> list[int]:
    available = [int(observation["index"]) for observation in panel["observations"]]
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested observation indices are missing from panel: {missing}")
    return requested


def collect_npe_rows(
    *,
    panel: dict[str, object],
    models: list[str],
    selected_indices: set[int],
) -> list[dict[str, object]]:
    available = [str(model) for model in panel.get("models", [])]
    missing = sorted(set(models) - set(available))
    if missing:
        raise ValueError(f"Requested NPE models missing from panel: {missing}; available={available}")
    rows = []
    for observation in panel["observations"]:
        obs_index = int(observation["index"])
        if obs_index not in selected_indices:
            continue
        timing = observation.get("timing_seconds", {}) or {}
        model_timing = timing.get("model_sampling", {}) or {}
        for model in models:
            rows.append(
                {
                    "observation_index": obs_index,
                    "sampler": f"NPE ({model})",
                    "mean_normalized_wasserstein": float(
                        observation["models"][model]["mean_normalized_wasserstein"]
                    ),
                    "runtime_seconds": model_timing.get(model),
                }
            )
    return rows


def run_mcmc_rows(
    *,
    panel: dict[str, object],
    selected_indices: list[int],
    output_csv: Path,
    device_name: str,
    chains: int,
    steps: int,
    burn_in: int,
    proposal_scale: tuple[float, float, float],
    grid_chunk_size: int,
    seed: int,
    resume: bool,
) -> list[dict[str, object]]:
    cached_by_index = load_cached_rows(output_csv) if resume else {}
    rows = list(cached_by_index.values())
    done = set(cached_by_index)

    device, dtype = choose_device(device_name)
    n_observations = int(panel["config"]["n_observations_per_curve"])
    t = np.linspace(0.0, 6.0, n_observations)
    t_tensor = torch.as_tensor(t, dtype=torch.float64)
    panel_by_index = {int(observation["index"]): observation for observation in panel["observations"]}

    remaining = [index for index in selected_indices if index not in done]
    if not remaining:
        return sorted(rows, key=lambda item: int(item["observation_index"]))

    start_all = time.perf_counter()
    for count, obs_index in enumerate(remaining, start=1):
        observation = panel_by_index[obs_index]
        y = np.asarray(observation["x"], dtype=np.float64)
        y_tensor = torch.as_tensor(y, dtype=torch.float64)
        z_ranges = np.asarray(
            [
                observation["grid_reference"]["z_ranges"][name]
                for name in PARAMETER_NAMES
            ],
            dtype=np.float64,
        )
        reference = build_grid_reference_from_ranges(
            t=t,
            y=y,
            z_ranges=z_ranges,
            grid_size=int(observation["grid_reference"]["grid_size"]),
            chunk_size=grid_chunk_size,
            restricted_region=None,
        )
        config = MCMCConfig(
            chains=chains,
            steps=steps,
            burn_in=burn_in,
            seed=seed + obs_index,
            proposal_scale=proposal_scale,
            requested_device=device_name,
            sampler_variant="low-overhead",
        )
        _, theta_samples, accepted, runtime_seconds = run_random_walk_metropolis(
            t=t_tensor,
            y=y_tensor,
            config=config,
            device=device,
            dtype=dtype,
        )
        posterior = theta_samples[:, burn_in:, :].reshape(-1, 3)
        metrics = compare_to_reference(posterior, reference)
        diagnostics = arviz_diagnostics(theta_samples, burn_in)
        flags = convergence_flags(diagnostics)
        row = {
            "observation_index": obs_index,
            "sampler": "MCMC",
            "mean_normalized_wasserstein": mean_normalized_wasserstein(metrics),
            "runtime_seconds": float(runtime_seconds),
            "acceptance_rate": float(accepted.mean()),
            "max_rhat": float(max(item["rhat"] for item in diagnostics.values())),
            "min_bulk_ess": float(min(item["ess_bulk"] for item in diagnostics.values())),
            "min_tail_ess": float(min(item["ess_tail"] for item in diagnostics.values())),
            "convergence_ok": bool(all(flags.values())),
        }
        rows.append(row)
        write_rows(rows, output_csv)
        if count == 1 or count % 10 == 0 or count == len(remaining):
            elapsed = time.perf_counter() - start_all
            rate = elapsed / count
            eta = rate * (len(remaining) - count)
            print(
                f"mcmc {count}/{len(remaining)} "
                f"obs={obs_index} W={row['mean_normalized_wasserstein']:.4f} "
                f"runtime={runtime_seconds:.3f}s elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return sorted(rows, key=lambda item: int(item["observation_index"]))


def plot_distribution(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    colors = {
        "NPE (mdn)": "#2f6fbb",
        "MCMC": "#b85c38",
    }
    labels = list(dict.fromkeys(str(row["sampler"]) for row in rows))
    all_values = np.asarray([row["mean_normalized_wasserstein"] for row in rows], dtype=np.float64)
    finite = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    if finite.size == 0:
        raise RuntimeError("No finite positive Wasserstein values to plot")
    use_log_x = finite.max() / max(finite.min(), 1e-12) > 80
    bins: int | np.ndarray
    if use_log_x:
        bins = np.geomspace(finite.min(), finite.max(), 36)
    else:
        bins = 30

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    hist_ax, ecdf_ax = axes
    for label in labels:
        values = np.asarray(
            [
                row["mean_normalized_wasserstein"]
                for row in rows
                if row["sampler"] == label
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        color = colors.get(label, "#374151")
        median = float(np.median(values))
        q90 = float(np.quantile(values, 0.90))
        plot_label = f"{label}: median {median:.3f}, q90 {q90:.3f}"
        hist_ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=plot_label,
        )
        hist_ax.axvline(median, color=color, linewidth=1.5, alpha=0.75)
        x, y = ecdf(values)
        ecdf_ax.step(x, y, where="post", color=color, linewidth=2.2, label=plot_label)
        ecdf_ax.axvline(median, color=color, linewidth=1.5, alpha=0.75)

    for ax in axes:
        ax.set_xlabel("mean normalized Wasserstein to grid60 posterior")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        if use_log_x:
            ax.set_xscale("log")
    hist_ax.set_ylabel("density")
    hist_ax.set_title("Distribution")
    ecdf_ax.set_ylabel("empirical CDF")
    ecdf_ax.set_title("ECDF")
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare decay NPE and MCMC Wasserstein distances to a grid posterior.",
    )
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--models", type=parse_model_list, default=["mdn"])
    parser.add_argument("--observation-indices", type=parse_int_list, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=24_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=6_000)
    parser.add_argument("--mcmc-proposal-scale", type=parse_proposal_scale, default=(0.030, 0.030, 0.040))
    parser.add_argument("--mcmc-seed", type=int, default=20261001)
    parser.add_argument("--grid-chunk-size", type=int, default=120_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    results_dir = args.output_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    panel = json.loads(args.panel_summary.read_text(encoding="utf-8"))
    grid_sizes = sorted(
        {
            int(observation["grid_reference"]["grid_size"])
            for observation in panel["observations"]
        }
    )
    if grid_sizes != [60]:
        raise ValueError(f"Expected a grid60 panel, found grid sizes: {grid_sizes}")

    indices = observation_indices(panel, args.observation_indices)
    selected_indices = set(indices)
    mcmc_csv = results_dir / "mcmc_wasserstein_rows.csv"
    mcmc_rows = run_mcmc_rows(
        panel=panel,
        selected_indices=indices,
        output_csv=mcmc_csv,
        device_name=args.device,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        burn_in=args.mcmc_burn_in,
        proposal_scale=args.mcmc_proposal_scale,
        grid_chunk_size=args.grid_chunk_size,
        seed=args.mcmc_seed,
        resume=args.resume,
    )
    npe_rows = collect_npe_rows(
        panel=panel,
        models=args.models,
        selected_indices=selected_indices,
    )
    rows = npe_rows + mcmc_rows

    combined_csv = results_dir / "npe_mcmc_wasserstein_rows.csv"
    fields = [
        "observation_index",
        "sampler",
        "mean_normalized_wasserstein",
        "runtime_seconds",
        "acceptance_rate",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "convergence_ok",
    ]
    with combined_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    figure_path = figure_dir / "npe_mcmc_wasserstein_distribution_grid60.png"
    title = (
        "Decay posterior distance to grid60 reference "
        f"(n={len(indices)}, MCMC {args.mcmc_chains}x{args.mcmc_steps})"
    )
    plot_distribution(rows=rows, output_path=figure_path, title=title)

    summary = {
        label: {
            "mean_normalized_wasserstein": quantile_summary(
                np.asarray(
                    [
                        row["mean_normalized_wasserstein"]
                        for row in rows
                        if row["sampler"] == label
                    ],
                    dtype=np.float64,
                )
            ),
            "runtime_seconds": quantile_summary(
                np.asarray(
                    [
                        row.get("runtime_seconds", np.nan)
                        for row in rows
                        if row["sampler"] == label
                    ],
                    dtype=np.float64,
                )
            ),
        }
        for label in sorted({str(row["sampler"]) for row in rows})
    }
    mcmc_convergence_flags = [
        bool(row.get("convergence_ok", False))
        for row in mcmc_rows
    ]
    output = {
        "panel_summary": args.panel_summary,
        "models": args.models,
        "grid_size": 60,
        "observation_indices": indices,
        "mcmc_config": {
            "chains": args.mcmc_chains,
            "steps": args.mcmc_steps,
            "burn_in": args.mcmc_burn_in,
            "proposal_scale": args.mcmc_proposal_scale,
            "seed": args.mcmc_seed,
            "device": args.device,
        },
        "mcmc_convergence_ok_fraction": float(np.mean(mcmc_convergence_flags))
        if mcmc_convergence_flags
        else None,
        "summary": summary,
        "outputs": {
            "figure": figure_path,
            "mcmc_csv": mcmc_csv,
            "combined_csv": combined_csv,
            "summary_json": results_dir / "npe_mcmc_wasserstein_distribution_summary.json",
        },
    }
    summary_json = Path(output["outputs"]["summary_json"])
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"figure: {figure_path}")
    print(f"summary_json: {summary_json}")
    for label, label_summary in summary.items():
        w = label_summary["mean_normalized_wasserstein"]
        print(
            f"{label}: W median={w['median']:.4f}, "
            f"q90={w['q90']:.4f}, q95={w['q95']:.4f}, max={w['max']:.4f}, n={w['n']}"
        )
    if mcmc_convergence_flags:
        print(
            "MCMC convergence_ok_fraction="
            f"{float(np.mean(mcmc_convergence_flags)):.3f}"
        )


if __name__ == "__main__":
    main()
