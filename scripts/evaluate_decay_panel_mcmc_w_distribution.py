from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_broad_panel_w_distribution import (
    compare_samples_to_marginals_detailed,
    load_panel_marginal_cache,
    quantile_summary,
)
from mcmc_decay_inference import (
    MCMCConfig,
    arviz_diagnostics,
    choose_device,
    convergence_flags,
    run_random_walk_metropolis,
)


DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel500_grid180_refined_marginals.npz"
)
DEFAULT_OUTPUT = (
    Path("runs/01_exponential_decay/15_broad_scaling/")
    / "200_panel_w_distribution_eval_mdn512k_spline4m_flow2_panel500/results/"
    "panel_w_mcmc_rows.csv"
)


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated proposal scales.")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("Proposal scales must be positive.")
    return pieces[0], pieces[1], pieces[2]


def parse_index_list(value: str) -> list[int]:
    indices = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("Expected at least one index.")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("Indices must be non-negative.")
    return sorted(set(indices))


def load_cached_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            rows[index] = {
                "model": row["model"],
                "model_label": row["model_label"],
                "index": index,
                "label": row["label"],
                "A": float(row["A"]),
                "k": float(row["k"]),
                "sigma": float(row["sigma"]),
                "target_wasserstein": float(row["target_wasserstein"]),
                "wasserstein": float(row["wasserstein"]),
                "target_ratio": float(row["target_ratio"]),
                "w_A": float(row["w_A"]),
                "w_k": float(row["w_k"]),
                "w_sigma": float(row["w_sigma"]),
                "seconds": float(row["seconds"]),
                "acceptance_rate": float(row["acceptance_rate"]),
                "max_rhat": float(row["max_rhat"]),
                "min_bulk_ess": float(row["min_bulk_ess"]),
                "min_tail_ess": float(row["min_tail_ess"]),
                "convergence_ok": row["convergence_ok"].lower() == "true",
            }
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "model",
        "model_label",
        "index",
        "label",
        "A",
        "k",
        "sigma",
        "target_wasserstein",
        "wasserstein",
        "target_ratio",
        "w_A",
        "w_k",
        "w_sigma",
        "seconds",
        "acceptance_rate",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "convergence_ok",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["index"])):
            writer.writerow({field: row.get(field) for field in fields})


def evaluate_mcmc_panel(args: argparse.Namespace) -> list[dict[str, object]]:
    panel, _ = load_panel_marginal_cache(args.panel_cache)
    x_panel = np.asarray(panel["x_panel"], dtype=np.float64)
    theta_panel = np.asarray(panel["theta_panel"], dtype=np.float64)
    theta_axes = np.asarray(panel["theta_axes"], dtype=np.float64)
    marginal_weights = np.asarray(panel["marginal_weights"], dtype=np.float64)
    targets = np.asarray(panel["target_wasserstein"], dtype=np.float64)
    labels = list(panel["labels"])

    indices = args.indices if args.indices is not None else list(range(x_panel.shape[0]))
    missing = sorted(set(indices) - set(range(x_panel.shape[0])))
    if missing:
        raise ValueError(f"Requested panel indices are out of range: {missing}")

    rows_by_index = load_cached_rows(args.output) if args.resume else {}
    rows = list(rows_by_index.values())

    device, dtype = choose_device(args.device)
    t = torch.as_tensor(np.linspace(0.0, 6.0, x_panel.shape[1]), dtype=torch.float64)
    pending = [index for index in indices if index not in rows_by_index]
    if not pending:
        return sorted(rows, key=lambda item: int(item["index"]))

    started = time.perf_counter()
    for count, index in enumerate(pending, start=1):
        signal_start = time.perf_counter()
        y = torch.as_tensor(x_panel[index], dtype=torch.float64)
        config = MCMCConfig(
            chains=args.chains,
            steps=args.steps,
            burn_in=args.burn_in,
            seed=args.seed + int(index),
            proposal_scale=args.proposal_scale,
            requested_device=args.device,
            sampler_variant="low-overhead",
        )
        _, theta_samples, accepted, runtime_seconds = run_random_walk_metropolis(
            t=t,
            y=y,
            config=config,
            device=device,
            dtype=dtype,
        )
        posterior = theta_samples[:, args.burn_in :, :].reshape(-1, 3)
        w_value, per_axis = compare_samples_to_marginals_detailed(
            theta_samples=posterior,
            theta_axes=theta_axes[index],
            marginal_weights=marginal_weights[index],
        )
        diagnostics = arviz_diagnostics(theta_samples, args.burn_in)
        flags = convergence_flags(diagnostics)
        target = float(targets[index])
        row = {
            "model": "mcmc",
            "model_label": "Random-walk MCMC",
            "index": int(index),
            "label": str(labels[index]),
            "A": float(theta_panel[index, 0]),
            "k": float(theta_panel[index, 1]),
            "sigma": float(theta_panel[index, 2]),
            "target_wasserstein": target,
            "wasserstein": w_value,
            "target_ratio": float(w_value / target) if target > 0 else float("nan"),
            "seconds": float(runtime_seconds),
            "acceptance_rate": float(accepted.mean()),
            "max_rhat": float(max(item["rhat"] for item in diagnostics.values())),
            "min_bulk_ess": float(min(item["ess_bulk"] for item in diagnostics.values())),
            "min_tail_ess": float(min(item["ess_tail"] for item in diagnostics.values())),
            "convergence_ok": bool(all(flags.values())),
        }
        row.update(per_axis)
        rows.append(row)
        write_rows(args.output, rows)

        if count == 1 or count == len(pending) or count % max(args.print_every, 1) == 0:
            elapsed = time.perf_counter() - started
            rate = elapsed / count
            eta = rate * (len(pending) - count)
            print(
                f"MCMC panel [{count}/{len(pending)}] index={index} "
                f"W={w_value:.5f} ratio={row['target_ratio']:.1f} "
                f"accept={row['acceptance_rate']:.3f} "
                f"seconds={time.perf_counter() - signal_start:.2f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return sorted(rows, key=lambda item: int(item["index"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate random-walk MCMC Wasserstein distances on the decay 500-signal panel.",
    )
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices", type=parse_index_list, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24_000)
    parser.add_argument("--burn-in", type=int, default=6_000)
    parser.add_argument("--proposal-scale", type=parse_proposal_scale, default=(0.030, 0.030, 0.040))
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_mcmc_panel(args)
    values = np.asarray([row["wasserstein"] for row in rows], dtype=np.float64)
    ratios = np.asarray([row["target_ratio"] for row in rows], dtype=np.float64)
    ok = np.asarray([bool(row["convergence_ok"]) for row in rows], dtype=bool)
    print(f"output: {args.output}")
    print(f"W summary: {quantile_summary(values)}")
    print(f"target-ratio summary: {quantile_summary(ratios)}")
    print(f"convergence_ok_fraction: {float(np.mean(ok)):.3f}")


if __name__ == "__main__":
    main()
