from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from npe_flow_decay import mean_normalized_wasserstein_value
from npe_metric_noise_floor_probe import (
    build_reference_cache,
    compare_samples_to_reference_fast,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_REFERENCE_NPZ = Path(
    "runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/"
    "decay_x0_grid300_reference.npz"
)
DEFAULT_REFERENCE_METADATA = Path(
    "runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/"
    "decay_x0_grid300_reference_metadata.json"
)
DEFAULT_SCALING_ROOT = Path(
    "runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples"
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


def load_reference(npz_path: Path, metadata_path: Path) -> dict[str, object]:
    arrays = np.load(npz_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "grid_size": int(metadata["grid_size"]),
        "grid_points": int(metadata["grid_points"]),
        "theta_grid": np.asarray(arrays["theta_grid"], dtype=np.float64),
        "weights": np.asarray(arrays["weights"], dtype=np.float64),
        "summary": metadata["summary"],
        "z_ranges": metadata["z_ranges"],
        "edge_mass": metadata["edge_mass"],
        "metadata": metadata,
    }


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q16": float(np.quantile(finite, 0.16)),
        "median": float(np.median(finite)),
        "q84": float(np.quantile(finite, 0.84)),
        "max": float(np.max(finite)),
    }


def flatten_summary(summary: dict[str, object], prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_summary(value, f"{name}."))
        else:
            output[name] = value
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_from_run(
    *,
    summary_path: Path,
    reference: dict[str, object],
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    target_wasserstein: float,
) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples_path = summary_path.with_name("local_scaling_samples.npz")
    if not samples_path.exists():
        raise FileNotFoundError(
            f"{samples_path} is missing. Rerun the sweep with --save-samples."
        )
    samples_npz = np.load(samples_path, allow_pickle=False)
    theta_samples = np.asarray(samples_npz["theta_samples"], dtype=np.float64)
    metrics = compare_samples_to_reference_fast(theta_samples, reference, reference_cache)
    corrected_w = mean_normalized_wasserstein_value(metrics)
    old_w = mean_normalized_wasserstein_value(summary["faithfulness_to_grid_reference"])
    sample_bytes = samples_path.stat().st_size
    return {
        "summary_json": str(summary_path),
        "samples_npz": str(samples_path),
        "samples_npz_bytes": int(sample_bytes),
        "samples_npz_mib": sample_bytes / (1024**2),
        "seed": int(summary["seed"]),
        "train_simulations": int(summary["train_simulations"]),
        "val_simulations": int(summary["val_simulations"]),
        "posterior_samples": int(theta_samples.shape[0]),
        "old_grid_size": int(summary["config"]["reference_grid_size"]),
        "old_wasserstein": float(old_w),
        "corrected_grid_size": int(reference["grid_size"]),
        "corrected_wasserstein": float(corrected_w),
        "corrected_target_wasserstein": float(target_wasserstein),
        "corrected_target_ratio": float(corrected_w / target_wasserstein),
        "old_target_wasserstein": float(summary["target_wasserstein"]),
        "old_target_ratio": float(summary["target_ratio"]),
        "best_val_nll": float(summary["training"]["best_val_nll"]),
        "best_val_nll_target_z": float(summary["training"]["best_val_nll_target_z"]),
        "final_val_nll_target_z": float(summary["training"]["final_val_nll_target_z"]),
        "epochs_completed": int(summary["training"]["epochs_completed"]),
        "best_epoch": int(summary["training"]["best_epoch"]),
        "training_seconds": float(summary["training"]["training_seconds"]),
        "total_train_eval_seconds": float(summary["training"]["total_train_eval_seconds"]),
        "corrected_metrics": metrics,
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for train_count in sorted({int(row["train_simulations"]) for row in rows}):
        group = [row for row in rows if int(row["train_simulations"]) == train_count]
        output.append({
            "train_simulations": train_count,
            "seed_count": len(group),
            "corrected_wasserstein": quantile_summary(
                np.asarray([row["corrected_wasserstein"] for row in group], dtype=np.float64)
            ),
            "corrected_target_ratio": quantile_summary(
                np.asarray([row["corrected_target_ratio"] for row in group], dtype=np.float64)
            ),
            "old_wasserstein": quantile_summary(
                np.asarray([row["old_wasserstein"] for row in group], dtype=np.float64)
            ),
            "best_val_nll_target_z": quantile_summary(
                np.asarray([row["best_val_nll_target_z"] for row in group], dtype=np.float64)
            ),
        })
    return output


def plot_corrected_scaling(
    *,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    target_wasserstein: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    x_all = np.asarray([row["train_simulations"] for row in rows], dtype=np.float64)

    panels = [
        ("corrected_wasserstein", "x0 W to grid-300", axes[0], "#2f6fbb", True),
        ("corrected_target_ratio", "grid-300 target ratio", axes[1], "#b85c38", True),
        ("best_val_nll_target_z", "best validation NLL in target-z units", axes[2], "#2f855a", False),
    ]
    for metric, ylabel, ax, color, log_y in panels:
        y_all = np.asarray([row[metric] for row in rows], dtype=np.float64)
        ax.scatter(x_all, y_all, color=color, alpha=0.45, s=30, label="seed")
        x_summary = np.asarray([row["train_simulations"] for row in summary_rows], dtype=np.float64)
        summary_key = metric
        median = np.asarray([row[summary_key]["median"] for row in summary_rows], dtype=np.float64)
        q16 = np.asarray([row[summary_key]["q16"] for row in summary_rows], dtype=np.float64)
        q84 = np.asarray([row[summary_key]["q84"] for row in summary_rows], dtype=np.float64)
        ax.plot(x_summary, median, color=color, linewidth=2.1, marker="o", label="median")
        ax.fill_between(x_summary, q16, q84, color=color, alpha=0.16, label="q16-q84")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("accepted local training simulations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)

    axes[0].axhline(target_wasserstein, color="#172033", linestyle="--", linewidth=1.35)
    axes[1].axhline(1.0, color="#172033", linestyle="--", linewidth=1.35)
    figure.suptitle("Controlled local NPE scaling corrected to x0 grid-300", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore saved local scaling posterior samples against a cached grid reference.",
    )
    parser.add_argument("--scaling-root", type=Path, default=DEFAULT_SCALING_ROOT)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--reference-metadata", type=Path, default=DEFAULT_REFERENCE_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.scaling_root / "results"
    figures_dir = args.scaling_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference_npz, args.reference_metadata)
    reference_cache = build_reference_cache(reference)
    target_wasserstein = float(reference["metadata"]["recommended_target"])

    summary_paths = sorted((args.scaling_root / "runs").glob("*/results/local_scaling_run_summary.json"))
    rows = [
        row_from_run(
            summary_path=path,
            reference=reference,
            reference_cache=reference_cache,
            target_wasserstein=target_wasserstein,
        )
        for path in summary_paths
    ]
    rows = sorted(rows, key=lambda row: (int(row["train_simulations"]), int(row["seed"])))
    summary_rows = summarize_rows(rows)

    rows_csv = results_dir / "local_data_scaling_grid300_rows.csv"
    summary_csv = results_dir / "local_data_scaling_grid300_summary.csv"
    summary_json = results_dir / "local_data_scaling_grid300_summary.json"
    figure_path = figures_dir / "local_data_scaling_grid300.png"

    write_csv(
        [
            {
                key: json.dumps(json_ready(value)) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            for row in rows
        ],
        rows_csv,
    )
    write_csv([flatten_summary(row) for row in summary_rows], summary_csv)
    plot_corrected_scaling(
        rows=rows,
        summary_rows=summary_rows,
        target_wasserstein=target_wasserstein,
        output_path=figure_path,
    )

    total_sample_bytes = sum(int(row["samples_npz_bytes"]) for row in rows)
    output = {
        "reference": {
            "npz": args.reference_npz,
            "metadata": args.reference_metadata,
            "grid_size": int(reference["grid_size"]),
            "grid_points": int(reference["grid_points"]),
            "target_wasserstein": target_wasserstein,
        },
        "storage": {
            "sample_file_count": len(rows),
            "sample_files_bytes": total_sample_bytes,
            "sample_files_mib": total_sample_bytes / (1024**2),
        },
        "rows": rows,
        "scale_summary": summary_rows,
        "outputs": {
            "rows_csv": rows_csv,
            "summary_csv": summary_csv,
            "summary_json": summary_json,
            "figure": figure_path,
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")
    print(f"rows: {len(rows)}")
    print(f"sample_files_mib: {total_sample_bytes / (1024**2):.1f}")


if __name__ == "__main__":
    main()
