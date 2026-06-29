from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from compare_decay_samplers import PARAMETER_NAMES
from npe_metric_noise_floor_probe import (
    build_reference_cache,
    compare_samples_to_reference_fast,
)
from npe_flow_decay import mean_normalized_wasserstein_value

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
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300")
DEFAULT_STAGE1_SUMMARIES = (
    Path("runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/results/npe_stage1_summary.json"),
    Path("runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/results/npe_stage1_summary.json"),
)
FLOW_SEARCH_ROOT = Path("runs/01_exponential_decay/03_npe_flow_search")


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


def parse_path_list(value: str) -> tuple[Path, ...]:
    return tuple(Path(piece.strip()) for piece in value.split(",") if piece.strip())


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


def old_mean_w(metrics: dict[str, object] | None) -> float | None:
    if not metrics:
        return None
    return mean_normalized_wasserstein_value(metrics)


def summarize_samples(samples: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        values = samples[:, index]
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


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_flow_config(summary_path: Path, sample_path: Path) -> dict[str, object]:
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = dict(summary.get("config", {}))
        return {
            "summary": summary,
            "config": config,
            "training": summary.get("training", {}),
            "old_wasserstein": old_mean_w(summary.get("faithfulness_to_grid_reference")),
            "sample_path": Path(summary["outputs"]["samples_npz"]),
        }
    model_path = sample_path.with_name("npe_flow_decay_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return {
        "summary": None,
        "config": dict(checkpoint.get("config", {})),
        "training": {},
        "old_wasserstein": None,
        "sample_path": sample_path,
    }


def include_local_flow(config: dict[str, object]) -> bool:
    return (
        config.get("training_mode") == "local_prior"
        and float(config.get("local_quantile", -1.0)) == 0.005
        and bool(config.get("linear_target_adjustment", False))
        and config.get("context_kind", "indirect") == "indirect"
    )


def include_broad_flow(config: dict[str, object]) -> bool:
    return config.get("training_mode", "weighted_proposal") != "local_prior"


def flow_label(config: dict[str, object], run_dir: Path) -> str:
    if config.get("training_mode") == "local_prior":
        transforms = config.get("transforms")
        hidden = config.get("hidden_features")
        seed = config.get("seed")
        return f"local q=0.005 t{transforms} seed {seed}"
    mix = config.get("proposal_prior_mixture", 0.0)
    transforms = config.get("transforms")
    return f"weighted proposal mix={mix} t{transforms} ({run_dir.name.split('_', 1)[0]})"


def collect_flow_rows(
    *,
    reference: dict[str, object],
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    flow_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_path in sorted(flow_root.glob("*/results/npe_flow_decay_samples.npz")):
        run_dir = sample_path.parents[1]
        summary_path = sample_path.with_name("npe_flow_decay_summary.json")
        loaded = read_flow_config(summary_path, sample_path)
        config = loaded["config"]
        group = None
        if include_local_flow(config):
            group = "local_flow"
        elif include_broad_flow(config):
            group = "broad_weighted_flow"
        if group is None:
            continue

        samples_npz = np.load(sample_path, allow_pickle=True)
        theta_samples = np.asarray(samples_npz["theta_samples"], dtype=np.float64)
        metrics = compare_samples_to_reference_fast(theta_samples, reference, reference_cache)
        training = loaded["training"]
        z_std = None
        if "z_std" in samples_npz.files:
            z_std = np.asarray(samples_npz["z_std"], dtype=np.float64)
        z_log_det = float(np.log(z_std).sum()) if z_std is not None else None
        best_val_nll = training.get("best_val_nll")
        best_val_nll_target_z = (
            float(best_val_nll + z_log_det)
            if best_val_nll is not None and z_log_det is not None
            else None
        )
        rows.append({
            "source": "npe_flow_decay",
            "group": group,
            "run": run_dir.name,
            "label": flow_label(config, run_dir),
            "train_simulations": int(config.get("train_simulations", -1)),
            "val_simulations": int(config.get("val_simulations", -1)),
            "posterior_samples": int(theta_samples.shape[0]),
            "sample_path": str(sample_path),
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "old_grid_size": config.get("reference_grid_size"),
            "old_wasserstein": loaded["old_wasserstein"],
            "corrected_grid_size": int(reference["grid_size"]),
            "corrected_wasserstein": mean_normalized_wasserstein_value(metrics),
            "best_val_nll": float(best_val_nll) if best_val_nll is not None else None,
            "best_val_nll_target_z": best_val_nll_target_z,
            "training_seconds": training.get("training_seconds"),
            "transforms": config.get("transforms"),
            "hidden_features": json.dumps(json_ready(config.get("hidden_features"))),
            "local_quantile": config.get("local_quantile"),
            "linear_target_adjustment": config.get("linear_target_adjustment"),
            "proposal_prior_mixture": config.get("proposal_prior_mixture", 0.0),
            "sample_summary": summarize_samples(theta_samples),
            "corrected_metrics": metrics,
        })
    return sorted(rows, key=lambda row: (str(row["group"]), int(row["train_simulations"]), str(row["run"])))


def collect_stage1_rows(
    *,
    reference: dict[str, object],
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    summary_paths: tuple[Path, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary["config"]
        samples_npz = np.load(summary["samples_npz"], allow_pickle=True)
        for family in config["families"]:
            theta_key = f"theta_samples_{family}"
            if theta_key not in samples_npz.files:
                continue
            theta_samples = np.asarray(samples_npz[theta_key], dtype=np.float64)
            metrics = compare_samples_to_reference_fast(theta_samples, reference, reference_cache)
            result = summary["results"][family]
            rows.append({
                "source": "npe_stage1_decay",
                "group": "broad_stage1",
                "run": summary_path.parents[1].name,
                "label": result.get("label", family),
                "family": family,
                "train_simulations": int(config["train_simulations"]),
                "val_simulations": int(config["val_simulations"]),
                "posterior_samples": int(theta_samples.shape[0]),
                "sample_path": str(summary["samples_npz"]),
                "summary_path": str(summary_path),
                "old_grid_size": config.get("reference_grid_size"),
                "old_wasserstein": old_mean_w(result.get("faithfulness_to_grid_reference")),
                "corrected_grid_size": int(reference["grid_size"]),
                "corrected_wasserstein": mean_normalized_wasserstein_value(metrics),
                "best_val_nll": float(result["best_val_nll"]),
                "best_val_nll_target_z": None,
                "training_seconds": result.get("training_seconds"),
                "hidden_dim": config.get("hidden_dim"),
                "hidden_layers": config.get("hidden_layers"),
                "corrected_metrics": metrics,
            })
    return sorted(rows, key=lambda row: (str(row["family"]), int(row["train_simulations"])))


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    return {
        key: json.dumps(json_ready(value)) if isinstance(value, dict) else value
        for key, value in row.items()
    }


def plot_stage1(rows: list[dict[str, object]], path: Path, target_wasserstein: float | None) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    families = sorted({str(row["family"]) for row in rows})
    colors = {
        "diag_gaussian": "#2f6fbb",
        "full_gaussian": "#2f855a",
        "mdn": "#b85c38",
        "affine_flow": "#7a5cc2",
    }
    for family in families:
        family_rows = sorted([row for row in rows if row["family"] == family], key=lambda row: int(row["train_simulations"]))
        x = np.asarray([row["train_simulations"] for row in family_rows], dtype=np.float64)
        w = np.asarray([row["corrected_wasserstein"] for row in family_rows], dtype=np.float64)
        nll = np.asarray([row["best_val_nll"] for row in family_rows], dtype=np.float64)
        label = str(family_rows[-1]["label"])
        color = colors.get(family, None)
        axes[0].plot(x, w, marker="o", linewidth=2.0, label=label, color=color)
        axes[1].plot(x, nll, marker="o", linewidth=2.0, label=label, color=color)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    if target_wasserstein is not None:
        axes[0].axhline(
            target_wasserstein,
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=f"grid-300 target {target_wasserstein:.3f}",
        )
    axes[0].set_xlabel("prior-predictive train simulations")
    axes[1].set_xlabel("prior-predictive train simulations")
    axes[0].set_ylabel("x0 mean normalized Wasserstein to grid-300")
    axes[1].set_ylabel("best validation NLL")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    figure.suptitle("Broad Stage-1 NPE x0 scaling corrected to grid-300", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def median_summary(rows: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["group"] == group]
    output = []
    for n in sorted({int(row["train_simulations"]) for row in selected}):
        group_rows = [row for row in selected if int(row["train_simulations"]) == n]
        w = np.asarray([row["corrected_wasserstein"] for row in group_rows], dtype=np.float64)
        nll_values = [row["best_val_nll_target_z"] for row in group_rows if row.get("best_val_nll_target_z") is not None]
        nll = np.asarray(nll_values, dtype=np.float64)
        output.append({
            "group": group,
            "train_simulations": n,
            "run_count": len(group_rows),
            "corrected_wasserstein_median": float(np.median(w)),
            "corrected_wasserstein_min": float(np.min(w)),
            "corrected_wasserstein_max": float(np.max(w)),
            "best_val_nll_target_z_median": float(np.median(nll)) if nll.size else None,
            "best_val_nll_target_z_min": float(np.min(nll)) if nll.size else None,
            "best_val_nll_target_z_max": float(np.max(nll)) if nll.size else None,
        })
    return output


def plot_flow(rows: list[dict[str, object]], path: Path, target_wasserstein: float | None) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    colors = {"local_flow": "#2f6fbb", "broad_weighted_flow": "#b85c38"}
    labels = {
        "local_flow": "local q=0.005 linear flow",
        "broad_weighted_flow": "weighted-proposal broad flow",
    }
    for group in ("local_flow", "broad_weighted_flow"):
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        x = np.asarray([row["train_simulations"] for row in group_rows], dtype=np.float64)
        w = np.asarray([row["corrected_wasserstein"] for row in group_rows], dtype=np.float64)
        axes[0].scatter(x, w, s=42, alpha=0.58, color=colors[group], label=f"{labels[group]} runs")
        summary = median_summary(rows, group)
        sx = np.asarray([row["train_simulations"] for row in summary], dtype=np.float64)
        sw = np.asarray([row["corrected_wasserstein_median"] for row in summary], dtype=np.float64)
        axes[0].plot(sx, sw, marker="o", linewidth=2.1, color=colors[group], label=f"{labels[group]} median")
        nll_rows = [row for row in group_rows if row.get("best_val_nll_target_z") is not None]
        if nll_rows:
            axes[1].scatter(
                [row["train_simulations"] for row in nll_rows],
                [row["best_val_nll_target_z"] for row in nll_rows],
                s=42,
                alpha=0.58,
                color=colors[group],
                label=f"{labels[group]} runs",
            )
            nll_summary = [
                row for row in summary if row.get("best_val_nll_target_z_median") is not None
            ]
            axes[1].plot(
                [row["train_simulations"] for row in nll_summary],
                [row["best_val_nll_target_z_median"] for row in nll_summary],
                marker="o",
                linewidth=2.1,
                color=colors[group],
                label=f"{labels[group]} median",
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    if target_wasserstein is not None:
        axes[0].axhline(
            target_wasserstein,
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=f"grid-300 target {target_wasserstein:.3f}",
        )
    axes[0].set_xlabel("train simulations")
    axes[1].set_xlabel("train simulations")
    axes[0].set_ylabel("x0 mean normalized Wasserstein to grid-300")
    axes[1].set_ylabel("best validation NLL in target-z units")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    figure.suptitle("Flow NPE x0 scaling corrected to grid-300", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate x0 broad/local NPE scaling plots against a cached grid reference.",
    )
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--reference-metadata", type=Path, default=DEFAULT_REFERENCE_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-summaries", type=parse_path_list, default=DEFAULT_STAGE1_SUMMARIES)
    parser.add_argument("--flow-search-root", type=Path, default=FLOW_SEARCH_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference_npz, args.reference_metadata)
    reference_cache = build_reference_cache(reference)

    stage1_rows = collect_stage1_rows(
        reference=reference,
        reference_cache=reference_cache,
        summary_paths=tuple(args.stage1_summaries),
    )
    flow_rows = collect_flow_rows(
        reference=reference,
        reference_cache=reference_cache,
        flow_root=args.flow_search_root,
    )
    flow_summary = (
        median_summary(flow_rows, "local_flow")
        + median_summary(flow_rows, "broad_weighted_flow")
    )

    stage1_csv = results_dir / "corrected_stage1_broad_x0_rows.csv"
    flow_csv = results_dir / "corrected_flow_x0_rows.csv"
    flow_summary_csv = results_dir / "corrected_flow_x0_summary.csv"
    summary_json = results_dir / "corrected_npe_x0_scaling_summary.json"
    stage1_png = figures_dir / "corrected_stage1_broad_x0_scaling.png"
    flow_png = figures_dir / "corrected_flow_x0_scaling.png"
    target_wasserstein = reference["metadata"].get("recommended_target")
    plot_stage1(
        stage1_rows,
        stage1_png,
        float(target_wasserstein) if target_wasserstein is not None else None,
    )
    plot_flow(
        flow_rows,
        flow_png,
        float(target_wasserstein) if target_wasserstein is not None else None,
    )
    write_csv([flatten_row(row) for row in stage1_rows], stage1_csv)
    write_csv([flatten_row(row) for row in flow_rows], flow_csv)
    write_csv(flow_summary, flow_summary_csv)

    output = {
        "reference": {
            "npz": args.reference_npz,
            "metadata": args.reference_metadata,
            "grid_size": int(reference["grid_size"]),
            "grid_points": int(reference["grid_points"]),
            "recommended_target": reference["metadata"].get("recommended_target"),
        },
        "stage1_rows": stage1_rows,
        "flow_rows": flow_rows,
        "flow_summary": flow_summary,
        "limitations": [
            "The controlled local scaling sweep did not save checkpoints or posterior samples, so this script corrects saved historical flow-search runs instead.",
            "All Wasserstein values are for the single original x0 observation.",
        ],
        "outputs": {
            "stage1_rows_csv": stage1_csv,
            "flow_rows_csv": flow_csv,
            "flow_summary_csv": flow_summary_csv,
            "summary_json": summary_json,
            "stage1_figure": stage1_png,
            "flow_figure": flow_png,
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"summary_json: {summary_json}")
    print(f"stage1_figure: {stage1_png}")
    print(f"flow_figure: {flow_png}")
    print(f"stage1_rows: {len(stage1_rows)}")
    print(f"flow_rows: {len(flow_rows)}")


if __name__ == "__main__":
    main()
