from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import wasserstein_distance

import npe_stage1_decay as stage1
from mcmc_decay_inference import PARAMETER_NAMES
from npe_posterior_viewer import load_stage1_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SPLINE_MODEL = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "34_mini_fixed_p_4m_diagnostic/spline/"
    "runs/n4096000_seed20260901/results/spline_flow_model.pt"
)
DEFAULT_MDN_MODEL = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "09_ui_best_broad_mdn_512k_seed20260902/"
    "runs/n512000_seed20260902/results/mdn_model.pt"
)
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel64_grid180_refined_marginals.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "38_panel_w_distribution_eval_mdn512k_spline4m"
)


MODEL_COLORS = {
    "spline": "#c45a2d",
    "mdn": "#6d4aff",
}
MODEL_LABELS = {
    "spline": "Broad spline 4.096M",
    "mdn": "Broad MDN 512k",
}


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


def load_panel_marginal_cache(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    arrays = np.load(path, allow_pickle=False)
    panel = {
        "x_panel": np.asarray(arrays["x_panel"], dtype=np.float64),
        "z_panel": np.asarray(arrays["z_panel"], dtype=np.float64),
        "theta_panel": np.asarray(arrays["theta_panel"], dtype=np.float64),
        "theta_axes": np.asarray(arrays["theta_axes"], dtype=np.float64),
        "marginal_weights": np.asarray(arrays["marginal_weights"], dtype=np.float64),
        "target_wasserstein": np.asarray(arrays["target_wasserstein"], dtype=np.float64),
        "labels": np.asarray(arrays["labels"]).astype(str).tolist(),
    }
    metadata_path = path.with_suffix(".json")
    metadata: dict[str, object] = {
        "path": str(path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "panel_size": int(panel["x_panel"].shape[0]),
        "grid_size": int(panel["theta_axes"].shape[-1]),
        "file_bytes": int(path.stat().st_size),
        "file_mib": path.stat().st_size / (1024**2),
    }
    if metadata_path.exists():
        metadata["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return panel, metadata


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
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    return finite, np.arange(1, finite.size + 1, dtype=np.float64) / finite.size


def seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def compare_samples_to_marginals_detailed(
    *,
    theta_samples: np.ndarray,
    theta_axes: np.ndarray,
    marginal_weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    values = []
    per_axis = {}
    for axis, name in enumerate(PARAMETER_NAMES):
        ref_axis = theta_axes[axis]
        ref_weights = marginal_weights[axis] / np.sum(marginal_weights[axis])
        mean = float(np.sum(ref_axis * ref_weights))
        sd = float(np.sqrt(max(np.sum((ref_axis - mean) ** 2 * ref_weights), 0.0)))
        w = wasserstein_distance(theta_samples[:, axis], ref_axis, v_weights=ref_weights)
        normalized = float(w / max(sd, 1e-12))
        values.append(normalized)
        per_axis[f"w_{name}"] = normalized
    return float(np.mean(values)), per_axis


def evaluate_model(
    *,
    model_key: str,
    model: torch.nn.Module,
    state: dict[str, object],
    panel: dict[str, object],
    posterior_samples: int,
    seed: int,
    device: torch.device,
    print_every: int,
) -> list[dict[str, object]]:
    x_panel = np.asarray(panel["x_panel"], dtype=np.float64)
    theta_panel = np.asarray(panel["theta_panel"], dtype=np.float64)
    theta_axes = np.asarray(panel["theta_axes"], dtype=np.float64)
    marginal_weights = np.asarray(panel["marginal_weights"], dtype=np.float64)
    targets = np.asarray(panel["target_wasserstein"], dtype=np.float64)
    labels = list(panel["labels"])
    rows: list[dict[str, object]] = []
    x_mean = np.asarray(state["x_mean"], dtype=np.float64)
    x_std = np.asarray(state["x_std"], dtype=np.float64)
    z_mean = np.asarray(state["z_mean"], dtype=np.float64)
    z_std = np.asarray(state["z_std"], dtype=np.float64)
    start = time.perf_counter()
    for index, label in enumerate(labels):
        signal_start = time.perf_counter()
        seed_torch(seed + index, device)
        _, theta_samples = stage1.sample_posterior_for_observation(
            model=model,
            observed_x=x_panel[index],
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            n=posterior_samples,
            device=device,
        )
        w_value, per_axis = compare_samples_to_marginals_detailed(
            theta_samples=theta_samples,
            theta_axes=theta_axes[index],
            marginal_weights=marginal_weights[index],
        )
        target = float(targets[index])
        row = {
            "model": model_key,
            "model_label": MODEL_LABELS[model_key],
            "index": int(index),
            "label": str(label),
            "A": float(theta_panel[index, 0]),
            "k": float(theta_panel[index, 1]),
            "sigma": float(theta_panel[index, 2]),
            "target_wasserstein": target,
            "wasserstein": w_value,
            "target_ratio": float(w_value / target) if target > 0 else float("nan"),
            "seconds": time.perf_counter() - signal_start,
        }
        row.update(per_axis)
        rows.append(row)
        if index == 0 or index + 1 == len(labels) or (index + 1) % max(print_every, 1) == 0:
            print(
                f"{MODEL_LABELS[model_key]} [{index + 1}/{len(labels)}] "
                f"W={w_value:.5f} ratio={row['target_ratio']:.1f} "
                f"seconds={row['seconds']:.2f}",
                flush=True,
            )
    print(
        f"{MODEL_LABELS[model_key]} total_seconds={time.perf_counter() - start:.1f}",
        flush=True,
    )
    return rows


def combine_rows(model_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    by_model = {
        model_key: {int(row["index"]): row for row in rows}
        for model_key, rows in model_rows.items()
    }
    common_indices = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    combined = []
    for index in common_indices:
        spline = by_model["spline"][index]
        mdn = by_model["mdn"][index]
        row = {
            "index": index,
            "label": spline["label"],
            "A": spline["A"],
            "k": spline["k"],
            "sigma": spline["sigma"],
            "target_wasserstein": spline["target_wasserstein"],
            "spline_wasserstein": spline["wasserstein"],
            "spline_target_ratio": spline["target_ratio"],
            "spline_w_A": spline["w_A"],
            "spline_w_k": spline["w_k"],
            "spline_w_sigma": spline["w_sigma"],
            "mdn_wasserstein": mdn["wasserstein"],
            "mdn_target_ratio": mdn["target_ratio"],
            "mdn_w_A": mdn["w_A"],
            "mdn_w_k": mdn["w_k"],
            "mdn_w_sigma": mdn["w_sigma"],
            "spline_minus_mdn": float(spline["wasserstein"] - mdn["wasserstein"]),
            "spline_over_mdn": (
                float(spline["wasserstein"] / mdn["wasserstein"])
                if float(mdn["wasserstein"]) > 0
                else float("nan")
            ),
            "better_model": "spline" if float(spline["wasserstein"]) < float(mdn["wasserstein"]) else "mdn",
        }
        combined.append(row)
    return combined


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(combined_rows: list[dict[str, object]], output_path: Path, *, posterior_samples: int) -> None:
    spline_w = np.asarray([row["spline_wasserstein"] for row in combined_rows], dtype=np.float64)
    mdn_w = np.asarray([row["mdn_wasserstein"] for row in combined_rows], dtype=np.float64)
    spline_ratio = np.asarray([row["spline_target_ratio"] for row in combined_rows], dtype=np.float64)
    mdn_ratio = np.asarray([row["mdn_target_ratio"] for row in combined_rows], dtype=np.float64)
    sigma = np.asarray([row["sigma"] for row in combined_rows], dtype=np.float64)
    indices = np.asarray([row["index"] for row in combined_rows], dtype=np.int64)
    all_w = np.concatenate([spline_w[np.isfinite(spline_w)], mdn_w[np.isfinite(mdn_w)]])
    all_ratio = np.concatenate([spline_ratio[np.isfinite(spline_ratio)], mdn_ratio[np.isfinite(mdn_ratio)]])
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    for values, key in [(spline_w, "spline"), (mdn_w, "mdn")]:
        x, y = ecdf(values)
        ax.step(x, y, where="post", lw=2.5, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
        ax.axvline(np.median(values), color=MODEL_COLORS[key], lw=1.6, alpha=0.68)
    if np.all(all_w > 0):
        ax.set_xscale("log")
    ax.set_xlabel("mean normalized marginal W")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Distribution across panel signals")
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    scatter = ax.scatter(
        mdn_w,
        spline_w,
        c=sigma,
        cmap="viridis",
        s=28,
        alpha=0.78,
        edgecolors="none",
    )
    lower = max(float(np.nanmin(all_w)) * 0.75, 1e-6)
    upper = float(np.nanmax(all_w)) * 1.25
    ax.plot([lower, upper], [lower, upper], color="#172033", lw=1.2, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Broad MDN 512k W")
    ax.set_ylabel("Broad spline 4.096M W")
    ax.set_title("Per-signal comparison")
    ax.grid(which="both", alpha=0.22)
    colorbar = figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("true sigma")
    worst_order = np.argsort(np.maximum(spline_w, mdn_w))[::-1][: min(6, len(indices))]
    for row_index in worst_order:
        ax.text(mdn_w[row_index], spline_w[row_index], str(int(indices[row_index])), fontsize=8)

    ax = axes[1, 0]
    if all_ratio.size and np.all(all_ratio > 0):
        bins = np.logspace(np.log10(np.nanmin(all_ratio) * 0.85), np.log10(np.nanmax(all_ratio) * 1.15), 24)
        ax.set_xscale("log")
    else:
        bins = 24
    ax.hist(spline_ratio, bins=bins, alpha=0.44, color=MODEL_COLORS["spline"], label=MODEL_LABELS["spline"])
    ax.hist(mdn_ratio, bins=bins, alpha=0.36, color=MODEL_COLORS["mdn"], label=MODEL_LABELS["mdn"])
    ax.axvline(np.median(spline_ratio), color=MODEL_COLORS["spline"], lw=2.0)
    ax.axvline(np.median(mdn_ratio), color=MODEL_COLORS["mdn"], lw=2.0)
    ax.set_xlabel("W / panel target numerical floor")
    ax.set_ylabel("signal count")
    ax.set_title("Distance to evaluation floor")
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(sigma, spline_w, color=MODEL_COLORS["spline"], s=26, alpha=0.68, label=MODEL_LABELS["spline"])
    ax.scatter(sigma, mdn_w, color=MODEL_COLORS["mdn"], s=26, alpha=0.60, label=MODEL_LABELS["mdn"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true sigma")
    ax.set_ylabel("mean normalized marginal W")
    ax.set_title("Failure concentration by noise level")
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False)

    figure.suptitle(
        f"Broad spline 4.096M vs Broad MDN 512k panel marginal W "
        f"(n={len(combined_rows)}, posterior samples={posterior_samples:,})",
        y=1.02,
    )
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def make_summary(
    *,
    combined_rows: list[dict[str, object]],
    model_rows: dict[str, list[dict[str, object]]],
    panel_metadata: dict[str, object],
    args: argparse.Namespace,
    outputs: dict[str, str],
) -> dict[str, object]:
    spline_w = np.asarray([row["spline_wasserstein"] for row in combined_rows], dtype=np.float64)
    mdn_w = np.asarray([row["mdn_wasserstein"] for row in combined_rows], dtype=np.float64)
    spline_ratio = np.asarray([row["spline_target_ratio"] for row in combined_rows], dtype=np.float64)
    mdn_ratio = np.asarray([row["mdn_target_ratio"] for row in combined_rows], dtype=np.float64)
    sorted_worst = sorted(
        combined_rows,
        key=lambda row: max(float(row["spline_wasserstein"]), float(row["mdn_wasserstein"])),
        reverse=True,
    )
    return {
        "panel": panel_metadata,
        "posterior_samples": int(args.posterior_samples),
        "device": str(args.device),
        "seed": int(args.seed),
        "models": {
            "spline": str(args.spline_model),
            "mdn": str(args.mdn_model),
        },
        "signal_count": len(combined_rows),
        "spline_wasserstein": quantile_summary(spline_w),
        "mdn_wasserstein": quantile_summary(mdn_w),
        "spline_target_ratio": quantile_summary(spline_ratio),
        "mdn_target_ratio": quantile_summary(mdn_ratio),
        "spline_better_count": int(np.sum(spline_w < mdn_w)),
        "mdn_better_count": int(np.sum(mdn_w < spline_w)),
        "tie_count": int(np.sum(spline_w == mdn_w)),
        "mean_improvement_mdn_minus_spline": float(np.mean(mdn_w - spline_w)),
        "median_improvement_mdn_minus_spline": float(np.median(mdn_w - spline_w)),
        "worst_signals": sorted_worst[: min(12, len(sorted_worst))],
        "seconds": {
            key: float(sum(row["seconds"] for row in rows))
            for key, rows in model_rows.items()
        },
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved broad NPE checkpoints against a cached panel of exact "
            "1D posterior marginals."
        ),
    )
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--spline-model", type=Path, default=DEFAULT_SPLINE_MODEL)
    parser.add_argument("--mdn-model", type=Path, default=DEFAULT_MDN_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = stage1.choose_training_device(str(args.device))
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel, panel_metadata = load_panel_marginal_cache(args.panel_cache)
    models = {}
    states = {}
    for key, path in [("spline", args.spline_model), ("mdn", args.mdn_model)]:
        model, state = load_stage1_checkpoint(path, device)
        models[key] = model
        states[key] = state

    model_rows = {}
    for offset, key in enumerate(("spline", "mdn")):
        model_rows[key] = evaluate_model(
            model_key=key,
            model=models[key],
            state=states[key],
            panel=panel,
            posterior_samples=int(args.posterior_samples),
            seed=int(args.seed) + 100_000 * offset,
            device=device,
            print_every=int(args.print_every),
        )

    combined_rows = combine_rows(model_rows)
    per_model_csv = results_dir / "panel_w_per_model_rows.csv"
    combined_csv = results_dir / "panel_w_combined_rows.csv"
    summary_json = results_dir / "panel_w_distribution_summary.json"
    figure_path = figures_dir / "panel_w_distribution_mdn512k_vs_spline4m.png"
    write_csv([row for rows in model_rows.values() for row in rows], per_model_csv)
    write_csv(combined_rows, combined_csv)
    plot(combined_rows, figure_path, posterior_samples=int(args.posterior_samples))
    outputs = {
        "per_model_csv": str(per_model_csv),
        "combined_csv": str(combined_csv),
        "summary_json": str(summary_json),
        "figure": str(figure_path),
    }
    summary = make_summary(
        combined_rows=combined_rows,
        model_rows=model_rows,
        panel_metadata=panel_metadata,
        args=args,
        outputs=outputs,
    )
    summary_json.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
