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
DEFAULT_FLOW2_ENSEMBLE_SUMMARY = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "199_nll63_randperm_e15_cosstep_ensemble4_saved/results/ensemble4_proof_summary.json"
)


MODEL_COLORS = {
    "spline": "#c45a2d",
    "mdn": "#6d4aff",
    "flow2_ensemble": "#0f766e",
}
MODEL_LABELS = {
    "spline": "Spline-flow NPE, 4.096M",
    "mdn": "MDN, 512k",
    "flow2_ensemble": "4-member Flow2 residual NSF",
}
SCATTER_AXIS_LABELS = {
    "spline": "spline-flow NPE distance",
    "mdn": "mixture density network distance",
    "flow2_ensemble": "Flow2 residual NSF ensemble distance",
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


def observed_features_for_state(state: dict[str, object], x: np.ndarray) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float64)
    x_mean = np.asarray(state["x_mean"], dtype=np.float64)
    if raw.shape[0] == x_mean.shape[0]:
        return raw
    config = state.get("config", {})
    mode = str(config.get("context_features", "raw")) if isinstance(config, dict) else "raw"
    features = stage1.transform_context_features(raw[None, :], mode)[0]
    if features.shape[0] != x_mean.shape[0]:
        raise ValueError(
            f"Context feature shape mismatch for context_features={mode!r}: "
            f"got {features.shape[0]}, expected {x_mean.shape[0]}"
        )
    return features


def weighted_sample_counts(total: int, weights: np.ndarray) -> np.ndarray:
    if total < weights.size:
        raise ValueError("--posterior-samples must be at least the number of ensemble members")
    raw_counts = weights * int(total)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def load_single_stage1_model(path: Path, device: torch.device) -> dict[str, object]:
    model, state = load_stage1_checkpoint(path, device)
    return {"type": "single", "model": model, "state": state, "path": str(path)}


def load_stage1_ensemble(summary_path: Path, device: torch.device) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    member_paths = [Path(path) for path in summary.get("model_paths", [])]
    if not member_paths:
        raise ValueError(f"Ensemble summary has no model_paths: {summary_path}")
    raw_weights = summary.get("ensemble_weights")
    if raw_weights is None:
        weights = np.full(len(member_paths), 1.0 / len(member_paths), dtype=np.float64)
    else:
        weights = np.asarray(raw_weights, dtype=np.float64)
        if weights.shape != (len(member_paths),):
            raise ValueError(
                f"Ensemble weights length {weights.shape[0]} does not match "
                f"model_paths length {len(member_paths)}: {summary_path}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"Invalid ensemble weights in {summary_path}")
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError(f"Ensemble weights sum to zero in {summary_path}")
        weights = weights / weight_sum
    members = []
    for path in member_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble member checkpoint: {path}")
        model, state = load_stage1_checkpoint(path, device)
        members.append({"model": model, "state": state, "path": str(path)})
    return {
        "type": "ensemble",
        "members": members,
        "weights": weights,
        "summary": summary,
        "summary_path": str(summary_path),
    }


def sample_loaded_stage1_model(
    *,
    loaded: dict[str, object],
    observed_x: np.ndarray,
    posterior_samples: int,
    device: torch.device,
) -> np.ndarray:
    if loaded["type"] == "single":
        state = loaded["state"]
        assert isinstance(state, dict)
        observed_features = observed_features_for_state(state, observed_x)
        _, theta_samples = stage1.sample_posterior_for_observation(
            model=loaded["model"],
            observed_x=observed_features,
            x_mean=np.asarray(state["x_mean"], dtype=np.float64),
            x_std=np.asarray(state["x_std"], dtype=np.float64),
            z_mean=np.asarray(state["z_mean"], dtype=np.float64),
            z_std=np.asarray(state["z_std"], dtype=np.float64),
            n=posterior_samples,
            device=device,
        )
        return theta_samples

    if loaded["type"] == "ensemble":
        weights = np.asarray(loaded["weights"], dtype=np.float64)
        counts = weighted_sample_counts(posterior_samples, weights)
        members = loaded["members"]
        assert isinstance(members, list)
        theta_parts = []
        for member, count in zip(members, counts, strict=True):
            if int(count) <= 0:
                continue
            state = member["state"]
            observed_features = observed_features_for_state(state, observed_x)
            _, theta_member = stage1.sample_posterior_for_observation(
                model=member["model"],
                observed_x=observed_features,
                x_mean=np.asarray(state["x_mean"], dtype=np.float64),
                x_std=np.asarray(state["x_std"], dtype=np.float64),
                z_mean=np.asarray(state["z_mean"], dtype=np.float64),
                z_std=np.asarray(state["z_std"], dtype=np.float64),
                n=int(count),
                device=device,
            )
            theta_parts.append(theta_member)
        return np.vstack(theta_parts)

    raise ValueError(f"Unknown loaded model type: {loaded['type']!r}")


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
    loaded: dict[str, object],
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
    start = time.perf_counter()
    for index, label in enumerate(labels):
        signal_start = time.perf_counter()
        seed_torch(seed + index, device)
        theta_samples = sample_loaded_stage1_model(
            loaded=loaded,
            observed_x=x_panel[index],
            posterior_samples=posterior_samples,
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
    model_keys = list(model_rows)
    by_model = {
        model_key: {int(row["index"]): row for row in rows}
        for model_key, rows in model_rows.items()
    }
    common_indices = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    combined = []
    for index in common_indices:
        first = by_model[model_keys[0]][index]
        row = {
            "index": index,
            "label": first["label"],
            "A": first["A"],
            "k": first["k"],
            "sigma": first["sigma"],
            "target_wasserstein": first["target_wasserstein"],
        }
        row["best_model"] = min(model_keys, key=lambda key: float(by_model[key][index]["wasserstein"]))
        for key in model_keys:
            model_row = by_model[key][index]
            row[f"{key}_wasserstein"] = model_row["wasserstein"]
            row[f"{key}_target_ratio"] = model_row["target_ratio"]
            row[f"{key}_w_A"] = model_row["w_A"]
            row[f"{key}_w_k"] = model_row["w_k"]
            row[f"{key}_w_sigma"] = model_row["w_sigma"]
        if "spline" in by_model and "mdn" in by_model:
            spline = by_model["spline"][index]
            mdn = by_model["mdn"][index]
            row["spline_minus_mdn"] = float(spline["wasserstein"] - mdn["wasserstein"])
            row["spline_over_mdn"] = (
                float(spline["wasserstein"] / mdn["wasserstein"])
                if float(mdn["wasserstein"]) > 0
                else float("nan")
            )
        if "flow2_ensemble" in by_model and "spline" in by_model:
            flow2 = by_model["flow2_ensemble"][index]
            spline = by_model["spline"][index]
            row["flow2_ensemble_minus_spline"] = float(flow2["wasserstein"] - spline["wasserstein"])
            row["flow2_ensemble_over_spline"] = (
                float(flow2["wasserstein"] / spline["wasserstein"])
                if float(spline["wasserstein"]) > 0
                else float("nan")
            )
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


def model_keys_from_combined_rows(combined_rows: list[dict[str, object]]) -> list[str]:
    if not combined_rows:
        return []
    suffix = "_wasserstein"
    present = {
        key[: -len(suffix)]
        for key in combined_rows[0]
        if key.endswith(suffix) and key != "target_wasserstein"
    }
    ordered = [key for key in MODEL_LABELS if key in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot(combined_rows: list[dict[str, object]], output_path: Path, *, posterior_samples: int) -> None:
    model_keys = model_keys_from_combined_rows(combined_rows)
    w_by_model = {
        key: np.asarray([row[f"{key}_wasserstein"] for row in combined_rows], dtype=np.float64)
        for key in model_keys
    }
    ratio_by_model = {
        key: np.asarray([row[f"{key}_target_ratio"] for row in combined_rows], dtype=np.float64)
        for key in model_keys
    }
    sigma = np.asarray([row["sigma"] for row in combined_rows], dtype=np.float64)
    indices = np.asarray([row["index"] for row in combined_rows], dtype=np.int64)
    all_w = np.concatenate([values[np.isfinite(values)] for values in w_by_model.values()])
    all_ratio = np.concatenate([values[np.isfinite(values)] for values in ratio_by_model.values()])
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)

    ax = axes[0, 0]
    for key, values in w_by_model.items():
        x, y = ecdf(values)
        ax.step(x, y, where="post", lw=2.5, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
        ax.axvline(np.median(values), color=MODEL_COLORS[key], lw=1.6, alpha=0.68)
    if np.all(all_w > 0):
        ax.set_xscale("log")
    ax.set_xlabel("mean normalized marginal Wasserstein distance")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Distribution across panel signals")
    ax.grid(which="both", alpha=0.22)

    ax = axes[0, 1]
    if "flow2_ensemble" in w_by_model and "spline" in w_by_model:
        x_key = "spline"
        y_key = "flow2_ensemble"
    else:
        x_key, y_key = model_keys[:2]
    x_values = w_by_model[x_key]
    y_values = w_by_model[y_key]
    scatter = ax.scatter(
        x_values,
        y_values,
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
    ax.set_xlabel(SCATTER_AXIS_LABELS.get(x_key, f"{x_key} distance"))
    ax.set_ylabel(SCATTER_AXIS_LABELS.get(y_key, f"{y_key} distance"))
    ax.set_title("Per-signal comparison")
    ax.grid(which="both", alpha=0.22)
    colorbar = figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("true sigma")
    worst_order = np.argsort(np.maximum(x_values, y_values))[::-1][: min(6, len(indices))]
    for row_index in worst_order:
        ax.text(x_values[row_index], y_values[row_index], str(int(indices[row_index])), fontsize=8)

    ax = axes[1, 0]
    if all_ratio.size and np.all(all_ratio > 0):
        bins = np.logspace(np.log10(np.nanmin(all_ratio) * 0.85), np.log10(np.nanmax(all_ratio) * 1.15), 24)
        ax.set_xscale("log")
    else:
        bins = 24
    for key, values in ratio_by_model.items():
        ax.hist(values, bins=bins, alpha=0.32, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
        ax.axvline(np.median(values), color=MODEL_COLORS[key], lw=2.0)
    ax.set_xlabel("distance / panel target numerical floor")
    ax.set_ylabel("signal count")
    ax.set_title("Distance to evaluation floor")
    ax.grid(which="both", alpha=0.22)

    ax = axes[1, 1]
    for key, values in w_by_model.items():
        ax.scatter(sigma, values, color=MODEL_COLORS[key], s=24, alpha=0.58, label=MODEL_LABELS[key])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true sigma")
    ax.set_ylabel("mean normalized marginal Wasserstein distance")
    ax.set_title("Failure concentration by noise level")
    ax.grid(which="both", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
    )

    figure.suptitle(
        f"Single-decay NPE panel marginal Wasserstein distribution "
        f"(n={len(combined_rows)}, posterior samples={posterior_samples:,})",
        y=1.075,
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
    model_keys = model_keys_from_combined_rows(combined_rows)
    w_by_model = {
        key: np.asarray([row[f"{key}_wasserstein"] for row in combined_rows], dtype=np.float64)
        for key in model_keys
    }
    ratio_by_model = {
        key: np.asarray([row[f"{key}_target_ratio"] for row in combined_rows], dtype=np.float64)
        for key in model_keys
    }
    sorted_worst = sorted(
        combined_rows,
        key=lambda row: max(float(row[f"{key}_wasserstein"]) for key in model_keys),
        reverse=True,
    )
    model_paths: dict[str, str] = {
        "spline": str(args.spline_model),
        "mdn": str(args.mdn_model),
    }
    if args.flow2_ensemble_summary is not None:
        model_paths["flow2_ensemble"] = str(args.flow2_ensemble_summary)
    summary = {
        "panel": panel_metadata,
        "posterior_samples": int(args.posterior_samples),
        "device": str(args.device),
        "seed": int(args.seed),
        "models": {key: model_paths[key] for key in model_keys},
        "signal_count": len(combined_rows),
        "wasserstein": {key: quantile_summary(values) for key, values in w_by_model.items()},
        "target_ratio": {key: quantile_summary(values) for key, values in ratio_by_model.items()},
        "best_model_counts": {
            key: int(sum(row["best_model"] == key for row in combined_rows))
            for key in model_keys
        },
        "worst_signals": sorted_worst[: min(12, len(sorted_worst))],
        "seconds": {
            key: float(sum(row["seconds"] for row in rows))
            for key, rows in model_rows.items()
        },
        "outputs": outputs,
    }
    if "spline" in w_by_model and "mdn" in w_by_model:
        spline_w = w_by_model["spline"]
        mdn_w = w_by_model["mdn"]
        summary.update(
            {
                "spline_wasserstein": quantile_summary(spline_w),
                "mdn_wasserstein": quantile_summary(mdn_w),
                "spline_target_ratio": quantile_summary(ratio_by_model["spline"]),
                "mdn_target_ratio": quantile_summary(ratio_by_model["mdn"]),
                "spline_better_count": int(np.sum(spline_w < mdn_w)),
                "mdn_better_count": int(np.sum(mdn_w < spline_w)),
                "tie_count": int(np.sum(spline_w == mdn_w)),
                "mean_improvement_mdn_minus_spline": float(np.mean(mdn_w - spline_w)),
                "median_improvement_mdn_minus_spline": float(np.median(mdn_w - spline_w)),
            }
        )
    if "flow2_ensemble" in w_by_model and "spline" in w_by_model:
        flow2_w = w_by_model["flow2_ensemble"]
        spline_w = w_by_model["spline"]
        summary.update(
            {
                "flow2_ensemble_wasserstein": quantile_summary(flow2_w),
                "flow2_ensemble_target_ratio": quantile_summary(ratio_by_model["flow2_ensemble"]),
                "flow2_ensemble_better_than_spline_count": int(np.sum(flow2_w < spline_w)),
                "spline_better_than_flow2_ensemble_count": int(np.sum(spline_w < flow2_w)),
                "mean_improvement_spline_minus_flow2_ensemble": float(np.mean(spline_w - flow2_w)),
                "median_improvement_spline_minus_flow2_ensemble": float(np.median(spline_w - flow2_w)),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved population-trained NPE checkpoints against a cached panel of exact "
            "1D posterior marginals."
        ),
    )
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--spline-model", type=Path, default=DEFAULT_SPLINE_MODEL)
    parser.add_argument("--mdn-model", type=Path, default=DEFAULT_MDN_MODEL)
    parser.add_argument("--flow2-ensemble-summary", type=Path, default=DEFAULT_FLOW2_ENSEMBLE_SUMMARY)
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
    loaded_models: dict[str, dict[str, object]] = {
        "spline": load_single_stage1_model(args.spline_model, device),
        "mdn": load_single_stage1_model(args.mdn_model, device),
    }
    if args.flow2_ensemble_summary is not None:
        loaded_models["flow2_ensemble"] = load_stage1_ensemble(args.flow2_ensemble_summary, device)

    model_rows = {}
    for offset, key in enumerate(loaded_models):
        model_rows[key] = evaluate_model(
            model_key=key,
            loaded=loaded_models[key],
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
    figure_name = (
        "panel_w_distribution_mdn512k_vs_spline4m_flow2ensemble.png"
        if "flow2_ensemble" in loaded_models
        else "panel_w_distribution_mdn512k_vs_spline4m.png"
    )
    figure_path = figures_dir / figure_name
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
