from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcmc_decay_inference import PARAMETER_NAMES  # noqa: E402
from npe_posterior_viewer import DEFAULT_BEST_BROAD_MODEL, DEFAULT_BEST_BROAD_SPLINE_MODEL, load_stage1_checkpoint  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CACHE = Path("runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz")
DEFAULT_OUTPUT = Path("runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/results/broad_nll_slice_analysis.json")


def nll_values(
    *,
    model: torch.nn.Module,
    state: dict[str, object],
    x_val: np.ndarray,
    z_val: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x_mean = np.asarray(state["x_mean"], dtype=np.float32)
    x_std = np.asarray(state["x_std"], dtype=np.float32)
    z_mean = np.asarray(state["z_mean"], dtype=np.float32)
    z_std = np.asarray(state["z_std"], dtype=np.float32)
    log_det = float(np.log(z_std.astype(np.float64)).sum())
    values = np.empty(x_val.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, x_val.shape[0], batch_size):
            stop = min(start + batch_size, x_val.shape[0])
            x_standardized = ((x_val[start:stop] - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
            z_standardized = ((z_val[start:stop] - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
            x_tensor = torch.from_numpy(x_standardized).to(device)
            z_tensor = torch.from_numpy(z_standardized).to(device)
            loss = -model.log_prob(z_tensor, x_tensor).detach().cpu().numpy().astype(np.float32)
            values[start:stop] = loss + log_det
    return values


def summarize(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = values[mask].astype(np.float64, copy=False)
    if selected.size == 0:
        return {
            "n": 0,
            "fraction": 0.0,
            "mean": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(selected.size),
        "fraction": float(selected.size / values.size),
        "mean": float(np.mean(selected)),
        "q50": float(np.quantile(selected, 0.50)),
        "q90": float(np.quantile(selected, 0.90)),
        "q99": float(np.quantile(selected, 0.99)),
        "max": float(np.max(selected)),
    }


def top_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    count = min(int(top_k), int(values.size))
    ranking_values = np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    candidate = np.argpartition(ranking_values, -count)[-count:]
    return candidate[np.argsort(ranking_values[candidate])[::-1]]


def top_rows(
    *,
    values: np.ndarray,
    theta: np.ndarray,
    indices: np.ndarray,
    masks: dict[str, np.ndarray],
    value_name: str,
) -> list[dict[str, object]]:
    rows = []
    for rank, index in enumerate(indices, start=1):
        row: dict[str, object] = {
            "rank": rank,
            "index": int(index),
            value_name: float(values[index]),
        }
        for param_name, param_value in zip(PARAMETER_NAMES, theta[index], strict=True):
            row[param_name] = float(param_value)
        for name, mask in masks.items():
            if name != "all":
                row[f"slice_{name}"] = bool(mask[index])
        rows.append(row)
    return rows


def top_slice_enrichment(
    *,
    indices: np.ndarray,
    masks: dict[str, np.ndarray],
    total_count: int,
) -> dict[str, dict[str, float | int]]:
    selected = np.zeros(total_count, dtype=bool)
    selected[indices] = True
    output = {}
    for name, mask in masks.items():
        if name == "all":
            continue
        population_fraction = float(mask.mean())
        top_fraction = float(mask[selected].mean()) if selected.any() else float("nan")
        output[name] = {
            "population_n": int(mask.sum()),
            "population_fraction": population_fraction,
            "top_n": int(np.count_nonzero(mask & selected)),
            "top_fraction": top_fraction,
            "enrichment": (
                float(top_fraction / population_fraction)
                if population_fraction > 0.0 and np.isfinite(top_fraction)
                else float("nan")
            ),
        }
    return output


def write_dict_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_top_signals(
    *,
    x_val: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    title: str,
    max_panels: int = 12,
) -> None:
    panel_indices = indices[:max_panels]
    if panel_indices.size == 0:
        return
    cols = min(3, int(panel_indices.size))
    rows = int(np.ceil(panel_indices.size / cols))
    t = np.linspace(0.0, 6.0, x_val.shape[1])
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for plot_index, ax in enumerate(axes.flat):
        if plot_index >= panel_indices.size:
            ax.axis("off")
            continue
        index = int(panel_indices[plot_index])
        params = theta[index]
        mean = params[0] * np.exp(-params[1] * t)
        ax.plot(t, x_val[index], color="#2f6fbb", linewidth=1.25, label="x")
        ax.plot(t, mean, color="#b85c38", linewidth=1.1, linestyle="--", label="true mean")
        ax.set_title(f"rank {plot_index + 1}; value={values[index]:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze broad NPE validation NLL by rare prior slices.")
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mdn-model", type=Path, default=DEFAULT_BEST_BROAD_MODEL)
    parser.add_argument("--spline-model", type=Path, default=DEFAULT_BEST_BROAD_SPLINE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(args.validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    z_val = np.asarray(data["z_val"], dtype=np.float32)
    theta = np.exp(z_val.astype(np.float64))
    A = theta[:, 0]
    k = theta[:, 1]
    sigma = theta[:, 2]
    snr = A / np.maximum(sigma, 1e-12)

    masks = {
        "all": np.ones(theta.shape[0], dtype=bool),
        "sigma_lt_0p05": sigma < 0.05,
        "sigma_lt_0p075": sigma < 0.075,
        "sigma_gt_2": sigma > 2.0,
        "sigma_gt_5": sigma > 5.0,
        "A_lt_0p2": A < 0.2,
        "A_lt_1": A < 1.0,
        "A_gt_40": A > 40.0,
        "A_gt_50": A > 50.0,
        "k_lt_0p1": k < 0.1,
        "k_gt_5": k > 5.0,
        "snr_lt_0p2": snr < 0.2,
        "snr_lt_1": snr < 1.0,
        "snr_gt_500": snr > 500.0,
        "A_gt_50_sigma_lt_0p05": (A > 50.0) & (sigma < 0.05),
        "A_gt_40_sigma_lt_0p075": (A > 40.0) & (sigma < 0.075),
        "low_sigma_mid_high_A": (A > 20.0) & (sigma < 0.075),
        "low_sigma_k_near_0p3": (sigma < 0.075) & (k > 0.22) & (k < 0.42),
        "low_snr_low_A": (snr < 1.0) & (A < 1.0),
        "high_noise_low_snr": (sigma > 2.0) & (snr < 1.0),
    }

    device = torch.device(args.device)
    model_specs = {
        "broad_mdn_512k": args.mdn_model,
        "broad_spline_4m": args.spline_model,
    }
    nll_by_model = {}
    model_summaries = {}
    top_failure_summary = {}
    top_failure_outputs = {}
    for model_id, path in model_specs.items():
        model, state = load_stage1_checkpoint(path, device)
        values = nll_values(
            model=model,
            state=state,
            x_val=x_val,
            z_val=z_val,
            device=device,
            batch_size=args.batch_size,
        )
        nll_by_model[model_id] = values
        model_summaries[model_id] = {
            name: summarize(values, mask)
            for name, mask in masks.items()
        }
        indices = top_indices(values, args.top_k)
        rows = top_rows(
            values=values,
            theta=theta,
            indices=indices,
            masks=masks,
            value_name="nll_z_units",
        )
        top_csv = args.output.with_name(f"{args.output.stem}_{model_id}_top{len(indices)}.csv")
        top_npz = args.output.with_name(f"{args.output.stem}_{model_id}_top{len(indices)}.npz")
        top_png = args.output.with_name(f"{args.output.stem}_{model_id}_top{min(len(indices), 12)}.png")
        write_dict_rows(rows, top_csv)
        np.savez_compressed(
            top_npz,
            index=indices.astype(np.int64),
            nll_z_units=values[indices],
            x=x_val[indices],
            z=z_val[indices],
            theta=theta[indices],
        )
        plot_top_signals(
            x_val=x_val,
            theta=theta,
            values=values,
            indices=indices,
            output_path=top_png,
            title=f"{model_id} worst validation examples",
        )
        top_failure_summary[model_id] = {
            "top_k": int(len(indices)),
            "worst_nll_z_units": float(values[indices[0]]) if indices.size else float("nan"),
            "best_of_top_nll_z_units": float(values[indices[-1]]) if indices.size else float("nan"),
            "slice_enrichment": top_slice_enrichment(
                indices=indices,
                masks=masks,
                total_count=theta.shape[0],
            ),
            "top_rows_preview": rows[:10],
        }
        top_failure_outputs[model_id] = {
            "csv": str(top_csv),
            "npz": str(top_npz),
            "figure": str(top_png),
        }

    deltas = {}
    spline_minus_mdn = nll_by_model["broad_spline_4m"].astype(np.float64) - nll_by_model["broad_mdn_512k"].astype(np.float64)
    for name, mask in masks.items():
        deltas[name] = summarize(spline_minus_mdn, mask)
    delta_tail_summary = {}
    for label, values in {
        "spline_much_worse": spline_minus_mdn,
        "spline_much_better": -spline_minus_mdn,
    }.items():
        indices = top_indices(values, args.top_k)
        rows = top_rows(
            values=spline_minus_mdn,
            theta=theta,
            indices=indices,
            masks=masks,
            value_name="spline_minus_mdn_nll_z_units",
        )
        delta_csv = args.output.with_name(f"{args.output.stem}_{label}_top{len(indices)}.csv")
        write_dict_rows(rows, delta_csv)
        delta_tail_summary[label] = {
            "top_k": int(len(indices)),
            "csv": str(delta_csv),
            "slice_enrichment": top_slice_enrichment(
                indices=indices,
                masks=masks,
                total_count=theta.shape[0],
            ),
            "top_rows_preview": rows[:10],
        }

    prior_slice_summary = {
        name: {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
        }
        for name, mask in masks.items()
    }
    result = {
        "validation_cache": str(args.validation_cache),
        "n": int(theta.shape[0]),
        "parameter_names": list(PARAMETER_NAMES),
        "prior_slice_summary": prior_slice_summary,
        "nll_z_units": model_summaries,
        "nll_delta_spline_minus_mdn": deltas,
        "top_failures": top_failure_summary,
        "top_failure_outputs": top_failure_outputs,
        "delta_tail_summary": delta_tail_summary,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice",
                "model",
                "n",
                "fraction",
                "mean",
                "q50",
                "q90",
                "q99",
                "max",
            ],
        )
        writer.writeheader()
        for model_id, summaries in model_summaries.items():
            for name, summary in summaries.items():
                writer.writerow({"slice": name, "model": model_id, **summary})
        for name, summary in deltas.items():
            writer.writerow({"slice": name, "model": "spline_minus_mdn", **summary})

    print(json.dumps({"json": str(args.output), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
