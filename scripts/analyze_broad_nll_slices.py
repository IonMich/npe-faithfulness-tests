from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcmc_decay_inference import PARAMETER_NAMES  # noqa: E402
from npe_posterior_viewer import DEFAULT_BEST_BROAD_MODEL, DEFAULT_BEST_BROAD_SPLINE_MODEL, load_stage1_checkpoint  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze broad NPE validation NLL by rare prior slices.")
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mdn-model", type=Path, default=DEFAULT_BEST_BROAD_MODEL)
    parser.add_argument("--spline-model", type=Path, default=DEFAULT_BEST_BROAD_SPLINE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=65_536)
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

    masks = {
        "all": np.ones(theta.shape[0], dtype=bool),
        "sigma_lt_0p05": sigma < 0.05,
        "sigma_lt_0p075": sigma < 0.075,
        "A_gt_40": A > 40.0,
        "A_gt_50": A > 50.0,
        "A_gt_50_sigma_lt_0p05": (A > 50.0) & (sigma < 0.05),
        "A_gt_40_sigma_lt_0p075": (A > 40.0) & (sigma < 0.075),
        "low_sigma_mid_high_A": (A > 20.0) & (sigma < 0.075),
        "low_sigma_k_near_0p3": (sigma < 0.075) & (k > 0.22) & (k < 0.42),
    }

    device = torch.device(args.device)
    model_specs = {
        "broad_mdn_512k": args.mdn_model,
        "broad_spline_4m": args.spline_model,
    }
    nll_by_model = {}
    model_summaries = {}
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

    deltas = {}
    spline_minus_mdn = nll_by_model["broad_spline_4m"].astype(np.float64) - nll_by_model["broad_mdn_512k"].astype(np.float64)
    for name, mask in masks.items():
        deltas[name] = summarize(spline_minus_mdn, mask)

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
