from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import npe_stage1_decay as stage1
from evaluate_npe_ensemble_nll import batch_log_prob_z, json_ready, load_checkpoint, resolve_repo_path


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)


def collect_globs(repo_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(repo_root.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def ensemble_log_probs(
    *,
    model_paths: list[Path],
    x_val: np.ndarray,
    z_val: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not model_paths:
        raise ValueError("Expected at least one model path.")
    sums: np.ndarray | None = None
    for path in model_paths:
        model, state = load_checkpoint(path, device)
        pieces: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_val.shape[0], batch_size):
                stop = min(start + batch_size, x_val.shape[0])
                value = batch_log_prob_z(
                    model=model,
                    state=state,
                    x_raw=x_val[start:stop],
                    z_raw=z_val[start:stop],
                    device=device,
                )
                pieces.append(value.detach().cpu().numpy().astype(np.float64, copy=False))
        log_prob = np.concatenate(pieces)
        sums = log_prob if sums is None else np.logaddexp(sums, log_prob)
        del model
    assert sums is not None
    return sums - math.log(len(model_paths))


def nll(values: np.ndarray, start: int, stop: int) -> float:
    return float(-np.mean(values[start:stop]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an x-gated mixture between base and specialist NPEs.")
    parser.add_argument("--base-glob", action="append", default=[])
    parser.add_argument("--specialist-glob", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--fit-examples", type=int, default=200_000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    validation_cache = resolve_repo_path(repo_root, args.validation_cache)
    if validation_cache is None or not validation_cache.exists():
        raise SystemExit(f"Missing validation cache: {validation_cache}")
    base_paths = collect_globs(repo_root, args.base_glob)
    specialist_paths = collect_globs(repo_root, args.specialist_glob)
    data = np.load(validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    z_val = np.asarray(data["z_val"], dtype=np.float32)
    if args.max_examples > 0:
        x_val = x_val[: int(args.max_examples)]
        z_val = z_val[: int(args.max_examples)]
    fit_stop = min(max(1, int(args.fit_examples)), x_val.shape[0])
    base_log_prob = ensemble_log_probs(
        model_paths=base_paths,
        x_val=x_val,
        z_val=z_val,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
    )
    specialist_log_prob = ensemble_log_probs(
        model_paths=specialist_paths,
        x_val=x_val,
        z_val=z_val,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
    )
    fit_log_sigma = stage1.decay_fit_summary_features(x_val.astype(np.float64))[:, 2]
    quantiles = np.linspace(0.02, 0.50, 25)
    thresholds = np.quantile(fit_log_sigma[:fit_stop], quantiles)
    temperatures = [0.05, 0.10, 0.20, 0.35, 0.50]
    max_weights = [0.10, 0.20, 0.35, 0.50, 0.75, 0.90]
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        for temperature in temperatures:
            gate_base = 1.0 / (1.0 + np.exp((fit_log_sigma - threshold) / temperature))
            for max_weight in max_weights:
                gate = np.clip(max_weight * gate_base, 1e-6, 1.0 - 1e-6)
                mixed = np.logaddexp(
                    np.log1p(-gate) + base_log_prob,
                    np.log(gate) + specialist_log_prob,
                )
                row = {
                    "threshold": float(threshold),
                    "temperature": float(temperature),
                    "max_weight": float(max_weight),
                    "fit_nll": nll(mixed, 0, fit_stop),
                    "holdout_nll": nll(mixed, fit_stop, x_val.shape[0]) if fit_stop < x_val.shape[0] else None,
                    "full_nll": nll(mixed, 0, x_val.shape[0]),
                    "mean_gate": float(np.mean(gate)),
                }
                if best is None or row["fit_nll"] < best["fit_nll"]:
                    best = row
    assert best is not None
    result = {
        "base_model_paths": [str(path) for path in base_paths],
        "specialist_model_paths": [str(path) for path in specialist_paths],
        "validation_examples": int(x_val.shape[0]),
        "fit_examples": int(fit_stop),
        "base_full_nll": nll(base_log_prob, 0, x_val.shape[0]),
        "specialist_full_nll": nll(specialist_log_prob, 0, x_val.shape[0]),
        "best_gate": best,
    }
    output = resolve_repo_path(repo_root, args.output)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
