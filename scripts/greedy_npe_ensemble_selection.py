from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_npe_ensemble_nll import batch_log_prob_z, json_ready, load_checkpoint, resolve_repo_path


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)


def collect_paths(args: argparse.Namespace) -> list[Path]:
    repo_root = args.repo_root.resolve()
    paths: list[Path] = []
    for value in args.model:
        path = resolve_repo_path(repo_root, value)
        if path is not None:
            paths.append(path)
    for pattern in args.model_glob:
        paths.extend(sorted(repo_root.glob(pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise SystemExit(f"Missing model checkpoint: {path}")
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise SystemExit("No model checkpoints were provided.")
    return unique


def compute_log_probs(
    *,
    model_paths: list[Path],
    validation_cache: Path,
    device: torch.device,
    batch_size: int,
    max_examples: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.load(validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    z_val = np.asarray(data["z_val"], dtype=np.float32)
    if max_examples > 0:
        x_val = x_val[:max_examples]
        z_val = z_val[:max_examples]

    rows: list[dict[str, Any]] = []
    log_probs = np.empty((len(model_paths), x_val.shape[0]), dtype=np.float32)
    for model_index, path in enumerate(model_paths):
        start_time = time.perf_counter()
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
                pieces.append(value.detach().cpu().numpy().astype(np.float32, copy=False))
        log_probs[model_index] = np.concatenate(pieces)
        rows.append(
            {
                "index": model_index,
                "model_path": str(path),
                "individual_full_val_nll_z_units": float(-log_probs[model_index].mean(dtype=np.float64)),
                "evaluation_seconds": float(time.perf_counter() - start_time),
            }
        )
        del model
    return log_probs, rows


def greedy_select(
    *,
    log_probs: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    max_size: int,
    min_delta: float,
) -> list[dict[str, Any]]:
    remaining = set(range(log_probs.shape[0]))
    selected: list[int] = []
    selected_logsumexp: np.ndarray | None = None
    current_nll = math.inf
    curve: list[dict[str, Any]] = []

    while remaining and len(selected) < max_size:
        best_index: int | None = None
        best_lse: np.ndarray | None = None
        best_nll = math.inf
        next_size = len(selected) + 1
        log_next_size = math.log(next_size)
        for index in remaining:
            if selected_logsumexp is None:
                mixture_log_prob = log_probs[index]
                candidate_lse = log_probs[index].astype(np.float32, copy=True)
            else:
                candidate_lse = np.logaddexp(selected_logsumexp, log_probs[index]).astype(
                    np.float32,
                    copy=False,
                )
                mixture_log_prob = candidate_lse - log_next_size
            nll = float(-mixture_log_prob.mean(dtype=np.float64))
            if nll < best_nll:
                best_nll = nll
                best_index = index
                best_lse = candidate_lse
        if best_index is None or best_lse is None:
            break
        improvement = current_nll - best_nll if math.isfinite(current_nll) else math.inf
        if math.isfinite(current_nll) and improvement < min_delta:
            break
        remaining.remove(best_index)
        selected.append(best_index)
        selected_logsumexp = best_lse
        current_nll = best_nll
        row = dict(candidate_rows[best_index])
        row.update(
            {
                "ensemble_size": len(selected),
                "ensemble_full_val_nll_z_units": best_nll,
                "ensemble_improvement": improvement,
            }
        )
        curve.append(row)
    return curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy selection for saved NPE log-density ensembles.")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-glob", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=32)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    validation_cache = resolve_repo_path(repo_root, args.validation_cache)
    if validation_cache is None or not validation_cache.exists():
        raise SystemExit(f"Missing validation cache: {validation_cache}")

    paths = collect_paths(args)
    log_probs, candidate_rows = compute_log_probs(
        model_paths=paths,
        validation_cache=validation_cache,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
        max_examples=int(args.max_examples),
    )
    order = np.argsort([row["individual_full_val_nll_z_units"] for row in candidate_rows])
    if args.candidate_cap > 0:
        order = order[: int(args.candidate_cap)]
    log_probs = log_probs[order]
    candidate_rows = [candidate_rows[int(index)] for index in order]
    curve = greedy_select(
        log_probs=log_probs,
        candidate_rows=candidate_rows,
        max_size=int(args.max_size),
        min_delta=float(args.min_delta),
    )
    result = {
        "candidate_count": len(candidate_rows),
        "validation_examples": int(log_probs.shape[1]),
        "greedy_curve": curve,
        "best_ensemble_full_val_nll_z_units": curve[-1]["ensemble_full_val_nll_z_units"] if curve else None,
        "selected_model_paths": [row["model_path"] for row in curve],
    }
    output = resolve_repo_path(repo_root, args.output)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
