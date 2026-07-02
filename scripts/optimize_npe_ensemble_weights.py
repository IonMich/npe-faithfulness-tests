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
from greedy_npe_ensemble_selection import collect_paths


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)


def compute_log_probs(
    *,
    model_paths: list[Path],
    validation_cache: Path,
    device: torch.device,
    batch_size: int,
    max_examples: int,
) -> np.ndarray:
    data = np.load(validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    z_val = np.asarray(data["z_val"], dtype=np.float32)
    if max_examples > 0:
        x_val = x_val[:max_examples]
        z_val = z_val[:max_examples]

    log_probs = np.empty((len(model_paths), x_val.shape[0]), dtype=np.float32)
    for model_index, path in enumerate(model_paths):
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
        del model
    return log_probs


def mixture_nll(log_probs: torch.Tensor, log_weights: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(log_probs + log_weights[:, None], dim=0).mean()


def fit_weights(
    *,
    log_probs: np.ndarray,
    fit_examples: int,
    iterations: int,
    lr: float,
) -> np.ndarray:
    if fit_examples <= 0 or fit_examples > log_probs.shape[1]:
        fit_examples = log_probs.shape[1]
    fit_tensor = torch.from_numpy(log_probs[:, :fit_examples])
    logits = torch.zeros(log_probs.shape[0], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        log_weights = torch.log_softmax(logits, dim=0)
        loss = mixture_nll(fit_tensor, log_weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64)


def evaluate_weights(log_probs: np.ndarray, weights: np.ndarray, start: int, stop: int) -> float:
    subset = log_probs[:, start:stop]
    log_weights = np.log(np.maximum(weights, 1e-30))[:, None]
    mixture = np.logaddexp.reduce(subset.astype(np.float64) + log_weights, axis=0)
    return float(-np.mean(mixture))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize convex weights for saved NPE ensembles.")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-glob", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--fit-examples", type=int, default=200_000)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    validation_cache = resolve_repo_path(repo_root, args.validation_cache)
    if validation_cache is None or not validation_cache.exists():
        raise SystemExit(f"Missing validation cache: {validation_cache}")
    model_paths = collect_paths(args)
    start = time.perf_counter()
    log_probs = compute_log_probs(
        model_paths=model_paths,
        validation_cache=validation_cache,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
        max_examples=int(args.max_examples),
    )
    individual = [-float(row.mean(dtype=np.float64)) for row in log_probs]
    order = np.argsort(individual)
    if args.candidate_cap > 0:
        order = order[: int(args.candidate_cap)]
        log_probs = log_probs[order]
        model_paths = [model_paths[int(index)] for index in order]
        individual = [individual[int(index)] for index in order]

    weights = fit_weights(
        log_probs=log_probs,
        fit_examples=int(args.fit_examples),
        iterations=int(args.iterations),
        lr=float(args.lr),
    )
    count = log_probs.shape[1]
    fit_stop = min(max(0, int(args.fit_examples)), count)
    if fit_stop == 0:
        fit_stop = count
    uniform = np.full(log_probs.shape[0], 1.0 / log_probs.shape[0], dtype=np.float64)
    result: dict[str, Any] = {
        "candidate_count": int(log_probs.shape[0]),
        "validation_examples": int(count),
        "fit_examples": int(fit_stop),
        "best_individual_full_nll": float(min(individual)),
        "uniform_full_nll": evaluate_weights(log_probs, uniform, 0, count),
        "weighted_full_nll": evaluate_weights(log_probs, weights, 0, count),
        "uniform_fit_nll": evaluate_weights(log_probs, uniform, 0, fit_stop),
        "weighted_fit_nll": evaluate_weights(log_probs, weights, 0, fit_stop),
        "uniform_holdout_nll": evaluate_weights(log_probs, uniform, fit_stop, count) if fit_stop < count else None,
        "weighted_holdout_nll": evaluate_weights(log_probs, weights, fit_stop, count) if fit_stop < count else None,
        "elapsed_seconds": float(time.perf_counter() - start),
        "members": [
            {
                "model_path": str(path),
                "individual_full_nll": float(nll),
                "weight": float(weight),
            }
            for path, nll, weight in zip(model_paths, individual, weights, strict=True)
            if weight > 1e-4
        ],
    }
    output = resolve_repo_path(repo_root, args.output)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
