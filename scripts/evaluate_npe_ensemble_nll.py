from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

import npe_stage1_decay as stage1


DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def model_paths_from_summary(summary_path: Path, repo_root: Path) -> list[Path]:
    payload = read_json(summary_path)
    if "model_pt" in payload:
        model_path = resolve_repo_path(repo_root, payload.get("model_pt"))
        return [model_path] if model_path is not None else []
    paths: list[Path] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        model_path = resolve_repo_path(repo_root, row.get("model_pt"))
        if model_path is not None:
            paths.append(model_path)
    return paths


def stage1_config_from_checkpoint(config: dict[str, Any]) -> stage1.Stage1Config:
    field_names = {field.name for field in fields(stage1.Stage1Config)}
    values = {key: value for key, value in config.items() if key in field_names}
    values["progress_jsonl"] = None
    return stage1.Stage1Config(**values)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    family = str(checkpoint["family"])
    config = stage1_config_from_checkpoint(dict(checkpoint["config"]))
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
    model = stage1.make_model(
        family,
        config,
        x_dim=int(x_mean.shape[0]),
        z_dim=int(z_mean.shape[0]),
    ).to(device)
    state_dict = checkpoint["state_dict"]
    if any(str(key).startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "family": family,
        "config": checkpoint["config"],
        "x_mean": x_mean,
        "x_std": np.asarray(checkpoint["x_std"], dtype=np.float64),
        "z_mean": z_mean,
        "z_std": np.asarray(checkpoint["z_std"], dtype=np.float64),
        "checkpoint_path": path,
    }


def collect_model_paths(args: argparse.Namespace) -> list[Path]:
    repo_root = args.repo_root.resolve()
    paths: list[Path] = []
    for value in args.model:
        model_path = resolve_repo_path(repo_root, value)
        if model_path is not None:
            paths.append(model_path)
    for value in args.summary:
        summary_path = resolve_repo_path(repo_root, value)
        if summary_path is not None:
            paths.extend(model_paths_from_summary(summary_path, repo_root))
    for value in args.summary_glob:
        paths.extend(
            model_path
            for summary_path in sorted(repo_root.glob(value))
            for model_path in model_paths_from_summary(summary_path, repo_root)
        )
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    missing = [path for path in unique_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing model path(s): " + ", ".join(str(path) for path in missing))
    if not unique_paths:
        raise SystemExit("No model checkpoints were provided.")
    return unique_paths


def batch_log_prob_z(
    *,
    model: torch.nn.Module,
    state: dict[str, object],
    x_raw: np.ndarray,
    z_raw: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    config = state.get("config")
    context_features = "raw"
    if isinstance(config, dict):
        context_features = str(config.get("context_features", "raw"))
    x_context = stage1.transform_context_features(x_raw, context_features)
    x_mean = np.asarray(state["x_mean"], dtype=np.float32)
    x_std = np.asarray(state["x_std"], dtype=np.float32)
    z_mean = np.asarray(state["z_mean"], dtype=np.float32)
    z_std = np.asarray(state["z_std"], dtype=np.float32)
    x_standardized = ((x_context - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    z_standardized = ((z_raw - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
    x_tensor = torch.from_numpy(x_standardized).to(device)
    z_tensor = torch.from_numpy(z_standardized).to(device)
    log_det = float(np.log(z_std.astype(np.float64)).sum())
    return model.log_prob(z_tensor, x_tensor) - log_det


def evaluate_ensemble(
    *,
    model_paths: list[Path],
    validation_cache: Path,
    device: torch.device,
    batch_size: int,
    max_examples: int,
) -> dict[str, Any]:
    data = np.load(validation_cache)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    z_val = np.asarray(data["z_val"], dtype=np.float32)
    if max_examples > 0:
        x_val = x_val[:max_examples]
        z_val = z_val[:max_examples]
    models_and_states = [load_checkpoint(path, device) for path in model_paths]
    individual_sums = np.zeros(len(models_and_states), dtype=np.float64)
    ensemble_sum = 0.0
    count = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, x_val.shape[0], batch_size):
            stop = min(start + batch_size, x_val.shape[0])
            batch_x = x_val[start:stop]
            batch_z = z_val[start:stop]
            log_probs = []
            for index, (model, state) in enumerate(models_and_states):
                log_prob = batch_log_prob_z(
                    model=model,
                    state=state,
                    x_raw=batch_x,
                    z_raw=batch_z,
                    device=device,
                )
                individual_sums[index] += float((-log_prob).sum().detach().cpu())
                log_probs.append(log_prob)
            stacked = torch.stack(log_probs, dim=0)
            mixture_log_prob = torch.logsumexp(stacked, dim=0) - math.log(len(log_probs))
            ensemble_sum += float((-mixture_log_prob).sum().detach().cpu())
            count += stop - start
    evaluation_seconds = time.perf_counter() - start_time
    individual_nlls = [float(value / count) for value in individual_sums]
    return {
        "ensemble_size": len(model_paths),
        "validation_examples": int(count),
        "ensemble_full_val_nll_z_units": float(ensemble_sum / count),
        "individual_full_val_nll_z_units": individual_nlls,
        "best_individual_full_val_nll_z_units": float(min(individual_nlls)),
        "ensemble_gain_vs_best_individual": float((ensemble_sum / count) - min(individual_nlls)),
        "evaluation_seconds": evaluation_seconds,
        "model_paths": [str(path) for path in model_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved NPE checkpoint ensemble with exact log-mean-exp NLL."
    )
    parser.add_argument("--model", action="append", default=[], help="Path to a saved model checkpoint.")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Path to a run or aggregate summary JSON containing model_pt entries.",
    )
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=[],
        help="Repo-root-relative glob for summary JSON files containing model_pt entries.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    validation_cache = resolve_repo_path(repo_root, args.validation_cache)
    if validation_cache is None or not validation_cache.exists():
        raise SystemExit(f"Missing validation cache: {validation_cache}")
    model_paths = collect_model_paths(args)
    result = evaluate_ensemble(
        model_paths=model_paths,
        validation_cache=validation_cache,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
        max_examples=int(args.max_examples),
    )
    if args.output is not None:
        output = resolve_repo_path(repo_root, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
