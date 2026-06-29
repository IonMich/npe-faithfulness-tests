from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import npe_stage1_decay as stage1


DEFAULT_MODEL = Path(
    "runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/"
    "results/mdn_model.pt"
)
DEFAULT_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
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


def config_from_checkpoint(config_dict: dict[str, object]) -> stage1.Stage1Config:
    return stage1.Stage1Config(
        train_simulations=int(config_dict["train_simulations"]),
        val_simulations=int(config_dict["val_simulations"]),
        epochs=int(config_dict["epochs"]),
        batch_size=int(config_dict["batch_size"]),
        learning_rate=float(config_dict["learning_rate"]),
        weight_decay=float(config_dict["weight_decay"]),
        hidden_dim=int(config_dict["hidden_dim"]),
        hidden_layers=int(config_dict["hidden_layers"]),
        mdn_components=int(config_dict["mdn_components"]),
        flow_layers=int(config_dict.get("flow_layers", 6)),
        flow_context_dim=int(config_dict.get("flow_context_dim", 64)),
        seed=int(config_dict["seed"]),
        observed_seed=int(config_dict["observed_seed"]),
        requested_device=str(config_dict.get("requested_device", "cpu")),
        families=list(config_dict.get("families", [])),
        posterior_samples=int(config_dict.get("posterior_samples", 0)),
        reference_grid_size=int(config_dict.get("reference_grid_size", 0)),
    )


def evaluate_cached_nll(
    *,
    model: torch.nn.Module,
    x_val: np.ndarray,
    z_val: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    total = 0.0
    count = 0
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for start_index in range(0, x_val.shape[0], batch_size):
            stop_index = min(start_index + batch_size, x_val.shape[0])
            x_batch = ((x_val[start_index:stop_index] - x_mean[None, :]) / x_std[None, :]).astype(
                np.float32,
                copy=False,
            )
            z_batch = ((z_val[start_index:stop_index] - z_mean[None, :]) / z_std[None, :]).astype(
                np.float32,
                copy=False,
            )
            x_tensor = torch.from_numpy(x_batch).to(device)
            z_tensor = torch.from_numpy(z_batch).to(device)
            loss = -model.log_prob(z_tensor, x_tensor)
            total += float(loss.detach().cpu().sum())
            count += int(stop_index - start_index)
    elapsed = time.perf_counter() - start
    nll_standardized = total / max(count, 1)
    nll_z_units = nll_standardized + float(np.log(z_std).sum())
    return {
        "pairs": int(count),
        "batch_size": int(batch_size),
        "nll_standardized": float(nll_standardized),
        "nll_z_units": float(nll_z_units),
        "seconds": float(elapsed),
        "pairs_per_second": float(count / elapsed) if elapsed > 0 else float("inf"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NLL evaluation on a cached broad validation set.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    family = str(checkpoint["family"])
    config = config_from_checkpoint(checkpoint["config"])
    config = replace(config, families=[family])
    model = stage1.make_model(family, config, x_dim=40, z_dim=3)
    model.load_state_dict(checkpoint["state_dict"])
    device = stage1.choose_training_device(args.device)
    model.to(device)

    arrays = np.load(args.validation_cache, allow_pickle=False)
    x_val = np.asarray(arrays["x_val"], dtype=np.float32)
    z_val = np.asarray(arrays["z_val"], dtype=np.float32)
    if args.max_pairs is not None:
        x_val = x_val[: args.max_pairs]
        z_val = z_val[: args.max_pairs]
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)

    result = {
        "model": args.model,
        "validation_cache": args.validation_cache,
        "validation_cache_mib": args.validation_cache.stat().st_size / (1024**2),
        "family": family,
        "device": str(device),
        "model_parameters": int(sum(param.numel() for param in model.parameters())),
        "evaluation": evaluate_cached_nll(
            model=model,
            x_val=x_val,
            z_val=z_val,
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            device=device,
            batch_size=args.batch_size,
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
