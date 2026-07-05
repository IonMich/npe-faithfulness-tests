from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp
from torch.utils.data import DataLoader, TensorDataset

import npe_stage1_decay as stage1


DEFAULT_OUTPUT_ROOT = Path("runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior")
FOLDED_SIGN_FLOOR = -1.426941782495585
FOLDED_SIGN_FLOOR_SE = 0.0011526154301947824


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def summarize(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "std_error": float(np.std(finite, ddof=1) / math.sqrt(finite.size))
        if finite.size > 1
        else 0.0,
        "min": float(np.min(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def runtime_metadata() -> dict[str, object]:
    return {
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def sample_sign_population(
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.8, size=(n, 2)).astype(np.float64)
    x = np.column_stack(
        [
            theta[:, 0] * theta[:, 0] + rng.normal(0.0, 0.22, size=n),
            theta[:, 1] + rng.normal(0.0, 0.16, size=n),
        ]
    )
    folded = np.column_stack([np.abs(theta[:, 0]), theta[:, 1]])
    return x.astype(np.float32), folded.astype(np.float32)


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean[None, :]) / std[None, :]).astype(np.float32)


def make_config(args: argparse.Namespace, *, seed: int, train_simulations: int) -> stage1.Stage1Config:
    return stage1.Stage1Config(
        train_simulations=int(train_simulations),
        val_simulations=int(args.val_simulations),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        mdn_components=5,
        flow_layers=int(args.flow_layers),
        flow_context_dim=64,
        seed=int(seed),
        observed_seed=int(seed),
        requested_device=str(args.device),
        families=["spline_flow"],
        posterior_samples=0,
        reference_grid_size=0,
        train_sampler="random",
        context_features="raw",
        spline_bins=int(args.spline_bins),
        lr_schedule=str(args.lr_schedule),
        lr_eta_min=float(args.lr_eta_min),
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_decay_epochs=int(args.lr_decay_epochs),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_eps=float(args.adam_eps),
        validation_every_epochs=int(args.validation_every_epochs),
        skip_training_validation=bool(args.skip_training_validation),
        torch_compile=str(args.torch_compile),
        grad_clip_norm=float(args.grad_clip_norm),
        ema_decay=float(args.ema_decay),
        batching_mode=str(args.batching_mode),
        max_optimizer_steps=int(args.max_optimizer_steps),
        loss_weight_mode="none",
        loss_tail_weight=3.0,
        target_transform="none",
        target_ridge=1e-3,
        flow_activation=str(args.flow_activation),
        flow_residual=bool(args.flow_residual),
        flow_randperm=bool(args.flow_randperm),
        flow_passes=int(args.flow_passes),
        flow_kind="nsf",
    )


def evaluate_model_log_prob(
    *,
    model: torch.nn.Module,
    x_raw: np.ndarray,
    z_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x_standardized = standardize(x_raw, x_mean, x_std)
    z_standardized = standardize(z_raw, z_mean, z_std)
    log_det = float(np.log(z_std.astype(np.float64)).sum())
    return model.log_prob(
        torch.from_numpy(z_standardized).to(device),
        torch.from_numpy(x_standardized).to(device),
    ) - log_det


@torch.no_grad()
def evaluate_population_nll(
    *,
    members: list[dict[str, object]],
    validation_examples: int,
    validation_seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    x_val, z_val = sample_sign_population(n=validation_examples, seed=validation_seed)
    individual_chunks: list[list[np.ndarray]] = [[] for _ in members]
    ensemble_chunks: list[np.ndarray] = []
    start_time = time.perf_counter()
    for start in range(0, validation_examples, batch_size):
        stop = min(start + batch_size, validation_examples)
        batch_x = x_val[start:stop]
        batch_z = z_val[start:stop]
        log_probs = []
        for index, member in enumerate(members):
            log_prob = evaluate_model_log_prob(
                model=member["model"],
                x_raw=batch_x,
                z_raw=batch_z,
                x_mean=member["x_mean"],
                x_std=member["x_std"],
                z_mean=member["z_mean"],
                z_std=member["z_std"],
                device=device,
            )
            log_prob_np = log_prob.detach().cpu().numpy().astype(np.float64)
            individual_chunks[index].append(-log_prob_np)
            log_probs.append(log_prob_np)
        stacked = np.stack(log_probs, axis=0)
        ensemble_log_prob = logsumexp(stacked, axis=0) - math.log(len(members))
        ensemble_chunks.append(-ensemble_log_prob)

    individual_nll = [np.concatenate(chunks) for chunks in individual_chunks]
    ensemble_nll = np.concatenate(ensemble_chunks)
    ensemble_summary = summarize(ensemble_nll)
    gap = float(ensemble_summary["mean"] - FOLDED_SIGN_FLOOR)
    combined_se = math.sqrt(float(ensemble_summary["std_error"]) ** 2 + FOLDED_SIGN_FLOOR_SE**2)
    return {
        "validation_examples": int(validation_examples),
        "validation_seed": int(validation_seed),
        "evaluation_seconds": float(time.perf_counter() - start_time),
        "individual_nll": [summarize(values) for values in individual_nll],
        "best_individual_nll": float(min(np.mean(values) for values in individual_nll)),
        "ensemble_nll": ensemble_summary,
        "floor": {
            "estimate": FOLDED_SIGN_FLOOR,
            "standard_error": FOLDED_SIGN_FLOOR_SE,
            "coordinate_target": "(abs(theta1), theta2)",
        },
        "ensemble_gap_to_floor": gap,
        "combined_standard_error": combined_se,
        "gap_z_score": gap / combined_se if combined_se > 0 else None,
    }


def train_member(
    *,
    args: argparse.Namespace,
    seed: int,
    member_index: int,
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    member_dir = output_root / f"member_{member_index:02d}_seed{seed}"
    results_dir = member_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = results_dir / "training_progress.jsonl"

    data_start = time.perf_counter()
    train_x, train_z = sample_sign_population(n=int(args.train_simulations), seed=seed)
    val_x, val_z = sample_sign_population(n=int(args.val_simulations), seed=seed + 1)
    x_mean = train_x.mean(axis=0).astype(np.float64)
    x_std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float64)
    z_mean = train_z.mean(axis=0).astype(np.float64)
    z_std = np.maximum(train_z.std(axis=0), 1e-6).astype(np.float64)
    train_x_std = standardize(train_x, x_mean, x_std)
    train_z_std = standardize(train_z, z_mean, z_std)
    val_x_std = standardize(val_x, x_mean, x_std)
    val_z_std = standardize(val_z, z_mean, z_std)
    data_seconds = time.perf_counter() - data_start

    config = replace(
        make_config(args, seed=seed, train_simulations=int(args.train_simulations)),
        progress_jsonl=progress_jsonl,
        progress_nll_offset=float(np.log(z_std).sum()),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)),
        batch_size=int(args.batch_size),
        shuffle=str(args.batching_mode) == "dataloader",
        generator=torch.Generator(device="cpu").manual_seed(seed + 2),
    )
    print(
        f"member {member_index} seed={seed} train={args.train_simulations} "
        f"batches={len(train_loader)} device={device}",
        flush=True,
    )
    model, metrics = stage1.train_one_model(
        family="spline_flow",
        config=config,
        train_loader=train_loader,
        val_x=torch.from_numpy(val_x_std),
        val_z=torch.from_numpy(val_z_std),
        device=device,
        x_dim=train_x_std.shape[1],
        z_dim=train_z_std.shape[1],
    )
    model_path = results_dir / "sign_population_spline_flow_model.pt"
    checkpoint = {
        "family": "spline_flow",
        "state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "config": asdict(config),
        "target": "(abs(theta1), theta2)",
        "runtime": runtime_metadata(),
    }
    torch.save(checkpoint, model_path)
    z_log_det = float(np.log(z_std).sum())
    summary = {
        "seed": int(seed),
        "member_index": int(member_index),
        "model_pt": str(model_path),
        "data_seconds": float(data_seconds),
        "model_parameters": int(sum(param.numel() for param in model.parameters())),
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "z_log_det": z_log_det,
        "best_val_nll_standardized": float(metrics["best_val_nll"]),
        "best_val_nll_folded_units": float(metrics["best_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["best_val_nll"]))
        else None,
        "final_train_nll_standardized": float(metrics["final_train_nll"]),
        "final_train_nll_folded_units": float(metrics["final_train_nll"] + z_log_det),
        "final_val_nll_standardized": float(metrics["final_val_nll"]),
        "final_val_nll_folded_units": float(metrics["final_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["final_val_nll"]))
        else None,
        "epochs_completed": int(metrics["epochs_completed"]),
        "optimizer_steps": int(metrics["optimizer_steps"]),
        "training_seconds": float(metrics["training_seconds"]),
        "history": metrics["history"],
        "config": asdict(config),
    }
    summary_path = results_dir / "sign_population_member_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "summary": summary,
        "summary_json": str(summary_path),
        "model_pt": str(model_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full-prior sign-symmetry population NPE with the single-decay Flow2 recipe."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=parse_int_list, default=(20260901, 20260902, 20260903, 20260904))
    parser.add_argument("--train-simulations", type=int, default=2_048_000)
    parser.add_argument("--val-simulations", type=int, default=65_536)
    parser.add_argument("--validation-examples", type=int, default=1_000_000)
    parser.add_argument("--validation-seed", type=int, default=20260705)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.00325)
    parser.add_argument("--weight-decay", type=float, default=0.0002)
    parser.add_argument("--hidden-dim", type=int, default=80)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--spline-bins", type=int, default=8)
    parser.add_argument("--flow-activation", choices=stage1.FLOW_ACTIVATIONS, default="relu")
    parser.add_argument("--flow-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-randperm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-passes", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine_epoch", "cosine_step", "one_cycle"), default="cosine_step")
    parser.add_argument("--lr-eta-min", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-decay-epochs", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--validation-every-epochs", type=int, default=1)
    parser.add_argument("--skip-training-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", choices=("none", "default", "reduce_overhead"), default="none")
    parser.add_argument("--grad-clip-norm", type=float, default=20.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--batching-mode", choices=("dataloader", "pre_shuffle", "sequential"), default="pre_shuffle")
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    started = time.perf_counter()
    members = []
    for member_index, seed in enumerate(args.seeds, start=1):
        members.append(
            train_member(
                args=args,
                seed=int(seed),
                member_index=member_index,
                device=device,
                output_root=output_root,
            )
        )
    evaluation = evaluate_population_nll(
        members=members,
        validation_examples=int(args.validation_examples),
        validation_seed=int(args.validation_seed),
        batch_size=int(args.eval_batch_size),
        device=device,
    )
    summary = {
        "kind": "sign_population_flow2_residual_nsf_ensemble",
        "description": (
            "Full-prior sign-symmetry population NPE using the single-decay "
            "Flow2 residual NSF/randperm training recipe, with folded target "
            "(abs(theta1), theta2)."
        ),
        "device": str(device),
        "wall_seconds": float(time.perf_counter() - started),
        "recipe": {
            "ensemble_size": len(members),
            "seeds": [int(seed) for seed in args.seeds],
            "train_simulations_per_member": int(args.train_simulations),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "flow_layers": int(args.flow_layers),
            "flow_residual": bool(args.flow_residual),
            "flow_randperm": bool(args.flow_randperm),
            "spline_bins": int(args.spline_bins),
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
            "lr_schedule": str(args.lr_schedule),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "batching_mode": str(args.batching_mode),
        },
        "members": [
            {
                "summary_json": member["summary_json"],
                "model_pt": member["model_pt"],
                "member_summary": member["summary"],
            }
            for member in members
        ],
        "evaluation": evaluation,
    }
    summary_path = results_dir / "sign_population_ensemble_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
