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


def set_torch_threads(thread_count: int | None) -> None:
    if thread_count is None:
        return
    torch.set_num_threads(int(thread_count))
    try:
        torch.set_num_interop_threads(max(1, int(thread_count)))
    except RuntimeError:
        pass


def stage1_config_from_checkpoint(config: dict[str, Any]) -> stage1.Stage1Config:
    field_names = {field.name for field in fields(stage1.Stage1Config)}
    values = {key: value for key, value in config.items() if key in field_names}
    values["progress_jsonl"] = None
    return stage1.Stage1Config(**values)


def load_checkpoint(path: Path, device: torch.device, *, from_scratch: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    family = str(checkpoint["family"])
    config = dict(checkpoint["config"])
    stage_config = stage1_config_from_checkpoint(config)
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
    if device.type == "mps":
        torch.mps.manual_seed(int(config["seed"]))
    model = stage1.make_model(
        family,
        stage_config,
        x_dim=int(x_mean.shape[0]),
        z_dim=int(z_mean.shape[0]),
    ).to(device)
    if not from_scratch:
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
        "config": config,
        "x_mean": x_mean,
        "x_std": np.asarray(checkpoint["x_std"], dtype=np.float64),
        "z_mean": z_mean,
        "z_std": np.asarray(checkpoint["z_std"], dtype=np.float64),
        "source_checkpoint": path,
    }


def save_checkpoint(
    *,
    model: torch.nn.Module,
    state: dict[str, Any],
    output_path: Path,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    continuation: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": state["family"],
            "state_dict": model.state_dict(),
            "x_mean": state["x_mean"],
            "x_std": state["x_std"],
            "z_mean": state["z_mean"],
            "z_std": state["z_std"],
            "config": config,
            "optimizer_state_dict": optimizer.state_dict(),
            "continuation": continuation,
            "runtime": {
                "created_at_unix": time.time(),
            },
        },
        output_path,
    )


def prepare_training_arrays(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    train_simulations: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    seed = int(config["seed"])
    sampler = str(config.get("train_sampler", "random"))
    context_features = str(config.get("context_features", "raw"))
    x_raw, z_raw, _ = stage1.sample_decay_pairs(
        n=int(train_simulations),
        seed=seed,
        sampler=sampler,
    )
    x_context = stage1.transform_context_features(x_raw, context_features)
    x_standardized = stage1.standardize(
        x_context,
        np.asarray(state["x_mean"], dtype=np.float64),
        np.asarray(state["x_std"], dtype=np.float64),
    ).astype(np.float32)
    z_standardized = stage1.standardize(
        z_raw,
        np.asarray(state["z_mean"], dtype=np.float64),
        np.asarray(state["z_std"], dtype=np.float64),
    ).astype(np.float32)
    z_log_det = float(np.log(np.asarray(state["z_std"], dtype=np.float64)).sum())
    return torch.from_numpy(x_standardized), torch.from_numpy(z_standardized), z_log_det


def train_continuation(args: argparse.Namespace) -> dict[str, Any]:
    set_torch_threads(args.torch_threads)
    device = stage1.choose_training_device(str(args.device))
    model, state = load_checkpoint(args.checkpoint, device, from_scratch=bool(args.from_scratch))
    config = dict(state["config"])
    base_epoch_default = 0 if args.from_scratch else int(config.get("epochs", 0))
    base_epoch = int(args.base_epoch if args.base_epoch is not None else base_epoch_default)
    train_simulations = int(args.train_simulations or config["train_simulations"])
    batch_size = int(args.batch_size or config["batch_size"])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else config["learning_rate"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.0))
    train_x, train_z, z_log_det = prepare_training_arrays(
        config=config,
        state=state,
        train_simulations=train_simulations,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.999))),
        eps=float(config.get("adam_eps", 1e-8)),
        weight_decay=weight_decay,
    )
    batches_per_epoch = math.ceil(train_simulations / batch_size)
    if args.schedule_unit == "step":
        total_steps = max(1, int(args.extra_epochs) * batches_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(args.lr_eta_min),
        )
    else:
        decay_epochs = max(1, int(args.extra_epochs))
        eta_ratio = float(args.lr_eta_min) / learning_rate if learning_rate > 0.0 else 0.0

        def cosine_epoch_factor(epoch_index: int) -> float:
            t = min(max(0, int(epoch_index)), decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * float(t) / float(decay_epochs)))
            return float(eta_ratio + (1.0 - eta_ratio) * cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_epoch_factor)
    progress_path = args.output_dir / "continuation_progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    snapshots: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    def append_progress(record: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready(record), sort_keys=True) + "\n")

    append_progress(
        {
            "event": "start",
            "source_checkpoint": args.checkpoint,
            "from_scratch": bool(args.from_scratch),
            "base_epoch": base_epoch,
            "extra_epochs": int(args.extra_epochs),
            "learning_rate": learning_rate,
            "lr_eta_min": float(args.lr_eta_min),
            "schedule_unit": args.schedule_unit,
            "batch_size": batch_size,
            "train_simulations": train_simulations,
            "batches_per_epoch": batches_per_epoch,
            "z_log_det": z_log_det,
        }
    )

    model.train()
    for extra_epoch in range(1, int(args.extra_epochs) + 1):
        absolute_epoch = base_epoch + extra_epoch
        epoch_start = time.perf_counter()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["seed"]) + 2 + train_simulations + absolute_epoch - 1
        )
        permutation = torch.randperm(train_simulations, generator=generator)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, train_simulations, batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x[indices].to(device)
            batch_z = train_z[indices].to(device)
            per_sample_loss = -model.log_prob(batch_z, batch_x)
            loss = per_sample_loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(config.get("grad_clip_norm", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if args.schedule_unit == "step":
                scheduler.step()
            batch_count = int(batch_x.shape[0])
            train_loss_sum += float(per_sample_loss.detach().sum().cpu())
            train_count += batch_count
        train_nll = train_loss_sum / max(train_count, 1)
        elapsed = time.perf_counter() - start_time
        append_progress(
            {
                "event": "epoch",
                "extra_epoch": extra_epoch,
                "absolute_epoch": absolute_epoch,
                "optimizer_steps": extra_epoch * batches_per_epoch,
                "lr": epoch_lr,
                "train_nll_standardized": train_nll,
                "train_nll_z_units": train_nll + z_log_det,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "elapsed_seconds": elapsed,
            }
        )
        if args.schedule_unit == "epoch":
            scheduler.step()
        should_snapshot = (
            extra_epoch == int(args.extra_epochs)
            or extra_epoch % max(1, int(args.snapshot_every)) == 0
        )
        if should_snapshot:
            snapshot_dir = args.output_dir / "snapshots" / f"epoch{absolute_epoch}"
            snapshot_path = snapshot_dir / f"{state['family']}_model.pt"
            snapshot_config = dict(config)
            snapshot_config.update(
                {
                    "epochs": absolute_epoch,
                    "learning_rate": learning_rate,
                    "lr_schedule": "cosine_step" if args.schedule_unit == "step" else "cosine_epoch",
                    "lr_eta_min": float(args.lr_eta_min),
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "train_simulations": train_simulations,
                }
            )
            continuation = {
                "source_checkpoint": str(args.checkpoint),
                "from_scratch": bool(args.from_scratch),
                "base_epoch": base_epoch,
                "extra_epoch": extra_epoch,
                "absolute_epoch": absolute_epoch,
                "learning_rate": learning_rate,
                "lr_eta_min": float(args.lr_eta_min),
                "train_nll_z_units": train_nll + z_log_det,
                "elapsed_seconds": elapsed,
                "progress_jsonl": str(progress_path),
            }
            save_checkpoint(
                model=model,
                state=state,
                output_path=snapshot_path,
                config=snapshot_config,
                optimizer=optimizer,
                continuation=continuation,
            )
            summary_path = snapshot_dir / "continuation_snapshot_summary.json"
            summary = {
                **continuation,
                "model_pt": str(snapshot_path),
                "seed": int(config["seed"]),
                "train_simulations": train_simulations,
                "batch_size": batch_size,
                "optimizer_steps": extra_epoch * batches_per_epoch,
            }
            summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
            snapshots.append(summary)

    summary = {
        "source_checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "from_scratch": bool(args.from_scratch),
        "base_epoch": base_epoch,
        "extra_epochs": int(args.extra_epochs),
        "learning_rate": learning_rate,
        "lr_eta_min": float(args.lr_eta_min),
        "schedule_unit": args.schedule_unit,
        "train_simulations": train_simulations,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "progress_jsonl": str(progress_path),
        "snapshots": snapshots,
        "total_seconds": time.perf_counter() - start_time,
    }
    (args.output_dir / "continuation_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue one saved stage1 NPE member and save epoch milestones.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-epochs", type=int, default=20)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-eta-min", type=float, default=0.0)
    parser.add_argument("--schedule-unit", choices=("epoch", "step"), default="step")
    parser.add_argument("--base-epoch", type=int, default=None)
    parser.add_argument("--train-simulations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = train_continuation(parse_args())
    print(json.dumps(json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
