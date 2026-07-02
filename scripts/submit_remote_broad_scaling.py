from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request


DEFAULT_ENDPOINT = "http://127.0.0.1:8877"
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/11_mdn_1m_remote")
DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
)
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
)


def parse_csv_ints(value: str) -> list[int]:
    items = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "run_name": args.run_name,
        "output_root": str(args.output_root),
        "train_simulations": args.train_simulations,
        "seeds": args.seeds,
        "family": args.family,
        "device": args.device,
        "standardization_simulations": args.standardization_simulations,
        "train_sampler": args.train_sampler,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "lr_eta_min": args.lr_eta_min,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_epochs": args.lr_decay_epochs,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "validation_every_epochs": args.validation_every_epochs,
        "skip_training_validation": args.skip_training_validation,
        "max_optimizer_steps": args.max_optimizer_steps,
        "torch_compile": args.torch_compile,
        "grad_clip_norm": args.grad_clip_norm,
        "ema_decay": args.ema_decay,
        "batching_mode": args.batching_mode,
        "loss_weight_mode": args.loss_weight_mode,
        "loss_tail_weight": args.loss_tail_weight,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "mdn_components": args.mdn_components,
        "flow_layers": args.flow_layers,
        "flow_context_dim": args.flow_context_dim,
        "flow_activation": args.flow_activation,
        "flow_residual": args.flow_residual,
        "flow_randperm": args.flow_randperm,
        "flow_passes": args.flow_passes,
        "flow_kind": args.flow_kind,
        "spline_bins": args.spline_bins,
        "target_transform": args.target_transform,
        "target_ridge": args.target_ridge,
        "context_features": args.context_features,
        "jobs": args.jobs,
        "parallel_backend": args.parallel_backend,
        "torch_threads": args.torch_threads,
        "eval_batch_size": args.eval_batch_size,
        "early_stop_val_simulations": args.early_stop_val_simulations,
        "validation_cache": str(args.validation_cache),
        "validation_cache_simulations": args.validation_cache_simulations,
        "early_val_cache_simulations": args.early_val_cache_simulations,
        "panel_marginal_cache": str(args.panel_marginal_cache),
        "panel_size": args.panel_size,
        "panel_grid_size": args.panel_grid_size,
        "panel_target_sample_count": args.panel_target_sample_count,
        "panel_posterior_samples": args.panel_posterior_samples,
        "posterior_samples": args.posterior_samples,
        "context_variants": args.context_variants,
        "tail_top_k": args.tail_top_k,
        "prepare_caches": args.prepare_caches,
        "save_models": not args.no_save_models,
        "train_only": args.train_only,
        "sync": not args.no_sync,
        "dry_run": args.remote_dry_run,
    }


def post_json(endpoint: str, token: str | None, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(
        endpoint.rstrip("/") + "/train/broad-scaling",
        data=body,
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a structured broad scaling-law request to a remote train endpoint.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--token", default=os.environ.get("TRAIN_REMOTE_TOKEN"))
    parser.add_argument("--run-name", default="broad_mdn_1m")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--train-simulations",
        type=parse_csv_ints,
        default=parse_csv_ints("64000,128000,256000,512000,1000000"),
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("20260901,20260902,20260903"))
    parser.add_argument(
        "--family",
        choices=("mdn", "affine_flow", "spline_flow", "full_gaussian", "diag_gaussian"),
        default="mdn",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "auto", "cuda"), default="cpu")
    parser.add_argument("--standardization-simulations", type=int, default=60_000)
    parser.add_argument("--train-sampler", choices=("random", "lhs", "sobol"), default="random")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine_epoch", "cosine_step", "one_cycle"),
        default="constant",
    )
    parser.add_argument("--lr-eta-min", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-epochs", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--validation-every-epochs", type=int, default=1)
    parser.add_argument("--skip-training-validation", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--torch-compile", choices=("none", "default", "reduce_overhead"), default="none")
    parser.add_argument("--grad-clip-norm", type=float, default=20.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--batching-mode",
        choices=("dataloader", "pre_shuffle", "sequential"),
        default="dataloader",
    )
    parser.add_argument("--loss-weight-mode", choices=("none", "tail_balanced"), default="none")
    parser.add_argument("--loss-tail-weight", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--flow-context-dim", type=int, default=64)
    parser.add_argument("--flow-activation", choices=("relu", "elu", "gelu", "silu", "tanh"), default="relu")
    parser.add_argument("--flow-residual", action="store_true")
    parser.add_argument("--flow-randperm", action="store_true")
    parser.add_argument("--flow-kind", choices=("nsf", "maf", "naf", "gf"), default="nsf")
    parser.add_argument("--flow-passes", type=int, default=0)
    parser.add_argument("--spline-bins", type=int, default=12)
    parser.add_argument("--target-transform", choices=("none", "linear_residual"), default="none")
    parser.add_argument("--target-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--context-features",
        choices=(
            "raw",
            "decay_summary",
            "fit_summary",
            "raw_decay_summary",
            "raw_fit_summary",
            "asinh",
            "asinh_decay_summary",
            "rms_normalized",
            "rms_normalized_decay_summary",
        ),
        default="raw",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--parallel-backend", choices=("subprocess", "threads"), default="subprocess")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--early-stop-val-simulations", type=int, default=100000)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--validation-cache-simulations", type=int, default=1_000_000)
    parser.add_argument("--early-val-cache-simulations", type=int, default=0)
    parser.add_argument("--panel-marginal-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--panel-grid-size", type=int, default=180)
    parser.add_argument("--panel-target-sample-count", type=int, default=20_000)
    parser.add_argument("--panel-posterior-samples", type=int, default=20_000)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--context-variants", default="real")
    parser.add_argument("--tail-top-k", type=int, default=20)
    parser.add_argument("--no-prepare-caches", dest="prepare_caches", action="store_false")
    parser.set_defaults(prepare_caches=True)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save checkpoints without per-member final validation or posterior diagnostics.",
    )
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument(
        "--remote-dry-run",
        action="store_true",
        help="Ask the remote endpoint to run the broad sweep with --dry-run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the request JSON without submitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    result = post_json(args.endpoint, args.token, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
