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
from scipy.special import logsumexp, roots_hermitenorm
from torch.utils.data import DataLoader, TensorDataset

import npe_stage1_decay as stage1
from npe_flow_stress_tests import StressCase, make_linear6_case


DEFAULT_OUTPUT_ROOT = Path("runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior")
DEFAULT_LINEAR6_OUTPUT_ROOT = Path("runs/05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4")
FOLDED_SIGN_FLOOR = -1.426941782495585
FOLDED_SIGN_FLOOR_SE = 0.0011526154301947824
LOG_2PI = math.log(2.0 * math.pi)


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


def default_output_root(model: str) -> Path:
    if model == "sign":
        return DEFAULT_OUTPUT_ROOT
    if model == "linear6":
        return DEFAULT_LINEAR6_OUTPUT_ROOT
    raise ValueError(f"Unsupported population model: {model}")


def population_target_description(model: str) -> str:
    if model == "sign":
        return "(abs(theta1), theta2)"
    if model == "linear6":
        return "(w1, ..., w6, log_sigma)"
    raise ValueError(f"Unsupported population model: {model}")


def population_kind(model: str) -> str:
    if model == "sign":
        return "sign_population_flow2_residual_nsf_ensemble"
    if model == "linear6":
        return "linear6_population_flow2_residual_nsf_ensemble"
    raise ValueError(f"Unsupported population model: {model}")


def population_description(model: str) -> str:
    if model == "sign":
        return (
            "Full-prior sign-symmetry population NPE using the single-decay "
            "Flow2 residual NSF/randperm training recipe, with folded target "
            "(abs(theta1), theta2)."
        )
    if model == "linear6":
        return (
            "Full-prior Linear6 population NPE using the single-decay Flow2 "
            "residual NSF/randperm training recipe, with target "
            "(w1, ..., w6, log_sigma)."
        )
    raise ValueError(f"Unsupported population model: {model}")


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


def sample_stress_population(
    case: StressCase,
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(n, case.z_dim),
    )
    x = case.simulate_x(z, rng)
    return case.context(x).astype(np.float32), z.astype(np.float32)


def sample_population(
    *,
    model: str,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model == "sign":
        return sample_sign_population(n=n, seed=seed)
    if model == "linear6":
        return sample_stress_population(make_linear6_case(), n=n, seed=seed)
    raise ValueError(f"Unsupported population model: {model}")


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


def linear6_sufficient_stats(x_context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = make_linear6_case()
    d_w = case.z_dim - 1
    coef = np.asarray(x_context[:, :d_w], dtype=np.float64)
    sigma_hat = np.exp(np.asarray(x_context[:, d_w], dtype=np.float64))
    n_obs = 32
    residual_sq = n_obs * sigma_hat * sigma_hat
    projected_sq = n_obs * np.sum(coef * coef, axis=1)
    return coef, projected_sq, residual_sq


def linear6_log_py_given_log_sigma(
    *,
    projected_sq: np.ndarray,
    residual_sq: np.ndarray,
    log_sigma: np.ndarray,
) -> np.ndarray:
    n_obs = 32
    d_w = 6
    prior_std_w = 1.25
    sigma2 = np.exp(2.0 * log_sigma)
    projected_var = sigma2 + n_obs * prior_std_w * prior_std_w
    return -0.5 * (
        n_obs * LOG_2PI
        + d_w * np.log(projected_var)
        + (n_obs - d_w) * np.log(sigma2)
        + projected_sq / projected_var
        + residual_sq / sigma2
    )


def normal_logpdf_1d(value: np.ndarray, mean: float, std: float) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized * standardized - math.log(std) - 0.5 * LOG_2PI


def linear6_log_evidence(
    x_context: np.ndarray,
    *,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    _, projected_sq, residual_sq = linear6_sufficient_stats(x_context)
    nodes, weights = roots_hermitenorm(int(quadrature_order))
    log_weights = np.log(weights) - 0.5 * LOG_2PI
    log_sigma_mean = math.log(0.25)
    log_sigma_std = 0.50
    log_sigma_nodes = log_sigma_mean + log_sigma_std * np.asarray(nodes, dtype=np.float64)
    result = np.empty(x_context.shape[0], dtype=np.float64)
    for start in range(0, x_context.shape[0], chunk_size):
        stop = min(start + chunk_size, x_context.shape[0])
        log_terms = (
            log_weights[None, :]
            + linear6_log_py_given_log_sigma(
                projected_sq=projected_sq[start:stop, None],
                residual_sq=residual_sq[start:stop, None],
                log_sigma=log_sigma_nodes[None, :],
            )
        )
        result[start:stop] = logsumexp(log_terms, axis=1)
    return result


def linear6_exact_posterior_nll(
    *,
    x_context: np.ndarray,
    z_raw: np.ndarray,
    quadrature_order: int,
    chunk_size: int,
) -> np.ndarray:
    coef, projected_sq, residual_sq = linear6_sufficient_stats(x_context)
    d_w = 6
    n_obs = 32
    prior_std_w = 1.25
    log_sigma_mean = math.log(0.25)
    log_sigma_std = 0.50
    log_sigma = np.asarray(z_raw[:, -1], dtype=np.float64)
    sigma2 = np.exp(2.0 * log_sigma)
    posterior_var = 1.0 / (1.0 / (prior_std_w * prior_std_w) + n_obs / sigma2)
    shrink = posterior_var * n_obs / sigma2
    posterior_mean = shrink[:, None] * coef
    delta = np.asarray(z_raw[:, :d_w], dtype=np.float64) - posterior_mean
    log_w_given_sigma_x = -0.5 * (
        d_w * LOG_2PI
        + d_w * np.log(posterior_var)
        + np.sum(delta * delta, axis=1) / posterior_var
    )
    log_py_sigma = linear6_log_py_given_log_sigma(
        projected_sq=projected_sq,
        residual_sq=residual_sq,
        log_sigma=log_sigma,
    )
    log_evidence = linear6_log_evidence(
        x_context,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
    )
    log_sigma_posterior = (
        normal_logpdf_1d(log_sigma, log_sigma_mean, log_sigma_std)
        + log_py_sigma
        - log_evidence
    )
    return -(log_w_given_sigma_x + log_sigma_posterior)


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
    model_name: str,
    members: list[dict[str, object]],
    validation_examples: int,
    validation_seed: int,
    batch_size: int,
    device: torch.device,
    linear6_quadrature_order: int,
) -> dict[str, Any]:
    x_val, z_val = sample_population(model=model_name, n=validation_examples, seed=validation_seed)
    individual_chunks: list[list[np.ndarray]] = [[] for _ in members]
    ensemble_chunks: list[np.ndarray] = []
    exact_chunks: list[np.ndarray] = []
    start_time = time.perf_counter()
    for start in range(0, validation_examples, batch_size):
        stop = min(start + batch_size, validation_examples)
        batch_x = x_val[start:stop]
        batch_z = z_val[start:stop]
        if model_name == "linear6":
            exact_chunks.append(
                linear6_exact_posterior_nll(
                    x_context=batch_x,
                    z_raw=batch_z,
                    quadrature_order=linear6_quadrature_order,
                    chunk_size=batch_size,
                )
            )
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
    output = {
        "validation_examples": int(validation_examples),
        "validation_seed": int(validation_seed),
        "evaluation_seconds": float(time.perf_counter() - start_time),
        "individual_nll": [summarize(values) for values in individual_nll],
        "best_individual_nll": float(min(np.mean(values) for values in individual_nll)),
        "ensemble_nll": ensemble_summary,
    }
    if model_name == "sign":
        gap = float(ensemble_summary["mean"] - FOLDED_SIGN_FLOOR)
        combined_se = math.sqrt(float(ensemble_summary["std_error"]) ** 2 + FOLDED_SIGN_FLOOR_SE**2)
        output.update({
            "floor": {
            "estimate": FOLDED_SIGN_FLOOR,
            "standard_error": FOLDED_SIGN_FLOOR_SE,
            "coordinate_target": "(abs(theta1), theta2)",
            },
            "ensemble_gap_to_floor": gap,
            "combined_standard_error": combined_se,
            "gap_z_score": gap / combined_se if combined_se > 0 else None,
        })
    elif model_name == "linear6":
        exact_nll = np.concatenate(exact_chunks)
        gap_samples = ensemble_nll - exact_nll
        paired_gap = summarize(gap_samples)
        floor_summary = summarize(exact_nll)
        output.update({
            "floor": {
                "estimate": float(floor_summary["mean"]),
                "standard_error": float(floor_summary["std_error"]),
                "coordinate_target": "(w1, ..., w6, log_sigma)",
                "method": (
                    "Linear-Gaussian conditional posterior with one-dimensional "
                    f"Gauss-Hermite evidence integration, order {linear6_quadrature_order}."
                ),
                "summary": floor_summary,
            },
            "ensemble_gap_to_floor": float(paired_gap["mean"]),
            "paired_gap_standard_error": float(paired_gap["std_error"]),
            "gap_z_score": float(paired_gap["mean"]) / float(paired_gap["std_error"])
            if float(paired_gap["std_error"]) > 0.0
            else None,
            "paired_gap_summary": paired_gap,
        })
    else:
        raise ValueError(f"Unsupported population model: {model_name}")
    return output


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
    train_x, train_z = sample_population(model=args.model, n=int(args.train_simulations), seed=seed)
    val_x, val_z = sample_population(model=args.model, n=int(args.val_simulations), seed=seed + 1)
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
        f"{args.model} member {member_index} seed={seed} train={args.train_simulations} "
        f"x_dim={train_x_std.shape[1]} z_dim={train_z_std.shape[1]} batches={len(train_loader)} device={device}",
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
    model_path = results_dir / f"{args.model}_population_spline_flow_model.pt"
    checkpoint = {
        "family": "spline_flow",
        "state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "config": asdict(config),
        "target": population_target_description(str(args.model)),
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
        "best_val_nll_target_units": float(metrics["best_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["best_val_nll"]))
        else None,
        "final_train_nll_standardized": float(metrics["final_train_nll"]),
        "final_train_nll_target_units": float(metrics["final_train_nll"] + z_log_det),
        "final_val_nll_standardized": float(metrics["final_val_nll"]),
        "final_val_nll_target_units": float(metrics["final_val_nll"] + z_log_det)
        if math.isfinite(float(metrics["final_val_nll"]))
        else None,
        "epochs_completed": int(metrics["epochs_completed"]),
        "optimizer_steps": int(metrics["optimizer_steps"]),
        "training_seconds": float(metrics["training_seconds"]),
        "history": metrics["history"],
        "config": asdict(config),
    }
    if args.model == "sign":
        summary["best_val_nll_folded_units"] = summary["best_val_nll_target_units"]
        summary["final_train_nll_folded_units"] = summary["final_train_nll_target_units"]
        summary["final_val_nll_folded_units"] = summary["final_val_nll_target_units"]
    summary_path = results_dir / f"{args.model}_population_member_summary.json"
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
        description="Train a full-prior stress-model population NPE with the single-decay Flow2 recipe."
    )
    parser.add_argument("--model", choices=("sign", "linear6"), default="sign")
    parser.add_argument("--output-root", type=Path, default=None)
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
    parser.add_argument("--linear6-quadrature-order", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(str(args.model))
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
        model_name=str(args.model),
        members=members,
        validation_examples=int(args.validation_examples),
        validation_seed=int(args.validation_seed),
        batch_size=int(args.eval_batch_size),
        device=device,
        linear6_quadrature_order=int(args.linear6_quadrature_order),
    )
    summary = {
        "kind": population_kind(str(args.model)),
        "description": population_description(str(args.model)),
        "target": population_target_description(str(args.model)),
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
    summary_path = results_dir / f"{args.model}_population_ensemble_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
