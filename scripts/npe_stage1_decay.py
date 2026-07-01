from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import artifact_paths as ap
from typing import Iterable

import corner
import matplotlib
import numpy as np
import torch
import zuko
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_decay_samplers import (
    build_grid_reference,
    compare_to_reference,
    load_samples,
    summarize_samples,
)
from corner_truth import overplot_true_values, true_theta_legend_handle
from mcmc_decay_inference import PARAMETER_NAMES, PRIOR_LOG_MEAN, PRIOR_LOG_STD, simulate_decay_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FAMILIES = ("diag_gaussian", "full_gaussian", "mdn", "affine_flow", "spline_flow")
CONTEXT_FEATURE_MODES = ("raw", "decay_summary", "raw_decay_summary")
TRAIN_SAMPLERS = ("random", "lhs", "sobol")
FAMILY_LABELS = {
    "diag_gaussian": "Diagonal Gaussian",
    "full_gaussian": "Full Gaussian",
    "mdn": "MDN",
    "affine_flow": "Affine flow",
    "spline_flow": "Spline flow",
}
FAMILY_COLORS = {
    "diag_gaussian": "#2f6fbb",
    "full_gaussian": "#3f8f5f",
    "mdn": "#c06f2d",
    "affine_flow": "#7a5cc2",
    "spline_flow": "#0f8b8d",
    "grid_reference": "#172033",
}
LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Stage1Config:
    train_simulations: int
    val_simulations: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    hidden_layers: int
    mdn_components: int
    flow_layers: int
    flow_context_dim: int
    seed: int
    observed_seed: int
    requested_device: str
    families: list[str]
    posterior_samples: int
    reference_grid_size: int
    train_sampler: str = "random"
    context_features: str = "raw"
    spline_bins: int = 12
    lr_schedule: str = "constant"
    lr_eta_min: float = 0.0
    lr_warmup_steps: int = 0
    validation_every_epochs: int = 1
    torch_compile: str = "none"
    grad_clip_norm: float = 20.0
    ema_decay: float = 0.0
    batching_mode: str = "dataloader"
    max_optimizer_steps: int = 0
    progress_jsonl: Path | None = None
    progress_nll_offset: float = 0.0


def choose_training_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def json_progress_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_progress_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_progress_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_progress_value(item) for item in value]
    return value


def append_progress_record(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_progress_value(record), sort_keys=True) + "\n")


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def sample_decay_pairs(
    *,
    n: int,
    seed: int,
    n_observations: int = 40,
    sampler: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(0.0, 6.0, n_observations, dtype=torch.float64)
    prior_mean = PRIOR_LOG_MEAN.to(dtype=torch.float64)
    prior_std = PRIOR_LOG_STD.to(dtype=torch.float64)
    if sampler == "random":
        z_unit = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    elif sampler in {"lhs", "sobol"}:
        z_unit_columns = []
        eps = torch.finfo(torch.float64).eps
        if sampler == "lhs":
            for _ in range(3):
                permutation = torch.randperm(n, generator=generator).to(dtype=torch.float64)
                jitter = torch.rand(n, generator=generator, dtype=torch.float64)
                quantiles = ((permutation + jitter) / float(n)).clamp(min=eps, max=1.0 - eps)
                z_unit_columns.append(math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0))
            z_unit = torch.stack(z_unit_columns, dim=1)
        else:
            engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
            quantiles = engine.draw(n).to(dtype=torch.float64).clamp(min=eps, max=1.0 - eps)
            z_unit = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
    else:
        raise ValueError(f"Unknown decay-pair sampler: {sampler}")
    z = prior_mean[None, :] + z_unit * prior_std[None, :]
    theta = torch.exp(z)
    amplitude = theta[:, 0:1]
    decay_rate = theta[:, 1:2]
    noise = theta[:, 2:3]
    mean = amplitude * torch.exp(-decay_rate * t[None, :])
    x = mean + torch.randn(n, n_observations, generator=generator, dtype=torch.float64) * noise
    return x.numpy(), z.numpy(), t.numpy()


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (value - mean[None, :]) / std[None, :]


def decay_context_summary_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D context array, got shape {x.shape}")
    t = np.linspace(0.0, 6.0, x.shape[1], dtype=np.float64)
    t_centered = t - float(np.mean(t))
    t_var = float(np.sum(t_centered * t_centered))
    clipped = np.clip(x, 1e-4, None)
    log_y = np.log(clipped)
    slope = (log_y @ t_centered) / max(t_var, 1e-12)
    intercept = np.mean(log_y, axis=1) - slope * float(np.mean(t))
    fitted = intercept[:, None] + slope[:, None] * t[None, :]
    log_resid = log_y - fitted
    early = np.mean(x[:, : max(1, x.shape[1] // 5)], axis=1)
    late = np.mean(x[:, -max(1, x.shape[1] // 5) :], axis=1)
    first = x[:, 0]
    last = x[:, -1]
    log_ratio = np.log(np.clip(first, 1e-4, None)) - np.log(np.clip(last, 1e-4, None))
    return np.column_stack(
        [
            intercept,
            -slope,
            np.std(log_resid, axis=1),
            np.mean(x, axis=1),
            np.std(x, axis=1),
            np.min(x, axis=1),
            np.max(x, axis=1),
            first,
            last,
            early,
            late,
            log_ratio,
        ]
    )


def transform_context_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.asarray(x, dtype=np.float64)
    summary = decay_context_summary_features(x)
    if mode == "decay_summary":
        return summary
    if mode == "raw_decay_summary":
        return np.concatenate([np.asarray(x, dtype=np.float64), summary], axis=1)
    raise ValueError(f"Unknown context feature mode: {mode}")


def lower_cholesky_from_params(params: torch.Tensor, dim: int = 3) -> torch.Tensor:
    leading_shape = params.shape[:-1]
    tril = torch.zeros(*leading_shape, dim, dim, device=params.device, dtype=params.dtype)
    tril[..., 0, 0] = torch.nn.functional.softplus(params[..., 0]) + 1e-4
    tril[..., 1, 0] = params[..., 1]
    tril[..., 1, 1] = torch.nn.functional.softplus(params[..., 2]) + 1e-4
    tril[..., 2, 0] = params[..., 3]
    tril[..., 2, 1] = params[..., 4]
    tril[..., 2, 2] = torch.nn.functional.softplus(params[..., 5]) + 1e-4
    return tril


def full_gaussian_log_prob(z: torch.Tensor, mean: torch.Tensor, tril: torch.Tensor) -> torch.Tensor:
    diff = (z - mean).unsqueeze(-1)
    solved = torch.linalg.solve_triangular(tril, diff, upper=False).squeeze(-1)
    maha = solved.square().sum(dim=-1)
    log_det = torch.log(torch.diagonal(tril, dim1=-2, dim2=-1)).sum(dim=-1)
    dim = z.shape[-1]
    return -0.5 * (dim * LOG_2PI + maha) - log_det


def batched_mixture_log_prob(
    z: torch.Tensor,
    logits: torch.Tensor,
    mean: torch.Tensor,
    tril: torch.Tensor,
) -> torch.Tensor:
    batch, components, dim = mean.shape
    diff = z[:, None, :] - mean
    solved = torch.linalg.solve_triangular(
        tril.reshape(batch * components, dim, dim),
        diff.reshape(batch * components, dim, 1),
        upper=False,
    ).reshape(batch, components, dim)
    maha = solved.square().sum(dim=-1)
    log_det = torch.log(torch.diagonal(tril, dim1=-2, dim2=-1)).sum(dim=-1)
    component_log_prob = -0.5 * (dim * LOG_2PI + maha) - log_det
    return torch.logsumexp(torch.log_softmax(logits, dim=-1) + component_log_prob, dim=-1)


class DiagonalGaussianPosterior(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.net = make_mlp(x_dim, 2 * z_dim, hidden_dim, hidden_layers)
        self.z_dim = z_dim

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mean, raw_std = self.net(x).chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + 1e-4
        return (-0.5 * ((z - mean) / std).square() - torch.log(std) - 0.5 * LOG_2PI).sum(dim=-1)

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        mean, raw_std = self.net(x).chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + 1e-4
        eps = torch.randn(n, self.z_dim, device=x.device, dtype=x.dtype)
        return mean.expand(n, -1) + eps * std.expand(n, -1)


class FullGaussianPosterior(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.net = make_mlp(x_dim, z_dim + 6, hidden_dim, hidden_layers)
        self.z_dim = z_dim

    def parameters_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(x)
        mean = output[..., : self.z_dim]
        tril = lower_cholesky_from_params(output[..., self.z_dim :], self.z_dim)
        return mean, tril

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mean, tril = self.parameters_from_x(x)
        return full_gaussian_log_prob(z, mean, tril)

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        mean, tril = self.parameters_from_x(x)
        eps = torch.randn(n, self.z_dim, 1, device=x.device, dtype=x.dtype)
        transformed = torch.matmul(tril.expand(n, -1, -1), eps).squeeze(-1)
        return mean.expand(n, -1) + transformed


class MixtureDensityPosterior(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        components: int,
    ) -> None:
        super().__init__()
        self.components = components
        self.z_dim = z_dim
        self.net = make_mlp(x_dim, components * (1 + z_dim + 6), hidden_dim, hidden_layers)

    def parameters_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.net(x).reshape(x.shape[0], self.components, 1 + self.z_dim + 6)
        logits = output[..., 0]
        mean = output[..., 1 : 1 + self.z_dim]
        tril = lower_cholesky_from_params(output[..., 1 + self.z_dim :], self.z_dim)
        return logits, mean, tril

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        logits, mean, tril = self.parameters_from_x(x)
        return batched_mixture_log_prob(z, logits, mean, tril)

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        logits, mean, tril = self.parameters_from_x(x)
        probs = torch.softmax(logits[0], dim=-1)
        component = torch.multinomial(probs, n, replacement=True)
        chosen_mean = mean[0, component]
        chosen_tril = tril[0, component]
        eps = torch.randn(n, self.z_dim, 1, device=x.device, dtype=x.dtype)
        transformed = torch.matmul(chosen_tril, eps).squeeze(-1)
        return chosen_mean + transformed


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        z_dim: int,
        context_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        mask: Iterable[float],
        max_log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", torch.tensor(list(mask), dtype=torch.float32))
        self.net = make_mlp(z_dim + context_dim, 2 * z_dim, hidden_dim, hidden_layers)
        self.max_log_scale = max_log_scale

    def scale_shift(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(device=z.device, dtype=z.dtype)
        output = self.net(torch.cat([z * mask, context], dim=-1))
        raw_scale, shift = output.chunk(2, dim=-1)
        scale = self.max_log_scale * torch.tanh(raw_scale) * (1.0 - mask)
        shift = shift * (1.0 - mask)
        return scale, shift

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(device=z.device, dtype=z.dtype)
        scale, shift = self.scale_shift(z, context)
        out = z * mask + (1.0 - mask) * (z * torch.exp(scale) + shift)
        log_det = scale.sum(dim=-1)
        return out, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(device=z.device, dtype=z.dtype)
        scale, shift = self.scale_shift(z, context)
        out = z * mask + (1.0 - mask) * ((z - shift) * torch.exp(-scale))
        log_det = -scale.sum(dim=-1)
        return out, log_det


class AffineFlowPosterior(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        flow_layers: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.context_encoder = make_mlp(x_dim, context_dim, hidden_dim, hidden_layers)
        masks = ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        self.layers = nn.ModuleList(
            ConditionalAffineCoupling(
                z_dim=z_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                hidden_layers=max(1, hidden_layers - 1),
                mask=masks[index % 2],
            )
            for index in range(flow_layers)
        )

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        context = self.context_encoder(x)
        current = z
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            current, layer_log_det = layer.inverse(current, context)
            log_det = log_det + layer_log_det
        base_log_prob = (-0.5 * current.square() - 0.5 * LOG_2PI).sum(dim=-1)
        return base_log_prob + log_det

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor) -> torch.Tensor:
        context = self.context_encoder(x).expand(n, -1)
        current = torch.randn(n, self.z_dim, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            current, _ = layer.forward(current, context)
        return current


class SplineFlowPosterior(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        flow_layers: int,
        bins: int,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.flow = zuko.flows.NSF(
            z_dim,
            context=x_dim,
            transforms=flow_layers,
            hidden_features=tuple(hidden_dim for _ in range(hidden_layers)),
            bins=bins,
        )

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.flow(x).log_prob(z)

    @torch.no_grad()
    def sample(self, n: int, x: torch.Tensor, chunk_size: int = 65_536) -> torch.Tensor:
        samples = []
        for start in range(0, n, chunk_size):
            current = min(chunk_size, n - start)
            drawn = self.flow(x).sample((current,))
            if drawn.ndim == 3:
                drawn = drawn[:, 0, :]
            samples.append(drawn)
        return torch.cat(samples, dim=0)


def make_model(family: str, config: Stage1Config, x_dim: int, z_dim: int) -> nn.Module:
    if family == "diag_gaussian":
        return DiagonalGaussianPosterior(x_dim, z_dim, config.hidden_dim, config.hidden_layers)
    if family == "full_gaussian":
        return FullGaussianPosterior(x_dim, z_dim, config.hidden_dim, config.hidden_layers)
    if family == "mdn":
        return MixtureDensityPosterior(
            x_dim,
            z_dim,
            config.hidden_dim,
            config.hidden_layers,
            config.mdn_components,
        )
    if family == "affine_flow":
        return AffineFlowPosterior(
            x_dim,
            z_dim,
            config.hidden_dim,
            config.hidden_layers,
            config.flow_layers,
            config.flow_context_dim,
        )
    if family == "spline_flow":
        return SplineFlowPosterior(
            x_dim,
            z_dim,
            config.hidden_dim,
            config.hidden_layers,
            config.flow_layers,
            config.spline_bins,
        )
    raise ValueError(f"Unknown family: {family}")


def first_tensor_dataset_batch(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor] | None:
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, TensorDataset) or len(dataset) == 0 or len(dataset.tensors) < 2:
        return None
    batch_size = int(loader.batch_size or len(dataset))
    stop = min(batch_size, len(dataset))
    return dataset.tensors[0][:stop], dataset.tensors[1][:stop]


def train_one_model(
    *,
    family: str,
    config: Stage1Config,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_z: torch.Tensor,
    device: torch.device,
    x_dim: int,
    z_dim: int,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(config.seed + 1000 + FAMILIES.index(family))
    model = make_model(family, config, x_dim, z_dim).to(device)
    if config.torch_compile == "default":
        model = torch.compile(model)
    elif config.torch_compile == "reduce_overhead":
        model = torch.compile(model, mode="reduce-overhead")
    elif config.torch_compile != "none":
        raise ValueError(f"Unknown torch_compile mode: {config.torch_compile}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    max_optimizer_steps = max(0, int(config.max_optimizer_steps))
    scheduler = None
    scheduler_step_unit = None
    if config.lr_schedule == "cosine_epoch":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=float(config.lr_eta_min),
        )
        scheduler_step_unit = "epoch"
    elif config.lr_schedule == "cosine_step":
        total_steps = max(1, config.epochs * len(train_loader))
        if max_optimizer_steps > 0:
            total_steps = min(total_steps, max_optimizer_steps)
        warmup_steps = max(0, min(int(config.lr_warmup_steps), total_steps - 1))
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=max(1.0 / warmup_steps, 1e-6),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=float(config.lr_eta_min),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=float(config.lr_eta_min),
            )
        scheduler_step_unit = "step"
    elif config.lr_schedule != "constant":
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")
    manual_batch_tensors = None
    manual_batch_shuffle = False
    if config.batching_mode in {"pre_shuffle", "sequential"}:
        if not isinstance(train_loader.dataset, TensorDataset):
            raise ValueError(
                f"batching_mode={config.batching_mode!r} requires a TensorDataset train_loader."
            )
        tensors = train_loader.dataset.tensors
        if len(tensors) < 2:
            raise ValueError(f"batching_mode={config.batching_mode!r} requires x and z tensors.")
        manual_batch_tensors = (tensors[0], tensors[1])
        manual_batch_shuffle = config.batching_mode == "pre_shuffle"
    elif config.batching_mode != "dataloader":
        raise ValueError(f"Unknown batching_mode: {config.batching_mode}")
    history = {
        "train_nll": [],
        "val_nll": [],
        "val_evaluated": [],
        "lr": [],
        "train_seconds": [],
        "val_seconds": [],
        "epoch_seconds": [],
        "optimizer_steps": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = max(20, config.epochs // 5)
    epochs_since_best = 0
    ema_decay = float(config.ema_decay)
    use_ema = 0.0 < ema_decay < 1.0
    ema_params = [param.detach().clone() for param in model.parameters()] if use_ema else []
    if config.progress_jsonl is not None:
        config.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        config.progress_jsonl.write_text("", encoding="utf-8")

    def evaluate_val_nll() -> float:
        model.eval()
        with torch.no_grad():
            val_loss = -model.log_prob(val_z.to(device), val_x.to(device)).mean()
            return float(val_loss.detach().cpu())

    def evaluate_ema_val_nll() -> float:
        if not use_ema:
            return evaluate_val_nll()
        backups = [param.detach().clone() for param in model.parameters()]
        try:
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_params, strict=True):
                    param.copy_(ema_param)
            return evaluate_val_nll()
        finally:
            with torch.no_grad():
                for param, backup in zip(model.parameters(), backups, strict=True):
                    param.copy_(backup)

    def ema_state_dict() -> dict[str, torch.Tensor]:
        if not use_ema:
            return copy.deepcopy(model.state_dict())
        backups = [param.detach().clone() for param in model.parameters()]
        try:
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_params, strict=True):
                    param.copy_(ema_param)
            return copy.deepcopy(model.state_dict())
        finally:
            with torch.no_grad():
                for param, backup in zip(model.parameters(), backups, strict=True):
                    param.copy_(backup)

    synchronize_device(device)
    initial_eval_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial_batch = first_tensor_dataset_batch(train_loader)
        if initial_batch is None:
            initial_train_batch_nll = float("nan")
        else:
            batch_x, batch_z = initial_batch[:2]
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)
            initial_train_batch_nll = float(
                (-model.log_prob(batch_z, batch_x).mean()).detach().cpu()
            )
        initial_val_nll = evaluate_val_nll()
    synchronize_device(device)
    initial_eval_seconds = time.perf_counter() - initial_eval_start
    append_progress_record(
        config.progress_jsonl,
        {
            "event": "initial_eval",
            "family": family,
            "epoch": 0,
            "optimizer_steps": 0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "initial_train_batch_nll_standardized": initial_train_batch_nll,
            "initial_train_batch_nll_z_units": initial_train_batch_nll + config.progress_nll_offset,
            "initial_val_nll_standardized": initial_val_nll,
            "initial_val_nll_z_units": initial_val_nll + config.progress_nll_offset,
            "initial_eval_seconds": initial_eval_seconds,
            "lr_schedule": config.lr_schedule,
            "lr_eta_min": float(config.lr_eta_min),
            "lr_warmup_steps": int(config.lr_warmup_steps),
            "validation_every_epochs": int(config.validation_every_epochs),
            "torch_compile": config.torch_compile,
            "grad_clip_norm": float(config.grad_clip_norm),
            "ema_decay": float(config.ema_decay),
            "batching_mode": config.batching_mode,
            "max_optimizer_steps": int(config.max_optimizer_steps),
        },
    )

    start = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(config.epochs):
        reached_max_optimizer_steps = False
        epoch_start = time.perf_counter()
        model.train()
        train_start = time.perf_counter()
        train_loss_sum = 0.0
        train_count = 0
        epoch_start_lr = float(optimizer.param_groups[0]["lr"])
        if manual_batch_tensors is None:
            batch_iterable = train_loader
        else:
            train_x_tensor, train_z_tensor = manual_batch_tensors
            if manual_batch_shuffle:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(config.seed + 2 + config.train_simulations + epoch)
                )
                permutation = torch.randperm(train_x_tensor.shape[0], generator=generator)
                batch_x_source = train_x_tensor[permutation]
                batch_z_source = train_z_tensor[permutation]
            else:
                batch_x_source = train_x_tensor
                batch_z_source = train_z_tensor
            batch_size = int(train_loader.batch_size or config.batch_size)
            batch_iterable = (
                (
                    batch_x_source[start_index : start_index + batch_size],
                    batch_z_source[start_index : start_index + batch_size],
                )
                for start_index in range(0, batch_x_source.shape[0], batch_size)
            )
        for batch_x, batch_z in batch_iterable:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)
            loss = -model.log_prob(batch_z, batch_x).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_params, model.parameters(), strict=True):
                        ema_param.mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)
            optimizer_steps += 1
            if scheduler is not None and scheduler_step_unit == "step":
                scheduler.step()
            train_loss_sum += float(loss.detach().cpu()) * batch_x.shape[0]
            train_count += batch_x.shape[0]
            if max_optimizer_steps > 0 and optimizer_steps >= max_optimizer_steps:
                reached_max_optimizer_steps = True
                break
        train_seconds = time.perf_counter() - train_start

        train_loss = train_loss_sum / train_count
        should_validate = (
            epoch == 0
            or (epoch + 1) % max(1, int(config.validation_every_epochs)) == 0
            or epoch == config.epochs - 1
            or reached_max_optimizer_steps
        )
        if should_validate:
            val_start = time.perf_counter()
            val_loss_float = evaluate_ema_val_nll()
            val_seconds = time.perf_counter() - val_start
        else:
            val_loss_float = float("nan")
            val_seconds = 0.0

        history["train_nll"].append(train_loss)
        history["val_nll"].append(val_loss_float)
        history["val_evaluated"].append(bool(should_validate))
        history["lr"].append(epoch_start_lr)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["epoch_seconds"].append(time.perf_counter() - epoch_start)
        history["optimizer_steps"].append(optimizer_steps)

        if should_validate:
            if val_loss_float < best_val:
                best_val = val_loss_float
                best_state = ema_state_dict()
                epochs_since_best = 0
            else:
                epochs_since_best += 1
        append_progress_record(
            config.progress_jsonl,
            {
                "event": "epoch",
                "family": family,
                "epoch": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "lr": epoch_start_lr,
                "train_nll_standardized": train_loss,
                "train_nll_z_units": train_loss + config.progress_nll_offset,
                "val_evaluated": bool(should_validate),
                "val_nll_standardized": val_loss_float,
                "val_nll_z_units": val_loss_float + config.progress_nll_offset
                if math.isfinite(val_loss_float)
                else float("nan"),
                "best_val_nll_standardized": best_val,
                "best_val_nll_z_units": best_val + config.progress_nll_offset
                if math.isfinite(best_val)
                else float("nan"),
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": history["epoch_seconds"][-1],
                "elapsed_training_seconds": time.perf_counter() - start,
                "max_optimizer_steps": int(config.max_optimizer_steps),
            },
        )
        if scheduler is not None and scheduler_step_unit == "epoch" and not reached_max_optimizer_steps:
            scheduler.step()
        if reached_max_optimizer_steps or epochs_since_best >= patience:
            break

    synchronize_device(device)
    runtime = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    final_val_nll = next(
        value for value in reversed(history["val_nll"]) if math.isfinite(float(value))
    )
    metrics = {
        "family": family,
        "label": FAMILY_LABELS[family],
        "initial_train_batch_nll": initial_train_batch_nll,
        "initial_val_nll": initial_val_nll,
        "initial_losses_finite": bool(
            math.isfinite(initial_train_batch_nll) and math.isfinite(initial_val_nll)
        ),
        "initial_eval_seconds": initial_eval_seconds,
        "lr_schedule": config.lr_schedule,
        "lr_eta_min": float(config.lr_eta_min),
        "lr_warmup_steps": int(config.lr_warmup_steps),
        "lr_scheduler_step_unit": scheduler_step_unit,
        "validation_every_epochs": int(config.validation_every_epochs),
        "validation_evaluations": int(sum(history["val_evaluated"])),
        "torch_compile": config.torch_compile,
        "grad_clip_norm": float(config.grad_clip_norm),
        "ema_decay": float(config.ema_decay),
        "batching_mode": config.batching_mode,
        "max_optimizer_steps": int(config.max_optimizer_steps),
        "progress_jsonl": str(config.progress_jsonl) if config.progress_jsonl is not None else None,
        "optimizer_steps": int(optimizer_steps),
        "batches_per_epoch": int(len(train_loader)),
        "epochs_completed": len(history["train_nll"]),
        "best_val_nll": best_val,
        "final_train_nll": history["train_nll"][-1],
        "final_val_nll": final_val_nll,
        "training_seconds": runtime,
        "history": history,
    }
    return model, metrics


@torch.no_grad()
def sample_posterior_for_observation(
    *,
    model: nn.Module,
    observed_x: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    n: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_standardized = standardize(observed_x[None, :], x_mean, x_std).astype(np.float32)
    x_tensor = torch.from_numpy(x_standardized).to(device)
    z_standardized = model.sample(n, x_tensor).detach().cpu().numpy()
    z = z_standardized * z_std[None, :] + z_mean[None, :]
    theta = np.exp(z)
    return z, theta


def load_reference_samples(mcmc_samples: Path, hmc_samples: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mcmc_samples.exists() and hmc_samples.exists():
        mcmc = load_samples(mcmc_samples, "MCMC")
        hmc = load_samples(hmc_samples, "HMC")
        combined_z = np.vstack([mcmc["posterior_z"], hmc["posterior_z"]])
        return combined_z, mcmc["t"], mcmc["y"], mcmc["true_theta"]
    t, y, true_theta = simulate_decay_data(seed=20260622)
    return np.log(true_theta.numpy()[None, :]), t.numpy(), y.numpy(), true_theta.numpy()


def sample_grid_reference(reference: dict[str, object], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = reference["weights"]
    index = rng.choice(len(weights), size=n, replace=True, p=weights)
    return reference["theta_grid"][index]


def subsample(values: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if values.shape[0] <= max_samples:
        return values
    rng = np.random.default_rng(seed)
    index = rng.choice(values.shape[0], size=max_samples, replace=False)
    return values[index]


def plot_training_curves(results: dict[str, dict[str, object]], outfile: Path) -> None:
    figure, ax = plt.subplots(figsize=(10, 6))
    for family, metrics in results.items():
        val = np.asarray(metrics["history"]["val_nll"], dtype=float)
        train = np.asarray(metrics["history"]["train_nll"], dtype=float)
        epochs = np.arange(1, len(val) + 1)
        color = FAMILY_COLORS[family]
        ax.plot(epochs, val, color=color, linewidth=2.0, label=f"{FAMILY_LABELS[family]} val")
        ax.plot(epochs, train, color=color, linewidth=1.0, alpha=0.35, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log likelihood on standardized z")
    ax.set_title("NPE Stage 1 training curves")
    ax.grid(alpha=0.22)
    ax.legend()
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_npe_corner_overlay(
    samples_by_family: dict[str, np.ndarray],
    reference_samples: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
    max_samples: int = 20_000,
) -> None:
    labels = [r"$A$", r"$k$", r"$\sigma$"]
    figure = corner.corner(
        subsample(reference_samples, max_samples, seed=1),
        labels=labels,
        color=FAMILY_COLORS["grid_reference"],
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.8},
        contour_kwargs={"linewidths": 1.5},
    )
    for offset, (family, samples) in enumerate(samples_by_family.items(), start=2):
        corner.corner(
            subsample(samples, max_samples, seed=offset),
            fig=figure,
            labels=labels,
            color=FAMILY_COLORS[family],
            plot_datapoints=False,
            fill_contours=False,
            levels=(0.50, 0.90),
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.5},
            contour_kwargs={"linewidths": 1.3},
        )

    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS["grid_reference"], lw=2, label="Grid reference"),
        true_theta_legend_handle(),
        *[
            plt.Line2D([0], [0], color=FAMILY_COLORS[family], lw=2, label=FAMILY_LABELS[family])
            for family in samples_by_family
        ],
    ]
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    figure.subplots_adjust(top=0.90)
    figure.suptitle("NPE Stage 1 posterior overlay", y=0.985, fontsize=15)
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def posterior_predictive_band(
    samples: np.ndarray,
    t_grid: np.ndarray,
    seed: int,
    max_draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = subsample(samples, max_draws, seed=seed)
    mean = selected[:, 0, None] * np.exp(-selected[:, 1, None] * t_grid[None, :])
    rng = np.random.default_rng(seed + 33)
    predictive = mean + rng.normal(0.0, selected[:, 2, None], size=mean.shape)
    return tuple(np.quantile(predictive, [0.05, 0.50, 0.95], axis=0))


def plot_npe_predictive_overlay(
    *,
    samples_by_family: dict[str, np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    true_theta: np.ndarray,
    outfile: Path,
) -> None:
    t_grid = np.linspace(float(t.min()), float(t.max()), 220)
    true_mean = true_theta[0] * np.exp(-true_theta[1] * t_grid)
    figure, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(t, y, color="#172033", s=28, zorder=5, label="observed data")
    ax.plot(t_grid, true_mean, color="#172033", linestyle="--", linewidth=1.8, label="true mean")
    for index, (family, samples) in enumerate(samples_by_family.items(), start=1):
        lower, median, upper = posterior_predictive_band(samples, t_grid, seed=100 + index, max_draws=900)
        color = FAMILY_COLORS[family]
        ax.fill_between(t_grid, lower, upper, color=color, alpha=0.10)
        ax.plot(t_grid, median, color=color, linewidth=2.0, label=FAMILY_LABELS[family])
    ax.set_xlabel("time t")
    ax.set_ylabel("replicated observation y")
    ax.set_title("NPE Stage 1 posterior predictive overlay")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(families) - set(FAMILIES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown families: {invalid}")
    return families


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 NPE models on the decay simulator.")
    parser.add_argument("--train-simulations", type=int, default=20_000)
    parser.add_argument("--val-simulations", type=int, default=5_000)
    parser.add_argument(
        "--train-sampler",
        choices=TRAIN_SAMPLERS,
        default="random",
        help="Sampler for training simulations. Validation remains random prior predictive.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine_epoch", "cosine_step"),
        default="constant",
        help="Learning-rate schedule. Default preserves the historical fixed-rate trainer.",
    )
    parser.add_argument(
        "--lr-eta-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine schedules. Default 0.0 preserves prior behavior.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Optimizer steps for linear LR warmup before cosine_step decay.",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Evaluate validation loss every N epochs, plus epoch 1 and the final epoch.",
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="Stop training after this many optimizer steps. Use 0 to train full epochs.",
    )
    parser.add_argument(
        "--torch-compile",
        choices=("none", "default", "reduce_overhead"),
        default="none",
        help="Optional torch.compile mode for the NPE model.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=20.0,
        help="Gradient clipping norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional weight EMA decay in [0, 1). Use 0 to disable EMA.",
    )
    parser.add_argument(
        "--batching-mode",
        choices=("dataloader", "pre_shuffle", "sequential"),
        default="dataloader",
        help=(
            "Batch source for training. pre_shuffle uses contiguous tensor slices after "
            "per-epoch shuffling; sequential slices the existing random simulation order."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--flow-context-dim", type=int, default=64)
    parser.add_argument("--spline-bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--observed-seed", type=int, default=20260622)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--families", type=parse_families, default=list(FAMILIES))
    parser.add_argument("--posterior-samples", type=int, default=60_000)
    parser.add_argument("--reference-grid-size", type=int, default=90)
    parser.add_argument(
        "--context-features",
        choices=CONTEXT_FEATURE_MODES,
        default="raw",
        help="Context representation used by the NPE.",
    )
    parser.add_argument("--output-dir", type=Path, default=ap.NPE_STAGE1_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=ap.NPE_STAGE1_FIGURES)
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = choose_training_device(args.device)
    config = Stage1Config(
        train_simulations=args.train_simulations,
        val_simulations=args.val_simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_eta_min=args.lr_eta_min,
        lr_warmup_steps=args.lr_warmup_steps,
        validation_every_epochs=args.validation_every_epochs,
        torch_compile=args.torch_compile,
        grad_clip_norm=args.grad_clip_norm,
        ema_decay=args.ema_decay,
        batching_mode=args.batching_mode,
        max_optimizer_steps=args.max_optimizer_steps,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        mdn_components=args.mdn_components,
        flow_layers=args.flow_layers,
        flow_context_dim=args.flow_context_dim,
        seed=args.seed,
        observed_seed=args.observed_seed,
        requested_device=args.device,
        families=args.families,
        posterior_samples=args.posterior_samples,
        reference_grid_size=args.reference_grid_size,
        train_sampler=args.train_sampler,
        context_features=args.context_features,
        spline_bins=args.spline_bins,
    )

    data_start = time.perf_counter()
    train_x, train_z, _ = sample_decay_pairs(
        n=args.train_simulations,
        seed=args.seed,
        sampler=args.train_sampler,
    )
    val_x, val_z, _ = sample_decay_pairs(n=args.val_simulations, seed=args.seed + 1)
    t_obs, y_obs, true_theta = simulate_decay_data(seed=args.observed_seed)
    observed_x_raw = y_obs.numpy()
    true_theta_np = true_theta.numpy()
    train_x_context = transform_context_features(train_x, args.context_features)
    val_x_context = transform_context_features(val_x, args.context_features)
    observed_x_context = transform_context_features(observed_x_raw[None, :], args.context_features)[0]

    x_mean = train_x_context.mean(axis=0)
    x_std = np.maximum(train_x_context.std(axis=0), 1e-6)
    z_mean = train_z.mean(axis=0)
    z_std = np.maximum(train_z.std(axis=0), 1e-6)

    train_x_std = standardize(train_x_context, x_mean, x_std).astype(np.float32)
    val_x_std = standardize(val_x_context, x_mean, x_std).astype(np.float32)
    train_z_std = standardize(train_z, z_mean, z_std).astype(np.float32)
    val_z_std = standardize(val_z, z_mean, z_std).astype(np.float32)
    data_seconds = time.perf_counter() - data_start

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_z_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_x_tensor = torch.from_numpy(val_x_std)
    val_z_tensor = torch.from_numpy(val_z_std)

    results: dict[str, dict[str, object]] = {}
    z_samples_by_family: dict[str, np.ndarray] = {}
    theta_samples_by_family: dict[str, np.ndarray] = {}
    model_paths: dict[str, str] = {}

    for family in args.families:
        print(f"training {family} on {device}")
        family_config = replace(
            config,
            progress_jsonl=args.output_dir / f"{family}_training_progress.jsonl",
            progress_nll_offset=float(np.log(z_std).sum()),
        )
        model, metrics = train_one_model(
            family=family,
            config=family_config,
            train_loader=train_loader,
            val_x=val_x_tensor,
            val_z=val_z_tensor,
            device=device,
            x_dim=train_x_std.shape[1],
            z_dim=train_z_std.shape[1],
        )
        z_samples, theta_samples = sample_posterior_for_observation(
            model=model,
            observed_x=observed_x_context,
            x_mean=x_mean,
            x_std=x_std,
            z_mean=z_mean,
            z_std=z_std,
            n=args.posterior_samples,
            device=device,
        )
        z_samples_by_family[family] = z_samples
        theta_samples_by_family[family] = theta_samples
        metrics["posterior_summary"] = summarize_samples(theta_samples)
        results[family] = metrics

        model_path = args.output_dir / f"{family}_model.pt"
        torch.save(
            {
                "family": family,
                "state_dict": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "z_mean": z_mean,
                "z_std": z_std,
                "config": asdict(config),
            },
            model_path,
        )
        model_paths[family] = str(model_path)

    reference_z, reference_t, reference_y, reference_true_theta = load_reference_samples(
        args.mcmc_samples,
        args.hmc_samples,
    )
    reference = build_grid_reference(
        t=reference_t,
        y=reference_y,
        combined_z_samples=reference_z,
        true_theta=reference_true_theta,
        grid_size=args.reference_grid_size,
        chunk_size=120_000,
    )
    reference_samples = sample_grid_reference(
        reference,
        n=min(args.posterior_samples, 80_000),
        seed=args.seed + 88,
    )
    for family, theta_samples in theta_samples_by_family.items():
        results[family]["faithfulness_to_grid_reference"] = compare_to_reference(theta_samples, reference)

    samples_npz = args.output_dir / "npe_stage1_samples.npz"
    np.savez_compressed(
        samples_npz,
        observed_x=observed_x_context,
        observed_x_raw=observed_x_raw,
        t=t_obs.numpy(),
        y=observed_x_raw,
        true_theta=true_theta_np,
        x_mean=x_mean,
        x_std=x_std,
        z_mean=z_mean,
        z_std=z_std,
        **{f"z_samples_{family}": samples for family, samples in z_samples_by_family.items()},
        **{f"theta_samples_{family}": samples for family, samples in theta_samples_by_family.items()},
    )

    training_curve_png = args.figure_dir / "npe_stage1_training_curves.png"
    corner_png = args.figure_dir / "npe_stage1_corner_overlay.png"
    predictive_png = args.figure_dir / "npe_stage1_predictive_overlay.png"
    plot_training_curves(results, training_curve_png)
    plot_npe_corner_overlay(theta_samples_by_family, reference_samples, true_theta_np, corner_png)
    plot_npe_predictive_overlay(
        samples_by_family=theta_samples_by_family,
        t=t_obs.numpy(),
        y=observed_x,
        true_theta=true_theta_np,
        outfile=predictive_png,
    )

    summary = {
        "config": asdict(config),
        "device": str(device),
        "data_seconds": data_seconds,
        "standardization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "z_mean": z_mean.tolist(),
            "z_std": z_std.tolist(),
        },
        "model_paths": model_paths,
        "samples_npz": str(samples_npz),
        "figures": {
            "training_curves": str(training_curve_png),
            "corner_overlay": str(corner_png),
            "predictive_overlay": str(predictive_png),
        },
        "grid_reference": {
            "grid_size": reference["grid_size"],
            "grid_points": reference["grid_points"],
            "edge_mass": reference["edge_mass"],
            "posterior_summary": reference["summary"],
        },
        "results": results,
    }
    summary_json = args.output_dir / "npe_stage1_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary_json: {summary_json}")
    print(f"samples_npz: {samples_npz}")
    print(f"corner_overlay: {corner_png}")
    print(f"predictive_overlay: {predictive_png}")
    print("mean normalized Wasserstein to grid reference:")
    for family in args.families:
        value = results[family]["faithfulness_to_grid_reference"]["mean_normalized_wasserstein"]["value"]
        print(f"  {family}: {value:.5f}")


if __name__ == "__main__":
    main()
