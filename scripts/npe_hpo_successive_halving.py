from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/89_next2x_hpo_successive_halving")
DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz"
)


@dataclass(frozen=True)
class Trial:
    name: str
    config: dict[str, Any]
    source: str


APPEND_LOCK = threading.Lock()


HIGH_BETA_BASE_CONFIG: dict[str, Any] = {
    "batch_size": 512,
    "learning_rate": 0.004,
    "lr_schedule": "cosine_epoch",
    "lr_eta_min": 0.0,
    "lr_warmup_steps": 0,
    "lr_decay_epochs": 0,
    "adam_beta1": 0.98,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    "weight_decay": 2e-4,
    "hidden_dim": 80,
    "hidden_layers": 2,
    "flow_layers": 3,
    "flow_context_dim": 64,
    "flow_activation": "relu",
    "flow_residual": False,
    "flow_randperm": False,
    "flow_passes": 0,
    "flow_kind": "nsf",
    "spline_bins": 8,
    "train_sampler": "random",
    "context_features": "raw",
    "batching_mode": "pre_shuffle",
    "loss_weight_mode": "none",
    "loss_tail_weight": 3.0,
    "target_transform": "none",
    "target_ridge": 1e-3,
    "grad_clip_norm": 0.0,
    "ema_decay": 0.0,
}


FRONTIER_BASE_CONFIG: dict[str, Any] = {
    **HIGH_BETA_BASE_CONFIG,
    "learning_rate": 0.003,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 2e-4,
    "hidden_dim": 80,
    "hidden_layers": 2,
    "flow_layers": 3,
    "flow_activation": "relu",
    "flow_residual": False,
    "flow_randperm": False,
    "flow_passes": 0,
    "flow_kind": "nsf",
    "spline_bins": 8,
}


NEXT2X_RESIDUAL_BASE_CONFIG: dict[str, Any] = {
    **FRONTIER_BASE_CONFIG,
    "batch_size": 512,
    "learning_rate": 0.003,
    "lr_schedule": "cosine_epoch",
    "lr_eta_min": 0.0,
    "lr_warmup_steps": 0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 2e-4,
    "hidden_dim": 80,
    "hidden_layers": 2,
    "flow_layers": 3,
    "flow_activation": "relu",
    "flow_residual": True,
    "flow_randperm": False,
    "flow_passes": 0,
    "flow_kind": "nsf",
    "spline_bins": 8,
}


NEXT8X_RAWFIT_BASE_CONFIG: dict[str, Any] = {
    **NEXT2X_RESIDUAL_BASE_CONFIG,
    "context_features": "raw_fit_summary",
}


BASE_PROFILES: dict[str, dict[str, Any]] = {
    "frontier": FRONTIER_BASE_CONFIG,
    "high_beta": HIGH_BETA_BASE_CONFIG,
    "next2x_residual": NEXT2X_RESIDUAL_BASE_CONFIG,
    "next8x_rawfit": NEXT8X_RAWFIT_BASE_CONFIG,
}


HIGH_BETA_ANCHORS: list[tuple[str, dict[str, Any]]] = [
    ("b512_lr004_b1098_b2099_wd2e4", {}),
    ("b512_lr0045_b1098_b2099_wd2e4", {"learning_rate": 0.0045}),
    ("b512_lr0035_b1098_b2099_wd2e4", {"learning_rate": 0.0035}),
    ("b768_lr0045_b1098_b2099_wd2e4", {"batch_size": 768, "learning_rate": 0.0045}),
    ("b896_lr0045_b1098_b2099_wd2e4", {"batch_size": 896, "learning_rate": 0.0045}),
    ("b1024_lr004_b1098_b2099_wd2e4", {"batch_size": 1024, "learning_rate": 0.004}),
    ("b512_lr004_b1095_b2099_wd2e4", {"adam_beta1": 0.95}),
    ("b512_lr004_b1098_b2098_wd2e4", {"adam_beta2": 0.98}),
    ("b512_lr004_gelu", {"flow_activation": "gelu"}),
    ("b512_lr004_silu", {"flow_activation": "silu"}),
    ("b512_lr004_elu", {"flow_activation": "elu"}),
    ("b512_lr004_residual", {"flow_residual": True}),
    ("b512_lr004_randperm", {"flow_randperm": True}),
    ("b512_lr004_coupling_passes2", {"flow_passes": 2}),
]


FRONTIER_ANCHORS: list[tuple[str, dict[str, Any]]] = [
    ("b512_lr003_b1090_b20999_wd2e4", {}),
    ("b512_lr00275_b1090_b20999_wd2e4", {"learning_rate": 0.00275}),
    ("b512_lr00325_b1090_b20999_wd2e4", {"learning_rate": 0.00325}),
    ("b512_lr0035_b1090_b20999_wd2e4", {"learning_rate": 0.0035}),
    ("b512_lr003_wd1e4", {"weight_decay": 1e-4}),
    ("b512_lr003_wd3e4", {"weight_decay": 3e-4}),
    ("b512_lr003_h96", {"hidden_dim": 96}),
    ("b512_lr003_h112", {"hidden_dim": 112}),
    ("b512_lr003_flow4", {"flow_layers": 4}),
    ("b512_lr003_bins10", {"spline_bins": 10}),
    ("b512_lr003_bins12", {"spline_bins": 12}),
    ("b384_lr0025", {"batch_size": 384, "learning_rate": 0.0025}),
    ("b640_lr00325", {"batch_size": 640, "learning_rate": 0.00325}),
    ("b768_lr0035", {"batch_size": 768, "learning_rate": 0.0035}),
    ("b512_lr003_beta095_0995", {"adam_beta1": 0.95, "adam_beta2": 0.995}),
    ("b512_lr003_beta093_0999", {"adam_beta1": 0.93, "adam_beta2": 0.999}),
    ("b512_lr003_gelu", {"flow_activation": "gelu"}),
    ("b512_lr003_silu", {"flow_activation": "silu"}),
    ("b512_lr003_residual", {"flow_residual": True}),
]


NEXT2X_RESIDUAL_ANCHORS: list[tuple[str, dict[str, Any]]] = [
    ("res_b512_lr003_wd2e4", {}),
    ("res_b512_lr0025_wd2e4", {"learning_rate": 0.0025}),
    ("res_b512_lr0035_wd2e4", {"learning_rate": 0.0035}),
    ("res_b512_lr004_wd2e4", {"learning_rate": 0.004}),
    ("res_b512_lr003_wd1e4", {"weight_decay": 1e-4}),
    ("res_b512_lr003_wd3e4", {"weight_decay": 3e-4}),
    ("res_b512_lr003_beta095_0995", {"adam_beta1": 0.95, "adam_beta2": 0.995}),
    ("res_b512_lr003_beta098_099", {"adam_beta1": 0.98, "adam_beta2": 0.99}),
    ("res_b512_lr003_eta5e5", {"lr_eta_min": 5e-5}),
    ("res_b512_lr003_eta1e4", {"lr_eta_min": 1e-4}),
    ("res_b512_lr003_decay20_eta5e5", {"lr_decay_epochs": 20, "lr_eta_min": 5e-5}),
    ("res_b512_lr003_decay20_eta1e4", {"lr_decay_epochs": 20, "lr_eta_min": 1e-4}),
    ("res_b512_lr003_decay20_eta2e4", {"lr_decay_epochs": 20, "lr_eta_min": 2e-4}),
    ("res_b512_lr003_cosstep_warm1k", {"lr_schedule": "cosine_step", "lr_warmup_steps": 1000}),
    ("res_b512_lr003_onecycle_warm1k", {"lr_schedule": "one_cycle", "lr_warmup_steps": 1000}),
    ("res_b384_lr0025_wd2e4", {"batch_size": 384, "learning_rate": 0.0025}),
    ("res_b640_lr00325_wd2e4", {"batch_size": 640, "learning_rate": 0.00325}),
    ("res_b768_lr0035_wd2e4", {"batch_size": 768, "learning_rate": 0.0035}),
    ("res_b1024_lr004_wd2e4", {"batch_size": 1024, "learning_rate": 0.004}),
    ("res_h64_f3_bins8", {"hidden_dim": 64}),
    ("res_h96_f3_bins8", {"hidden_dim": 96}),
    ("res_h80_f2_bins8", {"flow_layers": 2}),
    ("res_h80_f3_bins6", {"spline_bins": 6}),
    ("res_h80_f3_bins10", {"spline_bins": 10}),
    ("res_h80_f4_bins8", {"flow_layers": 4}),
    ("maf_b512_lr003_wd2e4", {"flow_residual": False, "flow_kind": "maf"}),
    ("gf_b512_lr003_wd2e4", {"flow_residual": False, "flow_kind": "gf"}),
    ("naf_b512_lr003_wd2e4", {"flow_residual": False, "flow_kind": "naf"}),
    ("res_raw_decay_summary", {"context_features": "raw_decay_summary"}),
    ("res_fit_summary", {"context_features": "fit_summary"}),
    ("res_raw_fit_summary", {"context_features": "raw_fit_summary"}),
    ("res_asinh", {"context_features": "asinh"}),
    ("res_rms_normalized", {"context_features": "rms_normalized"}),
    ("plain_b512_lr003_wd2e4", {"flow_residual": False}),
    ("res_randperm", {"flow_randperm": True}),
    ("res_passes2", {"flow_passes": 2}),
    ("res_ema0999", {"ema_decay": 0.999}),
]


NEXT8X_RAWFIT_ANCHORS: list[tuple[str, dict[str, Any]]] = [
    ("rawfit_res_b512_lr003_wd2e4", {}),
    ("rawfit_res_b512_lr0025_wd2e4", {"learning_rate": 0.0025}),
    ("rawfit_res_b512_lr00275_wd2e4", {"learning_rate": 0.00275}),
    ("rawfit_res_b512_lr00325_wd2e4", {"learning_rate": 0.00325}),
    ("rawfit_res_b512_lr0035_wd2e4", {"learning_rate": 0.0035}),
    ("rawfit_res_b512_lr004_wd2e4", {"learning_rate": 0.004}),
    ("rawfit_res_b512_wd0", {"weight_decay": 0.0}),
    ("rawfit_res_b512_wd1e4", {"weight_decay": 1e-4}),
    ("rawfit_res_b512_wd3e4", {"weight_decay": 3e-4}),
    ("rawfit_res_b512_wd5e4", {"weight_decay": 5e-4}),
    ("rawfit_res_b384_lr0025", {"batch_size": 384, "learning_rate": 0.0025}),
    ("rawfit_res_b640_lr00325", {"batch_size": 640, "learning_rate": 0.00325}),
    ("rawfit_res_b768_lr0035", {"batch_size": 768, "learning_rate": 0.0035}),
    ("rawfit_res_b1024_lr004", {"batch_size": 1024, "learning_rate": 0.004}),
    ("rawfit_res_beta095_0995", {"adam_beta1": 0.95, "adam_beta2": 0.995}),
    ("rawfit_res_beta098_099", {"adam_beta1": 0.98, "adam_beta2": 0.99}),
    ("rawfit_res_beta09_099", {"adam_beta2": 0.99}),
    ("rawfit_res_eta5e5", {"lr_eta_min": 5e-5}),
    ("rawfit_res_cosstep_warm500", {"lr_schedule": "cosine_step", "lr_warmup_steps": 500}),
    ("rawfit_res_cosstep_warm1000", {"lr_schedule": "cosine_step", "lr_warmup_steps": 1000}),
    ("rawfit_res_onecycle_warm500", {"lr_schedule": "one_cycle", "lr_warmup_steps": 500}),
    ("rawfit_res_h64", {"hidden_dim": 64}),
    ("rawfit_res_h96", {"hidden_dim": 96}),
    ("rawfit_res_h112", {"hidden_dim": 112}),
    ("rawfit_res_flow2", {"flow_layers": 2}),
    ("rawfit_res_flow4", {"flow_layers": 4}),
    ("rawfit_res_bins6", {"spline_bins": 6}),
    ("rawfit_res_bins10", {"spline_bins": 10}),
    ("rawfit_res_bins12", {"spline_bins": 12}),
    ("rawfit_plain_nsf", {"flow_residual": False}),
    ("rawfit_res_randperm", {"flow_randperm": True}),
    ("rawfit_res_passes2", {"flow_passes": 2}),
    ("fit_summary_residual", {"context_features": "fit_summary"}),
    ("raw_decay_summary_residual", {"context_features": "raw_decay_summary"}),
]


ANCHOR_PROFILES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "frontier": FRONTIER_ANCHORS,
    "high_beta": HIGH_BETA_ANCHORS,
    "next2x_residual": NEXT2X_RESIDUAL_ANCHORS,
    "next8x_rawfit": NEXT8X_RAWFIT_ANCHORS,
}


def slug(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.0e}" if abs(value) < 0.001 else f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def trial_name(config: dict[str, Any], index: int) -> str:
    parts = [
        f"t{index:03d}",
        f"b{config['batch_size']}",
        f"lr{slug(config['learning_rate'])}",
        f"s{slug(config['lr_schedule'])}",
        f"eta{slug(config['lr_eta_min'])}",
        f"de{config['lr_decay_epochs']}",
        f"b1{slug(config['adam_beta1'])}",
        f"b2{slug(config['adam_beta2'])}",
        f"wd{slug(config['weight_decay'])}",
        f"h{config['hidden_dim']}",
        f"hl{config['hidden_layers']}",
        f"f{config['flow_layers']}",
        f"bins{config['spline_bins']}",
        f"act{slug(config['flow_activation'])}",
        f"res{int(bool(config['flow_residual']))}",
        f"rp{int(bool(config['flow_randperm']))}",
        f"pass{config['flow_passes']}",
        f"kind{slug(config['flow_kind'])}",
    ]
    return "_".join(parts)


def anchored_trials(
    limit: int,
    *,
    anchor_count: int,
    anchor_offset: int,
    base_config: dict[str, Any],
    anchors: list[tuple[str, dict[str, Any]]],
) -> list[Trial]:
    trials: list[Trial] = []
    selected_anchors = anchors[max(0, anchor_offset) :]
    for name, overrides in selected_anchors[: min(limit, anchor_count)]:
        config = dict(base_config)
        config.update(overrides)
        trials.append(Trial(name=name, config=config, source="anchor"))
    return trials


def sample_trials(
    limit: int,
    seed: int,
    existing_overrides: list[dict[str, Any]],
    *,
    base_config: dict[str, Any],
    profile: str,
) -> list[Trial]:
    rng = random.Random(seed)
    trials: list[Trial] = []
    seen = {json.dumps(dict(base_config, **overrides), sort_keys=True) for overrides in existing_overrides}
    while len(trials) < limit:
        config = dict(base_config)
        if profile == "frontier":
            config.update(
                {
                    "batch_size": rng.choice([384, 512, 512, 640, 768, 896]),
                    "learning_rate": rng.choice([0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.004]),
                    "lr_schedule": rng.choice(["cosine_epoch", "cosine_epoch", "constant"]),
                    "lr_eta_min": rng.choice([0.0, 0.0, 1e-5, 5e-5]),
                    "adam_beta1": rng.choice([0.85, 0.9, 0.9, 0.93, 0.95, 0.98]),
                    "adam_beta2": rng.choice([0.99, 0.995, 0.999, 0.999]),
                    "weight_decay": rng.choice([0.0, 5e-5, 1e-4, 2e-4, 2e-4, 3e-4, 5e-4]),
                    "hidden_dim": rng.choice([64, 80, 80, 96, 112]),
                    "hidden_layers": rng.choice([1, 2, 2, 3]),
                    "flow_layers": rng.choice([2, 3, 3, 4]),
                    "spline_bins": rng.choice([6, 8, 8, 10, 12]),
                    "context_features": "raw",
                    "flow_activation": rng.choice(["relu", "relu", "elu", "gelu", "silu"]),
                    "flow_residual": rng.choice([False, False, False, True]),
                    "flow_randperm": rng.choice([False, False, True]),
                    "flow_passes": rng.choice([0, 0, 0, 2]),
                    "flow_kind": rng.choice(["nsf", "nsf", "nsf", "nsf", "maf"]),
                }
            )
        elif profile == "high_beta":
            config.update(
                {
                    "batch_size": rng.choice([384, 512, 640, 768, 896]),
                    "learning_rate": rng.choice([0.0035, 0.004, 0.0045, 0.005, 0.0055]),
                    "adam_beta1": rng.choice([0.93, 0.95, 0.97, 0.98, 0.99]),
                    "adam_beta2": rng.choice([0.97, 0.98, 0.99, 0.995]),
                    "weight_decay": rng.choice([1e-4, 2e-4, 3e-4, 5e-4]),
                    "hidden_dim": rng.choice([64, 80, 96, 112]),
                    "hidden_layers": rng.choice([2, 3]),
                    "flow_layers": rng.choice([3, 4]),
                    "spline_bins": rng.choice([8, 10, 12]),
                    "context_features": "raw",
                    "flow_activation": rng.choice(["relu", "elu", "gelu", "silu"]),
                    "flow_residual": rng.choice([False, False, True]),
                    "flow_randperm": rng.choice([False, True]),
                    "flow_passes": rng.choice([0, 0, 2]),
                    "flow_kind": rng.choice(["nsf", "nsf", "nsf", "maf", "gf"]),
                }
            )
        elif profile in {"next2x_residual", "next8x_rawfit"}:
            config.update(
                {
                    "batch_size": rng.choice([256, 384, 512, 512, 640, 768, 896, 1024]),
                    "learning_rate": rng.choice([0.002, 0.0025, 0.00275, 0.003, 0.00325, 0.0035, 0.004, 0.0045]),
                    "lr_schedule": rng.choice(["cosine_epoch", "cosine_epoch", "cosine_step", "one_cycle"]),
                    "lr_eta_min": rng.choice([0.0, 0.0, 1e-5, 5e-5, 1e-4]),
                    "lr_warmup_steps": rng.choice([0, 0, 500, 1000, 2000]),
                    "lr_decay_epochs": rng.choice([0, 0, 15, 20, 25]),
                    "adam_beta1": rng.choice([0.85, 0.9, 0.9, 0.93, 0.95, 0.98]),
                    "adam_beta2": rng.choice([0.98, 0.99, 0.995, 0.999, 0.999]),
                    "weight_decay": rng.choice([0.0, 5e-5, 1e-4, 2e-4, 2e-4, 3e-4, 5e-4]),
                    "hidden_dim": rng.choice([48, 64, 80, 80, 96, 112]),
                    "hidden_layers": rng.choice([1, 2, 2, 3]),
                    "flow_layers": rng.choice([2, 3, 3, 4]),
                    "spline_bins": rng.choice([4, 6, 8, 8, 10, 12]),
                    "context_features": rng.choice([
                        "raw",
                        "raw",
                        "raw_decay_summary",
                        "fit_summary",
                        "raw_fit_summary",
                        "asinh",
                        "rms_normalized",
                    ]),
                    "flow_activation": rng.choice(["relu", "relu", "elu", "gelu", "silu"]),
                    "flow_residual": rng.choice([True, True, True, False]),
                    "flow_randperm": rng.choice([False, False, True]),
                    "flow_passes": rng.choice([0, 0, 0, 2]),
                    "flow_kind": rng.choice(["nsf", "nsf", "nsf", "nsf", "maf", "gf", "naf"]),
                    "grad_clip_norm": rng.choice([0.0, 0.0, 5.0, 20.0]),
                    "ema_decay": rng.choice([0.0, 0.0, 0.999, 0.9995]),
                }
            )
            if profile == "next8x_rawfit":
                config["context_features"] = rng.choice(
                    [
                        "raw_fit_summary",
                        "raw_fit_summary",
                        "raw_fit_summary",
                        "raw_decay_summary",
                        "fit_summary",
                    ]
                )
                config["flow_residual"] = rng.choice([True, True, True, False])
        key = json.dumps(config, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append(Trial(name=trial_name(config, len(existing_overrides) + len(trials)), config=config, source="random"))
    return trials


def make_trials(count: int, seed: int, *, anchor_count: int, anchor_offset: int, profile: str) -> list[Trial]:
    base_config = BASE_PROFILES[profile]
    anchor_profile = ANCHOR_PROFILES[profile]
    anchors = anchored_trials(
        min(count, len(anchor_profile)),
        anchor_count=anchor_count,
        anchor_offset=anchor_offset,
        base_config=base_config,
        anchors=anchor_profile,
    )
    used_overrides = [
        overrides
        for _, overrides in anchor_profile[max(0, anchor_offset) : max(0, anchor_offset) + len(anchors)]
    ]
    return anchors + sample_trials(
        max(0, count - len(anchors)),
        seed=seed,
        existing_overrides=used_overrides,
        base_config=base_config,
        profile=profile,
    )


def stage_dir(root: Path, stage_name: str, trial: Trial) -> Path:
    return root / stage_name / trial.name


def build_command(args: argparse.Namespace, trial: Trial, *, stage_name: str, train_simulations: int, epochs: int) -> list[str]:
    config = trial.config
    return [
        str(args.uv),
        "run",
        "scripts/decay_broad_scaling_sweep.py",
        "--preset",
        "pilot",
        "--output-root",
        str(stage_dir(args.output_root, stage_name, trial)),
        "--train-simulations",
        str(train_simulations),
        "--seeds",
        str(args.seed),
        "--family",
        "spline_flow",
        "--val-simulations",
        str(args.val_simulations),
        "--standardization-simulations",
        str(args.standardization_simulations),
        "--train-sampler",
        str(config["train_sampler"]),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--lr-schedule",
        str(config["lr_schedule"]),
        "--lr-eta-min",
        str(config["lr_eta_min"]),
        "--lr-warmup-steps",
        str(config["lr_warmup_steps"]),
        "--lr-decay-epochs",
        str(config["lr_decay_epochs"]),
        "--adam-beta1",
        str(config["adam_beta1"]),
        "--adam-beta2",
        str(config["adam_beta2"]),
        "--adam-eps",
        str(config["adam_eps"]),
        "--validation-every-epochs",
        str(args.validation_every_epochs),
        "--max-optimizer-steps",
        "0",
        "--torch-compile",
        "none",
        "--grad-clip-norm",
        str(config["grad_clip_norm"]),
        "--ema-decay",
        str(config["ema_decay"]),
        "--batching-mode",
        str(config["batching_mode"]),
        "--loss-weight-mode",
        str(config["loss_weight_mode"]),
        "--loss-tail-weight",
        str(config["loss_tail_weight"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--hidden-layers",
        str(config["hidden_layers"]),
        "--mdn-components",
        "5",
        "--flow-layers",
        str(config["flow_layers"]),
        "--flow-context-dim",
        str(config["flow_context_dim"]),
        "--flow-activation",
        str(config["flow_activation"]),
        *(["--flow-residual"] if config["flow_residual"] else []),
        *(["--flow-randperm"] if config["flow_randperm"] else []),
        "--flow-kind",
        str(config["flow_kind"]),
        "--flow-passes",
        str(config["flow_passes"]),
        "--spline-bins",
        str(config["spline_bins"]),
        "--target-transform",
        str(config["target_transform"]),
        "--target-ridge",
        str(config["target_ridge"]),
        "--context-features",
        str(config["context_features"]),
        "--context-variants",
        "real",
        "--posterior-samples",
        "1",
        "--device",
        "cpu",
        "--validation-cache",
        str(args.validation_cache),
        "--early-val-cache-simulations",
        "0",
        "--panel-marginal-cache",
        str(args.panel_marginal_cache),
        "--panel-posterior-samples",
        "1",
        "--skip-x0-reference",
        "--skip-existing",
        "--jobs",
        "1",
        "--torch-threads",
        str(args.torch_threads),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--tail-top-k",
        "20",
        "--no-save-models",
    ]


def load_result(output_root: Path) -> dict[str, Any]:
    summaries = sorted(output_root.glob("runs/*/results/broad_scaling_run_summary.json"))
    if not summaries:
        return {"state": "missing"}
    data = json.loads(summaries[0].read_text(encoding="utf-8"))
    return {
        "state": "ok",
        "summary_path": str(summaries[0]),
        "full_val_nll_z_units": float(data["full_val_nll_z_units"]),
        "best_val_nll_z_units": float(data.get("best_val_nll_z_units", data["full_val_nll_z_units"])),
        "training_seconds": float(data["training_seconds"]),
        "optimizer_steps": int(data.get("optimizer_steps", -1)),
        "epochs_completed": int(data.get("epochs_completed", -1)),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_trial(args: argparse.Namespace, trial: Trial, *, stage_name: str, train_simulations: int, epochs: int) -> dict[str, Any]:
    output_root = stage_dir(args.output_root, stage_name, trial)
    command = build_command(args, trial, stage_name=stage_name, train_simulations=train_simulations, epochs=epochs)
    row: dict[str, Any] = {
        "stage": stage_name,
        "trial": trial.name,
        "source": trial.source,
        "config": trial.config,
        "train_simulations": train_simulations,
        "epochs": epochs,
        "command": command,
        "started_at": time.time(),
    }
    if args.dry_run:
        row["state"] = "dry_run"
        print(json.dumps(row, sort_keys=True))
        return row
    print(f"[{stage_name}] {trial.name}", flush=True)
    completed = subprocess.run(command, check=False)
    row["return_code"] = int(completed.returncode)
    row["finished_at"] = time.time()
    row["result"] = load_result(output_root)
    append_jsonl(args.output_root / "hpo_trials.jsonl", row)
    return row


def run_stage(
    args: argparse.Namespace,
    trials: list[Trial],
    *,
    stage_name: str,
    train_simulations: int,
    epochs: int,
) -> list[dict[str, Any]]:
    max_workers = max(1, min(int(args.trial_jobs), len(trials)))
    if max_workers == 1 or args.dry_run:
        return [
            run_trial(
                args,
                trial,
                stage_name=stage_name,
                train_simulations=train_simulations,
                epochs=epochs,
            )
            for trial in trials
        ]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trial = {
            executor.submit(
                run_trial,
                args,
                trial,
                stage_name=stage_name,
                train_simulations=train_simulations,
                epochs=epochs,
            ): trial
            for trial in trials
        }
        for future in as_completed(future_to_trial):
            rows.append(future.result())
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        result = row.get("result", {})
        if result.get("state") != "ok":
            return math.inf
        return float(result["full_val_nll_z_units"])

    return sorted(rows, key=key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-anchor successive-halving HPO driver for broad NPE efficiency probes."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--search-seed", type=int, default=20261031)
    parser.add_argument("--base-profile", choices=tuple(BASE_PROFILES), default="frontier")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--trial-jobs",
        type=int,
        default=1,
        help="Run this many independent HPO trials concurrently within each stage.",
    )
    parser.add_argument("--anchor-count", type=int, default=len(FRONTIER_ANCHORS))
    parser.add_argument("--anchor-offset", type=int, default=0)
    parser.add_argument("--promote-top", type=int, default=3)
    parser.add_argument("--stage1-train-simulations", type=int, default=128_000)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-train-simulations", type=int, default=512_000)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--promote-stage3-top", type=int, default=0)
    parser.add_argument("--stage3-train-simulations", type=int, default=0)
    parser.add_argument("--stage3-epochs", type=int, default=20)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--val-simulations", type=int, default=100_000)
    parser.add_argument("--standardization-simulations", type=int, default=60_000)
    parser.add_argument("--validation-every-epochs", type=int, default=5)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--panel-marginal-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16_384)
    parser.add_argument("--uv", default=os.environ.get("UV", "uv"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    trials = make_trials(
        args.trials,
        args.search_seed,
        anchor_count=max(0, int(args.anchor_count)),
        anchor_offset=max(0, int(args.anchor_offset)),
        profile=str(args.base_profile),
    )
    stage1_rows = run_stage(
        args,
        trials,
        stage_name="stage1_128k",
        train_simulations=int(args.stage1_train_simulations),
        epochs=int(args.stage1_epochs),
    )
    ranked_stage1 = rank_rows(stage1_rows)
    summary = {
        "stage1": ranked_stage1,
        "promoted": [row["trial"] for row in ranked_stage1[: max(0, int(args.promote_top))]],
    }
    if not args.skip_stage2 and args.promote_top > 0:
        trial_by_name = {trial.name: trial for trial in trials}
        stage2_trials = [trial_by_name[row["trial"]] for row in ranked_stage1[: int(args.promote_top)]]
        stage2_rows = run_stage(
            args,
            stage2_trials,
            stage_name="stage2_512k",
            train_simulations=int(args.stage2_train_simulations),
            epochs=int(args.stage2_epochs),
        )
        summary["stage2"] = rank_rows(stage2_rows)
        if args.promote_stage3_top > 0 and args.stage3_train_simulations > 0:
            stage3_source = summary["stage2"]
            stage3_trials = [trial_by_name[row["trial"]] for row in stage3_source[: int(args.promote_stage3_top)]]
            stage3_rows = run_stage(
                args,
                stage3_trials,
                stage_name=f"stage3_{int(args.stage3_train_simulations) // 1000}k",
                train_simulations=int(args.stage3_train_simulations),
                epochs=int(args.stage3_epochs),
            )
            summary["stage3"] = rank_rows(stage3_rows)
    summary_path = args.output_root / "hpo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
