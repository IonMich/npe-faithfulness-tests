from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TARGET_NLL = -3.6040911785998784
TARGET_SECONDS = 784.5767706038896
DEFAULT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling")
DEFAULT_MATRIX_ROOT = DEFAULT_ROOT / "59_2x_efficiency_search"
DEFAULT_VALIDATION_CACHE = (
    DEFAULT_ROOT / "validation_cache" / "broad_prior_val_1m_float32.npz"
)
DEFAULT_PANEL_CACHE = (
    DEFAULT_ROOT / "panel_marginal_cache" / "decay_panel16_grid180_marginals.npz"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    rationale: str
    overrides: dict[str, Any] = field(default_factory=dict)


BASE_RECIPE: dict[str, Any] = {
    "family": "spline_flow",
    "batch_size": 512,
    "learning_rate": 0.006,
    "lr_schedule": "cosine_step",
    "lr_eta_min": 0.0,
    "lr_warmup_steps": 0,
    "validation_every_epochs": 5,
    "max_optimizer_steps": 0,
    "early_val_cache_simulations": 0,
    "torch_compile": "none",
    "grad_clip_norm": 0.0,
    "ema_decay": 0.0,
    "batching_mode": "pre_shuffle",
    "weight_decay": 1e-5,
    "hidden_dim": 64,
    "hidden_layers": 2,
    "mdn_components": 5,
    "flow_layers": 4,
    "flow_context_dim": 64,
    "spline_bins": 8,
    "context_features": "raw",
    "train_sampler": "random",
}


def candidate_matrix() -> list[Candidate]:
    return [
        Candidate(
            "base_pre_shuffle",
            "control",
            "Current fastest recipe control, to keep local proxy comparisons anchored.",
        ),
        Candidate(
            "lr0055",
            "hyperparam",
            "Bracket the current LR from below after pre-shuffle changed step timing.",
            {"learning_rate": 0.0055},
        ),
        Candidate(
            "lr0065",
            "hyperparam",
            "Bracket the current LR from above after no-clip made larger LR viable.",
            {"learning_rate": 0.0065},
        ),
        Candidate(
            "lr0070",
            "hyperparam",
            "Test whether a slightly more aggressive step budget improves proxy NLL.",
            {"learning_rate": 0.007},
        ),
        Candidate(
            "wd3e5",
            "hyperparam",
            "Localize the positive weight-decay signal between 1e-5 and 1e-4.",
            {"weight_decay": 3e-5},
        ),
        Candidate(
            "wd1e4",
            "hyperparam",
            "Retest the strongest regularization signal with the faster batching path.",
            {"weight_decay": 1e-4},
        ),
        Candidate(
            "wd3e4_lr0065",
            "hyperparam",
            "Small two-factor test around the best LR/weight-decay neighborhood.",
            {"weight_decay": 3e-4, "learning_rate": 0.0065},
        ),
        Candidate(
            "eta3e5",
            "hyperparam",
            "Check whether a nonzero cosine floor helps late training under fewer epochs.",
            {"lr_eta_min": 3e-5},
        ),
        Candidate(
            "batch384_lr005",
            "hyperparam",
            "Test smaller batches for more optimizer updates without a large wall-time hit.",
            {"batch_size": 384, "learning_rate": 0.005},
        ),
        Candidate(
            "batch768_lr0085",
            "hyperparam",
            "Retest larger batches with proportional LR in the pre-shuffle implementation.",
            {"batch_size": 768, "learning_rate": 0.0085},
        ),
        Candidate(
            "batch1024_lr004",
            "hyperparam",
            "Reproduce the old near-miss batch1024 regime with the pre-shuffle path.",
            {"batch_size": 1024, "learning_rate": 0.004, "lr_schedule": "cosine_epoch"},
        ),
        Candidate(
            "batch1024_lr004_wd1e4",
            "hyperparam",
            "Test whether the regularization gain transfers to the batch1024 near-miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004",
            "combined",
            "Time-budget candidate: batch1024 near-miss plus hidden80 and wd1e-4.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr0035",
            "combined",
            "Lower LR bracket around the full-scale LR0.004 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0035,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr0045",
            "combined",
            "Midpoint LR bracket between the LR0.004 near miss and LR0.005 miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0045,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr0038",
            "hyperparam",
            "Fine LR bracket just below the full-scale LR0.004 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0038,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr0042",
            "hyperparam",
            "Fine LR bracket just above the full-scale LR0.004 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0042,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd5e5_lr004",
            "hyperparam",
            "Weight-decay bracket between 1e-5 and the successful 1e-4 signal.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 5e-5,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd2e4_lr004",
            "hyperparam",
            "Upper weight-decay bracket around the successful 1e-4 signal.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_step",
            "combined",
            "Use per-step cosine decay in the same near-miss configuration.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_step",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_eta5e6",
            "hyperparam",
            "Tiny cosine floor bracket between zero and eta1e-5.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "lr_eta_min": 5e-6,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_eta1e5",
            "combined",
            "Keep a tiny cosine floor so late training does not fully anneal to zero.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "lr_eta_min": 1e-5,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_valevery1",
            "combined",
            "Same near-miss training path with denser checkpoint selection.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "validation_every_epochs": 1,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_cacheval200k",
            "combined",
            "Select checkpoints on a fixed validation-cache subset closer to the hard metric.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "early_val_cache_simulations": 200_000,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_cacheval500k",
            "combined",
            "Larger fixed validation-cache checkpoint selector for the near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "early_val_cache_simulations": 500_000,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_sequential",
            "combined",
            "Skip per-epoch tensor reshuffling to test whether the near-miss can afford more epochs.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "batching_mode": "sequential",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden96_wd1e4_lr004",
            "architecture",
            "Width bracket above hidden80 inside the current full-scale near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 96,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_layers3_wd1e4_lr004",
            "architecture",
            "Context-network depth bracket inside the current full-scale near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "hidden_layers": 3,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_bins10_wd1e4_lr004",
            "architecture",
            "Test whether spline-bin capacity stacks with hidden80 in the near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "spline_bins": 10,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_bins10_wd1e4_lr004",
            "combined",
            "Batch1024 near-miss plus the strongest 256k capacity/regularization signal.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "spline_bins": 10,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_flow5_hidden80_wd1e4_lr004",
            "architecture",
            "Flow-depth bracket inside the current full-scale near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 5,
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd1e4_lr004",
            "architecture",
            "Fastest measured architecture under the final record recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004",
            "architecture",
            "Fast flow3/bins8 candidate plus the strongest proxy weight decay.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden96_bins8_wd1e4_lr004",
            "architecture",
            "Recover some flow3 capacity with width while keeping lower flow depth.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 96,
                "spline_bins": 8,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1280_flow3_hidden80_bins8_wd1e4_lr005",
            "combined",
            "Combine the faster flow3/bins8 architecture with the fastest plausible batch regime.",
            {
                "batch_size": 1280,
                "learning_rate": 0.005,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd2e4_lr004_step_warmup2k",
            "combined",
            "Step-based cosine with warmup and stronger weight decay for earlier learning.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_step",
                "lr_warmup_steps": 2000,
                "hidden_dim": 80,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_step_warmup2k",
            "combined",
            "Fast flow3/bins8 candidate with the most plausible step-efficiency schedule tweak.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_step",
                "lr_warmup_steps": 2000,
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_eta5e5",
            "combined",
            "Near-miss follow-up: keep late cosine learning alive for the fast flow3 model.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "lr_eta_min": 5e-5,
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_eta1e4",
            "combined",
            "Stronger eta-floor bracket for the fast flow3 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "lr_eta_min": 1e-4,
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_ema999",
            "combined",
            "EMA checkpointing probe for the fast flow3 near miss; tests whether smoothed weights improve full-cache NLL without changing architecture.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "ema_decay": 0.999,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_ema9995",
            "combined",
            "Slower EMA bracket for the fast flow3 near miss; useful if the final sparse selector is noisy late in training.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "ema_decay": 0.9995,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_valevery1",
            "combined",
            "Dense validation selector for the fast flow3 near miss; tests whether sparse five-epoch selection is leaving a better checkpoint unused.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "validation_every_epochs": 1,
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_sequential",
            "combined",
            "Fixed-order tensor batching for the fast flow3 near miss; tests whether avoiding per-epoch full-tensor reshuffle buys steps without hurting NLL.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "batching_mode": "sequential",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_cacheval200k",
            "combined",
            "Use a larger fixed validation-cache subset for early checkpoint selection in the fast flow3 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "early_val_cache_simulations": 200_000,
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_lhs",
            "combined",
            "Use Latin-hypercube prior draws to improve broad-prior coverage at the same simulation count.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "train_sampler": "lhs",
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_sobol",
            "combined",
            "Use scrambled Sobol prior draws as a lower-discrepancy broad-prior training set.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "train_sampler": "sobol",
            },
        ),
        Candidate(
            "batch1024_flow2_hidden80_bins8_wd2e4_lr004",
            "architecture",
            "Cheaper flow-depth branch: test whether two spline transforms retain enough posterior expressivity.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 2,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden64_bins8_wd2e4_lr004",
            "architecture",
            "Smaller context network for the successful flow3 recipe; useful only if speed wins without NLL loss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 64,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins10_wd2e4_lr004",
            "architecture",
            "Capacity trade: keep flow3 speed but increase spline-bin resolution for a stronger conditional density.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 10,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins12_wd2e4_lr004",
            "architecture",
            "Higher spline-bin bracket for flow3 to test whether binned capacity beats extra flow layers.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 12,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_rawsummary",
            "architecture",
            "Representation branch: append decay-summary features to raw traces for the fast flow3 model.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "context_features": "raw_decay_summary",
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr004_decaysummary",
            "architecture",
            "Aggressive representation compression: summary-only context for much cheaper input processing.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
                "context_features": "decay_summary",
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr0035",
            "hyperparam",
            "Lower LR bracket around the successful flow3/wd2e-4 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0035,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd2e4_lr0045",
            "hyperparam",
            "Upper LR bracket around the successful flow3/wd2e-4 near miss.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0045,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd3e4_lr004",
            "hyperparam",
            "Weight-decay bracket above wd2e-4 for the fast flow3 architecture.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 3e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins8_wd3e4_lr0045",
            "hyperparam",
            "Two-factor bracket: slightly higher LR with stronger regularization in flow3.",
            {
                "batch_size": 1024,
                "learning_rate": 0.0045,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 3e-4,
            },
        ),
        Candidate(
            "batch1152_flow3_hidden80_bins8_wd2e4_lr0045",
            "combined",
            "Larger-batch flow3 bracket to buy more epochs/steps under the hard wall-time ceiling.",
            {
                "batch_size": 1152,
                "learning_rate": 0.0045,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch896_flow3_hidden80_bins8_wd2e4_lr0035",
            "combined",
            "Smaller-batch flow3 bracket to test whether extra optimizer updates beat throughput loss.",
            {
                "batch_size": 896,
                "learning_rate": 0.0035,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 8,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_mdn80x2_comp10_wd2e4_lr004",
            "architecture",
            "Diverse NPE family probe: conditional MDN under the successful batch/LR/WD regime.",
            {
                "family": "mdn",
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "hidden_layers": 2,
                "mdn_components": 10,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_mdn128x3_comp10_wd2e4_lr004",
            "architecture",
            "Larger MDN probe to test whether mixture density can trade flow depth for cheap likelihood.",
            {
                "family": "mdn",
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 128,
                "hidden_layers": 3,
                "mdn_components": 10,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_full_gaussian80x2_wd2e4_lr004",
            "architecture",
            "Dumb density baseline: conditional full Gaussian with the same optimizer regime.",
            {
                "family": "full_gaussian",
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "hidden_layers": 2,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_diag_gaussian80x2_wd2e4_lr004",
            "architecture",
            "Cheapest conditional Gaussian baseline for speed/NLL calibration.",
            {
                "family": "diag_gaussian",
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "hidden_layers": 2,
                "weight_decay": 2e-4,
            },
        ),
        Candidate(
            "batch1024_flow3_hidden80_bins10_wd1e4_lr004",
            "architecture",
            "Cheaper flow-depth trade: fewer transforms but higher spline resolution.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "flow_layers": 3,
                "hidden_dim": 80,
                "spline_bins": 10,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_rawsummary",
            "architecture",
            "Retest engineered decay summaries only in the strong near-miss recipe.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
                "context_features": "raw_decay_summary",
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr004_decaysummary",
            "architecture",
            "Summary-only context ablation; useful as a cheap representation baseline.",
            {
                "batch_size": 1024,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
                "context_features": "decay_summary",
            },
        ),
        Candidate(
            "batch1152_hidden80_wd1e4_lr0045",
            "combined",
            "Slightly larger batch to buy more epochs under the hard wall-time ceiling.",
            {
                "batch_size": 1152,
                "learning_rate": 0.0045,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1056_hidden80_wd1e4_lr004",
            "combined",
            "Small batch increase to fit one more epoch with minimal quality change.",
            {
                "batch_size": 1056,
                "learning_rate": 0.004,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1088_hidden80_wd1e4_lr0042",
            "combined",
            "Fine-grained batch/LR point between batch1024 and batch1152.",
            {
                "batch_size": 1088,
                "learning_rate": 0.0042,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1120_hidden80_wd1e4_lr0043",
            "combined",
            "Fine-grained batch/LR point near the largest safe batch increase.",
            {
                "batch_size": 1120,
                "learning_rate": 0.0043,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1280_hidden80_wd1e4_lr005",
            "combined",
            "Middle ground between batch1024 quality and batch1536 throughput.",
            {
                "batch_size": 1280,
                "learning_rate": 0.005,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1024_hidden80_wd1e4_lr005",
            "combined",
            "Upper LR bracket for the batch1024 hidden80 regularized candidate.",
            {
                "batch_size": 1024,
                "learning_rate": 0.005,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1536_bins10_wd1e4_lr006",
            "combined",
            "Higher-throughput fallback for the bins10 regularized signal.",
            {
                "batch_size": 1536,
                "learning_rate": 0.006,
                "lr_schedule": "cosine_epoch",
                "spline_bins": 10,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "batch1536_hidden80_wd1e4_lr006",
            "combined",
            "Higher-throughput fallback if batch1024 cannot fit enough epochs under time.",
            {
                "batch_size": 1536,
                "learning_rate": 0.006,
                "lr_schedule": "cosine_epoch",
                "hidden_dim": 80,
                "weight_decay": 1e-4,
            },
        ),
        Candidate(
            "bins10",
            "architecture",
            "Bins improved Stage B NLL; remeasure with pre-shuffle time accounting.",
            {"spline_bins": 10},
        ),
        Candidate(
            "bins12",
            "architecture",
            "Capacity bracket above bins10 to see if likelihood gain is still efficient.",
            {"spline_bins": 12},
        ),
        Candidate(
            "flow5",
            "architecture",
            "Depth bracket above flow4; useful only if NLL gain beats speed loss.",
            {"flow_layers": 5},
        ),
        Candidate(
            "flow5_bins10",
            "architecture",
            "Two-axis capacity test for depth plus spline resolution.",
            {"flow_layers": 5, "spline_bins": 10},
        ),
        Candidate(
            "hidden80",
            "architecture",
            "Width bracket above hidden64 without changing flow depth.",
            {"hidden_dim": 80},
        ),
        Candidate(
            "hidden64_layers3",
            "architecture",
            "Embedding MLP depth bracket with constant width.",
            {"hidden_layers": 3},
        ),
        Candidate(
            "bins10_wd1e4",
            "combined",
            "Combine the best Stage B capacity signal with the best regularization signal.",
            {"spline_bins": 10, "weight_decay": 1e-4},
        ),
        Candidate(
            "flow5_wd1e4",
            "combined",
            "Depth plus regularization, checking whether flow5 overfits less at proxy scale.",
            {"flow_layers": 5, "weight_decay": 1e-4},
        ),
        Candidate(
            "hidden80_wd1e4",
            "combined",
            "Width plus regularization, a cheaper alternative to deeper/binnier flows.",
            {"hidden_dim": 80, "weight_decay": 1e-4},
        ),
        Candidate(
            "hidden96_wd1e4",
            "combined",
            "Width bracket above hidden80 with the same regularization.",
            {"hidden_dim": 96, "weight_decay": 1e-4},
        ),
        Candidate(
            "hidden80_bins10",
            "combined",
            "Check whether the hidden80 gain stacks with the bins10 capacity signal.",
            {"hidden_dim": 80, "spline_bins": 10},
        ),
        Candidate(
            "hidden80_lr0065_wd1e4",
            "combined",
            "Interaction test for the hidden80 regularized model and the upper LR bracket.",
            {"hidden_dim": 80, "learning_rate": 0.0065, "weight_decay": 1e-4},
        ),
    ]


def slug_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace(".", "p")


def recipe_for(candidate: Candidate) -> dict[str, Any]:
    recipe = dict(BASE_RECIPE)
    recipe.update(candidate.overrides)
    return recipe


def stage_root(args: argparse.Namespace, candidate: Candidate) -> Path:
    sims = slug_value(args.train_simulations)
    epochs = slug_value(args.epochs)
    return args.output_root / f"{candidate.name}_d{sims}_e{epochs}"


def local_command(args: argparse.Namespace, candidate: Candidate) -> list[str]:
    recipe = recipe_for(candidate)
    output_root = stage_root(args, candidate)
    command = [
        "uv",
        "run",
        "scripts/decay_broad_scaling_sweep.py",
        "--preset",
        "pilot",
        "--output-root",
        str(output_root),
        "--train-simulations",
        str(args.train_simulations),
        "--seeds",
        str(args.seed),
        "--family",
        str(recipe["family"]),
        "--val-simulations",
        str(args.val_simulations),
        "--standardization-simulations",
        str(args.standardization_simulations),
        "--train-sampler",
        str(recipe["train_sampler"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(recipe["batch_size"]),
        "--learning-rate",
        str(recipe["learning_rate"]),
        "--lr-schedule",
        str(recipe["lr_schedule"]),
        "--lr-eta-min",
        str(recipe["lr_eta_min"]),
        "--lr-warmup-steps",
        str(recipe["lr_warmup_steps"]),
        "--validation-every-epochs",
        str(recipe["validation_every_epochs"]),
        "--max-optimizer-steps",
        str(recipe["max_optimizer_steps"]),
        "--torch-compile",
        str(recipe["torch_compile"]),
        "--grad-clip-norm",
        str(recipe["grad_clip_norm"]),
        "--ema-decay",
        str(recipe["ema_decay"]),
        "--batching-mode",
        str(recipe["batching_mode"]),
        "--weight-decay",
        str(recipe["weight_decay"]),
        "--hidden-dim",
        str(recipe["hidden_dim"]),
        "--hidden-layers",
        str(recipe["hidden_layers"]),
        "--mdn-components",
        str(recipe["mdn_components"]),
        "--flow-layers",
        str(recipe["flow_layers"]),
        "--flow-context-dim",
        str(recipe["flow_context_dim"]),
        "--spline-bins",
        str(recipe["spline_bins"]),
        "--context-features",
        str(recipe["context_features"]),
        "--context-variants",
        "real",
        "--posterior-samples",
        str(args.posterior_samples),
        "--device",
        "cpu",
        "--validation-cache",
        str(args.validation_cache),
        "--early-val-cache-simulations",
        str(recipe["early_val_cache_simulations"]),
        "--panel-marginal-cache",
        str(args.panel_marginal_cache),
        "--panel-posterior-samples",
        str(args.posterior_samples),
        "--skip-x0-reference",
        "--skip-existing",
        "--jobs",
        "1",
        "--torch-threads",
        str(args.torch_threads),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--tail-top-k",
        str(args.tail_top_k),
        "--no-save-models",
    ]
    if args.dry_run_sweep:
        command.append("--dry-run")
    return command


def remote_command(args: argparse.Namespace, candidate: Candidate) -> list[str]:
    recipe = recipe_for(candidate)
    output_root = stage_root(args, candidate)
    command = [
        "uv",
        "run",
        "scripts/submit_remote_broad_scaling.py",
        "--endpoint",
        args.endpoint,
        "--run-name",
        args.run_name_prefix + candidate.name,
        "--output-root",
        str(output_root),
        "--train-simulations",
        str(args.train_simulations),
        "--seeds",
        str(args.seed),
        "--family",
        str(recipe["family"]),
        "--device",
        "cpu",
        "--standardization-simulations",
        str(args.standardization_simulations),
        "--train-sampler",
        str(recipe["train_sampler"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(recipe["batch_size"]),
        "--learning-rate",
        str(recipe["learning_rate"]),
        "--lr-schedule",
        str(recipe["lr_schedule"]),
        "--lr-eta-min",
        str(recipe["lr_eta_min"]),
        "--lr-warmup-steps",
        str(recipe["lr_warmup_steps"]),
        "--validation-every-epochs",
        str(recipe["validation_every_epochs"]),
        "--max-optimizer-steps",
        str(recipe["max_optimizer_steps"]),
        "--torch-compile",
        "reduce_overhead",
        "--grad-clip-norm",
        str(recipe["grad_clip_norm"]),
        "--ema-decay",
        str(recipe["ema_decay"]),
        "--batching-mode",
        str(recipe["batching_mode"]),
        "--weight-decay",
        str(recipe["weight_decay"]),
        "--hidden-dim",
        str(recipe["hidden_dim"]),
        "--hidden-layers",
        str(recipe["hidden_layers"]),
        "--mdn-components",
        str(recipe["mdn_components"]),
        "--flow-layers",
        str(recipe["flow_layers"]),
        "--flow-context-dim",
        str(recipe["flow_context_dim"]),
        "--spline-bins",
        str(recipe["spline_bins"]),
        "--context-features",
        str(recipe["context_features"]),
        "--jobs",
        "1",
        "--torch-threads",
        str(args.torch_threads),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--early-stop-val-simulations",
        str(args.val_simulations),
        "--validation-cache",
        str(args.validation_cache),
        "--early-val-cache-simulations",
        str(recipe["early_val_cache_simulations"]),
        "--validation-cache-simulations",
        "1000000",
        "--panel-marginal-cache",
        str(args.panel_marginal_cache),
        "--panel-posterior-samples",
        str(args.posterior_samples),
        "--posterior-samples",
        str(args.posterior_samples),
        "--context-variants",
        "real",
        "--tail-top-k",
        str(args.tail_top_k),
        "--no-save-models",
    ]
    if args.no_sync:
        command.append("--no-sync")
    return command


def is_completed(root: Path) -> bool:
    return (root / "results" / "broad_scaling_summary.json").exists()


def emit_candidates(args: argparse.Namespace) -> None:
    candidates = select_candidates(args)
    count = 0
    for candidate in candidates:
        root = stage_root(args, candidate)
        if args.skip_completed and is_completed(root):
            continue
        command = remote_command(args, candidate) if args.remote else local_command(args, candidate)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "name": candidate.name,
                        "kind": candidate.kind,
                        "rationale": candidate.rationale,
                        "output_root": str(root),
                        "recipe": recipe_for(candidate),
                        "command": command,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"# {candidate.name} [{candidate.kind}] - {candidate.rationale}")
            print(shlex.join(command))
        count += 1
        if args.limit is not None and count >= args.limit:
            break


def select_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = [
        candidate
        for candidate in candidate_matrix()
        if args.kind is None or candidate.kind == args.kind
    ]
    if args.names:
        wanted = set(args.names)
        candidates = [candidate for candidate in candidates if candidate.name in wanted]
    return candidates


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_candidates(args: argparse.Namespace) -> None:
    candidates = select_candidates(args)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "experiment_manifest.jsonl"
    pending = list(candidates)
    running: list[tuple[Candidate, Path, subprocess.Popen[bytes], Any]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < args.max_parallel:
            candidate = pending.pop(0)
            root = stage_root(args, candidate)
            if args.skip_completed and is_completed(root):
                append_manifest(
                    manifest,
                    {
                        "event": "skip_completed",
                        "name": candidate.name,
                        "output_root": str(root),
                        "time": time.time(),
                    },
                )
                continue
            root.mkdir(parents=True, exist_ok=True)
            log_path = root / "search_runner.log"
            command = local_command(args, candidate)
            log_handle = log_path.open("ab")
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((candidate, root, process, log_handle))
            append_manifest(
                manifest,
                {
                    "event": "start",
                    "name": candidate.name,
                    "kind": candidate.kind,
                    "rationale": candidate.rationale,
                    "output_root": str(root),
                    "recipe": recipe_for(candidate),
                    "command": command,
                    "log": str(log_path),
                    "pid": process.pid,
                    "time": time.time(),
                },
            )
            print(f"started {candidate.name}\tpid={process.pid}\tlog={log_path}", flush=True)

        still_running: list[tuple[Candidate, Path, subprocess.Popen[bytes], Any]] = []
        for candidate, root, process, log_handle in running:
            code = process.poll()
            if code is None:
                still_running.append((candidate, root, process, log_handle))
                continue
            log_handle.close()
            if code != 0:
                failures += 1
            append_manifest(
                manifest,
                {
                    "event": "finish",
                    "name": candidate.name,
                    "output_root": str(root),
                    "returncode": code,
                    "time": time.time(),
                    "summary_exists": is_completed(root),
                },
            )
            print(f"finished {candidate.name}\treturncode={code}\tsummary={is_completed(root)}", flush=True)
        running = still_running
        if pending or running:
            time.sleep(args.poll_seconds)
    if failures:
        raise SystemExit(f"{failures} candidate run(s) failed")


def iter_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/results/broad_scaling_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["summary_path"] = str(path)
            item["run_root"] = str(path.parent.parent)
            rows.append(item)
    return rows


def value_as_float(row: dict[str, Any], key: str) -> float | None:
    value = field_value(row, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_summary_row(path: Path) -> dict[str, Any] | None:
    summary_path = path / "results" / "broad_scaling_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("rows", [])
    if not rows or not isinstance(rows[-1], dict):
        return None
    row = dict(rows[-1])
    row["summary_path"] = str(summary_path)
    row["run_root"] = str(summary_path.parent.parent)
    return row


def format_float(value: float | None, *, digits: int = 12) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def field_value(row: dict[str, Any], key: str) -> Any:
    if key in row and row[key] is not None:
        return row[key]
    config = row.get("config")
    if isinstance(config, dict):
        return config.get(key)
    return None


def rank_rows(args: argparse.Namespace) -> None:
    rows = iter_summary_rows(args.root)
    if args.train_simulations is not None:
        rows = [
            row
            for row in rows
            if int(field_value(row, "train_simulations") or -1) == int(args.train_simulations)
        ]
    if args.epochs is not None:
        rows = [
            row
            for row in rows
            if int(field_value(row, "epochs") or -1) == int(args.epochs)
        ]
    if args.batching_mode is not None:
        rows = [
            row for row in rows if field_value(row, "batching_mode") == args.batching_mode
        ]
    keyed: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        nll = value_as_float(row, "full_val_nll_z_units")
        if nll is None:
            nll = value_as_float(row, "best_val_nll_z_units")
        if nll is None:
            continue
        keyed.append((nll, row))
    keyed.sort(key=lambda item: item[0])
    print(
        "rank\tfull_val_nll_z\ttarget_gap\tseconds\ttime_gap\tsims\tepochs\tbatch\tlr\twd\tema\tflow\tbins\thidden\tmode\trun"
    )
    for rank, (nll, row) in enumerate(keyed[: args.limit], start=1):
        seconds = value_as_float(row, "training_seconds")
        target_gap = nll - TARGET_NLL
        time_gap = None if seconds is None else seconds - TARGET_SECONDS
        print(
            "\t".join(
                [
                    str(rank),
                    f"{nll:.12f}",
                    f"{target_gap:+.12f}",
                    "" if seconds is None else f"{seconds:.3f}",
                    "" if time_gap is None else f"{time_gap:+.3f}",
                    str(field_value(row, "train_simulations") or ""),
                    str(field_value(row, "epochs") or row.get("epochs_completed", "")),
                    str(field_value(row, "batch_size") or ""),
                    str(field_value(row, "learning_rate") or field_value(row, "lr") or ""),
                    str(field_value(row, "weight_decay") or ""),
                    str(field_value(row, "ema_decay") or ""),
                    str(field_value(row, "flow_layers") or ""),
                    str(field_value(row, "spline_bins") or ""),
                    str(field_value(row, "hidden_dim") or ""),
                    str(field_value(row, "batching_mode") or ""),
                    str(row.get("run_root", "")),
                ]
            )
        )


def report_candidates(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for candidate in select_candidates(args):
        root = stage_root(args, candidate)
        row = first_summary_row(root)
        recipe = recipe_for(candidate)
        record: dict[str, Any] = {
            "name": candidate.name,
            "kind": candidate.kind,
            "status": "completed" if row is not None else "pending",
            "rationale": candidate.rationale,
            "output_root": str(root),
            "recipe": recipe,
        }
        if row is not None:
            nll = value_as_float(row, "full_val_nll_z_units")
            if nll is None:
                nll = value_as_float(row, "best_val_nll_z_units")
            seconds = value_as_float(row, "training_seconds")
            record.update(
                {
                    "full_val_nll_z_units": nll,
                    "target_gap": None if nll is None else nll - TARGET_NLL,
                    "training_seconds": seconds,
                    "time_gap": None if seconds is None else seconds - TARGET_SECONDS,
                    "epochs_completed": row.get("epochs_completed"),
                    "optimizer_steps": row.get("optimizer_steps"),
                    "progress_jsonl": row.get("progress_jsonl"),
                    "summary_path": row.get("summary_path"),
                }
            )
        records.append(record)

    completed = [record for record in records if record["status"] == "completed"]
    pending = [record for record in records if record["status"] != "completed"]
    completed.sort(
        key=lambda record: (
            float("inf")
            if record.get("full_val_nll_z_units") is None
            else float(record["full_val_nll_z_units"]),
            float("inf")
            if record.get("training_seconds") is None
            else float(record["training_seconds"]),
        )
    )
    ordered = completed + pending

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(
            json.dumps(ordered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.format == "json":
        for record in ordered[: args.limit]:
            print(json.dumps(record, sort_keys=True))
        return

    print(
        "status\tkind\tfull_val_nll_z\ttarget_gap\tseconds\ttime_gap\tepochs\tsteps\tbatch\tlr\twd\tema\tflow\tbins\thidden\tmode\tname\trationale"
    )
    for record in ordered[: args.limit]:
        recipe = record["recipe"]
        print(
            "\t".join(
                [
                    str(record["status"]),
                    str(record["kind"]),
                    format_float(record.get("full_val_nll_z_units")),
                    format_float(record.get("target_gap")),
                    format_seconds(record.get("training_seconds")),
                    format_seconds(record.get("time_gap")),
                    str(record.get("epochs_completed") or ""),
                    str(record.get("optimizer_steps") or ""),
                    str(recipe.get("batch_size", "")),
                    str(recipe.get("learning_rate", "")),
                    str(recipe.get("weight_decay", "")),
                    str(recipe.get("ema_decay", "")),
                    str(recipe.get("flow_layers", "")),
                    str(recipe.get("spline_bins", "")),
                    str(recipe.get("hidden_dim", "")),
                    str(recipe.get("batching_mode", "")),
                    str(record["name"]),
                    str(record["rationale"]),
                ]
            )
        )


def iter_progress_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def progress_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for path in args.paths:
        path = Path(path)
        if path.is_dir():
            paths.extend(sorted(path.glob("**/training_progress.jsonl")))
        else:
            paths.append(path)
    if not paths:
        paths.extend(sorted(args.root.glob("**/training_progress.jsonl")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def progress_report(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for path in progress_paths_from_args(args):
        records = iter_progress_records(path)
        if args.val_only:
            records = [record for record in records if record.get("val_evaluated")]
        if args.tail is not None:
            records = records[-args.tail :]
        run = str(path.parent.parent.parent)
        for record in records:
            rows.append(
                {
                    "run": run,
                    "progress_jsonl": str(path),
                    "event": record.get("event"),
                    "epoch": record.get("epoch"),
                    "optimizer_steps": record.get("optimizer_steps"),
                    "elapsed_training_seconds": record.get("elapsed_training_seconds"),
                    "train_nll_z_units": record.get("train_nll_z_units"),
                    "val_nll_z_units": record.get("val_nll_z_units"),
                    "best_val_nll_z_units": record.get("best_val_nll_z_units"),
                    "lr": record.get("lr"),
                    "val_evaluated": record.get("val_evaluated"),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["run"]),
            -1 if row["epoch"] is None else int(row["epoch"]),
            -1 if row["optimizer_steps"] is None else int(row["optimizer_steps"]),
        )
    )

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "run\tevent\tepoch\tsteps\telapsed_seconds\ttrain_nll_z\tval_nll_z\tbest_val_nll_z\tlr\tprogress_jsonl"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["run"]),
                    str(row.get("event") or ""),
                    str(row.get("epoch") or ""),
                    str(row.get("optimizer_steps") or ""),
                    format_seconds(
                        None
                        if row.get("elapsed_training_seconds") is None
                        else float(row["elapsed_training_seconds"])
                    ),
                    format_float(
                        None if row.get("train_nll_z_units") is None else float(row["train_nll_z_units"])
                    ),
                    format_float(
                        None if row.get("val_nll_z_units") is None else float(row["val_nll_z_units"])
                    ),
                    format_float(
                        None
                        if row.get("best_val_nll_z_units") is None
                        else float(row["best_val_nll_z_units"])
                    ),
                    "" if row.get("lr") is None else str(row["lr"]),
                    str(row["progress_jsonl"]),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit and rank systematic NPE efficiency-search experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit", help="Emit candidate commands from the matrix.")
    emit.add_argument("--output-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    emit.add_argument("--train-simulations", type=int, default=256_000)
    emit.add_argument("--epochs", type=int, default=25)
    emit.add_argument("--seed", type=int, default=20260901)
    emit.add_argument("--val-simulations", type=int, default=100_000)
    emit.add_argument("--standardization-simulations", type=int, default=60_000)
    emit.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    emit.add_argument("--panel-marginal-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    emit.add_argument("--posterior-samples", type=int, default=128)
    emit.add_argument("--torch-threads", type=int, default=2)
    emit.add_argument("--eval-batch-size", type=int, default=16384)
    emit.add_argument("--tail-top-k", type=int, default=0)
    emit.add_argument("--endpoint", default="http://127.0.0.1:8877")
    emit.add_argument("--run-name-prefix", default="npe_eff_matrix_")
    emit.add_argument("--kind", choices=("control", "hyperparam", "architecture", "combined"))
    emit.add_argument("--names", nargs="*", default=[])
    emit.add_argument("--limit", type=int)
    emit.add_argument("--format", choices=("shell", "json"), default="shell")
    emit.add_argument("--remote", action="store_true")
    emit.add_argument("--no-sync", action="store_true")
    emit.add_argument("--skip-completed", action="store_true")
    emit.add_argument("--dry-run-sweep", action="store_true")

    run = subparsers.add_parser("run", help="Run local candidate commands with logs and a manifest.")
    run.add_argument("--output-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    run.add_argument("--train-simulations", type=int, default=256_000)
    run.add_argument("--epochs", type=int, default=25)
    run.add_argument("--seed", type=int, default=20260901)
    run.add_argument("--val-simulations", type=int, default=100_000)
    run.add_argument("--standardization-simulations", type=int, default=60_000)
    run.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    run.add_argument("--panel-marginal-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    run.add_argument("--posterior-samples", type=int, default=128)
    run.add_argument("--torch-threads", type=int, default=1)
    run.add_argument("--eval-batch-size", type=int, default=16384)
    run.add_argument("--tail-top-k", type=int, default=0)
    run.add_argument("--kind", choices=("control", "hyperparam", "architecture", "combined"))
    run.add_argument("--names", nargs="*", default=[])
    run.add_argument("--limit", type=int)
    run.add_argument("--max-parallel", type=int, default=1)
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--skip-completed", action="store_true")
    run.add_argument("--dry-run-sweep", action="store_true")
    run.set_defaults(remote=False, endpoint="", run_name_prefix="", no_sync=True, format="shell")

    rank = subparsers.add_parser("rank", help="Rank completed broad-scaling summaries.")
    rank.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    rank.add_argument("--train-simulations", type=int)
    rank.add_argument("--epochs", type=int)
    rank.add_argument("--batching-mode")
    rank.add_argument("--limit", type=int, default=25)

    report = subparsers.add_parser("report", help="Show candidate status for a fixed proxy stage.")
    report.add_argument("--output-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    report.add_argument("--train-simulations", type=int, default=512_000)
    report.add_argument("--epochs", type=int, default=35)
    report.add_argument("--seed", type=int, default=20260901)
    report.add_argument("--kind", choices=("control", "hyperparam", "architecture", "combined"))
    report.add_argument("--names", nargs="*", default=[])
    report.add_argument("--limit", type=int, default=80)
    report.add_argument("--format", choices=("tsv", "json"), default="tsv")
    report.add_argument("--write-json", type=Path)

    progress = subparsers.add_parser("progress", help="Emit NLL-vs-steps records from training_progress.jsonl files.")
    progress.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    progress.add_argument("--paths", nargs="*", default=[])
    progress.add_argument("--tail", type=int)
    progress.add_argument("--val-only", action="store_true", default=True)
    progress.add_argument("--all-events", action="store_false", dest="val_only")
    progress.add_argument("--write-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "emit":
        emit_candidates(args)
        return
    if args.command == "run":
        run_candidates(args)
        return
    if args.command == "rank":
        rank_rows(args)
        return
    if args.command == "report":
        report_candidates(args)
        return
    if args.command == "progress":
        progress_report(args)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
