from __future__ import annotations

import argparse
import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import npe_stage1_decay as stage1
from calibrate_sign_target import build_grid_reference, compare_samples_to_reference, mode_summary
from npe_flow_stress_tests import make_sign_case, run_random_walk_mcmc
from npe_posterior_viewer import (
    DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
    DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
    DEFAULT_BEST_BROAD_MODEL,
    DEFAULT_BEST_BROAD_SPLINE_MODEL,
    DEFAULT_BROAD_MODEL,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
    NPEPosteriorViewer,
    SampleCornerLayer,
    WeightedCornerLayer,
    render_corner_layers,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_decay_posteriors"
SUMMARY_PATH = OUTPUT_DIR / "decay_population_readme_posteriors_summary.json"
SIGN_OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_sign_posteriors"
SIGN_ENSEMBLE_SUMMARY = (
    ROOT
    / "runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/"
    "results/sign_population_ensemble_summary.json"
)

MODEL_ID_MAP = {
    "broad_fresh_e15_ensemble4": "flow2_residual_nsf_ensemble4",
    "broad_weighted_checkpoint_pool": "convex_weighted_checkpoint_ensemble",
}

CASES = [
    {
        "key": "population_prior_predictive",
        "mode": "prior",
        "corner_path": OUTPUT_DIR / "decay_population_posterior_corner.png",
        "signal_path": OUTPUT_DIR / "decay_population_posterior_signal.png",
    },
    {
        "key": "low_prior_stress",
        "mode": "low_prior_very_low",
        "corner_path": OUTPUT_DIR / "decay_low_prior_stress_posterior_corner.png",
        "signal_path": OUTPUT_DIR / "decay_low_prior_stress_posterior_signal.png",
    },
]


def save_data_uri(uri: str, path: Path) -> None:
    prefix, payload = uri.split(",", 1)
    if not prefix.startswith("data:image/png;base64"):
        raise ValueError(f"Unexpected image data URI prefix: {prefix[:80]}")
    path.write_bytes(base64.b64decode(payload))


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


def metric_value(metrics: dict[str, object] | None) -> float | None:
    if metrics is None:
        return None
    value = metrics.get("mean_normalized_wasserstein")
    if not isinstance(value, dict):
        return None
    return float(value["value"])


def clean_model_id(model_id: str) -> str:
    return MODEL_ID_MAP.get(model_id, model_id)


def selected_model_summary(model: dict[str, object]) -> dict[str, object]:
    return {
        "id": clean_model_id(str(model["id"])),
        "label": str(model["label"]),
        "ensemble_size": model.get("ensemble_size"),
        "full_val_nll_z_units": model.get("full_val_nll_z_units"),
        "training_seconds": model.get("training_seconds"),
    }


def summarize_mcmc_diagnostics(diagnostics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "max_rhat": max(float(value["rhat"]) for value in diagnostics.values()),
        "min_bulk_ess": min(float(value["ess_bulk"]) for value in diagnostics.values()),
        "min_tail_ess": min(float(value["ess_tail"]) for value in diagnostics.values()),
    }


def sample_sign_prior_predictive_signal(*, seed: int, draw_index: int) -> tuple[np.ndarray, np.ndarray]:
    case = make_sign_case()
    rng = np.random.default_rng(seed)
    theta = rng.normal(
        case.prior_mean[None, :],
        case.prior_std[None, :],
        size=(draw_index + 1, case.z_dim),
    )
    x = case.simulate_x(theta, rng)
    return theta[draw_index].astype(np.float64), x[draw_index].astype(np.float64)


def standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean[None, :]) / std[None, :]).astype(np.float32)


def stage1_config_from_checkpoint(config_dict: dict[str, Any]) -> stage1.Stage1Config:
    config = stage1.Stage1Config(**config_dict)
    if config.progress_jsonl is not None:
        config = replace(config, progress_jsonl=Path(config.progress_jsonl))
    return config


def load_sign_member(member_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(member_path, map_location="cpu", weights_only=False)
    config = stage1_config_from_checkpoint(checkpoint["config"])
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float64)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float64)
    model = stage1.make_model(
        "spline_flow",
        config,
        x_dim=int(x_mean.shape[0]),
        z_dim=int(z_mean.shape[0]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "z_mean": z_mean,
        "z_std": z_std,
        "path": member_path,
    }


def load_sign_ensemble(summary_path: Path, device: torch.device) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return [load_sign_member(Path(item["model_pt"]), device) for item in summary["members"]]


@torch.no_grad()
def sample_sign_population_npe(
    *,
    members: list[dict[str, Any]],
    x: np.ndarray,
    samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    base_count = samples // len(members)
    remainder = samples % len(members)
    for index, member in enumerate(members):
        count = base_count + (1 if index < remainder else 0)
        x_standardized = standardize(
            x[None, :],
            np.asarray(member["x_mean"], dtype=np.float64),
            np.asarray(member["x_std"], dtype=np.float64),
        )
        x_tensor = torch.from_numpy(x_standardized).to(device)
        folded_standardized = member["model"].sample(count, x_tensor).detach().cpu().numpy()
        folded = (
            folded_standardized * np.asarray(member["z_std"], dtype=np.float64)[None, :]
            + np.asarray(member["z_mean"], dtype=np.float64)[None, :]
        )
        sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        chunks.append(np.column_stack([sign * np.maximum(folded[:, 0], 0.0), folded[:, 1]]))
    samples_raw = np.concatenate(chunks, axis=0)
    rng.shuffle(samples_raw, axis=0)
    return samples_raw


def sign_reference_layer(reference: dict[str, object]) -> WeightedCornerLayer:
    theta1 = np.asarray(reference["theta1_grid"], dtype=np.float64)
    theta2 = np.asarray(reference["theta2_grid"], dtype=np.float64)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="ij")
    weights = np.asarray(reference["weights"], dtype=np.float64)

    def widths(axis: np.ndarray) -> np.ndarray:
        if axis.size <= 1:
            return np.ones_like(axis)
        return np.full_like(axis, float(axis[1] - axis[0]), dtype=np.float64)

    return WeightedCornerLayer(
        label="Exact grid",
        color="#172033",
        values=np.column_stack([theta1_grid.ravel(), theta2_grid.ravel()]),
        weights=weights.ravel(),
        grid_shape=weights.shape,
        axes=(theta1, theta2),
        widths=(widths(theta1), widths(theta2)),
        hist_lw=2.0,
        contour_lw=1.55,
    )


def render_case(viewer: NPEPosteriorViewer, case: dict[str, object]) -> dict[str, object]:
    result = viewer.render(
        model_ids=["broad_fresh_e15_ensemble4", "broad_weighted_checkpoint_pool"],
        mode=str(case["mode"]),
        draw_id=None,
        reuse_current=False,
        refresh_layers=set(),
        npe_render_mode="sample",
        posterior_samples=7000,
        include_grid=True,
        include_mcmc=True,
        grid_size=60,
        npe_grid_size=None,
    )
    save_data_uri(str(result["corner"]), Path(case["corner_path"]))
    save_data_uri(str(result["signal"]), Path(case["signal_path"]))

    npe_metrics_raw = result["npe_grid_metrics"]
    assert isinstance(npe_metrics_raw, dict)
    mcmc_diagnostics = summarize_mcmc_diagnostics(result["mcmc_metadata"]["diagnostics"])
    return {
        "mode": case["mode"],
        "mode_metadata": result["mode_metadata"],
        "true_theta": result["true_theta"],
        "posterior_samples": result["posterior_samples"],
        "corner_path": str(Path(case["corner_path"]).relative_to(ROOT)),
        "signal_path": str(Path(case["signal_path"]).relative_to(ROOT)),
        "grid": {
            "grid_size": result["grid_metadata"]["grid_size"],
            "grid_points": result["grid_metadata"]["grid_points"],
            "max_edge_mass": result["grid_metadata"]["max_edge_mass"],
        },
        "mcmc": {
            "mean_normalized_wasserstein": metric_value(result["mcmc_grid_metrics"]),
            "acceptance_rate": result["mcmc_metadata"]["acceptance_rate"],
            "convergence_ok": result["mcmc_metadata"]["convergence_ok"],
            **mcmc_diagnostics,
        },
        "npe_mean_normalized_wasserstein": {
            clean_model_id(model_id): metric_value(metrics)
            for model_id, metrics in npe_metrics_raw.items()
        },
        "selected_models": [
            selected_model_summary(model)
            for model in result["selected_npe_models"]
        ],
    }


def render_decay_cases() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viewer = NPEPosteriorViewer(
        DEFAULT_MODEL,
        DEFAULT_BROAD_MODEL,
        DEFAULT_BEST_BROAD_MODEL,
        DEFAULT_BEST_BROAD_SPLINE_MODEL,
        DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
        DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
        DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
        seed=20260702,
        device="cpu",
        mcmc_device="cpu",
        mcmc_chains=8,
        mcmc_steps=24_000,
        mcmc_burn_in=6_000,
        mcmc_proposal_scale=(0.030, 0.030, 0.040),
    )
    summary = {
        str(case["key"]): render_case(viewer, case)
        for case in CASES
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)


def render_sign_population_case(args: argparse.Namespace) -> None:
    SIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = stage1.choose_training_device(args.device)
    theta, x = sample_sign_prior_predictive_signal(
        seed=args.signal_seed,
        draw_index=args.draw_index,
    )
    reference = build_grid_reference(x0=x, grid_size=args.grid_size, grid_limit=args.grid_limit)
    members = load_sign_ensemble(args.sign_ensemble_summary, device)
    npe_samples = sample_sign_population_npe(
        members=members,
        x=x,
        samples=args.npe_samples,
        seed=args.seed + 1000,
        device=device,
    )

    case = make_sign_case()
    mcmc_samples, mcmc_accept, mcmc_seconds = run_random_walk_mcmc(
        case,
        x,
        chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        seed=args.seed,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    mcmc_post = mcmc_samples[:, args.mcmc_burn_in :, :].reshape(-1, case.z_dim)

    figure = render_corner_layers(
        labels=[r"$\theta_1$", r"$\theta_2$"],
        true_values=theta,
        weighted_layers=[sign_reference_layer(reference)],
        sample_layers=[
            SampleCornerLayer("MCMC", "#b85c38", mcmc_post, hist_lw=1.5, contour_lw=1.35),
            SampleCornerLayer("Population NPE", "#0f766e", npe_samples, hist_lw=1.5, contour_lw=1.35),
        ],
        true_color="#172033",
        title="Sign population posterior: exact grid vs MCMC vs NPE",
        rng=np.random.default_rng(args.seed + 2000),
    )
    figure_path = SIGN_OUTPUT_DIR / "sign_population_prior_signal_corner.png"
    summary_path = SIGN_OUTPUT_DIR / "sign_population_prior_signal_summary.json"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    summary = {
        "description": (
            "Fresh full-prior sign posterior check for the population-trained "
            "folded-target NPE ensemble. This is not the old fixed-x0 sign run."
        ),
        "signal": {
            "seed": int(args.signal_seed),
            "draw_index": int(args.draw_index),
            "theta": theta,
            "x": x,
        },
        "grid": {
            "grid_size": int(args.grid_size),
            "grid_limit": float(args.grid_limit),
            "edge_mass": reference["edge_mass"],
        },
        "mcmc": {
            "chains": int(args.mcmc_chains),
            "steps": int(args.mcmc_steps),
            "burn_in": int(args.mcmc_burn_in),
            "posterior_samples": int(mcmc_post.shape[0]),
            "acceptance_rate": float(np.mean(mcmc_accept)),
            "seconds": float(mcmc_seconds),
            "to_grid_raw": compare_samples_to_reference(mcmc_post, reference, diagnostic=False),
            "to_grid_diagnostic": compare_samples_to_reference(mcmc_post, reference, diagnostic=True),
            "mode_mass": mode_summary(mcmc_post),
        },
        "npe": {
            "ensemble_summary": args.sign_ensemble_summary,
            "members": int(len(members)),
            "posterior_samples": int(npe_samples.shape[0]),
            "to_grid_raw": compare_samples_to_reference(npe_samples, reference, diagnostic=False),
            "to_grid_diagnostic": compare_samples_to_reference(npe_samples, reference, diagnostic=True),
            "mode_mass": mode_summary(npe_samples),
        },
        "outputs": {
            "figure": figure_path,
            "summary": summary_path,
        },
    }
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render README posterior comparison figures.")
    parser.add_argument(
        "--mode",
        choices=("single_decay", "sign_population"),
        default="single_decay",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--sign-ensemble-summary", type=Path, default=SIGN_ENSEMBLE_SUMMARY)
    parser.add_argument("--signal-seed", type=int, default=20260707)
    parser.add_argument("--draw-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--grid-limit", type=float, default=4.0)
    parser.add_argument("--npe-samples", type=int, default=80_000)
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=12_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=3_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single_decay":
        render_decay_cases()
    else:
        render_sign_population_case(args)


if __name__ == "__main__":
    main()
