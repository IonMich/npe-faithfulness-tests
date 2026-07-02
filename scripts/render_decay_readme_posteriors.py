from __future__ import annotations

import base64
import json
from pathlib import Path

from npe_posterior_viewer import (
    DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
    DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
    DEFAULT_BEST_BROAD_MODEL,
    DEFAULT_BEST_BROAD_SPLINE_MODEL,
    DEFAULT_BROAD_MODEL,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
    NPEPosteriorViewer,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs/00_shared_assets/readme_decay_posteriors"
SUMMARY_PATH = OUTPUT_DIR / "decay_population_readme_posteriors_summary.json"

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


def main() -> None:
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


if __name__ == "__main__":
    main()
