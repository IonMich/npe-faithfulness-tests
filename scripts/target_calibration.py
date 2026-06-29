from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import artifact_paths as ap


DEFAULT_DECAY_TARGET_SUMMARY = ap.FAITHFULNESS_TARGET_RESULTS / "faithfulness_target_check_summary.json"


def _first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _mean_normalized_wasserstein(data: dict[str, Any], dotted: str) -> float | None:
    value = _get_path(data, dotted)
    if isinstance(value, dict):
        value = value.get("value")
    return _first_float(value)


def resolve_target_wasserstein(
    explicit_target: float | None,
    *,
    summary_path: Path | None = None,
) -> tuple[float, str, dict[str, Any] | None]:
    if explicit_target is not None:
        return float(explicit_target), "explicit", None

    path = summary_path or DEFAULT_DECAY_TARGET_SUMMARY
    if not path.exists():
        raise FileNotFoundError(
            "No target was provided and no calibration summary exists at "
            f"{path}. Run `uv run scripts/check_faithfulness_target.py` first, "
            "or pass --target-wasserstein explicitly for reproduction."
        )

    data = json.loads(path.read_text())
    recommended = data.get("recommended_targets")
    mcmc_to_grid = _mean_normalized_wasserstein(
        data,
        "diagnostics.mcmc_full_to_grid.mean_normalized_wasserstein",
    )
    hmc_to_grid = _mean_normalized_wasserstein(
        data,
        "diagnostics.hmc_full_to_grid.mean_normalized_wasserstein",
    )
    computed_from_grid = None
    if mcmc_to_grid is not None and hmc_to_grid is not None:
        computed_from_grid = max(mcmc_to_grid, hmc_to_grid)
    target = _first_float(
        _get_path(data, "recommended_targets.mean_normalized_wasserstein"),
        computed_from_grid,
        data.get("target_wasserstein"),
    )
    if target is None:
        raise ValueError(f"Could not find a calibrated target in {path}")
    if isinstance(recommended, dict):
        recommended_targets = recommended
    elif computed_from_grid is not None:
        recommended_targets = {
            "mean_normalized_wasserstein": computed_from_grid,
            "rule": "max(full MCMC-to-grid, full HMC-to-grid)",
        }
    else:
        recommended_targets = None
    source = "mcmc_hmc_to_grid" if computed_from_grid is not None else data.get("target_source", "calibration_summary")
    return target, f"{source}:{path}", recommended_targets
