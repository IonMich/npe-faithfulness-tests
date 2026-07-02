from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluate_broad_panel_w_distribution import plot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/"
    "200_panel_w_distribution_eval_mdn512k_spline4m_flow2_panel500/results/"
    "panel_w_combined_rows.csv"
)
DEFAULT_SUMMARY = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/"
    "200_panel_w_distribution_eval_mdn512k_spline4m_flow2_panel500/results/"
    "panel_w_distribution_summary.json"
)
DEFAULT_MCMC_ROWS = (
    ROOT
    / "runs/01_exponential_decay/15_broad_scaling/"
    "200_panel_w_distribution_eval_mdn512k_spline4m_flow2_panel500/results/"
    "panel_w_mcmc_rows.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "runs/00_shared_assets/readme_scaling/decay_panel_w_distribution_mdn512k_vs_spline4m_500.png"
)


def parse_value(value: str) -> object:
    if value == "":
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def merge_mcmc_rows(
    combined_rows: list[dict[str, object]],
    mcmc_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    mcmc_by_index = {int(row["index"]): row for row in mcmc_rows}
    missing = sorted(int(row["index"]) for row in combined_rows if int(row["index"]) not in mcmc_by_index)
    if missing:
        raise ValueError(f"MCMC rows are missing panel indices: {missing[:12]}")

    merged = []
    for row in combined_rows:
        out = dict(row)
        mcmc = mcmc_by_index[int(row["index"])]
        for source, target in (
            ("wasserstein", "mcmc_wasserstein"),
            ("target_ratio", "mcmc_target_ratio"),
            ("w_A", "mcmc_w_A"),
            ("w_k", "mcmc_w_k"),
            ("w_sigma", "mcmc_w_sigma"),
            ("seconds", "mcmc_seconds"),
            ("acceptance_rate", "mcmc_acceptance_rate"),
            ("max_rhat", "mcmc_max_rhat"),
            ("min_bulk_ess", "mcmc_min_bulk_ess"),
            ("min_tail_ess", "mcmc_min_tail_ess"),
            ("convergence_ok", "mcmc_convergence_ok"),
        ):
            out[target] = mcmc[source]
        merged.append(out)
    return merged


def posterior_sample_count(path: Path) -> int:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return int(summary["posterior_samples"])


def main() -> None:
    rows = merge_mcmc_rows(load_rows(DEFAULT_ROWS), load_rows(DEFAULT_MCMC_ROWS))
    posterior_samples = posterior_sample_count(DEFAULT_SUMMARY)
    plot(rows, DEFAULT_OUTPUT, posterior_samples=posterior_samples)
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
