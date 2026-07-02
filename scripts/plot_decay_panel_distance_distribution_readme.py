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


def posterior_sample_count(path: Path) -> int:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return int(summary["posterior_samples"])


def main() -> None:
    rows = load_rows(DEFAULT_ROWS)
    posterior_samples = posterior_sample_count(DEFAULT_SUMMARY)
    plot(rows, DEFAULT_OUTPUT, posterior_samples=posterior_samples)
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
