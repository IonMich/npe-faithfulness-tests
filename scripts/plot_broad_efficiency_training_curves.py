from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "runs/00_shared_assets/readme_scaling/decay_broad_npe_training_efficiency_curves.png"
)
TARGET_NLL = -3.6040911785998784
TARGET_SECONDS = 784.5767706038896


RUNS = [
    {
        "label": "Spline L4 h64, D=4.096M, B=512",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/38_spline_lr_schedule_proof/cosine_4m_seed20260901/results/broad_scaling_summary.json",
        "progress": None,
        "color": "#7a7a7a",
    },
    {
        "label": "Spline L4 h80, D=4.096M, B=1024",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/58_partial_epoch_budget/batch1024_hidden80_wd1e4_lr004_e74_max294000_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/58_partial_epoch_budget/batch1024_hidden80_wd1e4_lr004_e74_max294000_seed20260901/runs/n4096000_seed20260901/results/training_progress.jsonl",
        "color": "#1f77b4",
    },
    {
        "label": "Spline L3 h80, D=4.096M, B=1024",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/67_flow3_partial_steps_clean/flow3_bins8_wd2e4_e54_max214800_threads1_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/67_flow3_partial_steps_clean/flow3_bins8_wd2e4_e54_max214800_threads1_seed20260901/runs/n4096000_seed20260901/results/training_progress.jsonl",
        "color": "#ff7f0e",
    },
    {
        "label": "Spline L3 h80, D=8.192M, B=1024",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/73_flow3_8m_datascale_capped/train8m_lr004_wd2e4_e27_max212000_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/73_flow3_8m_datascale_capped/train8m_lr004_wd2e4_e27_max212000_seed20260901/runs/n8192000_seed20260901/results/training_progress.jsonl",
        "color": "#2ca02c",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series_from_progress(path: Path) -> dict[str, np.ndarray]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "epoch":
            records.append(record)
    return {
        "epochs": np.array([record["epoch"] for record in records], dtype=float),
        "steps": np.array([record["optimizer_steps"] for record in records], dtype=float),
        "seconds": np.array(
            [record["elapsed_training_seconds"] for record in records],
            dtype=float,
        ),
        "train": np.array(
            [record["train_nll_z_units"] for record in records],
            dtype=float,
        ),
        "val": np.array(
            [
                record["val_nll_z_units"]
                if record.get("val_evaluated")
                and record.get("val_nll_z_units") is not None
                else np.nan
                for record in records
            ],
            dtype=float,
        ),
    }


def series_from_summary(summary: dict) -> dict[str, np.ndarray]:
    row = summary["rows"][0]
    history = row["history"]
    train_standardized = np.array(history["train_nll"], dtype=float)
    val_standardized = np.array(history["val_nll"], dtype=float)
    offset = float(row["final_train_nll_z_units"]) - float(train_standardized[-1])
    epochs = np.arange(1, len(train_standardized) + 1, dtype=float)

    if row.get("optimizer_steps") is None:
        batch_size = float(summary["config"]["batch_size"])
        n_train = float(summary["config"]["train_simulations"][0])
        batches_per_epoch = np.ceil(n_train / batch_size)
        steps = epochs * batches_per_epoch
    else:
        steps = epochs * (float(row["optimizer_steps"]) / len(train_standardized))
    seconds = epochs * (float(row["training_seconds"]) / len(train_standardized))
    return {
        "epochs": epochs,
        "steps": steps,
        "seconds": seconds,
        "train": train_standardized + offset,
        "val": val_standardized + offset,
    }


def load_run(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    progress = spec["progress"]
    if progress is not None and progress.exists():
        series = series_from_progress(progress)
    else:
        series = series_from_summary(summary)
    return {**spec, "summary_data": summary, "row": summary["rows"][0], "series": series}


def plot(output_path: Path = DEFAULT_OUTPUT) -> Path:
    runs = [load_run(spec) for spec in RUNS]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    panels = (
        (axes[0], "steps", "optimizer steps", "Training curves by optimizer step"),
        (axes[1], "seconds", "training seconds", "Training curves by wall time"),
    )
    for ax, x_key, xlabel, title in panels:
        for run in runs:
            series = run["series"]
            color = run["color"]
            ax.plot(
                series[x_key],
                series["train"],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=f"{run['label']} train",
            )
            val_mask = np.isfinite(series["val"])
            ax.plot(
                series[x_key][val_mask],
                series["val"][val_mask],
                color=color,
                linewidth=1.2,
                linestyle="--",
                marker="o",
                markersize=3.8,
                alpha=0.85,
                label=f"{run['label']} validation",
            )
            row = run["row"]
            final_x = (
                float(row.get("optimizer_steps") or series[x_key][-1])
                if x_key == "steps"
                else float(row["training_seconds"])
            )
            ax.scatter(
                [final_x],
                [float(row["full_val_nll_z_units"])],
                color=color,
                marker="*",
                s=150,
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
        ax.axhline(
            TARGET_NLL,
            color="#222222",
            linewidth=1.2,
            linestyle=":",
            label="previous record NLL",
        )
        if x_key == "seconds":
            ax.axvline(
                TARGET_SECONDS,
                color="#444444",
                linewidth=1.2,
                linestyle=":",
                label="2x wall-time target",
            )
            ax.set_xlim(left=0, right=1700)
        else:
            ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NLL, z units (lower is better)")
        ax.set_title(title)
        ax.set_ylim(-3.66, -2.95)

    handles, labels = axes[1].get_legend_handles_labels()
    seen = set()
    unique = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique.append((handle, label))
    fig.legend(
        [handle for handle, _ in unique],
        [label for _, label in unique],
        loc="outside lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.suptitle(
        "Broad-prior single-decay NPE training efficiency",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = plot()
    print(output_path)


if __name__ == "__main__":
    main()
