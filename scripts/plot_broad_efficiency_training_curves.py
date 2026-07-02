from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "runs/00_shared_assets/readme_scaling/decay_broad_npe_training_efficiency_curves.png"
)
SUCCESS_NLL = -3.60
SUB_MINUTE_SECONDS = 60.0


SINGLE_RUNS = [
    {
        "label": "Plain spline 8M single",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/train8m_lr004_wd2e4_e27_max212000_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/train8m_lr004_wd2e4_e27_max212000_seed20260901/runs/n8192000_seed20260901/results/training_progress.jsonl",
        "color": "#6f7378",
    },
    {
        "label": "Residual NSF 2M single",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/113_next2x_residual_2m_saved/residual_lr003_wd2e4_e20_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/113_next2x_residual_2m_saved/residual_lr003_wd2e4_e20_seed20260901/runs/n2048000_seed20260901/results/training_progress.jsonl",
        "color": "#1f77b4",
    },
]

ENSEMBLE_RUNS = [
    {
        "label": "Residual NSF 512k x4 ensemble",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/residual_512k_e20_lr003_wd2e4_seeds4/results/ensemble4_proof_summary.json",
        "member_progress_glob": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/residual_512k_e20_lr003_wd2e4_seeds4/runs/*/results/training_progress.jsonl",
        "color": "#2ca02c",
    },
    {
        "label": "Raw-fit residual 512k x4 ensemble",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/mixed_lr_rawfit_512k_e10_seeds2_6_3_5/results/ensemble4_proof_summary.json",
        "member_progress_glob": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/mixed_lr_rawfit_512k_e10_seeds2_6_3_5/runs/*/results/training_progress.jsonl",
        "color": "#d62728",
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


def load_single_run(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    progress = spec["progress"]
    if progress is not None and progress.exists():
        series = series_from_progress(progress)
    else:
        series = series_from_summary(summary)
    return {**spec, "summary_data": summary, "row": summary["rows"][0], "series": series}


def nan_column_mean(values: np.ndarray) -> np.ndarray:
    means = np.full(values.shape[1], np.nan, dtype=float)
    for index in range(values.shape[1]):
        column = values[:, index]
        finite = np.isfinite(column)
        if finite.any():
            means[index] = float(column[finite].mean())
    return means


def ensemble_series_from_progress(paths: list[Path]) -> dict[str, np.ndarray]:
    member_series = [series_from_progress(path) for path in paths]
    if not member_series:
        raise ValueError("No member progress files found for ensemble run.")
    length = min(len(series["epochs"]) for series in member_series)

    def stack(key: str) -> np.ndarray:
        return np.vstack([series[key][:length] for series in member_series])

    train_stack = stack("train")
    val_stack = stack("val")
    return {
        "epochs": stack("epochs").mean(axis=0),
        "steps": stack("steps").mean(axis=0),
        "seconds": stack("seconds").mean(axis=0),
        "train": train_stack.mean(axis=0),
        "train_lo": train_stack.min(axis=0),
        "train_hi": train_stack.max(axis=0),
        "val": nan_column_mean(val_stack),
        "member_steps": stack("steps"),
        "member_seconds": stack("seconds"),
    }


def load_ensemble_run(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    paths = sorted(spec["member_progress_glob"].parent.parent.parent.glob("*/results/training_progress.jsonl"))
    if not paths:
        paths = sorted(spec["member_progress_glob"].parent.glob("training_progress.jsonl"))
    series = ensemble_series_from_progress(paths)
    return {
        **spec,
        "summary_data": summary,
        "member_progress_paths": paths,
        "series": series,
    }


def final_x_for_single(run: dict, x_key: str) -> float:
    row = run["row"]
    series = run["series"]
    if x_key == "steps":
        return float(row.get("optimizer_steps") or series[x_key][-1])
    return float(row["training_seconds"])


def final_x_for_ensemble(run: dict, x_key: str) -> float:
    series = run["series"]
    if x_key == "steps":
        return float(series["steps"][-1])
    return float(run["summary_data"]["training_wall_seconds"])


def plot_single_run(ax: plt.Axes, run: dict, x_key: str) -> None:
    series = run["series"]
    color = run["color"]
    ax.plot(series[x_key], series["train"], color=color, linewidth=2.1, alpha=0.92)

    val_mask = np.isfinite(series["val"])
    if val_mask.any():
        ax.plot(
            series[x_key][val_mask],
            series["val"][val_mask],
            color=color,
            linewidth=1.2,
            linestyle="--",
            marker="o",
            markersize=3.8,
            alpha=0.9,
        )

    ax.scatter(
        [final_x_for_single(run, x_key)],
        [float(run["row"]["full_val_nll_z_units"])],
        color=color,
        marker="*",
        s=165,
        edgecolor="black",
        linewidth=0.5,
        zorder=6,
    )


def plot_ensemble_run(ax: plt.Axes, run: dict, x_key: str) -> None:
    series = run["series"]
    color = run["color"]
    ax.plot(series[x_key], series["train"], color=color, linewidth=2.4, alpha=0.95)
    ax.fill_between(
        series[x_key],
        series["train_lo"],
        series["train_hi"],
        color=color,
        alpha=0.12,
        linewidth=0,
    )

    val_mask = np.isfinite(series["val"])
    if val_mask.any():
        ax.plot(
            series[x_key][val_mask],
            series["val"][val_mask],
            color=color,
            linewidth=1.25,
            linestyle="--",
            marker="o",
            markersize=4,
            alpha=0.9,
        )

    summary = run["summary_data"]
    individual_nll = np.array(summary["individual_full_val_nll_z_units"], dtype=float)
    if x_key == "seconds":
        individual_x = np.array(summary["individual_training_seconds"], dtype=float)
    else:
        individual_x = np.full_like(individual_nll, series["steps"][-1], dtype=float)
    ax.scatter(
        individual_x,
        individual_nll,
        facecolors="white",
        edgecolors=color,
        marker="o",
        s=40,
        linewidth=1.1,
        alpha=0.9,
        zorder=5,
    )
    ax.scatter(
        [final_x_for_ensemble(run, x_key)],
        [float(summary["ensemble_full_val_nll_z_units"])],
        color=color,
        marker="*",
        s=220,
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )


def configure_axis(ax: plt.Axes, x_key: str, xlabel: str, title: str) -> None:
    ax.axhline(
        SUCCESS_NLL,
        color="#222222",
        linewidth=1.2,
        linestyle=":",
    )
    if x_key == "seconds":
        ax.axvline(
            SUB_MINUTE_SECONDS,
            color="#8b0000",
            linewidth=1.2,
            linestyle=":",
        )
        ax.set_xscale("log")
        ax.set_xlim(left=4.5, right=900)
        ax.set_xticks([5, 10, 20, 60, 120, 260, 780])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.annotate(
            "57.37s ensemble\n-3.6134 exact NLL",
            xy=(57.37, -3.61336271875),
            xytext=(92, -3.645),
            arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.0},
            fontsize=9,
            color="#222222",
        )
    else:
        ax.set_xlim(left=0, right=220000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NLL, z units (lower is better)")
    ax.set_title(title)
    ax.set_ylim(-3.66, -2.78)


def legend_handles(runs: list[dict]) -> list[Line2D]:
    recipe_handles = [
        Line2D([0], [0], color=run["color"], lw=3, label=run["label"])
        for run in runs
    ]
    style_handles = [
        Line2D([0], [0], color="#111111", lw=2, label="training NLL"),
        Line2D(
            [0],
            [0],
            color="#111111",
            lw=1.25,
            linestyle="--",
            marker="o",
            markersize=4,
            label="sparse cached validation",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="o",
            markerfacecolor="white",
            linestyle="None",
            markersize=5,
            label="individual exact full NLL",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="*",
            linestyle="None",
            markersize=11,
            label="final exact full NLL / ensemble NLL",
        ),
        Line2D([0], [0], color="#222222", lw=1.2, linestyle=":", label="NLL = -3.60"),
        Line2D([0], [0], color="#8b0000", lw=1.2, linestyle=":", label="60 seconds"),
    ]
    return recipe_handles + style_handles


def plot(output_path: Path = DEFAULT_OUTPUT) -> Path:
    runs = [load_single_run(spec) for spec in SINGLE_RUNS]
    runs.extend(load_ensemble_run(spec) for spec in ENSEMBLE_RUNS)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), constrained_layout=True)
    panels = (
        (axes[0], "steps", "optimizer steps per member", "Loss curves by optimizer step"),
        (axes[1], "seconds", "training wall seconds", "Loss curves by wall time"),
    )
    for ax, x_key, xlabel, title in panels:
        for run in runs:
            if "row" in run:
                plot_single_run(ax, run, x_key)
            else:
                plot_ensemble_run(ax, run, x_key)
        configure_axis(ax, x_key, xlabel, title)

    fig.legend(
        handles=legend_handles(runs),
        loc="outside lower center",
        ncol=3,
        fontsize=8.6,
        frameon=False,
    )
    fig.suptitle(
        "Broad-prior single-decay NPE loss frontier",
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
