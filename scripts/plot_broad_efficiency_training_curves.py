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
    / "runs/00_shared_assets/readme_scaling/decay_population_npe_training_efficiency_curves.png"
)
POPULATION_ENTROPY_NLL = -3.64122


RUNS = [
    {
        "kind": "single",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/38_spline_lr_schedule_proof/"
        "cosine_4m_seed20260901/results/broad_scaling_summary.json",
        "color": "#374151",
        "label": "Flow4 NSF, 4.096M simulations, 90 epochs",
    },
    {
        "kind": "single",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/58_partial_epoch_budget/"
        "batch1024_hidden80_wd1e4_lr004_e74_max294000_seed20260901/"
        "results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/58_partial_epoch_budget/"
        "batch1024_hidden80_wd1e4_lr004_e74_max294000_seed20260901/"
        "runs/n4096000_seed20260901/results/training_progress.jsonl",
        "color": "#64748b",
        "label": "Flow4 NSF, 4.096M simulations, batch 1024, 74 epochs",
    },
    {
        "kind": "single",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/"
        "train8m_lr004_wd2e4_e27_max212000_seed20260901/results/broad_scaling_summary.json",
        "progress": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/"
        "train8m_lr004_wd2e4_e27_max212000_seed20260901/runs/n8192000_seed20260901/"
        "results/training_progress.jsonl",
        "color": "#6f7378",
        "label": "Flow3 NSF, 8.192M simulations, 27 epochs",
    },
    {
        "kind": "ensemble",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/"
        "residual_512k_e20_lr003_wd2e4_seeds4/results/ensemble4_proof_summary.json",
        "progress_glob": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/"
        "residual_512k_e20_lr003_wd2e4_seeds4/runs/*/results/training_progress.jsonl",
        "color": "#16a34a",
        "label": "4-member residual NSF, 512k/member, 20 epochs",
    },
    {
        "kind": "ensemble",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/"
        "mixed_lr_rawfit_512k_e10_seeds2_6_3_5/results/ensemble4_proof_summary.json",
        "progress_glob": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/"
        "mixed_lr_rawfit_512k_e10_seeds2_6_3_5/runs/*/results/training_progress.jsonl",
        "color": "#d65f3d",
        "label": "4-member residual NSF + fit features, 512k/member, 10 epochs",
    },
    {
        "kind": "ensemble",
        "summary": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/199_nll63_randperm_e15_cosstep_ensemble4_saved/"
        "results/ensemble4_proof_summary.json",
        "progress_glob": ROOT
        / "runs/01_exponential_decay/15_broad_scaling/199_nll63_randperm_e15_cosstep_ensemble4_saved/"
        "rp_seed*/runs/*/results/training_progress.jsonl",
        "color": "#0f766e",
        "label": "4-member Flow2 residual NSF + fit features, 2.048M/member, 15 epochs",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def epoch_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "epoch":
            records.append(record)
    if not records:
        raise ValueError(f"No epoch records found in {path}")
    return records


def progress_series(path: Path) -> dict[str, np.ndarray]:
    records = epoch_records(path)
    return {
        "seconds": np.asarray([record["elapsed_training_seconds"] for record in records], dtype=float),
        "nll": np.asarray([record["train_nll_z_units"] for record in records], dtype=float),
    }


def final_nll(summary: dict) -> float:
    for key in ("full_val_nll_z_units", "ensemble_full_val_nll_z_units", "weighted_full_nll"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    if "rows" in summary:
        return float(summary["rows"][0]["full_val_nll_z_units"])
    raise KeyError("Could not find final NLL in summary.")


def final_seconds(summary: dict) -> float:
    for key in ("training_wall_seconds", "training_seconds"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    if "rows" in summary:
        return float(summary["rows"][0]["training_seconds"])
    raise KeyError("Could not find final wall seconds in summary.")


def first_summary_row(summary: dict) -> dict:
    rows = summary.get("rows")
    if isinstance(rows, list) and rows:
        return rows[0]
    return summary


def summary_history_series(summary: dict) -> dict[str, np.ndarray]:
    row = first_summary_row(summary)
    history = row.get("history")
    if not isinstance(history, dict):
        raise KeyError("Summary has no history block.")
    nll = history.get("train_nll")
    if not isinstance(nll, list) or not nll:
        raise KeyError("Summary history has no train_nll list.")
    nll_array = np.asarray(nll, dtype=float)
    if row.get("final_train_nll_z_units") is not None and row.get("final_train_nll_standardized") is not None:
        nll_array = nll_array + (
            float(row["final_train_nll_z_units"])
            - float(row["final_train_nll_standardized"])
        )
    if "train_seconds" in history and isinstance(history["train_seconds"], list):
        seconds = np.cumsum(np.asarray(history["train_seconds"], dtype=float))
    elif "epoch_seconds" in history and isinstance(history["epoch_seconds"], list):
        seconds = np.cumsum(np.asarray(history["epoch_seconds"], dtype=float))
    else:
        seconds = np.linspace(
            final_seconds(summary) / float(nll_array.size),
            final_seconds(summary),
            nll_array.size,
        )
    if seconds.size != nll_array.size:
        seconds = np.linspace(
            final_seconds(summary) / float(nll_array.size),
            final_seconds(summary),
            nll_array.size,
        )
    return {
        "seconds": seconds,
        "nll": nll_array,
    }


def load_single(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    progress_path = spec.get("progress")
    if isinstance(progress_path, Path) and progress_path.exists():
        series = progress_series(progress_path)
    else:
        series = summary_history_series(summary)
    return {
        **spec,
        "summary_data": summary,
        "series": series,
        "final_nll": final_nll(summary),
        "final_seconds": final_seconds(summary),
    }


def load_ensemble(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    paths = sorted(Path(ROOT).glob(str(spec["progress_glob"].relative_to(ROOT))))
    if not paths:
        raise FileNotFoundError(f"No progress files matched {spec['progress_glob']}")
    member_series = [progress_series(path) for path in paths]
    length = min(len(series["seconds"]) for series in member_series)
    seconds_stack = np.vstack([series["seconds"][:length] for series in member_series])
    nll_stack = np.vstack([series["nll"][:length] for series in member_series])
    seconds = seconds_stack.mean(axis=0)
    wall_seconds = final_seconds(summary)
    if seconds[-1] > 0.0:
        seconds = seconds * (wall_seconds / seconds[-1])
    return {
        **spec,
        "summary_data": summary,
        "series": {
            "seconds": seconds,
            "nll": nll_stack.mean(axis=0),
        },
        "final_nll": final_nll(summary),
        "final_seconds": wall_seconds,
    }


def load_point(spec: dict) -> dict:
    summary = load_json(spec["summary"])
    return {
        **spec,
        "summary_data": summary,
        "final_nll": final_nll(summary),
        "final_seconds": final_seconds(summary),
    }


def load_runs() -> list[dict]:
    loaded = []
    for spec in RUNS:
        if spec["kind"] == "single":
            loaded.append(load_single(spec))
        elif spec["kind"] == "ensemble":
            loaded.append(load_ensemble(spec))
        elif spec["kind"] == "point":
            loaded.append(load_point(spec))
        else:
            raise ValueError(f"Unknown run kind: {spec['kind']}")
    return loaded


def legend_label(run: dict) -> str:
    return f"{run['label']} ({float(run['final_nll']):.4f})"


def plot_curve(ax: plt.Axes, run: dict) -> None:
    color = run["color"]
    series = run["series"]
    ax.plot(
        series["seconds"],
        series["nll"],
        color=color,
        linewidth=2.4,
        alpha=0.95,
    )
    ax.scatter(
        [run["final_seconds"]],
        [run["final_nll"]],
        color=color,
        marker="*",
        s=220,
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )


def plot_point(ax: plt.Axes, run: dict) -> None:
    ax.scatter(
        [run["final_seconds"]],
        [run["final_nll"]],
        color=run["color"],
        marker="D",
        s=95,
        edgecolor="black",
        linewidth=0.55,
        zorder=8,
    )


def configure_axis(ax: plt.Axes) -> None:
    ax.axhline(POPULATION_ENTROPY_NLL, color="#172033", linewidth=1.2, linestyle=":")
    ax.annotate(
        "estimated population entropy floor",
        xy=(8.0, POPULATION_ENTROPY_NLL),
        xytext=(8.0, POPULATION_ENTROPY_NLL - 0.020),
        fontsize=8.8,
        color="#172033",
    )
    ax.set_xscale("log")
    ax.set_xlim(left=4.5, right=4000)
    ax.set_xticks([5, 10, 20, 60, 120, 260, 780, 1560, 3140])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(-3.67, -2.15)
    ax.set_xlabel("training wall seconds")
    ax.set_ylabel("negative log likelihood, z units (lower is better)")
    ax.set_title("Single-decay NPE loss by wall time")


def legend_handles(runs: list[dict]) -> list[Line2D]:
    handles = []
    for run in runs:
        marker = "D" if run["kind"] == "point" else "*"
        handles.append(
            Line2D(
                [0],
                [0],
                color=run["color"],
                marker=marker,
                linestyle="-" if run["kind"] != "point" else "None",
                linewidth=2.5 if run["kind"] != "point" else 0,
                markersize=8,
                label=legend_label(run),
            )
        )
    return handles


def plot(output_path: Path = DEFAULT_OUTPUT) -> Path:
    runs = load_runs()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.6, 7.0), constrained_layout=True)
    for run in runs:
        if run["kind"] == "point":
            plot_point(ax, run)
        else:
            plot_curve(ax, run)
    configure_axis(ax)
    ax.legend(
        handles=legend_handles(runs),
        title="final validation NLL",
        loc="upper right",
        fontsize=12.0,
        title_fontsize=12.5,
        ncol=1,
        frameon=True,
        framealpha=0.86,
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
