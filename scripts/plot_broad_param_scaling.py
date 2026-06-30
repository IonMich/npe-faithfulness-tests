from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis")


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


def quantile_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q16": float(np.quantile(finite, 0.16)),
        "median": float(np.median(finite)),
        "q84": float(np.quantile(finite, 0.84)),
        "max": float(np.max(finite)),
    }


def flatten(row: dict[str, object], prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in row.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten(value, f"{name}."))
        else:
            output[name] = value
    return output


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(summary_paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for summary_path in summary_paths:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in data["rows"]:
            item = dict(row)
            item["source_summary_json"] = str(summary_path)
            config = item.get("config", {})
            item["family"] = str(item.get("family", "unknown"))
            if isinstance(config, dict):
                item["hidden_dim"] = int(config.get("hidden_dim", 0))
                item["hidden_layers"] = int(config.get("hidden_layers", 0))
                item["mdn_components"] = int(config.get("mdn_components", 0))
                item["flow_layers"] = int(config.get("flow_layers", 0))
                item["flow_context_dim"] = int(config.get("flow_context_dim", 0))
                item["spline_bins"] = int(config.get("spline_bins", 0))
                if item["family"] == "affine_flow":
                    item["architecture_label"] = (
                        f"flow_h{item['hidden_dim']}_l{item['hidden_layers']}_"
                        f"t{item['flow_layers']}_ctx{item['flow_context_dim']}"
                    )
                elif item["family"] == "spline_flow":
                    item["architecture_label"] = (
                        f"spline_h{item['hidden_dim']}_l{item['hidden_layers']}_"
                        f"t{item['flow_layers']}_b{item['spline_bins']}"
                    )
                elif item["family"] == "mdn":
                    item["architecture_label"] = (
                        f"mdn_h{item['hidden_dim']}_l{item['hidden_layers']}_c{item['mdn_components']}"
                    )
                else:
                    item["architecture_label"] = f"{item['family']}_h{item['hidden_dim']}_l{item['hidden_layers']}"
            else:
                item["architecture_label"] = summary_path.parent.parent.name
            item["model_parameters"] = int(item["model_parameters"])
            item["train_simulations"] = int(item["train_simulations"])
            item["seed"] = int(item["seed"])
            rows.append(item)
    rows.sort(key=lambda row: (int(row["train_simulations"]), int(row["model_parameters"]), int(row["seed"])))
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(int(row["train_simulations"]), int(row["model_parameters"])) for row in rows})
    for train_simulations, model_parameters in keys:
        group = [
            row
            for row in rows
            if int(row["train_simulations"]) == train_simulations
            and int(row["model_parameters"]) == model_parameters
        ]
        first = group[0]
        item = {
            "train_simulations": train_simulations,
            "model_parameters": model_parameters,
            "seed_count": len(group),
            "family": first.get("family"),
            "architecture_label": first.get("architecture_label"),
            "hidden_dim": first.get("hidden_dim"),
            "hidden_layers": first.get("hidden_layers"),
            "mdn_components": first.get("mdn_components"),
            "flow_layers": first.get("flow_layers"),
            "flow_context_dim": first.get("flow_context_dim"),
            "spline_bins": first.get("spline_bins"),
            "panel_marginal_wasserstein_mean": quantile_summary(
                np.asarray([row["panel_marginal_wasserstein_mean"] for row in group], dtype=np.float64)
            ),
            "panel_marginal_target_ratio_mean": quantile_summary(
                np.asarray([row["panel_marginal_target_ratio_mean"] for row in group], dtype=np.float64)
            ),
            "x0_grid300_wasserstein": quantile_summary(
                np.asarray([row["x0_grid300_wasserstein"] for row in group], dtype=np.float64)
            ),
            "x0_grid300_target_ratio": quantile_summary(
                np.asarray([row["x0_grid300_target_ratio"] for row in group], dtype=np.float64)
            ),
            "full_val_nll_z_units": quantile_summary(
                np.asarray([row["full_val_nll_z_units"] for row in group], dtype=np.float64)
            ),
            "best_val_nll_z_units": quantile_summary(
                np.asarray([row["best_val_nll_z_units"] for row in group], dtype=np.float64)
            ),
            "training_seconds": quantile_summary(
                np.asarray([row["training_seconds"] for row in group], dtype=np.float64)
            ),
        }
        output.append(item)
    return output


def plot(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    panels = [
        ("panel_marginal_wasserstein_mean", "panel mean marginal W", axes[0, 0], "#2f6fbb", True),
        ("panel_marginal_target_ratio_mean", "panel mean target ratio", axes[0, 1], "#b85c38", True),
        ("full_val_nll_z_units", "validation NLL in z units", axes[1, 0], "#2f855a", False),
        ("training_seconds", "training seconds", axes[1, 1], "#5f4bb6", True),
    ]
    train_values = sorted({int(row["train_simulations"]) for row in summary_rows})
    palette = ["#2f6fbb", "#b85c38", "#2f855a", "#5f4bb6", "#6b7280"]
    colors = {train_simulations: palette[index % len(palette)] for index, train_simulations in enumerate(train_values)}
    for metric, ylabel, ax, fallback_color, log_y in panels:
        for train_simulations in train_values:
            group = [row for row in summary_rows if int(row["train_simulations"]) == train_simulations]
            group.sort(key=lambda row: int(row["model_parameters"]))
            x = np.asarray([row["model_parameters"] for row in group], dtype=np.float64)
            y = np.asarray([row[metric]["median"] for row in group], dtype=np.float64)
            q16 = np.asarray([row[metric]["q16"] for row in group], dtype=np.float64)
            q84 = np.asarray([row[metric]["q84"] for row in group], dtype=np.float64)
            color = colors.get(train_simulations, fallback_color)
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=f"D={train_simulations:,}")
            ax.fill_between(x, q16, q84, color=color, alpha=0.14)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("trainable NPE parameters")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 1].axhline(1.0, color="#172033", linestyle=":", linewidth=1.2)
    figure.suptitle("Broad NPE parameter-axis scaling", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate broad MDN parameter-axis scaling runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("summary_json", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.summary_json)
    summary_rows = summarize(rows)
    rows_csv = results_dir / "broad_param_scaling_rows.csv"
    summary_csv = results_dir / "broad_param_scaling_summary.csv"
    summary_json = results_dir / "broad_param_scaling_summary.json"
    figure_path = figures_dir / "broad_param_scaling.png"

    write_csv([flatten(row) for row in rows], rows_csv)
    write_csv([flatten(row) for row in summary_rows], summary_csv)
    plot(summary_rows, figure_path)

    output = {
        "sources": [str(path) for path in args.summary_json],
        "rows": rows,
        "summary_rows": summary_rows,
        "outputs": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "figure": str(figure_path),
        },
    }
    summary_json.write_text(json.dumps(json_ready(output), indent=2), encoding="utf-8")
    print(f"summary_json: {summary_json}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
