from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POPULATION_ENTROPY_NLL = -3.63865
POPULATION_ENTROPY_NLL_UNCERTAINTY = 0.0026
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/204_flow2_width_param_scaling_d512k_dense_combined")


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


def power_with_floor(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -alpha)


def power_no_floor(x: np.ndarray, amplitude: float, alpha: float) -> np.ndarray:
    return amplitude * np.power(x, -alpha)


def fit_power_with_floor(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-4)
    p0 = [float(np.nanmin(y) - 0.1 * y_span), float((np.nanmax(y) - np.nanmin(y)) * x[0] ** 0.5), 0.5]
    bounds = ([-np.inf, 0.0, 0.0], [np.inf, np.inf, 5.0])
    try:
        params, covariance = curve_fit(
            power_with_floor,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=100_000,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}
    y_hat = power_with_floor(x, *params)
    residual = y - y_hat
    total = y - float(np.mean(y))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(total * total))
    asymptote_std = float(np.sqrt(covariance[0, 0])) if covariance.shape == (3, 3) else float("nan")
    alpha_std = float(np.sqrt(covariance[2, 2])) if covariance.shape == (3, 3) else float("nan")
    return {
        "ok": True,
        "asymptote": float(params[0]),
        "asymptote_std_error": asymptote_std,
        "amplitude": float(params[1]),
        "alpha": float(params[2]),
        "alpha_std_error": alpha_std,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def fit_power_no_floor(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return {"ok": False, "error": "need at least three positive points"}
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    p0 = [float(np.exp(intercept)), float(max(-slope, 1e-3))]
    try:
        params, covariance = curve_fit(
            power_no_floor,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0], [np.inf, 5.0]),
            maxfev=100_000,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}
    y_hat = power_no_floor(x, *params)
    residual = y - y_hat
    total = y - float(np.mean(y))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(total * total))
    log_hat = np.log(np.maximum(y_hat, 1e-300))
    log_residual = log_y - log_hat
    log_total = log_y - float(np.mean(log_y))
    log_ss_res = float(np.sum(log_residual * log_residual))
    log_ss_tot = float(np.sum(log_total * log_total))
    alpha_std = float(np.sqrt(covariance[1, 1])) if covariance.shape == (2, 2) else float("nan")
    return {
        "ok": True,
        "amplitude": float(params[0]),
        "alpha": float(params[1]),
        "alpha_std_error": alpha_std,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "r2_log": float(1.0 - log_ss_res / log_ss_tot) if log_ss_tot > 0 else float("nan"),
    }


def capacity_fit_from_free_asymptote(fit: dict[str, object]) -> dict[str, object]:
    if not fit.get("ok"):
        return {"ok": False, "error": "raw-loss fit failed"}
    return {
        "ok": True,
        "source_fit": "full_val_nll_z_units",
        "floor": float(fit["asymptote"]),
        "floor_std_error": float(fit.get("asymptote_std_error", float("nan"))),
        "amplitude": float(fit["amplitude"]),
        "alpha": float(fit["alpha"]),
        "alpha_std_error": float(fit.get("alpha_std_error", float("nan"))),
    }


def read_run_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" in data:
        raise ValueError(f"{path} is an aggregate summary; pass individual broad_scaling_run_summary.json files")
    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return {
        "source_summary_json": str(path),
        "seed": int(data["seed"]),
        "family": str(data["family"]),
        "train_simulations": int(data["train_simulations"]),
        "model_parameters": int(data["model_parameters"]),
        "hidden_dim": int(config.get("hidden_dim", 0)),
        "hidden_layers": int(config.get("hidden_layers", 0)),
        "flow_layers": int(config.get("flow_layers", 0)),
        "spline_bins": int(config.get("spline_bins", 0)),
        "context_features": str(config.get("context_features", data.get("context_features", ""))),
        "flow_residual": bool(config.get("flow_residual", False)),
        "flow_randperm": bool(config.get("flow_randperm", False)),
        "full_val_nll_z_units": float(data["full_val_nll_z_units"]),
        "nll_excess_over_entropy_floor": float(data["full_val_nll_z_units"] - POPULATION_ENTROPY_NLL),
        "training_seconds": float(data.get("training_seconds", float("nan"))),
        "optimizer_steps": int(data.get("optimizer_steps", 0)),
    }


def quantile_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
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


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(int(row["train_simulations"]), int(row["model_parameters"])) for row in rows})
    summary_rows = []
    for train_simulations, model_parameters in keys:
        group = [
            row
            for row in rows
            if int(row["train_simulations"]) == train_simulations
            and int(row["model_parameters"]) == model_parameters
        ]
        first = group[0]
        summary_rows.append({
            "train_simulations": train_simulations,
            "model_parameters": model_parameters,
            "seed_count": len(group),
            "seeds": [int(row["seed"]) for row in group],
            "family": first["family"],
            "architecture_label": (
                f"spline_h{first['hidden_dim']}_l{first['hidden_layers']}_"
                f"t{first['flow_layers']}_b{first['spline_bins']}"
            ),
            "hidden_dim": first["hidden_dim"],
            "hidden_layers": first["hidden_layers"],
            "flow_layers": first["flow_layers"],
            "spline_bins": first["spline_bins"],
            "context_features": first["context_features"],
            "flow_residual": first["flow_residual"],
            "flow_randperm": first["flow_randperm"],
            "full_val_nll_z_units": quantile_summary([float(row["full_val_nll_z_units"]) for row in group]),
            "nll_excess_over_entropy_floor": quantile_summary(
                [float(row["nll_excess_over_entropy_floor"]) for row in group]
            ),
            "training_seconds": quantile_summary([float(row["training_seconds"]) for row in group]),
            "optimizer_steps": quantile_summary([float(row["optimizer_steps"]) for row in group]),
            "source_summary_jsons": [row["source_summary_json"] for row in group],
        })
    return summary_rows


def exponent_label(symbol: str, fit: dict[str, object]) -> str:
    value = float(fit["alpha"])
    error = float(fit.get("alpha_std_error", float("nan")))
    if np.isfinite(error):
        return rf"${symbol}={value:.2f}\pm{error:.2f}$"
    return rf"${symbol}={value:.2f}$"


def plot_scaling(summary: dict[str, object], output_path: Path) -> None:
    rows = list(summary["summary_rows"])
    x = np.asarray([row["model_parameters"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"]["median"] for row in rows], dtype=np.float64)
    nll_q16 = np.asarray([row["full_val_nll_z_units"]["q16"] for row in rows], dtype=np.float64)
    nll_q84 = np.asarray([row["full_val_nll_z_units"]["q84"] for row in rows], dtype=np.float64)
    excess = np.asarray([row["nll_excess_over_entropy_floor"]["median"] for row in rows], dtype=np.float64)
    excess_q16 = np.asarray([row["nll_excess_over_entropy_floor"]["q16"] for row in rows], dtype=np.float64)
    excess_q84 = np.asarray([row["nll_excess_over_entropy_floor"]["q84"] for row in rows], dtype=np.float64)
    entropy_floor = float(summary["population_entropy_nll"])
    entropy_uncertainty = float(summary["population_entropy_nll_uncertainty"])
    fits = summary.get("fits", {})
    train_simulations = int(summary["train_simulations"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, nll, color="#0f766e", marker="o", linewidth=2.3, label=r"median measured $L(N)$")
    ax.fill_between(x, nll_q16, nll_q84, color="#0f766e", alpha=0.14, label="seed q16-q84")
    fit = fits.get("full_val_nll_z_units") if isinstance(fits, dict) else None
    if isinstance(fit, dict) and fit.get("ok"):
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 300)
        ax.plot(
            x_dense,
            power_with_floor(
                x_dense,
                float(fit["asymptote"]),
                float(fit["amplitude"]),
                float(fit["alpha"]),
            ),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label=rf"fit $L_{{free}}+A N^{{-\alpha}}$; {exponent_label(r'\alpha', fit)}",
        )
        ax.axhline(
            float(fit["asymptote"]),
            color="#d97706",
            linestyle=":",
            linewidth=1.3,
            label=rf"free fitted $L_{{free}}={float(fit['asymptote']):.5f}$",
        )
        asymptote_std = float(fit.get("asymptote_std_error", float("nan")))
        if np.isfinite(asymptote_std) and asymptote_std > 0.0:
            ax.axhspan(
                float(fit["asymptote"]) - asymptote_std,
                float(fit["asymptote"]) + asymptote_std,
                color="#d97706",
                alpha=0.12,
                label=rf"free fitted $L_{{free}}\pm 1$ SE",
            )
    ax.axhspan(
        entropy_floor - entropy_uncertainty,
        entropy_floor + entropy_uncertainty,
        color="#b42318",
        alpha=0.12,
        label=r"independent $\hat H \pm s_H$",
    )
    ax.set_xscale("log")
    y_candidates = [
        float(np.nanmin(nll_q16)),
        float(np.nanmax(nll_q84)),
        entropy_floor - entropy_uncertainty,
        entropy_floor + entropy_uncertainty,
    ]
    fit = fits.get("full_val_nll_z_units") if isinstance(fits, dict) else None
    if isinstance(fit, dict) and fit.get("ok"):
        asymptote_std = float(fit.get("asymptote_std_error", float("nan")))
        if np.isfinite(asymptote_std):
            y_candidates.extend([
                float(fit["asymptote"]) - asymptote_std,
                float(fit["asymptote"]) + asymptote_std,
            ])
    y_min = min(y_candidates)
    y_max = max(y_candidates)
    margin = max(0.004, 0.16 * (y_max - y_min))
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("trainable parameters per member N")
    ax.set_ylabel(r"$L(N)$: validation NLL (z units)")
    ax.set_title("Raw loss with free asymptote")
    ax.text(
        0.02,
        0.04,
        rf"Fixed $D={train_simulations:,}$ simulations/member"
        "\n"
        rf"Independent floor: $\hat H={entropy_floor:.5f}\pm {entropy_uncertainty:.3f}$",
        transform=ax.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)

    ax = axes[1]
    fit = fits.get("capacity_excess_from_free_fit") if isinstance(fits, dict) else None
    if isinstance(fit, dict) and fit.get("ok"):
        floor = float(fit["floor"])
        capacity_excess = nll - floor
        capacity_q16 = nll_q16 - floor
        capacity_q84 = nll_q84 - floor
        ax.plot(
            x,
            capacity_excess,
            color="#0f766e",
            marker="o",
            linewidth=2.3,
            label=r"$\Delta_N(N)=L(N)-L_{free}$",
        )
        ax.fill_between(
            x,
            np.maximum(capacity_q16, 1e-5),
            capacity_q84,
            color="#0f766e",
            alpha=0.14,
            label="seed q16-q84",
        )
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 300)
        ax.plot(
            x_dense,
            float(fit["amplitude"]) * np.power(x_dense, -float(fit["alpha"])),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label="same fit residual from left panel",
        )
        floor_std = float(fit.get("floor_std_error", float("nan")))
        y_upper = capacity_q84
        if np.isfinite(floor_std) and floor_std > 0.0:
            lower = capacity_excess - floor_std
            upper = capacity_excess + floor_std
            if np.all(upper > 0.0):
                ax.fill_between(
                    x,
                    np.maximum(lower, 1e-5),
                    upper,
                    color="#d97706",
                    alpha=0.10,
                    label=r"$L_{free}$ uncertainty propagated",
                )
                y_upper = np.maximum(y_upper, upper)
        ax.set_ylim(max(float(np.nanmin(y_upper)) * 0.45, 1e-5), float(np.nanmax(y_upper)) * 1.9)
    else:
        ax.plot(
            x,
            excess,
            color="#0f766e",
            marker="o",
            linewidth=2.3,
            label=r"$\Delta_{\hat H}(N)=L(N)-\hat H$",
        )
        ax.fill_between(x, excess_q16, excess_q84, color="#0f766e", alpha=0.14, label="seed q16-q84")
        upper_excess = nll - (entropy_floor - entropy_uncertainty)
        lower = max(
            min(float(np.nanmin(excess_q16)) * 0.65, float(np.nanmin(upper_excess)) * 0.65),
            1e-4,
        )
        ax.set_ylim(lower, float(np.nanmax(upper_excess)) * 1.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("trainable parameters per member N")
    ax.set_ylabel(r"$\Delta_N(N)$: validation NLL minus $L_{free}$")
    ax.set_title("Capacity excess over fitted fixed-D floor")
    ax.text(
        0.02,
        0.04,
        r"Right panel removes the fitted fixed-D floor"
        "\n"
        r"$L_{free}\approx \hat H+$ finite-data contribution",
        transform=ax.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)
    fig.suptitle("Single-decay Flow2 residual NSF fixed-D parameter scaling", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(row: dict[str, object], prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in row.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten(value, f"{name}."))
        else:
            output[name] = value
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed-D Flow2 width/parameter scaling in README style.")
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

    rows = [read_run_summary(path) for path in args.summary_json]
    rows.sort(key=lambda row: (int(row["model_parameters"]), int(row["seed"]), str(row["source_summary_json"])))
    train_values = sorted({int(row["train_simulations"]) for row in rows})
    if len(train_values) != 1:
        raise ValueError(f"Expected one fixed D value, got {train_values}")
    summary_rows = summarize_rows(rows)
    x = np.asarray([row["model_parameters"] for row in summary_rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"]["median"] for row in summary_rows], dtype=np.float64)
    excess = np.asarray([row["nll_excess_over_entropy_floor"]["median"] for row in summary_rows], dtype=np.float64)
    raw_fit = fit_power_with_floor(x, nll)
    fits = {
        "full_val_nll_z_units": raw_fit,
        "capacity_excess_from_free_fit": capacity_fit_from_free_asymptote(raw_fit),
        "nll_excess_fixed_entropy_no_floor": fit_power_no_floor(x, excess),
    }
    outputs = {
        "rows_csv": str(results_dir / "flow2_width_param_scaling_rows.csv"),
        "summary_csv": str(results_dir / "flow2_width_param_scaling_summary.csv"),
        "summary_json": str(results_dir / "flow2_width_param_scaling_summary.json"),
        "figure": str(figures_dir / "flow2_width_param_scaling_weng_style.png"),
    }
    summary = {
        "description": "Fixed-D Flow2 residual NSF parameter scaling by conditioner width.",
        "train_simulations": train_values[0],
        "population_entropy_nll": POPULATION_ENTROPY_NLL,
        "population_entropy_nll_uncertainty": POPULATION_ENTROPY_NLL_UNCERTAINTY,
        "rows": rows,
        "summary_rows": summary_rows,
        "fits": fits,
        "outputs": outputs,
    }
    write_csv([flatten(row) for row in rows], Path(outputs["rows_csv"]))
    write_csv([flatten(row) for row in summary_rows], Path(outputs["summary_csv"]))
    Path(outputs["summary_json"]).write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    plot_scaling(summary, Path(outputs["figure"]))
    print(f"summary_json: {outputs['summary_json']}")
    print(f"figure: {outputs['figure']}")


if __name__ == "__main__":
    main()
