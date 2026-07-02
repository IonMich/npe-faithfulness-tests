from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

import evaluate_npe_ensemble_nll as ensemble_nll

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POPULATION_ENTROPY_NLL = -3.63865
POPULATION_ENTROPY_NLL_UNCERTAINTY = 0.0026
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/205_flow2_width_param_scaling_d8m_ensemble_combined")
DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz"
)


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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    try:
        params, covariance = curve_fit(
            power_with_floor,
            x,
            y,
            p0=p0,
            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, 5.0]),
            maxfev=100_000,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}
    y_hat = power_with_floor(x, *params)
    residual = y - y_hat
    total = y - float(np.mean(y))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(total * total))
    return {
        "ok": True,
        "asymptote": float(params[0]),
        "asymptote_std_error": float(np.sqrt(covariance[0, 0])) if covariance.shape == (3, 3) else float("nan"),
        "amplitude": float(params[1]),
        "alpha": float(params[2]),
        "alpha_std_error": float(np.sqrt(covariance[2, 2])) if covariance.shape == (3, 3) else float("nan"),
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
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    try:
        params, covariance = curve_fit(
            power_no_floor,
            x,
            y,
            p0=[float(np.exp(intercept)), float(max(-slope, 1e-3))],
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
    log_residual = np.log(y) - np.log(np.maximum(y_hat, 1e-300))
    log_total = np.log(y) - float(np.mean(np.log(y)))
    log_ss_res = float(np.sum(log_residual * log_residual))
    log_ss_tot = float(np.sum(log_total * log_total))
    return {
        "ok": True,
        "amplitude": float(params[0]),
        "alpha": float(params[1]),
        "alpha_std_error": float(np.sqrt(covariance[1, 1])) if covariance.shape == (2, 2) else float("nan"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_width_groups(roots: list[Path], train_simulations: int, ensemble_size: int) -> list[dict[str, object]]:
    groups = []
    for root in roots:
        for summary_path in sorted(root.glob(f"h*/runs/n{train_simulations}_seed*/results/broad_scaling_run_summary.json")):
            row = read_json(summary_path)
            model_path = row.get("model_pt")
            if not model_path:
                continue
            resolved = Path(str(model_path))
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            if not resolved.exists():
                continue
            config = row.get("config", {})
            if not isinstance(config, dict):
                config = {}
            groups.append({
                "summary_path": str(summary_path),
                "model_path": str(resolved),
                "seed": int(row["seed"]),
                "train_simulations": int(row["train_simulations"]),
                "model_parameters": int(row["model_parameters"]),
                "hidden_dim": int(config.get("hidden_dim", 0)),
                "hidden_layers": int(config.get("hidden_layers", 0)),
                "flow_layers": int(config.get("flow_layers", 0)),
                "spline_bins": int(config.get("spline_bins", 0)),
                "context_features": str(config.get("context_features", "")),
                "training_seconds": float(row.get("training_seconds", float("nan"))),
            })
    by_width: dict[int, list[dict[str, object]]] = {}
    for item in groups:
        by_width.setdefault(int(item["model_parameters"]), []).append(item)
    output = []
    for model_parameters, items in sorted(by_width.items()):
        items = sorted(items, key=lambda item: int(item["seed"]))
        if len(items) < ensemble_size:
            print(
                f"skipping model_parameters={model_parameters}: "
                f"found {len(items)} members, need {ensemble_size}",
                file=sys.stderr,
            )
            continue
        output.append({
            "model_parameters": model_parameters,
            "members": items[:ensemble_size],
        })
    return output


def evaluate_or_load_group(
    *,
    group: dict[str, object],
    results_dir: Path,
    validation_cache: Path,
    batch_size: int,
    max_examples: int,
    device: str,
    force: bool,
) -> dict[str, object]:
    members = list(group["members"])
    first = members[0]
    model_parameters = int(group["model_parameters"])
    hidden_dim = int(first["hidden_dim"])
    nll_path = results_dir / f"flow2_width_ensemble_nll_h{hidden_dim}_p{model_parameters}.json"
    if nll_path.exists() and not force:
        nll = read_json(nll_path)
    else:
        nll = ensemble_nll.evaluate_ensemble(
            model_paths=[Path(str(member["model_path"])) for member in members],
            validation_cache=validation_cache,
            device=ensemble_nll.torch.device(device),
            batch_size=batch_size,
            max_examples=max_examples,
        )
        nll_path.write_text(json.dumps(json_ready(nll), indent=2, sort_keys=True), encoding="utf-8")
    member_seconds = np.asarray([float(member["training_seconds"]) for member in members], dtype=np.float64)
    return {
        "train_simulations_per_member": int(first["train_simulations"]),
        "model_parameters_per_member": model_parameters,
        "ensemble_size": len(members),
        "seeds": [int(member["seed"]) for member in members],
        "family": "spline_flow",
        "architecture_label": (
            f"spline_h{hidden_dim}_l{first['hidden_layers']}_t{first['flow_layers']}_b{first['spline_bins']}"
        ),
        "hidden_dim": hidden_dim,
        "hidden_layers": int(first["hidden_layers"]),
        "flow_layers": int(first["flow_layers"]),
        "spline_bins": int(first["spline_bins"]),
        "context_features": str(first["context_features"]),
        "full_val_nll_z_units": float(nll["ensemble_full_val_nll_z_units"]),
        "nll_excess_over_entropy_floor": float(nll["ensemble_full_val_nll_z_units"] - POPULATION_ENTROPY_NLL),
        "best_individual_full_val_nll_z_units": float(nll["best_individual_full_val_nll_z_units"]),
        "ensemble_gain_vs_best_individual": float(nll["ensemble_gain_vs_best_individual"]),
        "individual_full_val_nll_z_units": list(nll["individual_full_val_nll_z_units"]),
        "member_training_seconds_sum": float(np.nansum(member_seconds)),
        "member_training_seconds_max": float(np.nanmax(member_seconds)),
        "member_summary_paths": [str(member["summary_path"]) for member in members],
        "model_paths": [str(member["model_path"]) for member in members],
        "nll_json": str(nll_path),
    }


def exponent_label(symbol: str, fit: dict[str, object]) -> str:
    value = float(fit["alpha"])
    error = float(fit.get("alpha_std_error", float("nan")))
    if np.isfinite(error):
        return rf"${symbol}={value:.2f}\pm{error:.2f}$"
    return rf"${symbol}={value:.2f}$"


def plot_scaling(summary: dict[str, object], output_path: Path) -> None:
    rows = list(summary["rows"])
    x = np.asarray([row["model_parameters_per_member"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"] for row in rows], dtype=np.float64)
    excess = np.asarray([row["nll_excess_over_entropy_floor"] for row in rows], dtype=np.float64)
    entropy_floor = float(summary["population_entropy_nll"])
    entropy_uncertainty = float(summary["population_entropy_nll_uncertainty"])
    fits = summary.get("fits", {})
    train_simulations = int(summary["train_simulations_per_member"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, nll, color="#0f766e", marker="o", linewidth=2.3, label=r"measured ensemble $L(N)$")
    fit = fits.get("full_val_nll_z_units") if isinstance(fits, dict) else None
    if isinstance(fit, dict) and fit.get("ok"):
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 300)
        ax.plot(
            x_dense,
            power_with_floor(x_dense, float(fit["asymptote"]), float(fit["amplitude"]), float(fit["alpha"])),
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
    y_candidates = [float(np.nanmin(nll)), float(np.nanmax(nll)), entropy_floor - entropy_uncertainty, entropy_floor + entropy_uncertainty]
    fit = fits.get("full_val_nll_z_units") if isinstance(fits, dict) else None
    if isinstance(fit, dict) and fit.get("ok"):
        asymptote_std = float(fit.get("asymptote_std_error", float("nan")))
        if np.isfinite(asymptote_std):
            y_candidates.extend([float(fit["asymptote"]) - asymptote_std, float(fit["asymptote"]) + asymptote_std])
    y_min = min(y_candidates)
    y_max = max(y_candidates)
    margin = max(0.004, 0.16 * (y_max - y_min))
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("trainable parameters per member N")
    ax.set_ylabel(r"$L(N)$: ensemble validation NLL (z units)")
    ax.set_title("Raw loss with free asymptote")
    ax.text(
        0.02,
        0.04,
        rf"Fixed $D={train_simulations:,}$ simulations/member"
        "\n"
        rf"Ensemble size {int(summary['ensemble_size'])}",
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
        ax.plot(
            x,
            capacity_excess,
            color="#0f766e",
            marker="o",
            linewidth=2.3,
            label=r"$\Delta_N(N)=L(N)-L_{free}$",
        )
        x_dense = np.geomspace(float(np.min(x)), float(np.max(x)), 300)
        ax.plot(
            x_dense,
            power_no_floor(x_dense, float(fit["amplitude"]), float(fit["alpha"])),
            color="#172033",
            linestyle="--",
            linewidth=1.3,
            label="same fit residual from left panel",
        )
        floor_std = float(fit.get("floor_std_error", float("nan")))
        y_upper = capacity_excess
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
                y_upper = upper
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
        upper_excess = nll - (entropy_floor - entropy_uncertainty)
        ax.set_ylim(max(float(np.nanmin(upper_excess)) * 0.55, 1e-4), float(np.nanmax(upper_excess)) * 1.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("trainable parameters per member N")
    ax.set_ylabel(r"$\Delta_N(N)$: ensemble validation NLL minus $L_{free}$")
    ax.set_title("Capacity excess over fitted fixed-D floor")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(which="both", alpha=0.24)
    fig.suptitle("Single-decay Flow2 residual NSF ensemble fixed-D parameter scaling", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and plot fixed-D Flow2 ensemble parameter scaling.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-root", type=Path, action="append", required=True)
    parser.add_argument("--train-simulations", type=int, required=True)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation_cache = args.validation_cache.resolve()
    groups = collect_width_groups(
        roots=[path.resolve() for path in args.train_root],
        train_simulations=int(args.train_simulations),
        ensemble_size=int(args.ensemble_size),
    )
    if not groups:
        raise SystemExit("No complete width groups found.")
    rows = [
        evaluate_or_load_group(
            group=group,
            results_dir=results_dir,
            validation_cache=validation_cache,
            batch_size=int(args.batch_size),
            max_examples=int(args.max_examples),
            device=str(args.device),
            force=bool(args.force_eval),
        )
        for group in groups
    ]
    rows.sort(key=lambda row: int(row["model_parameters_per_member"]))
    x = np.asarray([row["model_parameters_per_member"] for row in rows], dtype=np.float64)
    nll = np.asarray([row["full_val_nll_z_units"] for row in rows], dtype=np.float64)
    excess = np.asarray([row["nll_excess_over_entropy_floor"] for row in rows], dtype=np.float64)
    raw_fit = fit_power_with_floor(x, nll)
    fits = {
        "full_val_nll_z_units": raw_fit,
        "capacity_excess_from_free_fit": capacity_fit_from_free_asymptote(raw_fit),
        "nll_excess_fixed_entropy_no_floor": fit_power_no_floor(x, excess),
    }
    outputs = {
        "rows_csv": str(results_dir / "flow2_ensemble_width_param_scaling_rows.csv"),
        "summary_json": str(results_dir / "flow2_ensemble_width_param_scaling_summary.json"),
        "figure": str(figures_dir / "flow2_ensemble_width_param_scaling_weng_style.png"),
    }
    summary = {
        "description": "Fixed-D 4-member Flow2 residual NSF ensemble parameter scaling by conditioner width.",
        "train_roots": [str(path) for path in args.train_root],
        "train_simulations_per_member": int(args.train_simulations),
        "ensemble_size": int(args.ensemble_size),
        "validation_cache": str(validation_cache),
        "population_entropy_nll": POPULATION_ENTROPY_NLL,
        "population_entropy_nll_uncertainty": POPULATION_ENTROPY_NLL_UNCERTAINTY,
        "rows": rows,
        "fits": fits,
        "outputs": outputs,
    }
    write_csv([flatten(row) for row in rows], Path(outputs["rows_csv"]))
    Path(outputs["summary_json"]).write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    plot_scaling(summary, Path(outputs["figure"]))
    print(f"summary_json: {outputs['summary_json']}")
    print(f"figure: {outputs['figure']}")


if __name__ == "__main__":
    main()
