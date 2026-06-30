from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_decay_samplers import compare_to_reference  # noqa: E402
from evaluate_decay_amortization_panel import initial_z_ranges, max_edge_mass  # noqa: E402
from mcmc_decay_inference import PARAMETER_NAMES  # noqa: E402
from npe_posterior_viewer import (  # noqa: E402
    DEFAULT_BEST_BROAD_MODEL,
    DEFAULT_BEST_BROAD_SPLINE_MODEL,
    DEFAULT_MODEL,
    NPEPosteriorViewer,
    grid_theta_axes_and_widths,
    simulate_x_from_z,
)


OUTPUT_ROOT = Path("runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid")


def normalize_mass(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    peak = float(np.nanmax(values)) if values.size else 0.0
    return values / peak if peak > 0 else values


def reference_marginal(reference: dict[str, object], parameter_index: int) -> tuple[np.ndarray, np.ndarray]:
    grid_size = int(reference["grid_size"])
    axes, _widths = grid_theta_axes_and_widths(reference)
    cube = np.asarray(reference["weights"], dtype=np.float64).reshape((grid_size, grid_size, grid_size))
    reduce_axes = tuple(axis for axis in range(3) if axis != parameter_index)
    mass = cube.sum(axis=reduce_axes)
    return axes[parameter_index], normalize_mass(mass)


def sample_histogram(
    samples: np.ndarray,
    parameter_index: int,
    *,
    low: float,
    high: float,
    bins: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    values = samples[:, parameter_index]
    hist, edges = np.histogram(values, bins=bins, range=(low, high), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, normalize_mass(hist.astype(np.float64))


def theta_step_at_truth(reference: dict[str, object], true_theta: np.ndarray) -> dict[str, float]:
    grid_size = int(reference["grid_size"])
    steps = {}
    for index, name in enumerate(PARAMETER_NAMES):
        low, high = [float(value) for value in reference["z_ranges"][name]]
        z_axis = np.linspace(low, high, grid_size)
        nearest = int(np.argmin(np.abs(z_axis - np.log(true_theta[index]))))
        if nearest == 0:
            step = np.exp(z_axis[1]) - np.exp(z_axis[0])
        elif nearest == grid_size - 1:
            step = np.exp(z_axis[-1]) - np.exp(z_axis[-2])
        else:
            step = 0.5 * (np.exp(z_axis[nearest + 1]) - np.exp(z_axis[nearest - 1]))
        steps[name] = float(abs(step))
    return steps


def compact_reference_summary(reference: dict[str, object], true_theta: np.ndarray) -> dict[str, object]:
    return {
        "grid_size": int(reference["grid_size"]),
        "edge_mass": float(max_edge_mass(reference)),
        "theta_step_at_truth": theta_step_at_truth(reference, true_theta),
        "summary": reference["summary"],
        "z_ranges": reference["z_ranges"],
    }


def build_viewer(args: argparse.Namespace) -> NPEPosteriorViewer:
    return NPEPosteriorViewer(
        args.model,
        None,
        args.best_broad_model,
        args.best_broad_spline_model,
        seed=args.seed,
        device=args.device,
        mcmc_device="cpu",
        mcmc_chains=4,
        mcmc_steps=2_000,
        mcmc_burn_in=500,
        mcmc_proposal_scale=(0.030, 0.030, 0.040),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate diagnostics for the extreme NPE UI failure case.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--best-broad-model", type=Path, default=DEFAULT_BEST_BROAD_MODEL)
    parser.add_argument("--best-broad-spline-model", type=Path, default=DEFAULT_BEST_BROAD_SPLINE_MODEL)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--signal-seed", type=int, default=0)
    parser.add_argument("--posterior-samples", type=int, default=100_000)
    parser.add_argument("--ui-grid-size", type=int, default=60)
    parser.add_argument("--ui-large-grid-size", type=int, default=180)
    parser.add_argument("--focused-grid-size", type=int, default=180)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    figures_dir = args.output_root / "figures"
    results_dir = args.output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    viewer = build_viewer(args)
    true_theta = np.asarray([52.4, 0.306, 0.045], dtype=np.float64)
    z_true = np.log(true_theta)
    signal_rng = np.random.default_rng(args.signal_seed)
    x = simulate_x_from_z(z_true[None, :], viewer.t, signal_rng)[0]
    context = viewer.context_for_signal(x)

    samples: dict[str, np.ndarray] = {}
    labels = {
        "broad_spline_4m": "Broad spline 4.096M",
        "broad_mdn_512k": "Broad MDN 512k",
    }
    colors = {
        "broad_spline_4m": "#c45a2d",
        "broad_mdn_512k": "#6d4aff",
    }
    z_samples = {}
    for model_id in labels:
        z_model, theta_model = viewer.sample_estimator(
            model_id=model_id,
            x=x,
            context=context,
            posterior_samples=args.posterior_samples,
        )
        z_samples[model_id] = z_model
        samples[model_id] = theta_model

    ui_ranges = initial_z_ranges(
        z_samples_by_model={"broad_spline_4m": z_samples["broad_spline_4m"]},
        true_z=z_true,
        padding_fraction=0.45,
        min_padding=0.16,
    )
    focused_ranges = np.column_stack([
        z_true - np.asarray([0.030, 0.030, 1.250], dtype=np.float64),
        z_true + np.asarray([0.030, 0.030, 1.250], dtype=np.float64),
    ])

    ui_grid = viewer.build_grid_comparison(
        x=x,
        z_true=z_true,
        grid_size=args.ui_grid_size,
        z_ranges=ui_ranges,
    )["reference"]
    ui_large_grid = viewer.build_grid_comparison(
        x=x,
        z_true=z_true,
        grid_size=args.ui_large_grid_size,
        z_ranges=ui_ranges,
    )["reference"]
    focused_grid = viewer.build_grid_comparison(
        x=x,
        z_true=z_true,
        grid_size=args.focused_grid_size,
        z_ranges=focused_ranges,
    )["reference"]

    references = {
        f"ui_range_{args.ui_grid_size}": ("UI range 60^3", ui_grid, "#5f6b7a"),
        f"ui_range_{args.ui_large_grid_size}": ("UI range 180^3", ui_large_grid, "#0f766e"),
        f"focused_{args.focused_grid_size}": ("Focused 180^3", focused_grid, "#111827"),
    }
    metrics = {
        model_id: {
            key: compare_to_reference(theta_samples, reference)
            for key, (_label, reference, _color) in references.items()
        }
        for model_id, theta_samples in samples.items()
    }

    xlims = {
        "A": (50.5, 54.2),
        "k": (0.294, 0.326),
        "sigma": (0.0, 0.42),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), constrained_layout=True)
    for index, name in enumerate(PARAMETER_NAMES):
        ax = axes[0, index]
        low, high = xlims[name]
        for ref_label, reference, color in references.values():
            x_ref, y_ref = reference_marginal(reference, index)
            mask = (x_ref >= low) & (x_ref <= high)
            ax.step(x_ref[mask], y_ref[mask], where="mid", lw=2.0, color=color, label=ref_label)
        for model_id, theta_samples in samples.items():
            centers, hist = sample_histogram(theta_samples, index, low=low, high=high)
            ax.plot(centers, hist, lw=1.8, alpha=0.85, color=colors[model_id], label=labels[model_id])
        ax.axvline(true_theta[index], color="#111827", linestyle="--", lw=1.4, label="true" if index == 0 else None)
        ax.set_title(f"{name} marginal, max-normalized mass")
        ax.set_xlim(low, high)
        ax.set_ylim(-0.02, 1.08)
        ax.grid(alpha=0.18)
        if index == 0:
            ax.set_ylabel("relative mass")

    ref_keys = list(references)
    x_positions = np.arange(len(ref_keys), dtype=np.float64)
    width = 0.34
    for col, model_id in enumerate(labels):
        ax = axes[1, col]
        values = [
            metrics[model_id][key]["mean_normalized_wasserstein"]["value"]
            for key in ref_keys
        ]
        ax.bar(x_positions, values, width=0.62, color=colors[model_id], alpha=0.86)
        ax.set_yscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([references[key][0] for key in ref_keys], rotation=18, ha="right")
        ax.set_title(labels[model_id])
        ax.set_ylabel("mean normalized W")
        ax.grid(axis="y", which="both", alpha=0.2)
        for x_pos, value in zip(x_positions, values):
            ax.text(x_pos, value * 1.15, f"{value:.2g}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "Reference-grid resolution",
        "",
        f"true theta: A={true_theta[0]:.3g}, k={true_theta[1]:.3g}, sigma={true_theta[2]:.3g}",
        "",
    ]
    for ref_label, reference, _color in references.values():
        steps = theta_step_at_truth(reference, true_theta)
        summary = reference["summary"]
        lines.extend([
            ref_label,
            f"  edge mass: {max_edge_mass(reference):.1e}",
            f"  step @ truth: A {steps['A']:.3g}, k {steps['k']:.3g}, sigma {steps['sigma']:.3g}",
            f"  medians: A {summary['A']['median']:.4g}, k {summary['k']['median']:.4g}, sigma {summary['sigma']['median']:.4g}",
            "",
        ])
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)

    handles, handle_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, handle_labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.08))
    output_path = figures_dir / "controlled_failure_resolution_v2.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result = {
        "case": "controlled_low_noise_high_A_prior_predictive_like_ui_failure",
        "true_theta": dict(zip(PARAMETER_NAMES, [float(value) for value in true_theta])),
        "posterior_samples": int(args.posterior_samples),
        "references": {
            key: compact_reference_summary(reference, true_theta)
            for key, (_label, reference, _color) in references.items()
        },
        "metrics": metrics,
    }
    result_path = results_dir / "controlled_failure_diagnostics_v2.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(output_path), "results": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
