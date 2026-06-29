from __future__ import annotations

import argparse
from pathlib import Path

import artifact_paths as ap

import corner
import matplotlib
import numpy as np

from compare_decay_samplers import load_samples, subsample
from corner_truth import overplot_true_values, true_theta_legend_handle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay MCMC, HMC, and NPE posterior corner plots.")
    parser.add_argument("--mcmc-samples", type=Path, default=ap.MCMC_DECAY_SAMPLES)
    parser.add_argument("--hmc-samples", type=Path, default=ap.HMC_DECAY_SAMPLES)
    parser.add_argument(
        "--npe-samples",
        type=Path,
        default=ap.NPE_FLOW_TARGET_PASS_RESULTS / "npe_flow_decay_samples.npz",
    )
    parser.add_argument("--max-samples", type=int, default=35_000)
    parser.add_argument("--output", type=Path, default=ap.MCMC_HMC_NPE_CORNER_OVERLAY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mcmc = load_samples(args.mcmc_samples, "MCMC")
    hmc = load_samples(args.hmc_samples, "HMC")
    npe_data = np.load(args.npe_samples, allow_pickle=True)
    npe_theta = np.asarray(npe_data["theta_samples"])
    true_theta = np.asarray(npe_data["true_theta"])

    labels = [r"$A$", r"$k$", r"$\sigma$"]
    figure = corner.corner(
        subsample(mcmc["posterior_theta"], args.max_samples, seed=101),
        labels=labels,
        color="#2f6fbb",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.7},
        contour_kwargs={"linewidths": 1.5},
    )
    corner.corner(
        subsample(hmc["posterior_theta"], args.max_samples, seed=202),
        fig=figure,
        labels=labels,
        color="#b85c38",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.7},
        contour_kwargs={"linewidths": 1.5},
    )
    corner.corner(
        subsample(npe_theta, args.max_samples, seed=303),
        fig=figure,
        labels=labels,
        color="#3f8f5f",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.90),
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.7},
        contour_kwargs={"linewidths": 1.5},
    )

    handles = [
        plt.Line2D([0], [0], color="#2f6fbb", lw=2, label="Random-walk MCMC"),
        plt.Line2D([0], [0], color="#b85c38", lw=2, label="HMC"),
        plt.Line2D([0], [0], color="#3f8f5f", lw=2, label="Spline-flow NPE"),
        true_theta_legend_handle(),
    ]
    overplot_true_values(figure, true_theta)
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    figure.subplots_adjust(top=0.90)
    figure.suptitle("Posterior overlay: MCMC vs HMC vs spline-flow NPE", y=0.985, fontsize=15)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"corner_overlay: {args.output}")


if __name__ == "__main__":
    main()
