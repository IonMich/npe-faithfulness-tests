from __future__ import annotations

import argparse
import base64
import errno
import io
import json
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import matplotlib
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from abc_faithfulness_decay import make_k_grid
from compare_decay_samplers import compare_to_reference, summarize_samples, weighted_quantile
from corner_truth import true_theta_legend_handle
from evaluate_decay_amortization_panel import (
    build_adaptive_grid_reference,
    initial_z_ranges,
    max_edge_mass,
)
from mcmc_decay_inference import (
    MCMCConfig,
    PARAMETER_NAMES,
    PRIOR_LOG_MEAN,
    PRIOR_LOG_STD,
    arviz_diagnostics,
    choose_device as choose_mcmc_device,
    convergence_flags,
    run_random_walk_metropolis,
    simulate_decay_data,
)
from npe_flow_decay import (
    ConditionalSplineFlow,
    context_distances,
    make_context_summaries,
    sample_flow_posterior,
    sample_prior_z,
)
from npe_stage1_decay import (
    FAMILY_LABELS,
    Stage1Config,
    make_model,
    posterior_predictive_band,
    sample_posterior_for_observation,
    transform_context_features,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL = Path(
    "runs/01_exponential_decay/03_npe_flow_search/"
    "11_npe_flow_local_q0005_linear_150k_t8_seed20260706/"
    "results/npe_flow_decay_model.pt"
)
DEFAULT_BROAD_MODEL: Path | None = None
DEFAULT_BEST_BROAD_MODEL: Path | None = None
DEFAULT_BEST_BROAD_SPLINE_MODEL = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "34_mini_fixed_p_4m_diagnostic/spline/"
    "runs/n4096000_seed20260901/results/spline_flow_model.pt"
)
DEFAULT_BEST_BROAD_EFFICIENCY_MODEL = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "74_ui_best_8m_checkpoint/train8m_lr004_wd2e4_e27_max212000_seed20260901/"
    "runs/n8192000_seed20260901/results/spline_flow_model.pt"
)
DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "199_nll63_randperm_e15_cosstep_ensemble4_saved/"
    "results/ensemble4_proof_summary.json"
)
DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY = Path(
    "runs/01_exponential_decay/15_broad_scaling/"
    "187_nll63_weighted_broad_pool/results/weighted_ensemble_summary.json"
)
DEFAULT_UI_DIST = Path("viewer-ui/dist")
DEFAULT_PORT = 8876
DEFAULT_NPE_EVAL_GRID_SIZE = 60
MAX_NPE_EVAL_GRID_SIZE = 180
MAX_POSTERIOR_SAMPLES = 500_000

GRID_COLOR = "#172033"
NPE_COLOR = "#2f6fbb"
BROAD_NPE_COLOR = "#276749"
MCMC_COLOR = "#b85c38"
NPE_LAYER_COLORS = {
    "local_flow": NPE_COLOR,
    "broad_mdn": BROAD_NPE_COLOR,
    "broad_spline_4m": "#c45a2d",
    "broad_spline_8m": "#6d4aff",
    "broad_fresh_e15_ensemble4": "#0f766e",
    "broad_weighted_checkpoint_pool": "#7c3aed",
}
HIDDEN_UI_MODEL_IDS = {"local_flow", "broad_spline_4m"}
LOW_PRIOR_SIGNAL_SPECS = {
    "low_prior_very_low": {
        "label": "very_low",
        "standardized_offset": np.asarray([2.5, -2.5, 2.5], dtype=np.float64),
        "noise_seed": 2026070201,
    },
    "low_prior_extreme": {
        "label": "extremely_low",
        "standardized_offset": np.asarray([4.0, -4.0, 3.5], dtype=np.float64),
        "noise_seed": 2026070202,
    },
}


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


def simulate_x_from_z(z: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.exp(z)
    mean = theta[:, 0:1] * np.exp(-theta[:, 1:2] * t[None, :])
    return mean + rng.normal(0.0, theta[:, 2:3], size=mean.shape)


def figure_to_data_uri(figure: plt.Figure, *, dpi: int = 150) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sample_summary_rows(samples: np.ndarray) -> list[dict[str, str]]:
    summary = summarize_samples(samples)
    rows = []
    for name in PARAMETER_NAMES:
        item = summary[name]
        rows.append(
            {
                "parameter": name,
                "median": f"{item['median']:.4g}",
                "q16": f"{item['q16']:.4g}",
                "q84": f"{item['q84']:.4g}",
                "q05": f"{item['q05']:.4g}",
                "q95": f"{item['q95']:.4g}",
            }
        )
    return rows


def grid_summary_rows(reference: dict[str, object]) -> list[dict[str, str]]:
    rows = []
    summary = reference["summary"]
    for name in PARAMETER_NAMES:
        item = summary[name]
        rows.append(
            {
                "parameter": name,
                "median": f"{item['median']:.4g}",
                "q16": f"{item['q16']:.4g}",
                "q84": f"{item['q84']:.4g}",
                "q05": f"{item['q05']:.4g}",
                "q95": f"{item['q95']:.4g}",
            }
        )
    return rows


def summarize_weighted_reference(theta_grid: np.ndarray, weights: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        values = theta_grid[:, index]
        q05, q16, q50, q84, q95 = weighted_quantile(values, weights, [0.05, 0.16, 0.50, 0.84, 0.95])
        mean = float(np.sum(values * weights))
        variance = float(np.sum((values - mean) ** 2 * weights))
        summary[name] = {
            "mean": mean,
            "sd": float(np.sqrt(max(variance, 0.0))),
            "q05": float(q05),
            "q16": float(q16),
            "median": float(q50),
            "q84": float(q84),
            "q95": float(q95),
        }
    return summary


def reference_from_log_density_grid(
    *,
    z_grid: np.ndarray,
    log_density: np.ndarray,
    z_ranges: np.ndarray,
    grid_size: int,
) -> dict[str, object]:
    log_weights = log_density - logsumexp(log_density)
    weights = np.exp(log_weights)
    theta_grid = np.exp(z_grid)
    weight_cube = weights.reshape(grid_size, grid_size, grid_size)
    edge_mass = {}
    for index, name in enumerate(PARAMETER_NAMES):
        edge_mass[name] = {
            "lower": float(np.take(weight_cube, indices=0, axis=index).sum()),
            "upper": float(np.take(weight_cube, indices=grid_size - 1, axis=index).sum()),
        }
    return {
        "grid_size": int(grid_size),
        "grid_points": int(z_grid.shape[0]),
        "z_ranges": {
            name: [float(z_ranges[index, 0]), float(z_ranges[index, 1])]
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "edge_mass": edge_mass,
        "theta_grid": theta_grid,
        "weights": weights,
        "summary": summarize_weighted_reference(theta_grid, weights),
    }


def z_grid_from_ranges(z_ranges: np.ndarray, grid_size: int) -> np.ndarray:
    axes = [np.linspace(low, high, grid_size) for low, high in z_ranges]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.reshape(-1) for axis in mesh])


def default_npe_grid_ranges(true_z: np.ndarray) -> np.ndarray:
    half_width = np.full(3, 1.10, dtype=np.float64)
    return np.column_stack([true_z - half_width, true_z + half_width])


def range_cache_key(z_ranges: np.ndarray) -> tuple[float, ...]:
    return tuple(float(np.round(value, 8)) for value in z_ranges.reshape(-1))


def compare_weighted_to_reference(
    estimate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, dict[str, float]]:
    estimate_values = np.asarray(estimate["theta_grid"], dtype=np.float64)
    estimate_weights = np.asarray(estimate["weights"], dtype=np.float64)
    reference_values = np.asarray(reference["theta_grid"], dtype=np.float64)
    reference_weights = np.asarray(reference["weights"], dtype=np.float64)
    estimate_summary = estimate["summary"]
    reference_summary = reference["summary"]
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        ref_sd = max(float(reference_summary[name]["sd"]), 1e-12)
        wasserstein = wasserstein_distance(
            estimate_values[:, index],
            reference_values[:, index],
            u_weights=estimate_weights,
            v_weights=reference_weights,
        )
        metrics[name] = {
            "wasserstein_to_grid": float(wasserstein),
            "wasserstein_to_grid_in_ref_sd": float(wasserstein / ref_sd),
            "median_error": float(estimate_summary[name]["median"] - reference_summary[name]["median"]),
            "q05_error": float(estimate_summary[name]["q05"] - reference_summary[name]["q05"]),
            "q95_error": float(estimate_summary[name]["q95"] - reference_summary[name]["q95"]),
        }
    metrics["mean_normalized_wasserstein"] = {
        "value": float(np.mean([
            metrics[name]["wasserstein_to_grid_in_ref_sd"]
            for name in PARAMETER_NAMES
        ]))
    }
    return metrics


def grid_theta_axes_and_widths(reference: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    grid_size = int(reference["grid_size"])
    axes = []
    widths = []
    for name in PARAMETER_NAMES:
        low, high = [float(value) for value in reference["z_ranges"][name]]
        z_axis = np.linspace(low, high, grid_size)
        if grid_size > 1:
            step = float(z_axis[1] - z_axis[0])
        else:
            step = 1.0
        z_edges = np.concatenate([
            [z_axis[0] - 0.5 * step],
            0.5 * (z_axis[:-1] + z_axis[1:]),
            [z_axis[-1] + 0.5 * step],
        ])
        theta_axis = np.exp(z_axis)
        theta_edges = np.exp(z_edges)
        axes.append(theta_axis)
        widths.append(np.maximum(np.diff(theta_edges), 1e-12))
    return axes, widths


def contour_thresholds_by_mass(
    density: np.ndarray,
    mass: np.ndarray,
    masses: tuple[float, ...],
) -> list[float]:
    density_flat = np.asarray(density, dtype=np.float64).ravel()
    mass_flat = np.asarray(mass, dtype=np.float64).ravel()
    mask = (density_flat > 0.0) & (mass_flat > 0.0) & np.isfinite(density_flat)
    if not np.any(mask):
        return []
    density_positive = density_flat[mask]
    mass_positive = mass_flat[mask]
    order = np.argsort(density_positive)[::-1]
    density_sorted = density_positive[order]
    mass_sorted = mass_positive[order]
    cumulative = np.cumsum(mass_sorted)
    cumulative /= cumulative[-1]
    thresholds = []
    for mass in sorted(masses, reverse=True):
        index = int(np.searchsorted(cumulative, mass, side="left"))
        index = min(index, density_sorted.size - 1)
        thresholds.append(float(density_sorted[index]))
    return sorted(set(thresholds))


def apply_corner_axis_labels(axes: np.ndarray, labels: list[str]) -> None:
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == 2:
                ax.set_xlabel(labels[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row])
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.14)


def stage1_config_from_dict(config: dict[str, object]) -> Stage1Config:
    return Stage1Config(
        train_simulations=int(config["train_simulations"]),
        val_simulations=int(config["val_simulations"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        mdn_components=int(config["mdn_components"]),
        flow_layers=int(config["flow_layers"]),
        flow_context_dim=int(config["flow_context_dim"]),
        seed=int(config["seed"]),
        observed_seed=int(config["observed_seed"]),
        requested_device=str(config["requested_device"]),
        families=[str(item) for item in config["families"]],
        posterior_samples=int(config["posterior_samples"]),
        reference_grid_size=int(config["reference_grid_size"]),
        train_sampler=str(config.get("train_sampler", "random")),
        context_features=str(config.get("context_features", "raw")),
        spline_bins=int(config.get("spline_bins", 12)),
        lr_schedule=str(config.get("lr_schedule", "constant")),
        lr_eta_min=float(config.get("lr_eta_min", 0.0)),
        lr_warmup_steps=int(config.get("lr_warmup_steps", 0)),
        lr_decay_epochs=int(config.get("lr_decay_epochs", 0)),
        adam_beta1=float(config.get("adam_beta1", 0.9)),
        adam_beta2=float(config.get("adam_beta2", 0.999)),
        adam_eps=float(config.get("adam_eps", 1e-8)),
        validation_every_epochs=int(config.get("validation_every_epochs", 1)),
        skip_training_validation=bool(config.get("skip_training_validation", False)),
        torch_compile=str(config.get("torch_compile", "none")),
        grad_clip_norm=float(config.get("grad_clip_norm", 20.0)),
        ema_decay=float(config.get("ema_decay", 0.0)),
        batching_mode=str(config.get("batching_mode", "dataloader")),
        max_optimizer_steps=int(config.get("max_optimizer_steps", 0)),
        loss_weight_mode=str(config.get("loss_weight_mode", "none")),
        loss_tail_weight=float(config.get("loss_tail_weight", 3.0)),
        target_transform=str(config.get("target_transform", "none")),
        target_ridge=float(config.get("target_ridge", 1e-3)),
        flow_activation=str(config.get("flow_activation", "relu")),
        flow_residual=bool(config.get("flow_residual", False)),
        flow_randperm=bool(config.get("flow_randperm", False)),
        flow_passes=int(config.get("flow_passes", 0)),
        flow_kind=str(config.get("flow_kind", "nsf")),
    )


def parse_proposal_scale(value: str) -> tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("proposal scale must contain three comma-separated floats")
    if any(piece <= 0.0 for piece in pieces):
        raise argparse.ArgumentTypeError("proposal scales must be positive")
    return pieces[0], pieces[1], pieces[2]


def static_path_for_request(ui_dist: Path, request_path: str) -> Path | None:
    root = ui_dist.resolve()
    relative = "index.html" if request_path == "/" else request_path.lstrip("/")
    if not relative or relative.endswith("/"):
        relative = f"{relative}index.html"
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


@dataclass
class DrawRecord:
    draw_id: str
    z_true: np.ndarray
    x: np.ndarray
    context: np.ndarray
    distance: float | None
    metadata: dict[str, object]
    npe_cache: dict[tuple[str, int], dict[str, object]] = field(default_factory=dict)
    npe_grid_cache: dict[tuple[str, int, tuple[float, ...]], dict[str, object]] = field(default_factory=dict)
    grid_cache: dict[int, dict[str, object]] = field(default_factory=dict)
    mcmc_cache: dict[str, object] | None = None


def make_server(
    *,
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
    port_retries: int,
    strict_port: bool,
) -> tuple[ThreadingHTTPServer, int]:
    if port == 0:
        server = ReusableThreadingHTTPServer((host, port), handler)
        return server, int(server.server_address[1])

    ports = [port] if strict_port else list(range(port, port + max(port_retries, 0) + 1))
    last_error: OSError | None = None
    for candidate in ports:
        try:
            server = ReusableThreadingHTTPServer((host, candidate), handler)
            if candidate != port:
                print(f"port {port} is in use; using {candidate} instead", flush=True)
            return server, int(server.server_address[1])
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE:
                raise
    assert last_error is not None
    raise OSError(
        last_error.errno,
        f"Could not bind {host}:{port}; tried through port {ports[-1]}. "
        "Use --port with a free port or --port 0 for an OS-assigned port.",
    ) from last_error


def load_stage1_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    family = str(checkpoint["family"])
    config = stage1_config_from_dict(checkpoint["config"])
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
    model = make_model(
        family,
        config,
        x_dim=int(x_mean.shape[0]),
        z_dim=int(z_mean.shape[0]),
    ).to(device)
    state_dict = checkpoint["state_dict"]
    if any(str(key).startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    state = {
        "family": family,
        "label": FAMILY_LABELS.get(family, family),
        "config": checkpoint["config"],
        "x_mean": x_mean,
        "x_std": np.asarray(checkpoint["x_std"], dtype=np.float64),
        "z_mean": z_mean,
        "z_std": np.asarray(checkpoint["z_std"], dtype=np.float64),
        "checkpoint_path": path,
    }
    return model, state


def load_stage1_run_metadata(path: Path) -> dict[str, object]:
    keys = (
        "full_val_nll_z_units",
        "best_val_nll_z_units",
        "training_seconds",
        "panel_marginal_wasserstein_mean",
        "panel_marginal_wasserstein_median",
        "optimizer_steps",
        "epochs_completed",
        "batches_per_epoch",
        "validation_evaluations",
        "model_parameters",
    )

    def selected_metadata(row: dict[str, object], source_path: Path) -> dict[str, object]:
        metadata = {key: row[key] for key in keys if row.get(key) is not None}
        metadata["run_summary"] = str(row.get("summary_json") or source_path)
        return metadata

    run_summary_path = path.parent / "broad_scaling_run_summary.json"
    if run_summary_path.exists():
        return selected_metadata(
            json.loads(run_summary_path.read_text(encoding="utf-8")),
            run_summary_path,
        )

    checkpoint = str(path)
    for parent in path.parents:
        summary_path = parent / "results" / "broad_scaling_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("rows", []):
            if row.get("model_pt") == checkpoint:
                return selected_metadata(row, summary_path)
    return {}


class NPEPosteriorViewer:
    def __init__(
        self,
        model_path: Path,
        broad_model_path: Path | None,
        best_broad_model_path: Path | None,
        best_broad_spline_model_path: Path | None,
        best_broad_efficiency_model_path: Path | None,
        best_broad_ensemble_summary_path: Path | None,
        weighted_broad_ensemble_summary_path: Path | None,
        seed: int,
        device: str,
        mcmc_device: str,
        mcmc_chains: int,
        mcmc_steps: int,
        mcmc_burn_in: int,
        mcmc_proposal_scale: tuple[float, float, float],
    ) -> None:
        self.model_path = model_path
        self.model_dir = model_path.parent
        self.summary_path = self.model_dir / "npe_flow_decay_summary.json"
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.device = torch.device(device)
        self.mcmc_device_name = mcmc_device
        self.mcmc_device, self.mcmc_dtype = choose_mcmc_device(mcmc_device)
        self.mcmc_chains = mcmc_chains
        self.mcmc_steps = mcmc_steps
        self.mcmc_burn_in = mcmc_burn_in
        self.mcmc_proposal_scale = mcmc_proposal_scale
        self.draw_cache: dict[str, DrawRecord] = {}
        self.draw_cache_order: list[str] = []
        self.current_draw_id: str | None = None

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]
        self.config = config
        self.context_kind = str(config.get("context_kind", "indirect"))
        self.k_grid = make_k_grid(
            int(config.get("k_grid_points", 260)),
            float(config.get("k_min", 0.04)),
            float(config.get("k_max", 3.0)),
        )
        self.t = np.linspace(0.0, 6.0, 40)
        self.context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float64)
        self.context_std = np.asarray(checkpoint["context_std"], dtype=np.float64)
        self.z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float64)
        self.z_std = np.asarray(checkpoint["z_std"], dtype=np.float64)
        self.linear_adjustment = checkpoint.get("linear_adjustment")

        self.model = ConditionalSplineFlow(
            z_dim=3,
            context_dim=int(self.context_mean.shape[0]),
            transforms=int(config["transforms"]),
            hidden_features=tuple(int(v) for v in config["hidden_features"]),
            bins=int(config["bins"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.summary = None
        self.local_region = None
        self.observed_context = None
        if self.summary_path.exists():
            self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            self.local_region = self.summary.get("local_training", {}).get("region")
            observed_context = self.summary.get("observed_context")
            if observed_context is not None:
                self.observed_context = np.asarray(observed_context, dtype=np.float64)

        self.model_registry: dict[str, dict[str, object]] = {
            "local_flow": {
                "id": "local_flow",
                "label": "Local q0.005 spline flow, t8 h192x2 bins16, 150k",
                "plot_label": "Local q0.005 spline t8 h192 bins16",
                "color": NPE_LAYER_COLORS["local_flow"],
                "kind": "flow_decay",
                "training_scope": "local_prior",
                "training_description": "prior predictive filtered to the local region near x_0",
                "train_simulations": config.get("train_simulations"),
                "local_quantile": config.get("local_quantile"),
                "checkpoint": str(model_path),
                "has_local_region": self.local_region is not None,
            }
        }
        self.stage1_models: dict[str, torch.nn.Module] = {}
        self.stage1_states: dict[str, dict[str, object]] = {}
        self.stage1_ensembles: dict[str, list[str]] = {}
        self.stage1_ensemble_weights: dict[str, np.ndarray] = {}
        # Keep the legacy MDN arguments loadable for explicit debugging, but omit
        # them from the default UI model set after the older spline-flow records.
        for model_id, path, label, plot_label, color in (
            (
                "broad_mdn",
                broad_model_path,
                "Population-trained MDN, 5 components, h128x3, 100k",
                "Population-trained MDN 5c h128x3 100k",
                NPE_LAYER_COLORS["broad_mdn"],
            ),
            (
                "broad_mdn_512k",
                best_broad_model_path,
                "Population-trained MDN, 5 components, h128x3, 512k, seed 20260902",
                "Population-trained MDN 5c h128x3 512k",
                "#6d4aff",
            ),
        ):
            if path is not None and path.exists():
                self.register_stage1_checkpoint(
                    model_id=model_id,
                    path=path,
                    label=label,
                    plot_label=plot_label,
                    color=color,
                    training_description=(
                        "legacy MDN population-trained checkpoint retained for explicit "
                        "debugging, not part of the default comparison set"
                    ),
                )
        if best_broad_spline_model_path is not None and best_broad_spline_model_path.exists():
            self.register_stage1_checkpoint(
                model_id="broad_spline_4m",
                path=best_broad_spline_model_path,
                label="Population-trained Flow4 NSF, raw curve, h64x2 bins8, 4.096M x e90",
                plot_label="Population-trained Flow4 NSF 4.096M e90",
                color=NPE_LAYER_COLORS["broad_spline_4m"],
                training_description=(
                    "best panel-distance fixed parameter-count NPE from the scaling diagnostics "
                    "(4.096M prior-predictive simulations, seed 20260901)"
                ),
            )
        if best_broad_ensemble_summary_path is not None and best_broad_ensemble_summary_path.exists():
            self.register_stage1_ensemble(
                model_id="broad_fresh_e15_ensemble4",
                summary_path=best_broad_ensemble_summary_path,
                label="4-member Flow2 residual NSF ensemble, random permutations, raw curve plus fit features, 2.048M/member, 15 epochs",
                plot_label="4-member Flow2 residual NSF, 15 epochs",
                color=NPE_LAYER_COLORS["broad_fresh_e15_ensemble4"],
                training_description=(
                    "4-member equal-density ensemble trained from initialization; exact full-validation "
                    "NLL -3.630690 in 246 seconds of remote training wall time"
                ),
            )
        if weighted_broad_ensemble_summary_path is not None and weighted_broad_ensemble_summary_path.exists():
            self.register_stage1_ensemble(
                model_id="broad_weighted_checkpoint_pool",
                summary_path=weighted_broad_ensemble_summary_path,
                label="16-member convex-weighted checkpoint ensemble, mixed NSF checkpoints",
                plot_label="16-member weighted checkpoint ensemble",
                color=NPE_LAYER_COLORS["broad_weighted_checkpoint_pool"],
                training_description=(
                    "reference NLL record only: convex weighted density ensemble over saved "
                    "population-trained NPE checkpoints, not a direct training run"
                ),
            )
        elif best_broad_ensemble_summary_path is None and best_broad_efficiency_model_path is not None and best_broad_efficiency_model_path.exists():
            self.register_stage1_checkpoint(
                model_id="broad_spline_8m",
                path=best_broad_efficiency_model_path,
                label="Population-trained Flow3 NSF, raw curve, h80x2 bins8, 8.192M x e27",
                plot_label="Population-trained Flow3 NSF 8.192M e27",
                color=NPE_LAYER_COLORS["broad_spline_8m"],
                training_description=(
                    "superseded population-validation NLL record from the efficiency sweep "
                    "(8.192M prior-predictive simulations, 212k optimizer steps)"
                ),
            )

    def model_options(self) -> list[dict[str, object]]:
        return [
            json_ready(self.model_registry[model_id])
            for model_id in self.model_registry
            if model_id not in HIDDEN_UI_MODEL_IDS
        ]

    def register_stage1_checkpoint(
        self,
        *,
        model_id: str,
        path: Path,
        label: str,
        plot_label: str,
        color: str,
        training_description: str,
    ) -> None:
        model, state = load_stage1_checkpoint(path, self.device)
        config = state["config"]
        self.stage1_models[model_id] = model
        self.stage1_states[model_id] = state
        self.model_registry[model_id] = {
            "id": model_id,
            "label": label,
            "plot_label": plot_label,
            "color": color,
            "kind": f"stage1_{state['family']}",
            "training_scope": "broad_prior_predictive",
            "training_description": training_description,
            "family": state["family"],
            "family_label": state["label"],
            "train_simulations": config.get("train_simulations"),
            "local_quantile": None,
            "checkpoint": str(path),
            "has_local_region": False,
            **load_stage1_run_metadata(path),
        }

    def register_stage1_ensemble(
        self,
        *,
        model_id: str,
        summary_path: Path,
        label: str,
        plot_label: str,
        color: str,
        training_description: str,
    ) -> None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        member_paths = [Path(path) for path in summary.get("model_paths", [])]
        if not member_paths:
            raise ValueError(f"Ensemble summary has no model_paths: {summary_path}")
        raw_weights = summary.get("ensemble_weights")
        if raw_weights is None:
            weights = np.full(len(member_paths), 1.0 / len(member_paths), dtype=np.float64)
        else:
            weights = np.asarray(raw_weights, dtype=np.float64)
            if weights.shape != (len(member_paths),):
                raise ValueError(
                    f"Ensemble weights length {weights.shape[0]} does not match "
                    f"model_paths length {len(member_paths)}: {summary_path}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError(f"Invalid ensemble weights in {summary_path}")
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                raise ValueError(f"Ensemble weights sum to zero in {summary_path}")
            weights = weights / weight_sum
        member_ids: list[str] = []
        for index, path in enumerate(member_paths, start=1):
            if not path.exists():
                raise FileNotFoundError(f"Missing ensemble member checkpoint: {path}")
            member_id = f"{model_id}__member{index}"
            model, state = load_stage1_checkpoint(path, self.device)
            self.stage1_models[member_id] = model
            self.stage1_states[member_id] = state
            member_ids.append(member_id)
        self.stage1_ensembles[model_id] = member_ids
        self.stage1_ensemble_weights[model_id] = weights
        self.model_registry[model_id] = {
            "id": model_id,
            "label": label,
            "plot_label": plot_label,
            "color": color,
            "kind": "stage1_ensemble",
            "training_scope": "broad_prior_predictive",
            "training_description": training_description,
            "family": "spline_flow",
            "family_label": "Residual spline flow ensemble",
            "train_simulations": summary.get("train_simulations"),
            "full_val_nll_z_units": summary.get("full_val_nll_z_units"),
            "panel_marginal_wasserstein_mean": summary.get("panel_marginal_wasserstein_mean"),
            "panel_marginal_wasserstein_median": summary.get("panel_marginal_wasserstein_median"),
            "training_seconds": summary.get("training_wall_seconds"),
            "run_summary": str(summary_path),
            "ensemble_size": summary.get("ensemble_size", len(member_ids)),
            "ensemble_weights": weights.tolist(),
            "model_paths": [str(path) for path in member_paths],
            "checkpoint": str(summary_path),
            "has_local_region": False,
            "local_quantile": None,
        }

    def stage1_observed_features(self, model_id: str, x: np.ndarray) -> np.ndarray:
        state = self.stage1_states[model_id]
        x_mean = np.asarray(state["x_mean"], dtype=np.float64)
        raw = np.asarray(x, dtype=np.float64)
        if raw.shape[0] == x_mean.shape[0]:
            return raw
        config = state.get("config", {})
        mode = str(config.get("context_features", "raw"))
        features = transform_context_features(raw[None, :], mode)[0]
        if features.shape[0] != x_mean.shape[0]:
            raise ValueError(
                f"Context feature shape mismatch for {model_id}: "
                f"got {features.shape[0]}, expected {x_mean.shape[0]}"
            )
        return features

    def context_for_signal(self, x: np.ndarray) -> np.ndarray:
        return make_context_summaries(
            x[None, :],
            self.t,
            self.k_grid,
            kind=self.context_kind,
            chunk_size=1,
        )[0]

    def local_distance(self, context: np.ndarray) -> float | None:
        if self.local_region is None or self.observed_context is None:
            return None
        center = np.asarray(self.local_region["center"], dtype=np.float64)
        scale = np.asarray(self.local_region["scale"], dtype=np.float64)
        return float(context_distances(context[None, :], self.observed_context, center, scale)[0])

    def draw_prior_signal(self) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, object]]:
        z = sample_prior_z(1, self.rng)
        x = simulate_x_from_z(z, self.t, self.rng)[0]
        context = self.context_for_signal(x)
        distance = self.local_distance(context)
        return z[0], x, distance, {"mode": "prior_predictive"}

    def draw_x0_signal(self) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, object]]:
        seed = int(self.config.get("observed_seed", 20260622))
        t, y, true_theta = simulate_decay_data(seed=seed, n_observations=len(self.t))
        self.t = t.numpy()
        x = y.numpy()
        z = np.log(true_theta.numpy())
        context = self.context_for_signal(x)
        distance = self.local_distance(context)
        return z, x, distance, {"mode": "x0", "observed_seed": seed}

    def draw_low_prior_signal(self, mode: str) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, object]]:
        spec = LOW_PRIOR_SIGNAL_SPECS[mode]
        offset = np.asarray(spec["standardized_offset"], dtype=np.float64)
        z_mean = PRIOR_LOG_MEAN.detach().cpu().numpy().astype(np.float64)
        z_std = PRIOR_LOG_STD.detach().cpu().numpy().astype(np.float64)
        z = z_mean + z_std * offset
        theta = np.exp(z)
        rng = np.random.default_rng(int(spec["noise_seed"]))
        x = simulate_x_from_z(z[None, :], self.t, rng)[0]
        context = self.context_for_signal(x)
        distance = self.local_distance(context)
        prior_mahalanobis = float(np.linalg.norm(offset))
        return (
            z,
            x,
            distance,
            {
                "mode": mode,
                "theta": theta.tolist(),
                "standardized_log_prior_offset": offset.tolist(),
                "prior_mahalanobis": prior_mahalanobis,
                "log_prior_density_delta_vs_mean": float(-0.5 * prior_mahalanobis**2),
                "noise_seed": int(spec["noise_seed"]),
                "tail_label": str(spec["label"]),
            },
        )

    def draw_local_signal(self) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, object]]:
        if self.local_region is None or self.observed_context is None:
            return self.draw_prior_signal()

        center = np.asarray(self.local_region["center"], dtype=np.float64)
        scale = np.asarray(self.local_region["scale"], dtype=np.float64)
        radius = float(self.local_region["radius"])
        candidates = 0
        while candidates < 1_000_000:
            chunk = 4096
            z = sample_prior_z(chunk, self.rng)
            x = simulate_x_from_z(z, self.t, self.rng)
            context = make_context_summaries(
                x,
                self.t,
                self.k_grid,
                kind=self.context_kind,
                chunk_size=chunk,
            )
            distances = context_distances(context, self.observed_context, center, scale)
            accepted = np.flatnonzero(distances <= radius)
            candidates += chunk
            if accepted.size:
                index = int(accepted[0])
                return (
                    z[index],
                    x[index],
                    float(distances[index]),
                    {
                        "mode": "local_region",
                        "radius": radius,
                        "candidate_count": candidates,
                    },
                )
        raise RuntimeError("Could not draw a local-region signal within 1,000,000 candidates.")

    def draw_signal(self, mode: str) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, object]]:
        if mode == "prior":
            return self.draw_prior_signal()
        if mode == "x0":
            return self.draw_x0_signal()
        if mode in LOW_PRIOR_SIGNAL_SPECS:
            return self.draw_low_prior_signal(mode)
        return self.draw_local_signal()

    def create_draw(self, mode: str) -> DrawRecord:
        z_true, x, distance, metadata = self.draw_signal(mode)
        draw = DrawRecord(
            draw_id=uuid4().hex,
            z_true=z_true.copy(),
            x=x.copy(),
            context=self.context_for_signal(x),
            distance=distance,
            metadata=dict(metadata),
        )
        self.draw_cache[draw.draw_id] = draw
        self.draw_cache_order.append(draw.draw_id)
        self.current_draw_id = draw.draw_id
        while len(self.draw_cache_order) > 12:
            old_draw_id = self.draw_cache_order.pop(0)
            self.draw_cache.pop(old_draw_id, None)
            if self.current_draw_id == old_draw_id:
                self.current_draw_id = None
        return draw

    def resolve_draw(
        self,
        *,
        mode: str,
        draw_id: str | None,
        reuse_current: bool,
    ) -> DrawRecord:
        if draw_id:
            draw = self.draw_cache.get(draw_id)
            if draw is None:
                raise ValueError("The requested draw is no longer available. Draw a new signal and try again.")
            self.current_draw_id = draw.draw_id
            return draw
        if reuse_current and self.current_draw_id is not None:
            draw = self.draw_cache.get(self.current_draw_id)
            if draw is not None:
                return draw
        return self.create_draw(mode)

    def build_grid_comparison(
        self,
        *,
        x: np.ndarray,
        z_true: np.ndarray,
        grid_size: int,
        z_samples: np.ndarray | None = None,
        z_ranges: np.ndarray | None = None,
    ) -> dict[str, object]:
        start = time.perf_counter()
        if z_ranges is None:
            if z_samples is None:
                z_ranges = default_npe_grid_ranges(z_true)
            else:
                z_ranges = initial_z_ranges(
                    z_samples_by_model={"NPE": z_samples},
                    true_z=z_true,
                    padding_fraction=0.45,
                    min_padding=0.16,
                )
        reference, expansions = build_adaptive_grid_reference(
            t=self.t,
            y=x,
            z_ranges=z_ranges,
            grid_size=grid_size,
            chunk_size=120_000,
            edge_mass_tolerance=1e-4,
            max_expansions=2,
            restricted_region=None,
        )
        return {
            "reference": reference,
            "metadata": {
                "grid_size": int(reference["grid_size"]),
                "grid_points": int(reference["grid_points"]),
                "deterministic": True,
                "grid_expansions": int(expansions),
                "max_edge_mass": max_edge_mass(reference),
                "elapsed_seconds": time.perf_counter() - start,
            },
        }

    def sample_mcmc(
        self,
        *,
        x: np.ndarray,
    ) -> dict[str, object]:
        start = time.perf_counter()
        seed = int(self.rng.integers(1, 2_000_000_000))
        config = MCMCConfig(
            chains=self.mcmc_chains,
            steps=self.mcmc_steps,
            burn_in=self.mcmc_burn_in,
            seed=seed,
            proposal_scale=self.mcmc_proposal_scale,
            requested_device=self.mcmc_device_name,
            sampler_variant="low-overhead",
        )
        _, theta_raw, accepted, sampler_seconds = run_random_walk_metropolis(
            t=torch.as_tensor(self.t, dtype=torch.float64),
            y=torch.as_tensor(x, dtype=torch.float64),
            config=config,
            device=self.mcmc_device,
            dtype=self.mcmc_dtype,
        )
        posterior = theta_raw[:, self.mcmc_burn_in :, :].reshape(-1, 3)
        diagnostics = arviz_diagnostics(theta_raw, self.mcmc_burn_in)
        flags = convergence_flags(diagnostics)
        return {
            "samples": posterior,
            "metadata": {
                "chains": self.mcmc_chains,
                "steps": self.mcmc_steps,
                "burn_in": self.mcmc_burn_in,
                "proposal_scale": list(self.mcmc_proposal_scale),
                "seed": seed,
                "device": str(self.mcmc_device),
                "dtype": str(self.mcmc_dtype).replace("torch.", ""),
                "runtime_seconds": float(sampler_seconds),
                "elapsed_seconds": time.perf_counter() - start,
                "draws_after_burn_in": int(posterior.shape[0]),
                "acceptance_rate": float(accepted.mean()),
                "diagnostics": diagnostics,
                "convergence_flags": flags,
                "convergence_ok": bool(all(flags.values())),
            },
        }

    def sample_estimator(
        self,
        *,
        model_id: str,
        x: np.ndarray,
        context: np.ndarray,
        posterior_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if model_id == "local_flow":
            return sample_flow_posterior(
                model=self.model,
                observed_context=context,
                context_mean=self.context_mean,
                context_std=self.context_std,
                z_mean=self.z_mean,
                z_std=self.z_std,
                linear_adjustment=self.linear_adjustment,
                n=posterior_samples,
                device=self.device,
            )
        if model_id in self.stage1_models:
            stage1_state = self.stage1_states[model_id]
            observed_features = self.stage1_observed_features(model_id, x)
            return sample_posterior_for_observation(
                model=self.stage1_models[model_id],
                observed_x=observed_features,
                x_mean=stage1_state["x_mean"],
                x_std=stage1_state["x_std"],
                z_mean=stage1_state["z_mean"],
                z_std=stage1_state["z_std"],
                n=posterior_samples,
                device=self.device,
            )
        if model_id in self.stage1_ensembles:
            member_ids = self.stage1_ensembles[model_id]
            weights = self.stage1_ensemble_weights.get(
                model_id,
                np.full(len(member_ids), 1.0 / len(member_ids), dtype=np.float64),
            )
            raw_counts = weights * int(posterior_samples)
            counts = np.floor(raw_counts).astype(int)
            remainder = int(posterior_samples) - int(counts.sum())
            if remainder > 0:
                order = np.argsort(raw_counts - counts)[::-1]
                counts[order[:remainder]] += 1
            z_parts = []
            theta_parts = []
            for member_id, count in zip(member_ids, counts, strict=True):
                if count <= 0:
                    continue
                stage1_state = self.stage1_states[member_id]
                observed_features = self.stage1_observed_features(member_id, x)
                z_member, theta_member = sample_posterior_for_observation(
                    model=self.stage1_models[member_id],
                    observed_x=observed_features,
                    x_mean=stage1_state["x_mean"],
                    x_std=stage1_state["x_std"],
                    z_mean=stage1_state["z_mean"],
                    z_std=stage1_state["z_std"],
                    n=count,
                    device=self.device,
                )
                z_parts.append(z_member)
                theta_parts.append(theta_member)
            return np.vstack(z_parts), np.vstack(theta_parts)
        available = ", ".join(self.model_registry)
        raise ValueError(f"Unknown model_id {model_id!r}. Available models: {available}")

    def estimator_log_prob_on_z_grid(
        self,
        *,
        model_id: str,
        x: np.ndarray,
        context: np.ndarray,
        z_grid: np.ndarray,
        chunk_size: int = 120_000,
    ) -> np.ndarray:
        log_prob = np.empty(z_grid.shape[0], dtype=np.float64)
        if model_id == "local_flow":
            context_std_value = ((context - self.context_mean) / self.context_std).astype(np.float32)
            adjustment = np.zeros(3, dtype=np.float64)
            if self.linear_adjustment is not None:
                observed_context_std = np.asarray(self.linear_adjustment["observed_context_std"], dtype=np.float64)
                slope = np.asarray(self.linear_adjustment["slope"], dtype=np.float64)
                delta = context_std_value.astype(np.float64) - observed_context_std
                adjustment = delta @ slope
            context_row = torch.from_numpy(context_std_value[None, :]).to(self.device)
            for start in range(0, z_grid.shape[0], chunk_size):
                stop = min(start + chunk_size, z_grid.shape[0])
                z_standardized = ((z_grid[start:stop] - adjustment[None, :] - self.z_mean[None, :]) / self.z_std[None, :]).astype(np.float32)
                z_tensor = torch.from_numpy(z_standardized).to(self.device)
                context_tensor = context_row.expand(z_tensor.shape[0], -1)
                with torch.no_grad():
                    values = self.model.log_prob(z_tensor, context_tensor).detach().cpu().numpy()
                log_prob[start:stop] = values
            return log_prob

        if model_id in self.stage1_models:
            stage1_state = self.stage1_states[model_id]
            observed_features = self.stage1_observed_features(model_id, x)
            x_standardized = ((observed_features[None, :] - stage1_state["x_mean"][None, :]) / stage1_state["x_std"][None, :]).astype(np.float32)
            x_row = torch.from_numpy(x_standardized).to(self.device)
            z_mean = np.asarray(stage1_state["z_mean"], dtype=np.float64)
            z_std = np.asarray(stage1_state["z_std"], dtype=np.float64)
            for start in range(0, z_grid.shape[0], chunk_size):
                stop = min(start + chunk_size, z_grid.shape[0])
                z_standardized = ((z_grid[start:stop] - z_mean[None, :]) / z_std[None, :]).astype(np.float32)
                z_tensor = torch.from_numpy(z_standardized).to(self.device)
                x_tensor = x_row.expand(z_tensor.shape[0], -1)
                with torch.no_grad():
                    values = self.stage1_models[model_id].log_prob(z_tensor, x_tensor).detach().cpu().numpy()
                log_prob[start:stop] = values
            return log_prob

        if model_id in self.stage1_ensembles:
            member_log_probs = []
            for member_id in self.stage1_ensembles[model_id]:
                member_log_probs.append(
                    self.estimator_log_prob_on_z_grid(
                        model_id=member_id,
                        x=x,
                        context=context,
                        z_grid=z_grid,
                        chunk_size=chunk_size,
                    )
                )
            weights = self.stage1_ensemble_weights.get(
                model_id,
                np.full(len(member_log_probs), 1.0 / len(member_log_probs), dtype=np.float64),
            )
            return logsumexp(
                np.vstack(member_log_probs) + np.log(weights)[:, None],
                axis=0,
            )

        available = ", ".join(self.model_registry)
        raise ValueError(f"Unknown model_id {model_id!r}. Available models: {available}")

    def evaluate_estimator_grid(
        self,
        *,
        model_id: str,
        x: np.ndarray,
        context: np.ndarray,
        z_ranges: np.ndarray,
        grid_size: int,
    ) -> dict[str, object]:
        z_grid = z_grid_from_ranges(z_ranges, grid_size)
        log_density = self.estimator_log_prob_on_z_grid(
            model_id=model_id,
            x=x,
            context=context,
            z_grid=z_grid,
        )
        return reference_from_log_density_grid(
            z_grid=z_grid,
            log_density=log_density,
            z_ranges=z_ranges,
            grid_size=grid_size,
        )

    def render_signal_plot(
        self,
        *,
        x: np.ndarray,
        true_theta: np.ndarray,
        npe_layers: list[dict[str, object]],
        mcmc_samples: np.ndarray | None,
    ) -> str:
        t_grid = np.linspace(float(self.t.min()), float(self.t.max()), 220)
        true_mean = true_theta[0] * np.exp(-true_theta[1] * t_grid)
        figure, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.scatter(self.t, x, s=28, color=GRID_COLOR, label="signal")
        ax.plot(t_grid, true_mean, color=GRID_COLOR, linestyle="--", lw=1.6, label="true mean")
        for layer in npe_layers:
            color = str(layer["color"])
            label = str(layer["label"])
            if "reference" in layer:
                reference = layer["reference"]
                theta_grid = np.asarray(reference["theta_grid"], dtype=np.float64)
                weights = np.asarray(reference["weights"], dtype=np.float64)
                t_band = np.linspace(float(self.t.min()), float(self.t.max()), 110)
                mean_lower = np.empty_like(t_band)
                mean_median = np.empty_like(t_band)
                mean_upper = np.empty_like(t_band)
                for index, value in enumerate(t_band):
                    mean_values = theta_grid[:, 0] * np.exp(-theta_grid[:, 1] * value)
                    mean_lower[index], mean_median[index], mean_upper[index] = weighted_quantile(
                        mean_values,
                        weights,
                        [0.05, 0.50, 0.95],
                    )
                ax.fill_between(t_band, mean_lower, mean_upper, color=color, alpha=0.15, label=f"{label} 90% mean")
                ax.plot(t_band, mean_median, color=color, lw=2.0, label=f"{label} median")
            else:
                posterior_samples = np.asarray(layer["samples"], dtype=np.float64)
                mean_draws = posterior_samples[:, 0, None] * np.exp(-posterior_samples[:, 1, None] * t_grid[None, :])
                mean_lower, mean_median, mean_upper = np.quantile(mean_draws, [0.05, 0.50, 0.95], axis=0)
                lower, _, upper = posterior_predictive_band(
                    posterior_samples,
                    t_grid,
                    seed=int(self.rng.integers(1, 2_000_000_000)),
                    max_draws=800,
                )
                ax.fill_between(t_grid, lower, upper, color=color, alpha=0.08, label=f"{label} 90% replicated y")
                ax.fill_between(t_grid, mean_lower, mean_upper, color=color, alpha=0.15, label=f"{label} 90% mean")
                ax.plot(t_grid, mean_median, color=color, lw=2.0, label=f"{label} median")
        if mcmc_samples is not None:
            mcmc_mean_draws = mcmc_samples[:, 0, None] * np.exp(-mcmc_samples[:, 1, None] * t_grid[None, :])
            mcmc_mean_lower, mcmc_mean_median, mcmc_mean_upper = np.quantile(
                mcmc_mean_draws,
                [0.05, 0.50, 0.95],
                axis=0,
            )
            ax.fill_between(
                t_grid,
                mcmc_mean_lower,
                mcmc_mean_upper,
                color=MCMC_COLOR,
                alpha=0.16,
                label="MCMC 90% mean curve",
            )
            ax.plot(t_grid, mcmc_mean_median, color=MCMC_COLOR, lw=1.9, label="MCMC median mean curve")
        ax.set_xlabel("time")
        ax.set_ylabel("observation")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", frameon=False)
        figure.tight_layout()
        return figure_to_data_uri(figure)

    def render_corner_plot(
        self,
        *,
        true_theta: np.ndarray,
        npe_layers: list[dict[str, object]],
        grid_reference: dict[str, object] | None,
        mcmc_samples: np.ndarray | None,
    ) -> str:
        labels = [r"$A$", r"$k$", r"$\sigma$"]

        weighted_layers = []
        if grid_reference is not None:
            weighted_layers.append(("Grid reference", GRID_COLOR, grid_reference, 1.9, 1.55))
        sample_layers = []
        for npe_layer in npe_layers:
            if "reference" in npe_layer:
                weighted_layers.append(
                    (
                        str(npe_layer["label"]),
                        str(npe_layer["color"]),
                        npe_layer["reference"],
                        1.7,
                        1.4,
                    )
                )
            else:
                plot_samples = np.asarray(npe_layer["samples"], dtype=np.float64)
                if plot_samples.shape[0] > 12_000:
                    indices = self.rng.choice(plot_samples.shape[0], size=12_000, replace=False)
                    plot_samples = plot_samples[indices]
                sample_layers.append((str(npe_layer["label"]), str(npe_layer["color"]), plot_samples, 1.5, 1.35))
        if mcmc_samples is not None:
            mcmc_plot = mcmc_samples
            if mcmc_plot.shape[0] > 12_000:
                indices = self.rng.choice(mcmc_plot.shape[0], size=12_000, replace=False)
                mcmc_plot = mcmc_plot[indices]
            sample_layers.append(("MCMC posterior", MCMC_COLOR, mcmc_plot, 1.5, 1.35))

        if not weighted_layers and not sample_layers:
            figure, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
            for row in range(3):
                for col in range(3):
                    ax = axes[row, col]
                    if row < col:
                        ax.axis("off")
                        continue
                    if row == col:
                        ax.axvline(true_theta[col], color=GRID_COLOR, linestyle="--", lw=1.6)
                    else:
                        ax.axvline(true_theta[col], color=GRID_COLOR, linestyle="--", lw=1.3)
                        ax.axhline(true_theta[row], color=GRID_COLOR, linestyle="--", lw=1.3)
                    ax.grid(alpha=0.16)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row == 2:
                        ax.set_xlabel(labels[col])
                    if col == 0 and row > 0:
                        ax.set_ylabel(labels[row])
            figure.legend(handles=[true_theta_legend_handle()], loc="upper right", bbox_to_anchor=(0.97, 0.96))
            figure.suptitle("True theta only", y=0.985, fontsize=15)
            figure.tight_layout()
            return figure_to_data_uri(figure, dpi=145)

        ranges = []
        for index in range(3):
            bounds = [float(true_theta[index])]
            for _, _, reference, _, _ in weighted_layers:
                grid_values = np.asarray(reference["theta_grid"], dtype=np.float64)[:, index]
                grid_weights = np.asarray(reference["weights"], dtype=np.float64)
                bounds.extend(weighted_quantile(grid_values, grid_weights, [0.002, 0.998]).tolist())
            for _, _, samples, _, _ in sample_layers:
                bounds.extend(np.quantile(samples[:, index], [0.002, 0.998]).tolist())
            low = min(bounds)
            high = max(bounds)
            width = max(high - low, 1e-9)
            ranges.append((low - 0.08 * width, high + 0.08 * width))

        figure, axes = plt.subplots(3, 3, figsize=(7.4, 7.4))
        apply_corner_axis_labels(axes, labels)
        handles = []

        for label, color, reference, hist_lw, contour_lw in weighted_layers:
            weights = np.asarray(reference["weights"], dtype=np.float64)
            grid_size = int(reference["grid_size"])
            weight_cube = weights.reshape(grid_size, grid_size, grid_size)
            theta_axes, theta_widths = grid_theta_axes_and_widths(reference)
            for col in range(3):
                ax = axes[col, col]
                sum_axes = tuple(axis for axis in range(3) if axis != col)
                marginal_mass = weight_cube.sum(axis=sum_axes)
                density = marginal_mass / theta_widths[col]
                ax.plot(theta_axes[col], density, color=color, linewidth=hist_lw)
            for row in range(1, 3):
                for col in range(row):
                    ax = axes[row, col]
                    sum_axes = tuple(axis for axis in range(3) if axis not in {col, row})
                    marginal_mass_2d = weight_cube.sum(axis=sum_axes)
                    density = marginal_mass_2d / (theta_widths[col][:, None] * theta_widths[row][None, :])
                    levels = contour_thresholds_by_mass(density, marginal_mass_2d, (0.50, 0.90))
                    if levels:
                        ax.contour(
                            theta_axes[col],
                            theta_axes[row],
                            density.T,
                            levels=levels,
                            colors=[color],
                            linewidths=contour_lw,
                        )
            handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

        for label, color, samples, hist_lw, contour_lw in sample_layers:
            for col in range(3):
                ax = axes[col, col]
                hist, edges = np.histogram(samples[:, col], bins=70, range=ranges[col], density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, hist, where="mid", color=color, linewidth=hist_lw)
            for row in range(1, 3):
                for col in range(row):
                    ax = axes[row, col]
                    counts, x_edges, y_edges = np.histogram2d(
                        samples[:, col],
                        samples[:, row],
                        bins=48,
                        range=[ranges[col], ranges[row]],
                    )
                    x_widths = np.diff(x_edges)
                    y_widths = np.diff(y_edges)
                    total = max(float(counts.sum()), 1.0)
                    mass = counts / total
                    density = mass / (x_widths[:, None] * y_widths[None, :])
                    levels = contour_thresholds_by_mass(density, mass, (0.50, 0.90))
                    if levels:
                        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                        ax.contour(
                            x_centers,
                            y_centers,
                            density.T,
                            levels=levels,
                            colors=[color],
                            linewidths=contour_lw,
                        )
            handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

        for row in range(3):
            for col in range(row + 1):
                ax = axes[row, col]
                if row == col:
                    ax.set_xlim(*ranges[col])
                    ax.axvline(true_theta[col], color=GRID_COLOR, linestyle="--", lw=1.2, alpha=0.8)
                else:
                    ax.set_xlim(*ranges[col])
                    ax.set_ylim(*ranges[row])
                    ax.axvline(true_theta[col], color=GRID_COLOR, linestyle="--", lw=1.0, alpha=0.75)
                    ax.axhline(true_theta[row], color=GRID_COLOR, linestyle="--", lw=1.0, alpha=0.75)

        handles.append(true_theta_legend_handle())
        figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.96))
        figure.subplots_adjust(top=0.88, hspace=0.08, wspace=0.08)
        title_parts = [label for label, _, _, _, _ in weighted_layers]
        title_parts.extend(label for label, _, _, _, _ in sample_layers)
        title = " vs ".join(title_parts)
        figure.suptitle(title, y=0.985, fontsize=15)
        return figure_to_data_uri(figure, dpi=145)

    def render(
        self,
        *,
        model_ids: list[str],
        mode: str,
        draw_id: str | None,
        reuse_current: bool,
        refresh_layers: set[str],
        npe_render_mode: str,
        posterior_samples: int,
        include_grid: bool,
        include_mcmc: bool,
        grid_size: int,
        npe_grid_size: int | None,
    ) -> dict[str, object]:
        unknown_model_ids = [model_id for model_id in model_ids if model_id not in self.model_registry]
        if unknown_model_ids:
            available = ", ".join(self.model_registry)
            raise ValueError(f"Unknown model_id(s) {unknown_model_ids!r}. Available models: {available}")
        allowed_refresh_layers = set(self.model_registry) | {"grid", "mcmc"}
        unknown_refresh_layers = refresh_layers.difference(allowed_refresh_layers)
        if unknown_refresh_layers:
            available = ", ".join(sorted(allowed_refresh_layers))
            raise ValueError(f"Unknown refresh layer(s) {sorted(unknown_refresh_layers)!r}. Available layers: {available}")
        if npe_render_mode not in {"sample", "grid"}:
            raise ValueError("npe_render_mode must be 'sample' or 'grid'.")
        start = time.perf_counter()
        with self.lock:
            draw = self.resolve_draw(mode=mode, draw_id=draw_id, reuse_current=reuse_current)
            z_true = draw.z_true
            x = draw.x
            distance = draw.distance
            metadata = dict(draw.metadata)
            true_theta = np.exp(z_true)
            observed_context = draw.context
            npe_results = []
            npe_seconds = 0.0
            sampled_npe_count = 0
            npe_eval_grid_size = (
                min(grid_size, DEFAULT_NPE_EVAL_GRID_SIZE)
                if npe_grid_size is None
                else npe_grid_size
            )

            grid = None
            grid_seconds = None
            if include_grid and npe_render_mode == "grid":
                grid_key = grid_size
                grid = draw.grid_cache.get(grid_key)
                if grid is None or "grid" in refresh_layers:
                    grid = self.build_grid_comparison(
                        x=x,
                        z_true=z_true,
                        grid_size=grid_size,
                        z_ranges=default_npe_grid_ranges(z_true),
                    )
                    draw.grid_cache[grid_key] = grid
                    grid_seconds = float(grid["metadata"]["elapsed_seconds"])
                else:
                    grid_seconds = 0.0

            if npe_render_mode == "sample":
                for selected_model_id in model_ids:
                    cache_key = (selected_model_id, posterior_samples)
                    cached_result = draw.npe_cache.get(cache_key)
                    layer_seconds = 0.0
                    if cached_result is None or selected_model_id in refresh_layers:
                        npe_start = time.perf_counter()
                        z_samples, theta_samples = self.sample_estimator(
                            model_id=selected_model_id,
                            x=x,
                            context=observed_context,
                            posterior_samples=posterior_samples,
                        )
                        layer_seconds = time.perf_counter() - npe_start
                        npe_seconds += layer_seconds
                        sampled_npe_count += 1
                        model_metadata = self.model_registry[selected_model_id]
                        cached_result = {
                            "model_id": selected_model_id,
                            "label": str(model_metadata.get("plot_label", model_metadata["label"])),
                            "full_label": str(model_metadata["label"]),
                            "color": str(model_metadata.get("color", NPE_LAYER_COLORS.get(selected_model_id, NPE_COLOR))),
                            "metadata": model_metadata,
                            "render_kind": "sample",
                            "z_samples": z_samples,
                            "theta_samples": theta_samples,
                            "summary": sample_summary_rows(theta_samples),
                        }
                        draw.npe_cache[cache_key] = cached_result
                    npe_results.append(
                        {
                            **cached_result,
                            "seconds": layer_seconds,
                        }
                    )

            else:
                if grid is not None:
                    eval_ranges = np.asarray(
                        [
                            grid["reference"]["z_ranges"][name]
                            for name in PARAMETER_NAMES
                        ],
                        dtype=np.float64,
                    )
                else:
                    eval_ranges = default_npe_grid_ranges(z_true)
                eval_range_key = range_cache_key(eval_ranges)
                for selected_model_id in model_ids:
                    cache_key = (selected_model_id, npe_eval_grid_size, eval_range_key)
                    cached_result = draw.npe_grid_cache.get(cache_key)
                    layer_seconds = 0.0
                    if cached_result is None or selected_model_id in refresh_layers:
                        npe_start = time.perf_counter()
                        reference = self.evaluate_estimator_grid(
                            model_id=selected_model_id,
                            x=x,
                            context=observed_context,
                            z_ranges=eval_ranges,
                            grid_size=npe_eval_grid_size,
                        )
                        layer_seconds = time.perf_counter() - npe_start
                        npe_seconds += layer_seconds
                        model_metadata = self.model_registry[selected_model_id]
                        cached_result = {
                            "model_id": selected_model_id,
                            "label": str(model_metadata.get("plot_label", model_metadata["label"])),
                            "full_label": str(model_metadata["label"]),
                            "color": str(model_metadata.get("color", NPE_LAYER_COLORS.get(selected_model_id, NPE_COLOR))),
                            "metadata": model_metadata,
                            "render_kind": "grid",
                            "reference": reference,
                            "summary": grid_summary_rows(reference),
                        }
                        draw.npe_grid_cache[cache_key] = cached_result
                    npe_results.append(
                        {
                            **cached_result,
                            "seconds": layer_seconds,
                        }
                    )

            if include_grid and npe_render_mode == "sample":
                grid_key = grid_size
                grid = draw.grid_cache.get(grid_key)
                if grid is None or "grid" in refresh_layers:
                    grid_anchor = npe_results[0] if npe_results else None
                    if grid_anchor is None:
                        anchor_model_id = "local_flow" if "local_flow" in self.model_registry else next(iter(self.model_registry))
                        z_samples, theta_samples = self.sample_estimator(
                            model_id=anchor_model_id,
                            x=x,
                            context=observed_context,
                            posterior_samples=posterior_samples,
                        )
                        grid_anchor = {
                            "model_id": anchor_model_id,
                            "z_samples": z_samples,
                            "theta_samples": theta_samples,
                        }
                    grid = self.build_grid_comparison(
                        x=x,
                        z_true=z_true,
                        z_samples=grid_anchor["z_samples"],
                        grid_size=grid_size,
                    )
                    draw.grid_cache[grid_key] = grid
                    grid_seconds = float(grid["metadata"]["elapsed_seconds"])
                else:
                    grid_seconds = 0.0
            npe_grid_metrics = {}
            if grid is not None:
                for result in npe_results:
                    if result.get("render_kind") == "grid":
                        npe_grid_metrics[str(result["model_id"])] = compare_weighted_to_reference(
                            result["reference"],
                            grid["reference"],
                        )
                    else:
                        npe_grid_metrics[str(result["model_id"])] = compare_to_reference(
                            result["theta_samples"],
                            grid["reference"],
                        )
            mcmc = None
            mcmc_seconds = None
            if include_mcmc:
                if draw.mcmc_cache is None or "mcmc" in refresh_layers:
                    mcmc = self.sample_mcmc(x=x)
                    draw.mcmc_cache = mcmc
                    mcmc_seconds = float(mcmc["metadata"]["runtime_seconds"])
                else:
                    mcmc = draw.mcmc_cache
                    mcmc_seconds = 0.0
                if grid is not None:
                    mcmc = {
                        **mcmc,
                        "metrics": compare_to_reference(
                            mcmc["samples"],
                            grid["reference"],
                        ),
                    }
            plot_start = time.perf_counter()
            plot_npe_layers = []
            for result in npe_results:
                layer = {
                    "label": result["label"],
                    "color": result["color"],
                }
                if result.get("render_kind") == "grid":
                    layer["reference"] = result["reference"]
                else:
                    layer["samples"] = result["theta_samples"]
                plot_npe_layers.append(layer)
            corner_uri = self.render_corner_plot(
                true_theta=true_theta,
                npe_layers=plot_npe_layers,
                grid_reference=None if grid is None else grid["reference"],
                mcmc_samples=None if mcmc is None else mcmc["samples"],
            )
            signal_uri = self.render_signal_plot(
                x=x,
                true_theta=true_theta,
                npe_layers=plot_npe_layers,
                mcmc_samples=None if mcmc is None else mcmc["samples"],
            )
            plot_seconds = time.perf_counter() - plot_start

        radius = None
        if self.local_region is not None:
            radius = float(self.local_region["radius"])
        total_seconds = time.perf_counter() - start
        first_npe = npe_results[0] if npe_results else None
        first_model_metadata = None if first_npe is None else first_npe["metadata"]
        sampled_npe_draws = posterior_samples * sampled_npe_count
        return {
            "draw_id": draw.draw_id,
            "corner": corner_uri,
            "signal": signal_uri,
            "true_theta": {
                name: float(true_theta[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "signal_data": {
                "t": [float(value) for value in self.t],
                "x": [float(value) for value in x],
                "z_true": {
                    name: float(z_true[index])
                    for index, name in enumerate(PARAMETER_NAMES)
                },
            },
            "posterior_summary": [] if first_npe is None else first_npe["summary"],
            "npe_summaries": [
                {
                    "model_id": result["model_id"],
                    "label": result["label"],
                    "full_label": result["full_label"],
                    "summary": result["summary"],
                }
                for result in npe_results
            ],
            "grid_summary": None if grid is None else grid_summary_rows(grid["reference"]),
            "mcmc_summary": None if mcmc is None else sample_summary_rows(mcmc["samples"]),
            "grid_metrics": (
                None
                if grid is None or first_npe is None
                else json_ready(npe_grid_metrics[str(first_npe["model_id"])])
            ),
            "npe_grid_metrics": json_ready(npe_grid_metrics),
            "mcmc_grid_metrics": (
                None
                if mcmc is None or "metrics" not in mcmc
                else json_ready(mcmc["metrics"])
            ),
            "grid_metadata": None if grid is None else json_ready(grid["metadata"]),
            "npe_grid_metadata": (
                None
                if npe_render_mode != "grid" or not npe_results
                else {
                    "grid_size": int(npe_eval_grid_size),
                    "grid_points": int(npe_eval_grid_size**3),
                    "resolution_cap": int(MAX_NPE_EVAL_GRID_SIZE),
                    "uses_reference_ranges": bool(grid is not None),
                }
            ),
            "mcmc_metadata": None if mcmc is None else json_ready(mcmc["metadata"]),
            "mode_metadata": metadata,
            "local_distance": distance,
            "local_radius": radius,
            "inside_local_region": None if distance is None or radius is None else bool(distance <= radius),
            "posterior_samples": posterior_samples,
            "npe_render_mode": npe_render_mode,
            "include_npe": bool(npe_results),
            "elapsed_seconds": total_seconds,
            "timing": {
                "npe_sampling_seconds": npe_seconds,
                "npe_samples_per_second": sampled_npe_draws / max(npe_seconds, 1e-12) if sampled_npe_draws else 0.0,
                "npe_timings": {
                    str(result["model_id"]): float(result["seconds"])
                    for result in npe_results
                },
                "grid_seconds": grid_seconds,
                "mcmc_seconds": mcmc_seconds,
                "mcmc_elapsed_seconds": None if mcmc is None else float(mcmc["metadata"]["elapsed_seconds"]),
                "grid_points_per_second": (
                    None
                    if grid is None
                    else int(grid["metadata"]["grid_points"]) / max(grid_seconds or 1e-12, 1e-12) if grid_seconds else 0.0
                ),
                "plot_seconds": plot_seconds,
                "total_seconds": total_seconds,
            },
            "model": "" if first_model_metadata is None else first_model_metadata["checkpoint"],
            "model_id": "" if first_npe is None else first_npe["model_id"],
            "model_metadata": None if first_model_metadata is None else json_ready(first_model_metadata),
            "selected_npe_model_ids": [result["model_id"] for result in npe_results],
            "selected_npe_models": [
                json_ready(result["metadata"])
                for result in npe_results
            ],
            "summary": {
                "checkpoint_context": self.context_kind,
                "training_mode": None if first_model_metadata is None else first_model_metadata.get("training_scope"),
                "train_simulations": None if first_model_metadata is None else first_model_metadata.get("train_simulations"),
                "local_quantile": None if first_model_metadata is None else first_model_metadata.get("local_quantile"),
            },
            "z_sample_shape": (
                []
                if first_npe is None
                else list(first_npe["z_samples"].shape)
                if first_npe.get("render_kind") == "sample"
                else list(first_npe["reference"]["theta_grid"].shape)
            ),
        }


def make_handler(
    viewer: NPEPosteriorViewer,
    *,
    ui_dist: Path,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

        def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, path: Path) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {
                "application/javascript",
                "application/json",
                "image/svg+xml",
            }:
                content_type = f"{content_type}; charset=utf-8"
            self.send_bytes(path.read_bytes(), content_type)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/models":
                payload = json.dumps(json_ready(viewer.model_options())).encode("utf-8")
                self.send_bytes(payload, "application/json")
                return
            if parsed.path == "/api/new":
                query = parse_qs(parsed.query, keep_blank_values=True)
                model_id = query.get("model_id", ["local_flow"])[0]
                model_ids_raw = query.get("model_ids", [None])[0]
                mode = query.get("mode", ["local"])[0]
                draw_id = query.get("draw_id", [None])[0] or None
                reuse_current_raw = query.get("reuse_current", ["0"])[0]
                refresh_layers_raw = query.get("refresh_layers", [""])[0]
                samples_raw = query.get("samples", ["7000"])[0]
                reference = query.get("reference", ["grid"])[0]
                include_npe_raw = query.get("npe", ["1"])[0]
                include_mcmc_raw = query.get("mcmc", ["0"])[0]
                npe_render_mode = query.get("npe_mode", ["sample"])[0]
                grid_size_raw = query.get("grid_size", ["60"])[0]
                npe_grid_size_raw = query.get("npe_grid_size", [None])[0]
                try:
                    posterior_samples = min(max(int(samples_raw), 1000), MAX_POSTERIOR_SAMPLES)
                    grid_size = min(max(int(grid_size_raw), 25), 180)
                    npe_grid_size = (
                        None
                        if npe_grid_size_raw in {None, ""}
                        else min(max(int(npe_grid_size_raw), 25), MAX_NPE_EVAL_GRID_SIZE)
                    )
                    reuse_current = reuse_current_raw.lower() in {"1", "true", "yes", "on"}
                    include_npe = include_npe_raw.lower() in {"1", "true", "yes", "on"}
                    include_mcmc = include_mcmc_raw.lower() in {"1", "true", "yes", "on"}
                    if model_ids_raw is None:
                        selected_model_ids = [model_id] if include_npe else []
                    else:
                        selected_model_ids = [
                            item.strip()
                            for item in model_ids_raw.split(",")
                            if item.strip()
                        ]
                    refresh_layers = {
                        item.strip()
                        for item in refresh_layers_raw.split(",")
                        if item.strip()
                    }
                    output = viewer.render(
                        model_ids=selected_model_ids,
                        mode=mode,
                        draw_id=draw_id,
                        reuse_current=reuse_current,
                        refresh_layers=refresh_layers,
                        npe_render_mode=npe_render_mode,
                        posterior_samples=posterior_samples,
                        include_grid=reference == "grid",
                        include_mcmc=include_mcmc,
                        grid_size=grid_size,
                        npe_grid_size=npe_grid_size,
                    )
                    payload = json.dumps(json_ready(output)).encode("utf-8")
                    self.send_bytes(payload, "application/json")
                except Exception as exc:  # noqa: BLE001 - surface errors to the local UI.
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    self.send_bytes(payload, "application/json", HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if parsed.path.startswith("/api/"):
                self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return

            static_path = static_path_for_request(ui_dist, parsed.path)
            if static_path is None and not parsed.path.startswith("/assets/"):
                static_path = static_path_for_request(ui_dist, "/")
            if static_path is not None:
                self.send_file(static_path)
                return

            message = (
                f"React UI build not found at {ui_dist.resolve()}.\n"
                "Run: cd viewer-ui && npm install && npm run build\n"
            )
            self.send_bytes(
                message.encode("utf-8"),
                "text/plain; charset=utf-8",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an interactive viewer for a saved decay NPE posterior.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--broad-model",
        type=Path,
        default=DEFAULT_BROAD_MODEL,
        help=(
            "Optional older population-trained decay NPE checkpoint to expose "
            "in the posterior-estimator dropdown. Omitted by default."
        ),
    )
    parser.add_argument(
        "--best-broad-model",
        type=Path,
        default=DEFAULT_BEST_BROAD_MODEL,
        help=(
            "Optional legacy population-trained MDN checkpoint from earlier scaling-law runs. "
            "Omitted by default."
        ),
    )
    parser.add_argument(
        "--best-broad-spline-model",
        type=Path,
        default=DEFAULT_BEST_BROAD_SPLINE_MODEL,
        help=(
            "Optional older spline-flow checkpoint from the fixed parameter-count scaling "
            "diagnostics. If the path is missing, it is omitted from the model dropdown."
        ),
    )
    parser.add_argument(
        "--best-broad-efficiency-model",
        type=Path,
        default=DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
        help=(
            "Optional superseded population-trained spline-flow efficiency-record checkpoint. "
            "Used as a fallback only when --best-broad-ensemble-summary is missing."
        ),
    )
    parser.add_argument(
        "--best-broad-ensemble-summary",
        type=Path,
        default=DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
        help=(
            "Optional current residual neural spline-flow ensemble summary. "
            "If present, it replaces the superseded 8.192M efficiency model in the picker."
        ),
    )
    parser.add_argument(
        "--weighted-broad-ensemble-summary",
        type=Path,
        default=DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
        help=(
            "Optional convex-weighted saved-checkpoint reference ensemble. "
            "Kept separate from directly trained ensemble records."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--port-retries",
        type=int,
        default=20,
        help="When --port is occupied, try this many subsequent ports before failing.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of trying subsequent ports when --port is occupied.",
    )
    parser.add_argument(
        "--ui-dist",
        type=Path,
        default=DEFAULT_UI_DIST,
        help="Directory containing the built React viewer UI.",
    )
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--mcmc-device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=24_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=6_000)
    parser.add_argument("--mcmc-proposal-scale", type=parse_proposal_scale, default=(0.030, 0.030, 0.040))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = NPEPosteriorViewer(
        args.model,
        args.broad_model,
        args.best_broad_model,
        args.best_broad_spline_model,
        args.best_broad_efficiency_model,
        args.best_broad_ensemble_summary,
        args.weighted_broad_ensemble_summary,
        seed=args.seed,
        device=args.device,
        mcmc_device=args.mcmc_device,
        mcmc_chains=args.mcmc_chains,
        mcmc_steps=args.mcmc_steps,
        mcmc_burn_in=args.mcmc_burn_in,
        mcmc_proposal_scale=args.mcmc_proposal_scale,
    )
    ui_dist = args.ui_dist.resolve()
    handler = make_handler(
        viewer,
        ui_dist=ui_dist,
    )
    server, actual_port = make_server(
        host=args.host,
        port=args.port,
        handler=handler,
        port_retries=args.port_retries,
        strict_port=args.strict_port,
    )
    url = f"http://{args.host}:{actual_port}/"
    print(f"NPE posterior viewer: {url}", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"ui: {ui_dist}", flush=True)
    print(f"models: {', '.join(item['id'] for item in viewer.model_options())}", flush=True)
    print(
        f"mcmc: {args.mcmc_chains} chains x {args.mcmc_steps} steps "
        f"(burn-in {args.mcmc_burn_in}) on {viewer.mcmc_device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
