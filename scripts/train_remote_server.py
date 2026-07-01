from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_REPO_DIR = Path("/Users/ioannism/repos/npe")
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/11_mdn_1m_remote")
DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
)
DEFAULT_VALIDATION_SEED = 20260990
DEFAULT_VALIDATION_N_OBSERVATIONS = 40
DEFAULT_VALIDATION_DTYPE = "float32"
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
)
DEFAULT_PANEL_SEED = 20261001
DEFAULT_PANEL_TARGET_REPEATS = 5
DEFAULT_TRAIN_SIMULATIONS = [64_000, 128_000, 256_000, 512_000, 1_000_000]
DEFAULT_SEEDS = [20260901, 20260902, 20260903]
FAMILY_CHOICES = {"mdn", "affine_flow", "spline_flow", "full_gaussian", "diag_gaussian"}
DEVICE_CHOICES = {"cpu", "mps", "auto", "cuda"}
LR_SCHEDULE_CHOICES = {"constant", "cosine_epoch", "cosine_step"}
TORCH_COMPILE_CHOICES = {"none", "default", "reduce_overhead"}
CONTEXT_VARIANT_CHOICES = {"real", "zero_x", "shuffled_x"}
CONTEXT_FEATURE_CHOICES = {"raw", "decay_summary", "raw_decay_summary"}
BATCHING_MODE_CHOICES = {"dataloader", "pre_shuffle", "sequential"}
TRAIN_SAMPLER_CHOICES = {"random", "lhs", "sobol"}
RUN_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def project_python_version(repo_dir: Path) -> str | None:
    version_file = repo_dir / ".python-version"
    if not version_file.exists():
        return None
    value = version_file.read_text(encoding="utf-8").strip()
    return value or None


def runtime_metadata(repo_dir: Path) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "project_python_version": project_python_version(repo_dir),
    }


def process_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def safe_run_name(value: object) -> str:
    name = "broad_scaling" if value in (None, "") else str(value)
    return RUN_NAME_RE.sub("_", name).strip("._-") or "broad_scaling"


def parse_bool(value: object, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean")


def parse_int(value: object, *, default: int, name: str, minimum: int = 1) -> int:
    output = default if value is None else int(value)
    if output < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return output


def parse_float(value: object, *, default: float, name: str, minimum: float | None = None) -> float:
    output = default if value is None else float(value)
    if minimum is not None and output < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return output


def parse_int_list(value: object, *, default: list[int], name: str) -> list[int]:
    if value is None:
        values = list(default)
    elif isinstance(value, str):
        values = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    elif isinstance(value, list):
        values = [int(item) for item in value]
    else:
        raise ValueError(f"{name} must be a comma-separated string or list of integers")
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    return values


def parse_choice(value: object, *, default: str, name: str, choices: set[str]) -> str:
    output = default if value in (None, "") else str(value)
    if output not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return output


def parse_str_list(
    value: object,
    *,
    default: list[str],
    name: str,
    choices: set[str] | None = None,
) -> list[str]:
    if value is None:
        values = list(default)
    elif isinstance(value, str):
        values = [piece.strip() for piece in value.split(",") if piece.strip()]
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        raise ValueError(f"{name} must be a comma-separated string or list of strings")
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if choices is not None:
        invalid = sorted(set(values) - choices)
        if invalid:
            allowed = ", ".join(sorted(choices))
            raise ValueError(f"{name} contains invalid values {invalid}; allowed: {allowed}")
    return values


def repo_relative_path(value: object, *, default: Path, name: str) -> Path:
    path = default if value in (None, "") else Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{name} must be relative to the repo")
    if ".." in path.parts:
        raise ValueError(f"{name} cannot contain '..'")
    return path


def run_checked(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    append_log(log_path, "$ " + " ".join(command))
    with log_path.open("ab") as handle:
        subprocess.run(command, cwd=cwd, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)


def metadata_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        metadata = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def setup_cache_is_current(repo_dir: Path, item: dict[str, object]) -> bool:
    skip_path = item.get("skip_if_exists")
    if not skip_path:
        return False
    cache_path = repo_dir / Path(str(skip_path))
    if not cache_path.exists():
        return False
    expected = item.get("expected_metadata")
    if not isinstance(expected, dict):
        return True
    return metadata_matches(cache_path.with_suffix(".json"), expected)


def broad_scaling_config(payload: dict[str, object]) -> dict[str, object]:
    output_root = repo_relative_path(payload.get("output_root"), default=DEFAULT_OUTPUT_ROOT, name="output_root")
    validation_cache = repo_relative_path(
        payload.get("validation_cache"),
        default=DEFAULT_VALIDATION_CACHE,
        name="validation_cache",
    )
    panel_cache = repo_relative_path(
        payload.get("panel_marginal_cache"),
        default=DEFAULT_PANEL_CACHE,
        name="panel_marginal_cache",
    )
    return {
        "run_name": safe_run_name(payload.get("run_name")),
        "output_root": output_root,
        "train_simulations": parse_int_list(
            payload.get("train_simulations"),
            default=DEFAULT_TRAIN_SIMULATIONS,
            name="train_simulations",
        ),
        "seeds": parse_int_list(payload.get("seeds"), default=DEFAULT_SEEDS, name="seeds"),
        "family": parse_choice(payload.get("family"), default="mdn", name="family", choices=FAMILY_CHOICES),
        "device": parse_choice(payload.get("device"), default="cpu", name="device", choices=DEVICE_CHOICES),
        "standardization_simulations": parse_int(
            payload.get("standardization_simulations"),
            default=60_000,
            name="standardization_simulations",
        ),
        "train_sampler": parse_choice(
            payload.get("train_sampler"),
            default="random",
            name="train_sampler",
            choices=TRAIN_SAMPLER_CHOICES,
        ),
        "epochs": parse_int(payload.get("epochs"), default=90, name="epochs"),
        "batch_size": parse_int(payload.get("batch_size"), default=512, name="batch_size"),
        "learning_rate": parse_float(payload.get("learning_rate"), default=2e-3, name="learning_rate", minimum=0.0),
        "lr_schedule": parse_choice(
            payload.get("lr_schedule"),
            default="constant",
            name="lr_schedule",
            choices=LR_SCHEDULE_CHOICES,
        ),
        "lr_eta_min": parse_float(payload.get("lr_eta_min"), default=0.0, name="lr_eta_min", minimum=0.0),
        "lr_warmup_steps": parse_int(
            payload.get("lr_warmup_steps"),
            default=0,
            name="lr_warmup_steps",
            minimum=0,
        ),
        "validation_every_epochs": parse_int(
            payload.get("validation_every_epochs"),
            default=1,
            name="validation_every_epochs",
        ),
        "max_optimizer_steps": parse_int(
            payload.get("max_optimizer_steps"),
            default=0,
            name="max_optimizer_steps",
            minimum=0,
        ),
        "torch_compile": parse_choice(
            payload.get("torch_compile"),
            default="none",
            name="torch_compile",
            choices=TORCH_COMPILE_CHOICES,
        ),
        "grad_clip_norm": parse_float(
            payload.get("grad_clip_norm"),
            default=20.0,
            name="grad_clip_norm",
            minimum=0.0,
        ),
        "ema_decay": parse_float(
            payload.get("ema_decay"),
            default=0.0,
            name="ema_decay",
            minimum=0.0,
        ),
        "batching_mode": parse_choice(
            payload.get("batching_mode"),
            default="dataloader",
            name="batching_mode",
            choices=BATCHING_MODE_CHOICES,
        ),
        "weight_decay": parse_float(payload.get("weight_decay"), default=1e-5, name="weight_decay", minimum=0.0),
        "hidden_dim": parse_int(payload.get("hidden_dim"), default=128, name="hidden_dim"),
        "hidden_layers": parse_int(payload.get("hidden_layers"), default=3, name="hidden_layers"),
        "mdn_components": parse_int(payload.get("mdn_components"), default=5, name="mdn_components"),
        "flow_layers": parse_int(payload.get("flow_layers"), default=6, name="flow_layers"),
        "flow_context_dim": parse_int(payload.get("flow_context_dim"), default=64, name="flow_context_dim"),
        "spline_bins": parse_int(payload.get("spline_bins"), default=12, name="spline_bins", minimum=2),
        "context_features": parse_choice(
            payload.get("context_features"),
            default="raw",
            name="context_features",
            choices=CONTEXT_FEATURE_CHOICES,
        ),
        "jobs": parse_int(payload.get("jobs"), default=2, name="jobs"),
        "torch_threads": parse_int(payload.get("torch_threads"), default=2, name="torch_threads"),
        "eval_batch_size": parse_int(payload.get("eval_batch_size"), default=16_384, name="eval_batch_size"),
        "early_stop_val_simulations": parse_int(
            payload.get("early_stop_val_simulations"),
            default=100_000,
            name="early_stop_val_simulations",
        ),
        "validation_cache": validation_cache,
        "validation_cache_simulations": parse_int(
            payload.get("validation_cache_simulations"),
            default=1_000_000,
            name="validation_cache_simulations",
        ),
        "early_val_cache_simulations": parse_int(
            payload.get("early_val_cache_simulations"),
            default=0,
            name="early_val_cache_simulations",
            minimum=0,
        ),
        "panel_marginal_cache": panel_cache,
        "panel_size": parse_int(payload.get("panel_size"), default=16, name="panel_size"),
        "panel_grid_size": parse_int(payload.get("panel_grid_size"), default=180, name="panel_grid_size", minimum=2),
        "panel_target_sample_count": parse_int(
            payload.get("panel_target_sample_count"),
            default=20_000,
            name="panel_target_sample_count",
        ),
        "panel_posterior_samples": parse_int(
            payload.get("panel_posterior_samples"),
            default=20_000,
            name="panel_posterior_samples",
        ),
        "posterior_samples": parse_int(
            payload.get("posterior_samples"),
            default=20_000,
            name="posterior_samples",
        ),
        "context_variants": parse_str_list(
            payload.get("context_variants"),
            default=["real"],
            name="context_variants",
            choices=CONTEXT_VARIANT_CHOICES,
        ),
        "tail_top_k": parse_int(payload.get("tail_top_k"), default=20, name="tail_top_k", minimum=0),
        "prepare_caches": parse_bool(payload.get("prepare_caches"), default=True, name="prepare_caches"),
        "save_models": parse_bool(payload.get("save_models"), default=True, name="save_models"),
        "sync": parse_bool(payload.get("sync"), default=True, name="sync"),
        "dry_run": parse_bool(payload.get("dry_run"), default=False, name="dry_run"),
    }


def broad_scaling_commands(config: dict[str, object], *, uv: str) -> tuple[list[dict[str, object]], list[str]]:
    validation_cache = Path(str(config["validation_cache"]))
    panel_cache = Path(str(config["panel_marginal_cache"]))
    setup_commands: list[dict[str, object]] = []
    if config["prepare_caches"]:
        setup_commands = [
            {
                "name": "validation_cache",
                "skip_if_exists": validation_cache,
                "expected_metadata": {
                    "simulations": int(config["validation_cache_simulations"]),
                    "seed": DEFAULT_VALIDATION_SEED,
                    "n_observations": DEFAULT_VALIDATION_N_OBSERVATIONS,
                    "dtype": DEFAULT_VALIDATION_DTYPE,
                },
                "command": [
                    uv,
                    "run",
                    "scripts/cache_decay_broad_validation.py",
                    "--output",
                    str(validation_cache),
                    "--simulations",
                    str(config["validation_cache_simulations"]),
                    "--force",
                ],
            },
            {
                "name": "panel_marginal_cache",
                "skip_if_exists": panel_cache,
                "expected_metadata": {
                    "panel_size": int(config["panel_size"]),
                    "prior_panel_size": int(config["panel_size"]),
                    "include_x0": False,
                    "panel_seed": DEFAULT_PANEL_SEED,
                    "grid_size": int(config["panel_grid_size"]),
                    "target_sample_count": int(config["panel_target_sample_count"]),
                    "target_repeats": DEFAULT_PANEL_TARGET_REPEATS,
                },
                "command": [
                    uv,
                    "run",
                    "scripts/cache_decay_panel_marginals.py",
                    "--output",
                    str(panel_cache),
                    "--panel-size",
                    str(config["panel_size"]),
                    "--grid-size",
                    str(config["panel_grid_size"]),
                    "--target-sample-count",
                    str(config["panel_target_sample_count"]),
                    "--force",
                ],
            },
        ]

    command = [
        uv,
        "run",
        "scripts/decay_broad_scaling_sweep.py",
        "--preset",
        "pilot",
        "--output-root",
        str(config["output_root"]),
        "--train-simulations",
        ",".join(str(value) for value in config["train_simulations"]),
        "--seeds",
        ",".join(str(value) for value in config["seeds"]),
        "--family",
        str(config["family"]),
        "--val-simulations",
        str(config["early_stop_val_simulations"]),
        "--standardization-simulations",
        str(config["standardization_simulations"]),
        "--train-sampler",
        str(config["train_sampler"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--lr-schedule",
        str(config["lr_schedule"]),
        "--lr-eta-min",
        str(config["lr_eta_min"]),
        "--lr-warmup-steps",
        str(config["lr_warmup_steps"]),
        "--validation-every-epochs",
        str(config["validation_every_epochs"]),
        "--max-optimizer-steps",
        str(config["max_optimizer_steps"]),
        "--torch-compile",
        str(config["torch_compile"]),
        "--grad-clip-norm",
        str(config["grad_clip_norm"]),
        "--ema-decay",
        str(config["ema_decay"]),
        "--batching-mode",
        str(config["batching_mode"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--hidden-layers",
        str(config["hidden_layers"]),
        "--mdn-components",
        str(config["mdn_components"]),
        "--flow-layers",
        str(config["flow_layers"]),
        "--flow-context-dim",
        str(config["flow_context_dim"]),
        "--spline-bins",
        str(config["spline_bins"]),
        "--context-features",
        str(config["context_features"]),
        "--context-variants",
        ",".join(str(value) for value in config["context_variants"]),
        "--posterior-samples",
        str(config["posterior_samples"]),
        "--device",
        str(config["device"]),
        "--validation-cache",
        str(validation_cache),
        "--early-val-cache-simulations",
        str(config["early_val_cache_simulations"]),
        "--panel-marginal-cache",
        str(panel_cache),
        "--panel-posterior-samples",
        str(config["panel_posterior_samples"]),
        "--skip-x0-reference",
        "--skip-existing",
        "--jobs",
        str(config["jobs"]),
        "--torch-threads",
        str(config["torch_threads"]),
        "--eval-batch-size",
        str(config["eval_batch_size"]),
        "--tail-top-k",
        str(config["tail_top_k"]),
    ]
    if not config["save_models"]:
        command.append("--no-save-models")
    if config["dry_run"]:
        command.append("--dry-run")
    return setup_commands, command


def build_broad_scaling_spec(payload: dict[str, object], *, repo_dir: Path, uv: str) -> dict[str, object]:
    config = broad_scaling_config(payload)
    setup_commands, command = broad_scaling_commands(config, uv=uv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['run_name']}_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = repo_dir / "logs" / "train_remote" / run_id
    return {
        "run_id": run_id,
        "run_name": config["run_name"],
        "job_type": "broad_scaling",
        "created_at": utc_now(),
        "repo_dir": repo_dir,
        "uv": uv,
        "runtime": runtime_metadata(repo_dir),
        "config": config,
        "sync": config["sync"],
        "setup_commands": setup_commands,
        "command": command,
        "output_root": config["output_root"],
        "run_dir": run_dir,
        "spec_path": run_dir / "spec.json",
        "status_path": run_dir / "status.json",
        "log_path": run_dir / "train.log",
    }


def update_status(spec: dict[str, object], **updates: object) -> None:
    status_path = Path(str(spec["status_path"]))
    current = read_json(status_path) if status_path.exists() else {}
    current.update({
        "run_id": spec["run_id"],
        "run_name": spec["run_name"],
        "job_type": spec["job_type"],
        "created_at": spec["created_at"],
        "updated_at": utc_now(),
        "runtime": spec.get("runtime"),
        "log_path": spec["log_path"],
        "output_root": spec["output_root"],
    })
    current.update(updates)
    write_json(status_path, current)


def worker_main(spec_path: Path) -> int:
    spec = read_json(spec_path)
    repo_dir = Path(str(spec["repo_dir"]))
    log_path = Path(str(spec["log_path"]))
    env = os.environ.copy()
    update_status(spec, state="preparing", worker_pid=os.getpid(), worker_runtime=runtime_metadata(repo_dir))
    append_log(log_path, f"[{utc_now()}] train worker starting")
    try:
        if spec.get("sync", True):
            command = [str(spec["uv"]), "sync"]
            python_version = project_python_version(repo_dir)
            if python_version:
                command.extend(["--python", python_version])
            run_checked(command, cwd=repo_dir, log_path=log_path, env=env)

        for item in spec["setup_commands"]:
            skip_path = item.get("skip_if_exists")
            if setup_cache_is_current(repo_dir, item):
                append_log(log_path, f"[{utc_now()}] setup skip {item['name']}: {skip_path} metadata matches")
                continue
            if skip_path and (repo_dir / Path(str(skip_path))).exists():
                append_log(log_path, f"[{utc_now()}] setup refresh {item['name']}: {skip_path} metadata mismatch")
            update_status(spec, state="running_setup", active_setup=item["name"], worker_pid=os.getpid())
            run_checked([str(part) for part in item["command"]], cwd=repo_dir, log_path=log_path, env=env)

        update_status(spec, state="launching", worker_pid=os.getpid())
        append_log(log_path, f"[{utc_now()}] launching train command")
        handle = log_path.open("ab")
        process = subprocess.Popen(
            [str(part) for part in spec["command"]],
            cwd=repo_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        update_status(
            spec,
            state="running",
            worker_pid=os.getpid(),
            train_pid=process.pid,
            command=spec["command"],
        )
        append_log(log_path, f"[{utc_now()}] train_pid={process.pid}")
        return_code = process.wait()
        handle.close()
        if return_code == 0:
            update_status(spec, state="completed", worker_pid=os.getpid(), train_pid=process.pid, return_code=0)
            append_log(log_path, f"[{utc_now()}] completed return_code=0")
        else:
            update_status(
                spec,
                state="failed",
                worker_pid=os.getpid(),
                train_pid=process.pid,
                return_code=return_code,
            )
            append_log(log_path, f"[{utc_now()}] failed return_code={return_code}")
        return int(return_code)
    except subprocess.CalledProcessError as exc:
        update_status(spec, state="failed", worker_pid=os.getpid(), return_code=exc.returncode)
        append_log(log_path, f"[{utc_now()}] failed return_code={exc.returncode}")
        return int(exc.returncode)
    except Exception as exc:  # noqa: BLE001
        update_status(spec, state="failed", worker_pid=os.getpid(), error=repr(exc))
        append_log(log_path, f"[{utc_now()}] failed error={exc!r}")
        return 1


def enrich_status(status: dict[str, object]) -> dict[str, object]:
    output = dict(status)
    terminal = output.get("state") in {"completed", "failed"}
    output["worker_running"] = False if terminal else process_is_running(int(output.get("worker_pid") or 0))
    output["train_running"] = False if terminal else process_is_running(int(output.get("train_pid") or 0))
    return output


class TrainRemoteHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        repo_dir: Path,
        uv: str,
        token: str | None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.repo_dir = repo_dir
        self.uv = uv
        self.token = token


class Handler(BaseHTTPRequestHandler):
    server: TrainRemoteHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def authorized(self) -> bool:
        if not self.server.token:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.server.token}"

    def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(json_ready(payload), indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, status: HTTPStatus, text: str) -> None:
        body = text.encode("utf-8", errors="replace")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_payload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        pieces = [piece for piece in parsed.path.split("/") if piece]
        if parsed.path == "/health":
            uv_path = shutil.which(self.server.uv) or (
                self.server.uv if Path(self.server.uv).exists() else None
            )
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "repo_dir": self.server.repo_dir,
                    "uv": self.server.uv,
                    "uv_path": uv_path,
                    "runtime": runtime_metadata(self.server.repo_dir),
                    "token_required": bool(self.server.token),
                    "train_endpoints": ["/train/broad-scaling"],
                },
            )
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if parsed.path == "/train":
            run_root = self.server.repo_dir / "logs" / "train_remote"
            runs = [
                enrich_status(read_json(path))
                for path in sorted(run_root.glob("*/status.json"), reverse=True)
            ]
            self.send_json(HTTPStatus.OK, {"runs": runs})
            return
        if len(pieces) >= 2 and pieces[0] == "train":
            run_id = pieces[1]
            run_dir = self.server.repo_dir / "logs" / "train_remote" / run_id
            status_path = run_dir / "status.json"
            if not status_path.exists():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "unknown run_id", "run_id": run_id})
                return
            if len(pieces) == 2:
                self.send_json(HTTPStatus.OK, enrich_status(read_json(status_path)))
                return
            if len(pieces) == 3 and pieces[2] == "log":
                query = parse_qs(parsed.query)
                tail = int(query.get("tail", ["120"])[0])
                log_path = run_dir / "train.log"
                if not log_path.exists():
                    self.send_text(HTTPStatus.NOT_FOUND, "")
                    return
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                self.send_text(HTTPStatus.OK, "\n".join(lines[-tail:]) + ("\n" if lines else ""))
                return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if parsed.path != "/train/broad-scaling":
            self.send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not found", "supported_post_endpoints": ["/train/broad-scaling"]},
            )
            return
        try:
            spec = build_broad_scaling_spec(self.read_payload(), repo_dir=self.server.repo_dir, uv=self.server.uv)
            spec_path = Path(str(spec["spec_path"]))
            write_json(spec_path, spec)
            update_status(spec, state="queued")
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)],
                cwd=self.server.repo_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            update_status(spec, state="queued", worker_pid=worker.pid)
            self.send_json(
                HTTPStatus.ACCEPTED,
                {
                    "status": "accepted",
                    "run_id": spec["run_id"],
                    "worker_pid": worker.pid,
                    "status_url": f"/train/{spec['run_id']}",
                    "log_url": f"/train/{spec['run_id']}/log",
                    "log_path": spec["log_path"],
                    "output_root": spec["output_root"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": repr(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP endpoint for launching allowlisted training jobs.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the train endpoint.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8877)
    serve.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    serve.add_argument("--uv", default=os.environ.get("UV", "uv"))
    serve.add_argument("--token", default=os.environ.get("TRAIN_REMOTE_TOKEN"))

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--spec", type=Path, required=True)

    args = parser.parse_args()
    if args.command is None:
        args.command = "serve"
        args.host = "127.0.0.1"
        args.port = 8877
        args.repo_dir = DEFAULT_REPO_DIR
        args.uv = os.environ.get("UV", "uv")
        args.token = os.environ.get("TRAIN_REMOTE_TOKEN")
    return args


def serve_main(args: argparse.Namespace) -> int:
    repo_dir = args.repo_dir.resolve()
    if not repo_dir.exists():
        raise SystemExit(f"repo dir does not exist: {repo_dir}")
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    server = TrainRemoteHTTPServer(
        (args.host, int(args.port)),
        Handler,
        repo_dir=repo_dir,
        uv=str(args.uv),
        token=args.token,
    )
    print(f"serving train endpoint on http://{args.host}:{args.port} repo_dir={repo_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        raise SystemExit(worker_main(args.spec))
    raise SystemExit(serve_main(args))


if __name__ == "__main__":
    main()
