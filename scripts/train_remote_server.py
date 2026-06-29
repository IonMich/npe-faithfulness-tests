from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
)
DEFAULT_TRAIN_SIMULATIONS = [64_000, 128_000, 256_000, 512_000, 1_000_000]
DEFAULT_SEEDS = [20260901, 20260902, 20260903]
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
                "command": [
                    uv,
                    "run",
                    "scripts/cache_decay_broad_validation.py",
                    "--output",
                    str(validation_cache),
                    "--simulations",
                    str(config["validation_cache_simulations"]),
                ],
            },
            {
                "name": "panel_marginal_cache",
                "skip_if_exists": panel_cache,
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
        "--val-simulations",
        str(config["early_stop_val_simulations"]),
        "--validation-cache",
        str(validation_cache),
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
    update_status(spec, state="preparing", worker_pid=os.getpid())
    append_log(log_path, f"[{utc_now()}] train worker starting")
    try:
        if spec.get("sync", True):
            run_checked([str(spec["uv"]), "sync"], cwd=repo_dir, log_path=log_path, env=env)

        for item in spec["setup_commands"]:
            skip_path = item.get("skip_if_exists")
            if skip_path and (repo_dir / Path(str(skip_path))).exists():
                append_log(log_path, f"[{utc_now()}] setup skip {item['name']}: {skip_path} exists")
                continue
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
