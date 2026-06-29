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
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
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
    name = "train" if value in (None, "") else str(value)
    return RUN_NAME_RE.sub("_", name).strip("._-") or "train"


def safe_branch(value: object) -> str | None:
    if value in (None, ""):
        return None
    branch = str(value)
    if not BRANCH_RE.fullmatch(branch):
        raise ValueError(f"Unsafe branch name: {branch!r}")
    return branch


def require_command(value: object, *, uv: str, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of command arguments")
    command = [str(part) for part in value]
    if command[0] == "uv":
        command[0] = uv
    if command[0] != uv:
        raise ValueError(f"{name} must start with 'uv' or the configured uv path")
    if len(command) < 4 or command[1] != "run" or not command[2].startswith("scripts/"):
        raise ValueError(f"{name} must be a repo-local 'uv run scripts/...' command")
    return command


def require_setup_commands(value: object, *, uv: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("setup_commands must be a list")
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"setup_commands[{index}] must be an object")
        output.append({
            "name": str(item.get("name") or f"setup_{index}"),
            "skip_if_exists": str(item["skip_if_exists"]) if item.get("skip_if_exists") else None,
            "command": require_command(item.get("command"), uv=uv, name=f"setup_commands[{index}].command"),
        })
    return output


def require_env(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("env must be an object")
    return {str(key): str(item) for key, item in value.items()}


def run_checked(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    append_log(log_path, "$ " + " ".join(command))
    with log_path.open("ab") as handle:
        subprocess.run(command, cwd=cwd, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)


def build_train_spec(payload: dict[str, object], *, repo_dir: Path, uv: str) -> dict[str, object]:
    run_name = safe_run_name(payload.get("run_name"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name}_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = repo_dir / "logs" / "train_remote" / run_id
    command = require_command(payload.get("command"), uv=uv, name="command")
    setup_commands = require_setup_commands(payload.get("setup_commands"), uv=uv)
    env = require_env(payload.get("env"))
    output_root = payload.get("output_root")
    return {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": utc_now(),
        "repo_dir": repo_dir,
        "uv": uv,
        "branch": safe_branch(payload.get("branch")),
        "sync": bool(payload.get("sync", True)),
        "setup_commands": setup_commands,
        "command": command,
        "env": env,
        "output_root": str(output_root) if output_root is not None else None,
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
        "created_at": spec["created_at"],
        "updated_at": utc_now(),
        "branch": spec.get("branch"),
        "log_path": spec["log_path"],
        "output_root": spec.get("output_root"),
    })
    current.update(updates)
    write_json(status_path, current)


def worker_main(spec_path: Path) -> int:
    spec = read_json(spec_path)
    repo_dir = Path(str(spec["repo_dir"]))
    log_path = Path(str(spec["log_path"]))
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in dict(spec.get("env") or {}).items()})
    update_status(spec, state="preparing", worker_pid=os.getpid())
    append_log(log_path, f"[{utc_now()}] train worker starting")
    try:
        branch = spec.get("branch")
        if branch:
            run_checked(["git", "fetch", "origin", str(branch), "--prune"], cwd=repo_dir, log_path=log_path, env=env)
            run_checked(["git", "checkout", "-B", str(branch), f"origin/{branch}"], cwd=repo_dir, log_path=log_path, env=env)

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
        if parsed.path != "/train":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            spec = build_train_spec(self.read_payload(), repo_dir=self.server.repo_dir, uv=self.server.uv)
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
                    "output_root": spec.get("output_root"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": repr(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP endpoint for launching repo-local training commands.")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the train endpoint.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    serve.add_argument("--uv", default=os.environ.get("UV", "uv"))
    serve.add_argument("--token", default=os.environ.get("TRAIN_REMOTE_TOKEN"))

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--spec", type=Path, required=True)

    args = parser.parse_args()
    if args.command is None:
        args.command = "serve"
        args.host = "127.0.0.1"
        args.port = 8765
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
