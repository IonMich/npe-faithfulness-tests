from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = "https://github.com/IonMich/npe-faithfulness-tests.git"
DEFAULT_REMOTE_DIR = Path("/Users/ioannism/repos/npe")
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/11_mdn_1m_macmini")
DEFAULT_VALIDATION_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/validation_cache/"
    "broad_prior_val_1m_float32.npz"
)
DEFAULT_PANEL_CACHE = Path(
    "runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/"
    "decay_panel16_grid180_marginals.npz"
)


def parse_csv_ints(value: str) -> list[int]:
    items = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def run_local(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def current_branch() -> str:
    result = run_local(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def remote_script(args: argparse.Namespace) -> str:
    remote_dir = str(args.remote_dir)
    branch = args.branch
    cache_commands: list[str] = []
    if args.prepare_caches:
        cache_commands.extend([
            (
                f"test -f {shlex.quote(str(args.validation_cache))} || "
                "uv run scripts/cache_decay_broad_validation.py "
                f"--output {shlex.quote(str(args.validation_cache))} "
                f"--simulations {int(args.validation_cache_simulations)}"
            ),
            (
                f"test -f {shlex.quote(str(args.panel_marginal_cache))} || "
                "uv run scripts/cache_decay_panel_marginals.py "
                f"--output {shlex.quote(str(args.panel_marginal_cache))} "
                f"--panel-size {int(args.panel_size)} "
                f"--grid-size {int(args.panel_grid_size)} "
                f"--target-sample-count {int(args.panel_target_sample_count)}"
            ),
        ])

    train_simulations = ",".join(str(value) for value in args.train_simulations)
    seeds = ",".join(str(value) for value in args.seeds)
    sweep_command = [
        "uv",
        "run",
        "scripts/decay_broad_scaling_sweep.py",
        "--preset",
        "pilot",
        "--output-root",
        str(args.output_root),
        "--train-simulations",
        train_simulations,
        "--seeds",
        seeds,
        "--val-simulations",
        str(args.early_stop_val_simulations),
        "--validation-cache",
        str(args.validation_cache),
        "--panel-marginal-cache",
        str(args.panel_marginal_cache),
        "--panel-posterior-samples",
        str(args.panel_posterior_samples),
        "--skip-x0-reference",
        "--skip-existing",
        "--jobs",
        str(args.jobs),
        "--torch-threads",
        str(args.torch_threads),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    if args.no_save_models:
        sweep_command.append("--no-save-models")

    setup_lines = [
        "set -euo pipefail",
        "mkdir -p /Users/ioannism/repos",
        f"if [ -d {shlex.quote(remote_dir)}/.git ]; then",
        f"  cd {shlex.quote(remote_dir)}",
        f"  git fetch origin {shlex.quote(branch)} --prune",
        f"  git checkout -B {shlex.quote(branch)} {shlex.quote(f'origin/{branch}')}",
        "else",
        f"  git clone --branch {shlex.quote(branch)} {shlex.quote(args.repo)} {shlex.quote(remote_dir)}",
        f"  cd {shlex.quote(remote_dir)}",
        "fi",
    ]
    if args.install_uv:
        setup_lines.extend([
            "if ! command -v uv >/dev/null 2>&1; then",
            "  curl -LsSf https://astral.sh/uv/install.sh | sh",
            "  export PATH=\"$HOME/.local/bin:$PATH\"",
            "fi",
        ])
    else:
        setup_lines.extend([
            "if ! command -v uv >/dev/null 2>&1; then",
            "  echo 'uv is not installed on the remote host. Re-run with --install-uv or install uv manually.' >&2",
            "  exit 12",
            "fi",
        ])
    setup_lines.append("uv sync")
    setup_lines.extend(cache_commands)

    run_lines = setup_lines
    if args.prepare_only:
        return "\n".join(run_lines)

    log_dir = args.remote_dir / "logs" / "macmini_scaling"
    log_path = log_dir / f"{args.run_name}.log"
    run_lines.append(f"mkdir -p {shlex.quote(str(log_dir))}")
    if args.detach:
        run_lines.extend([
            f"nohup {quote_command(sweep_command)} > {shlex.quote(str(log_path))} 2>&1 &",
            "pid=$!",
            f"echo {shlex.quote(json.dumps({'status': 'started', 'log': str(log_path)}))}",
            "echo pid=$pid",
        ])
    else:
        run_lines.append(quote_command(sweep_command))
    return "\n".join(run_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Mac mini and launch a broad NPE scaling sweep over SSH.",
    )
    parser.add_argument("--host", default="mini")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--remote-dir", type=Path, default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="broad_mdn_1m")
    parser.add_argument("--train-simulations", type=parse_csv_ints, default=parse_csv_ints("64000,128000,256000,512000,1000000"))
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("20260901,20260902,20260903"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--early-stop-val-simulations", type=int, default=100000)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--validation-cache-simulations", type=int, default=1_000_000)
    parser.add_argument("--panel-marginal-cache", type=Path, default=DEFAULT_PANEL_CACHE)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--panel-grid-size", type=int, default=180)
    parser.add_argument("--panel-target-sample-count", type=int, default=20_000)
    parser.add_argument("--panel-posterior-samples", type=int, default=20_000)
    parser.add_argument("--no-prepare-caches", dest="prepare_caches", action="store_false")
    parser.set_defaults(prepare_caches=True)
    parser.add_argument("--install-uv", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", dest="detach", action="store_false")
    parser.set_defaults(detach=True)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.branch is None:
        args.branch = current_branch()
    script = remote_script(args)
    if args.dry_run:
        print(script)
        return
    command = ["ssh", args.host, "bash", "-lc", script]
    try:
        result = subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
