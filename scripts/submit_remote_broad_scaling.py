from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib import request


DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
DEFAULT_OUTPUT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling/11_mdn_1m_remote")
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


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    train_simulations = ",".join(str(value) for value in args.train_simulations)
    seeds = ",".join(str(value) for value in args.seeds)
    command = [
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
        command.append("--no-save-models")

    setup_commands = []
    if args.prepare_caches:
        setup_commands.extend([
            {
                "name": "validation_cache",
                "skip_if_exists": str(args.validation_cache),
                "command": [
                    "uv",
                    "run",
                    "scripts/cache_decay_broad_validation.py",
                    "--output",
                    str(args.validation_cache),
                    "--simulations",
                    str(args.validation_cache_simulations),
                ],
            },
            {
                "name": "panel_marginal_cache",
                "skip_if_exists": str(args.panel_marginal_cache),
                "command": [
                    "uv",
                    "run",
                    "scripts/cache_decay_panel_marginals.py",
                    "--output",
                    str(args.panel_marginal_cache),
                    "--panel-size",
                    str(args.panel_size),
                    "--grid-size",
                    str(args.panel_grid_size),
                    "--target-sample-count",
                    str(args.panel_target_sample_count),
                ],
            },
        ])

    env = {}
    if args.path_prefix:
        env["PATH"] = f"{args.path_prefix}:{os.environ.get('PATH', '')}"

    return {
        "run_name": args.run_name,
        "branch": args.branch,
        "sync": not args.no_sync,
        "setup_commands": setup_commands,
        "command": command,
        "env": env,
        "output_root": str(args.output_root),
    }


def post_json(endpoint: str, token: str | None, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(endpoint.rstrip("/") + "/train", data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a broad scaling-law training request to a remote /train endpoint.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--token", default=os.environ.get("TRAIN_REMOTE_TOKEN"))
    parser.add_argument("--branch", default=None)
    parser.add_argument("--run-name", default="broad_mdn_1m")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Optional PATH prefix for the remote training process, e.g. /Users/ioannism/.local/bin.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.branch is None:
        args.branch = current_branch()
    return args


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    result = post_json(args.endpoint, args.token, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
