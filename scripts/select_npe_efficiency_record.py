from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("runs/01_exponential_decay/15_broad_scaling")
DEFAULT_TARGET_NLL = -3.6058692668472965
DEFAULT_TARGET_SECONDS = 387.8054779791273


@dataclass(frozen=True)
class Candidate:
    kind: str
    summary_path: Path
    train_simulations: int
    full_val_nll_z_units: float
    best_val_nll_z_units: float | None
    training_seconds: float
    optimizer_steps: int | None
    epochs_completed: int | None
    model_pt: str | None
    model_exists: bool
    model_bytes: int | None
    saved_model_count: int


def resolve_model_path(repo_root: Path, model_pt: str | None) -> Path | None:
    if not model_pt:
        return None
    path = Path(model_pt)
    if path.is_absolute():
        return path
    return repo_root / path


def load_candidate(summary_path: Path, repo_root: Path) -> Candidate | None:
    try:
        data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("kind") == "ensemble" or "ensemble_full_val_nll_z_units" in data:
        try:
            full_val_nll = float(data.get("full_val_nll_z_units", data["ensemble_full_val_nll_z_units"]))
            training_seconds = float(data["training_wall_seconds"])
        except Exception:
            return None
        model_paths = [
            resolve_model_path(repo_root, str(path))
            for path in data.get("model_paths", [])
        ]
        existing_model_paths = [path for path in model_paths if path is not None and path.exists()]
        model_exists = bool(model_paths) and len(existing_model_paths) == len(model_paths)
        model_bytes = (
            sum(path.stat().st_size for path in existing_model_paths)
            if model_exists
            else None
        )
        return Candidate(
            kind="ensemble",
            summary_path=summary_path,
            train_simulations=int(data.get("train_simulations", -1)),
            full_val_nll_z_units=full_val_nll,
            best_val_nll_z_units=None,
            training_seconds=training_seconds,
            optimizer_steps=None,
            epochs_completed=None,
            model_pt=None,
            model_exists=model_exists,
            model_bytes=model_bytes,
            saved_model_count=len(model_paths),
        )

    try:
        full_val_nll = float(data["full_val_nll_z_units"])
        training_seconds = float(data["training_seconds"])
    except Exception:
        return None

    model_pt = data.get("model_pt")
    model_path = resolve_model_path(repo_root, model_pt)
    model_exists = bool(model_path and model_path.exists())
    model_bytes = model_path.stat().st_size if model_exists and model_path else None
    return Candidate(
        kind="single",
        summary_path=summary_path,
        train_simulations=int(data.get("train_simulations", -1)),
        full_val_nll_z_units=full_val_nll,
        best_val_nll_z_units=(
            float(data["best_val_nll_z_units"])
            if data.get("best_val_nll_z_units") is not None
            else None
        ),
        training_seconds=training_seconds,
        optimizer_steps=(
            int(data["optimizer_steps"]) if data.get("optimizer_steps") is not None else None
        ),
        epochs_completed=(
            int(data["epochs_completed"]) if data.get("epochs_completed") is not None else None
        ),
        model_pt=str(model_pt) if model_pt else None,
        model_exists=model_exists,
        model_bytes=model_bytes,
        saved_model_count=1 if model_exists else 0,
    )


def find_candidates(root: Path, repo_root: Path) -> list[Candidate]:
    return [
        candidate
        for path in sorted(
            [
                *root.glob("**/results/broad_scaling_run_summary.json"),
                *root.glob("**/results/ensemble*_proof_summary.json"),
            ]
        )
        if (candidate := load_candidate(path, repo_root)) is not None
    ]


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "kind": candidate.kind,
        "summary_path": str(candidate.summary_path),
        "train_simulations": candidate.train_simulations,
        "full_val_nll_z_units": candidate.full_val_nll_z_units,
        "best_val_nll_z_units": candidate.best_val_nll_z_units,
        "training_seconds": candidate.training_seconds,
        "optimizer_steps": candidate.optimizer_steps,
        "epochs_completed": candidate.epochs_completed,
        "model_pt": candidate.model_pt,
        "model_exists": candidate.model_exists,
        "model_bytes": candidate.model_bytes,
        "saved_model_count": candidate.saved_model_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the fastest NPE run that satisfies an NLL/time target. "
            "Use --require-saved-model for final proof claims."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--target-nll", type=float, default=DEFAULT_TARGET_NLL)
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument("--require-saved-model", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    root = args.root if args.root.is_absolute() else repo_root / args.root
    candidates = find_candidates(root, repo_root)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.full_val_nll_z_units <= args.target_nll
        and candidate.training_seconds <= args.target_seconds
        and (candidate.model_exists or not args.require_saved_model)
    ]
    eligible.sort(key=lambda candidate: (candidate.training_seconds, candidate.full_val_nll_z_units))

    if args.as_json:
        print(
            json.dumps(
                {
                    "target_nll": args.target_nll,
                    "target_seconds": args.target_seconds,
                    "require_saved_model": bool(args.require_saved_model),
                    "eligible_count": len(eligible),
                    "best": candidate_to_dict(eligible[0]) if eligible else None,
                    "top": [candidate_to_dict(candidate) for candidate in eligible[: args.top]],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"target_nll <= {args.target_nll}")
        print(f"target_seconds <= {args.target_seconds}")
        print(f"require_saved_model = {bool(args.require_saved_model)}")
        print(f"eligible_count = {len(eligible)}")
        for index, candidate in enumerate(eligible[: args.top], start=1):
            marker = "BEST" if index == 1 else f"#{index}"
            print(
                "\t".join(
                    [
                        marker,
                        f"kind={candidate.kind}",
                        f"seconds={candidate.training_seconds:.12g}",
                        f"full_nll={candidate.full_val_nll_z_units:.15g}",
                        f"train={candidate.train_simulations}",
                        f"steps={candidate.optimizer_steps}",
                        f"model_exists={candidate.model_exists}",
                        f"saved_models={candidate.saved_model_count}",
                        str(candidate.summary_path),
                    ]
                )
            )

    if not eligible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
