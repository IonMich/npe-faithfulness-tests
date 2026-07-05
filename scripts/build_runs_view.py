from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(".")
NOTES = ROOT / "notes"
SCRIPTS = ROOT / "scripts"
RUNS = ROOT / "runs"

STRESS_GROUPS = {
    "02_stress_sign",
    "03_stress_banana",
    "04_stress_label_switch",
    "05_stress_linear6",
    "06_two_exponential",
}


GROUP_TITLES = {
    "01_exponential_decay": "Exponential Decay",
    "02_stress_sign": "Stress Test: Sign",
    "03_stress_banana": "Stress Test: Banana",
    "04_stress_label_switch": "Stress Test: Label Switching",
    "05_stress_linear6": "Stress Test: Linear 6D",
    "06_two_exponential": "Two-Exponential",
}

SCRIPT_BY_GROUP = {
    "01_exponential_decay": "compare_decay_samplers.py",
    "02_stress_sign": "npe_flow_stress_tests.py",
    "03_stress_banana": "npe_flow_stress_tests.py",
    "04_stress_label_switch": "npe_flow_stress_tests.py",
    "05_stress_linear6": "npe_flow_stress_tests.py",
    "06_two_exponential": "npe_flow_stress_tests.py",
}


@dataclass
class Run:
    group: str
    run_rel: Path
    summary_path: Path
    result_dir: Path
    figure_dir: Path | None
    status: str
    score: float | None
    target: float | None
    metric_label: str
    reason: str
    script: Path | None
    notes: list[Path]

    @property
    def run_dir(self) -> Path:
        return RUNS / self.group / self.run_rel

    @property
    def name(self) -> str:
        parts = [p for p in self.run_rel.parts if p]
        return " / ".join(parts)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def get_path(data: Any, dotted: str) -> Any:
    current = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def iter_dicts(data: Any, prefix: str = ""):
    if isinstance(data, dict):
        yield prefix, data
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else key
            yield from iter_dicts(value, child)
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            yield from iter_dicts(value, f"{prefix}[{idx}]")


def iter_scalars(data: Any, prefix: str = ""):
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else key
            yield from iter_scalars(value, child)
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            yield from iter_scalars(value, f"{prefix}[{idx}]")
    else:
        yield prefix, data


def find_note_candidates(run_rel: Path, summary_path: Path) -> list[Path]:
    notes: list[Path] = []
    rel_text = "_".join(run_rel.parts)
    stem = rel_text
    if stem.startswith("01_npe_flow_stress_tests_"):
        stem = stem.removeprefix("01_")
    elif stem.startswith(tuple(f"{i:02d}_" for i in range(2, 20))):
        stem = stem[3:]
    stem_dash = stem.replace("_", "-")

    candidates = [
        NOTES / f"{stem_dash}.md",
        NOTES / f"npe-flow-stress-test-{stem_dash.removeprefix('npe-flow-stress-tests-')}.md",
    ]
    if "two-exp-ordered" in stem_dash:
        cleaned = stem_dash.replace("npe-flow-stress-tests-", "npe-flow-stress-test-")
        candidates.append(NOTES / f"{cleaned}.md")
    if "npe-flow-local-q0005-linear-150k-t8-seed20260706" in stem_dash:
        candidates.append(NOTES / "npe-flow-target-pass-results.md")
    if "npe-stage1" in stem_dash or "npe-stage-1" in stem_dash:
        candidates.append(NOTES / "npe-stage-1-results.md")
    if "local-region" in stem_dash:
        candidates.append(NOTES / "npe-local-region-results.md")
    if "multi-x" in stem_dash:
        candidates.append(NOTES / "npe-multi-x-faithfulness-results.md")
    if "abc-faithfulness" in stem_dash:
        candidates.append(NOTES / "abc-faithfulness-repair-results.md")
    if "snpe" in stem_dash or "sbi" in stem_dash:
        candidates.append(NOTES / "snpe-sequential-results.md")

    summary_text = str(summary_path)
    for note in NOTES.glob("*.md"):
        try:
            if summary_text in note.read_text():
                candidates.append(note)
        except UnicodeDecodeError:
            pass

    seen = set()
    for note in candidates:
        if note.exists() and note not in seen:
            notes.append(note)
            seen.add(note)
    return notes[:4]


def script_for(group: str, run_rel: Path, summary_name: str) -> Path | None:
    run_text = "/".join(run_rel.parts)
    combined = f"{run_text}/{summary_name}"
    if group in {
        "02_stress_sign",
        "03_stress_banana",
        "04_stress_label_switch",
        "05_stress_linear6",
    }:
        return SCRIPTS / "npe_flow_stress_tests.py"
    if group == "06_two_exponential":
        if "02_sbi" in run_rel.parts or "sbi_two_exp" in summary_name:
            return SCRIPTS / "sbi_two_exp_ordered.py"
        return SCRIPTS / "npe_flow_stress_tests.py"
    if "mcmc_hmc_reference" in combined or "sampler_comparison" in summary_name:
        return SCRIPTS / "compare_decay_samplers.py"
    if "npe_flow" in combined:
        return SCRIPTS / "npe_flow_decay.py"
    if "abc_faithfulness" in combined:
        return SCRIPTS / "abc_faithfulness_decay.py"
    if "faithfulness_target" in combined:
        return SCRIPTS / "check_faithfulness_target.py"
    if "sbi_alternate" in summary_name:
        return SCRIPTS / "sbi_alternate_decay.py"
    if "sbi_two_exp" in summary_name:
        return SCRIPTS / "sbi_two_exp_ordered.py"
    if "snpe_sbi_summary" in summary_name:
        return SCRIPTS / "snpe_sbi_summary_decay.py"
    if "snpe_sbi" in summary_name:
        return SCRIPTS / "snpe_sbi_decay.py"
    if "snpe_sequential" in summary_name:
        return SCRIPTS / "snpe_sequential_decay.py"
    if "oracle" in summary_name:
        return SCRIPTS / "oracle_posterior_density_fit.py"
    if "npe_multi_x" in summary_name:
        return SCRIPTS / "evaluate_npe_multi_x.py"
    if "npe_local_region" in summary_name:
        return SCRIPTS / "npe_local_region_decay.py"
    if "npe_stage1" in combined:
        return SCRIPTS / "npe_stage1_decay.py"
    if "npe_summary" in combined:
        return SCRIPTS / "npe_summary_context_decay.py"
    script = SCRIPTS / SCRIPT_BY_GROUP.get(group, "")
    return script if script.exists() else None


def first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def is_inherited_pairwise_target(group: str, target: float | None) -> bool:
    return group in STRESS_GROUPS and target is not None and abs(target - 0.034) < 1e-12


def near_target(score: float | None, target: float | None) -> bool:
    return score is not None and target is not None and score <= target * 1.25


def target_text(target: float | None) -> str:
    return "uncalibrated" if target is None else f"{target:.4g}"


def derive_status(data: dict[str, Any], summary_path: Path, group: str) -> tuple[str, float | None, float | None, str, str]:
    target = first_float(
        get_path(data, "target_wasserstein"),
        get_path(data, "config.target_wasserstein"),
        get_path(data, "agreement_flags.target"),
        get_path(data, "config.agreement_target"),
    )

    calibrated_target = first_float(get_path(data, "recommended_targets.diagnostic_mean_normalized_wasserstein"))
    calibrated_score = first_float(get_path(data, "diagnostics.npe_to_grid.diagnostic.mean_normalized_wasserstein"))
    calibrated_pass = get_path(data, "target_checks.npe_passes_diagnostic_target")
    if calibrated_target is not None and calibrated_score is not None and calibrated_pass is not None:
        status = "grid-faithful" if calibrated_pass else "fail"
        if not calibrated_pass and near_target(calibrated_score, calibrated_target):
            status = "near"
        return (
            status,
            calibrated_score,
            calibrated_target,
            "NPE-to-grid diagnostic Wasserstein",
            "exact-grid calibrated MCMC/HMC faithfulness target",
        )

    agreement_met = get_path(data, "agreement_flags.diagnostic_target_met")
    agreement_score = first_float(get_path(data, "agreement_flags.max_mean_diagnostic_wasserstein"))
    if agreement_met is not None:
        if target is None:
            return (
                "diagnostic",
                agreement_score,
                None,
                "pairwise max diagnostic Wasserstein",
                "pairwise agreement only; no calibrated target",
            )
        if is_inherited_pairwise_target(group, target) and agreement_met:
            return (
                "legacy_pairwise_pass",
                agreement_score,
                target,
                "pairwise max diagnostic Wasserstein",
                "inherited 0.034 pairwise target; model-specific calibration pending or superseded",
            )
        status = "pass" if agreement_met else "fail"
        if not agreement_met and near_target(agreement_score, target):
            status = "near"
        return status, agreement_score, target, "max diagnostic Wasserstein", "MCMC, HMC, and NPE agreement target"

    target_pass = get_path(data, "target_pass")
    score = first_float(
        get_path(data, "faithfulness_to_grid_reference.mean_normalized_wasserstein.value"),
        get_path(data, "target_ratio"),
    )
    label = "mean normalized Wasserstein"
    if get_path(data, "target_ratio") == score:
        label = "target ratio"
    if target_pass is not None:
        if target is None:
            return "diagnostic", score, None, label, "target pass flag present but no calibrated target was recorded"
        status = "pass" if target_pass else "fail"
        ratio = first_float(get_path(data, "target_ratio"))
        if not target_pass and ratio is not None and ratio <= 1.25:
            status = "near"
        return status, score, target, label, "NPE target pass flag"

    best = get_path(data, "best_result")
    if isinstance(best, dict):
        faithful = best.get("faithful")
        best_score = first_float(get_path(best, "metrics.mean_normalized_wasserstein.value"))
        if target is None:
            return "diagnostic", best_score, None, "best mean normalized Wasserstein", "ABC result without calibrated target"
        status = "pass" if faithful else "fail"
        if not faithful and near_target(best_score, target):
            status = "near"
        return status, best_score, target, "best mean normalized Wasserstein", "ABC best-result faithfulness flag"

    final_flags = get_path(data, "final_agreement_flags")
    if isinstance(final_flags, dict):
        met = final_flags.get("diagnostic_target_met") or final_flags.get("target_met")
        final_score = first_float(final_flags.get("max_mean_diagnostic_wasserstein"))
        if met is not None:
            if target is None:
                return "diagnostic", final_score, None, "pairwise max diagnostic Wasserstein", "final agreement only; no calibrated target"
            if is_inherited_pairwise_target(group, target) and met:
                return (
                    "legacy_pairwise_pass",
                    final_score,
                    target,
                    "pairwise max diagnostic Wasserstein",
                    "inherited 0.034 pairwise target; model-specific calibration pending",
                )
            status = "pass" if met else "fail"
            if not met and near_target(final_score, target):
                status = "near"
            return status, final_score, target, "max diagnostic Wasserstein", "final agreement flag"

    bool_hits = [(path, value) for path, value in iter_scalars(data) if path.endswith(("target_pass", "target_met", "faithful")) and isinstance(value, bool)]
    if bool_hits:
        any_pass = any(value for _, value in bool_hits)
        scores = [
            float(value)
            for path, value in iter_scalars(data)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (
                "mean_normalized_wasserstein.value" in path
                or path.endswith("target_ratio")
                or path.endswith("agreement.display.mcmc_npe.mean")
                or path.endswith("agreement.display.hmc_npe.mean")
            )
        ]
        best_score = min(scores) if scores else None
        if target is None:
            return "diagnostic", best_score, None, "best discovered target metric", "nested target flag without calibrated target"
        if any_pass:
            return "pass", best_score, target, "best discovered target metric", "nested target flag"
        status = "fail"
        if near_target(best_score, target):
            status = "near"
        return status, best_score, target, "best discovered target metric", "nested target flags were false"

    convergence = get_path(data, "convergence_flags")
    if isinstance(convergence, dict):
        scalar_flags = [value for _, value in iter_scalars(convergence) if isinstance(value, bool)]
        if scalar_flags and all(scalar_flags):
            return "reference", None, target, "convergence", "sampler convergence reference"

    grid_scores = [
        float(value)
        for path, value in iter_scalars(data)
        if isinstance(value, (int, float)) and "mean_normalized_wasserstein.value" in path
    ]
    if grid_scores:
        best_score = min(grid_scores)
        status = "diagnostic_pass" if target is not None and best_score <= target else "diagnostic"
        return status, best_score, target, "best mean normalized Wasserstein", "diagnostic/reference metric"

    return "diagnostic", None, target, "summary", "no target metric found"


def relative_link_target(link: Path, target: Path) -> Path:
    return Path("../" * len(link.parent.relative_to(ROOT).parts)) / target


def replace_symlink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(relative_link_target(link, target))


def write_run_readme(run: Run) -> None:
    lines = [
        f"# {run.name}",
        "",
        f"Status: `{run.status}`",
        f"Reason: {run.reason}",
    ]
    if run.score is not None:
        lines.append(f"Metric: {run.metric_label} = `{run.score:.6g}`")
    if run.target is not None:
        lines.append(f"Target: `{run.target:.6g}`")
    lines.extend(
        [
            "",
            "Artifacts:",
            "- `results/` - result files and summary JSON for this run",
            "- `figures/` - plots for this run, when available",
        ]
    )
    if run.script:
        lines.append(f"- script: `{run.script.as_posix()}`")
    for note in run.notes:
        lines.append(f"- note: `{note.as_posix()}`")
    lines.append("")
    run.run_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


def collect_runs() -> list[Run]:
    runs: list[Run] = []
    for summary_path in sorted(RUNS.rglob("results/*summary.json")):
        if "00_successful_runs" in summary_path.parts:
            continue
        if (
            summary_path.name in {"mcmc_decay_summary.json", "hmc_decay_summary.json"}
            and (summary_path.parent / "sampler_comparison_summary.json").exists()
        ):
            continue
        if len(summary_path.parts) < 4:
            continue
        group = summary_path.parts[1]
        if group not in GROUP_TITLES:
            continue
        result_dir = summary_path.parent
        run_dir = result_dir.parent
        run_rel = run_dir.relative_to(RUNS / group)
        figure_dir = run_dir / "figures"
        data = load_json(summary_path)
        status, score, target, metric_label, reason = derive_status(data, summary_path, group)
        script = script_for(group, run_rel, summary_path.name)
        runs.append(
            Run(
                group=group,
                run_rel=run_rel,
                summary_path=summary_path,
                result_dir=result_dir,
                figure_dir=figure_dir if figure_dir.exists() else None,
                status=status,
                score=score,
                target=target,
                metric_label=metric_label,
                reason=reason,
                script=script,
                notes=find_note_candidates(run_rel, summary_path),
            )
        )
    return runs


def materialize_run(run: Run) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(run.run_dir.iterdir()):
        if stale.is_symlink() and (
            stale.name in {"results", "figures", "summary.json", "script.py", "note.md"}
            or (stale.name.startswith("note_") and stale.suffix == ".md")
        ):
            stale.unlink()
    write_run_readme(run)


def format_score(run: Run) -> str:
    if run.score is None:
        return ""
    return f"{run.score:.4g}"


def status_order(status: str) -> int:
    return {
        "grid-faithful": 0,
        "floor_pass": 1,
        "pass": 2,
        "diagnostic_pass": 3,
        "reference": 4,
        "near_floor": 5,
        "legacy_pairwise_pass": 6,
        "near": 7,
        "diagnostic": 8,
        "fail": 9,
    }.get(status, 99)


def link_text(path: Path, label: str | None = None) -> str:
    label = label or str(path)
    return f"[{label}]({path})"


def link_from(base_dir: Path, target: Path, label: str | None = None) -> str:
    label = label or str(target)
    if target.is_relative_to(base_dir):
        rel = target.relative_to(base_dir)
    else:
        rel = Path("..") / target.relative_to(RUNS)
    return f"[{label}]({rel})"


def write_indexes(runs: list[Run]) -> None:
    RUNS.mkdir(exist_ok=True)
    successful = [r for r in runs if r.status in {"grid-faithful", "floor_pass", "pass", "diagnostic_pass", "reference"}]
    target_pass = [r for r in runs if r.status in {"grid-faithful", "floor_pass", "pass"}]
    reference = [r for r in runs if r.status in {"reference", "diagnostic_pass"}]
    near = [r for r in runs if r.status == "near"]
    legacy_pairwise = [r for r in runs if r.status == "legacy_pairwise_pass"]

    success_dir = RUNS / "00_successful_runs"
    if success_dir.exists():
        shutil.rmtree(success_dir)
    success_dir.mkdir(parents=True, exist_ok=True)
    for run in successful:
        group_dir = success_dir / run.group
        group_dir.mkdir(parents=True, exist_ok=True)
        link_name = "__".join(run.run_rel.parts)
        replace_symlink(group_dir / link_name, run.run_dir)

    lines = [
        "# Runs",
        "",
        "This directory is the canonical artifact store for the project.",
        "Top-level numbered folders are statistical models. Method folders sit inside each model, and each run folder contains real `results/` and `figures/` directories plus a README with status, metric, script, and note references.",
        "The only symlinked convenience layer is `00_successful_runs/`, which points to canonical run folders that passed a calibrated target or serve as references.",
        "",
        "Model folders:",
        "- `01_exponential_decay`: original exponential-decay likelihood and all decay-focused methods.",
        "- `02_stress_sign`: sign-symmetry stress likelihood.",
        "- `03_stress_banana`: banana-shaped posterior stress likelihood.",
        "- `04_stress_label_switch`: label-switching stress likelihood.",
        "- `05_stress_linear6`: higher-dimensional linear-Gaussian stress likelihood.",
        "- `06_two_exponential`: ordered two-exponential likelihood and SBI variants.",
        "",
        "Status labels:",
        "- `grid-faithful`: NPE matched a model-specific exact/reference posterior target.",
        "- `pass`: calibrated target metric was met by the run summary.",
        "- `reference`: sampler/reference run with convergence or baseline agreement.",
        "- `diagnostic_pass`: diagnostic/reference metric met the target, but it is not a direct NPE success claim.",
        "- `floor_pass`: full-prior population NLL is statistically indistinguishable from the model-specific entropy floor under the documented comparison criterion.",
        "- `near_floor`: full-prior population NLL is close to the model-specific entropy floor, but the remaining gap is still resolved or otherwise not a strict floor hit.",
        "- `legacy_pairwise_pass`: passed an inherited pairwise agreement target, but has not been calibrated against a model-specific truth target.",
        "- `near`: missed the target but stayed within 25% of it.",
        "- `fail`: explicit target metric was not met.",
        "- `diagnostic`: no direct target metric was found.",
        "",
        "Start here:",
        f"- {link_from(RUNS, RUNS / '00_successful_runs' / 'README.md', 'successful and reference runs')}",
        "",
        "Best run by model:",
        "",
        "| Model | Best status | Run | Metric |",
        "| --- | --- | --- | --- |",
    ]
    for group in sorted(GROUP_TITLES):
        group_runs = [r for r in runs if r.group == group]
        if not group_runs:
            continue
        best = sorted(group_runs, key=lambda r: (status_order(r.status), float("inf") if r.score is None else r.score))[0]
        lines.append(
            f"| `{group}` | `{best.status}` | {link_from(RUNS, best.run_dir, best.name)} | {best.metric_label}: {format_score(best)} |"
        )
    (RUNS / "README.md").write_text("\n".join(lines) + "\n")

    success_lines = [
        "# Calibrated Successful And Reference Runs",
        "",
        "Use this page when you want the shortest path to runs that either passed a calibrated faithfulness target or establish a reference baseline.",
        "",
        "Runs that only passed the inherited `0.034` pairwise agreement threshold for a different model are not listed as successful here.",
        "",
        "## Calibrated Target-Passing Runs",
        "",
        "| Group | Run | Metric | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for run in sorted(target_pass, key=lambda r: (r.group, r.run_rel.parts)):
        success_lines.append(
            f"| `{run.group}` | {link_from(success_dir, run.run_dir, run.name)} | {run.metric_label}: {format_score(run)} / target {target_text(run.target)} | {run.reason} |"
        )
    if not target_pass:
        success_lines.append("| | No target-passing runs found. | | |")

    success_lines.extend(
        [
            "",
            "## Reference Or Diagnostic-Passing Runs",
            "",
            "| Group | Run | Metric | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for run in sorted(reference, key=lambda r: (r.group, r.run_rel.parts)):
        success_lines.append(
            f"| `{run.group}` | {link_from(success_dir, run.run_dir, run.name)} | {run.metric_label}: {format_score(run)} | {run.reason} |"
        )

    success_lines.extend(
        [
            "",
            "## Near Misses",
            "",
            "These runs did not meet the target but are close enough to be useful for comparison.",
            "",
            "| Group | Run | Metric | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for run in sorted(near, key=lambda r: (r.group, float("inf") if r.score is None else r.score)):
        success_lines.append(
            f"| `{run.group}` | {link_from(success_dir, run.run_dir, run.name)} | {run.metric_label}: {format_score(run)} / target {target_text(run.target)} | {run.reason} |"
        )

    success_lines.extend(
        [
            "",
            "## Legacy Pairwise Passes",
            "",
            "These runs passed an inherited pairwise agreement threshold, but they are not categorized as successful until their model has a calibrated truth or reference target.",
            "",
            "| Group | Run | Metric | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for run in sorted(legacy_pairwise, key=lambda r: (r.group, r.run_rel.parts)):
        success_lines.append(
            f"| `{run.group}` | {link_from(success_dir, run.run_dir, run.name)} | {run.metric_label}: {format_score(run)} / inherited target {target_text(run.target)} | {run.reason} |"
        )
    (success_dir / "README.md").write_text("\n".join(success_lines) + "\n")

    for group in sorted(GROUP_TITLES):
        group_runs = [r for r in runs if r.group == group]
        if not group_runs:
            continue
        group_dir = RUNS / group
        group_dir.mkdir(parents=True, exist_ok=True)
        group_lines = [
            f"# {GROUP_TITLES[group]}",
            "",
            "Runs are grouped by method folder inside this model.",
            "",
            "| Status | Run | Metric | Target | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for run in sorted(group_runs, key=lambda r: (status_order(r.status), r.run_rel.parts)):
            target = "" if run.target is None else target_text(run.target)
            group_lines.append(
                f"| `{run.status}` | {link_from(group_dir, run.run_dir, run.name)} | {run.metric_label}: {format_score(run)} | {target} | {run.reason} |"
            )
        (group_dir / "README.md").write_text("\n".join(group_lines) + "\n")

        method_names = sorted({run.run_rel.parts[0] for run in group_runs if len(run.run_rel.parts) > 1})
        for method_name in method_names:
            method_dir = group_dir / method_name
            method_runs = [run for run in group_runs if run.run_rel.parts[0] == method_name]
            method_lines = [
                f"# {GROUP_TITLES[group]} / {method_name}",
                "",
                "| Status | Run | Metric | Target | Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
            for run in sorted(method_runs, key=lambda r: (status_order(r.status), r.run_rel.parts)):
                target = "" if run.target is None else target_text(run.target)
                run_label = " / ".join(run.run_rel.parts[1:])
                method_lines.append(
                    f"| `{run.status}` | {link_from(method_dir, run.run_dir, run_label)} | {run.metric_label}: {format_score(run)} | {target} | {run.reason} |"
                )
            (method_dir / "README.md").write_text("\n".join(method_lines) + "\n")


def main() -> None:
    runs = collect_runs()
    for run in runs:
        materialize_run(run)
    write_indexes(runs)
    print(f"Built runs view for {len(runs)} runs.")
    print(f"Target/reference successes: {sum(r.status in {'grid-faithful', 'pass', 'reference', 'diagnostic_pass'} for r in runs)}")
    print(f"Near misses: {sum(r.status == 'near' for r in runs)}")


if __name__ == "__main__":
    main()
