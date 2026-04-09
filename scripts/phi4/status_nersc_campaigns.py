#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


NUMERIC_RE = re.compile(r"[-+]?[0-9]+(?:\.[0-9]+)?")

STATUS_ORDER = {
    "failed": 0,
    "running": 1,
    "pending": 2,
    "missing_output": 3,
    "done": 4,
}

PENDING_STATES = {
    "PENDING",
    "CONFIGURING",
    "SUSPENDED",
}

RUNNING_STATES = {
    "RUNNING",
    "COMPLETING",
    "STAGE_OUT",
    "SIGNALING",
}

FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "STOPPED",
    "TIMEOUT",
}

DONE_STATES = {"COMPLETED"}


@dataclass(frozen=True)
class SlurmRecord:
    job_id: str
    job_name: str
    state: str
    source: str
    exit_code: str = ""
    elapsed: str = ""
    reason: str = ""
    start: str = ""
    end: str = ""


@dataclass(frozen=True)
class ExpectedRun:
    campaign_id: str
    title: str
    mode: str
    run_key: str
    job_name: str
    run_dir: Path
    completion_kind: str
    source_files: tuple[str, ...]
    expected_files: tuple[str, ...] = ()
    expected_epoch: int | None = None


def normalize_state(state: str) -> str:
    normalized = state.strip().upper()
    if not normalized:
        return ""
    normalized = normalized.split()[0]
    normalized = normalized.split("+", 1)[0]
    return normalized


def parse_timestamp(value: str) -> datetime:
    if not value or value in {"Unknown", "None"}:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only inventory for the active phi4 NERSC campaigns. "
            "The tool reads repo campaign specs, inspects run roots, queries "
            "squeue/sacct when available, and classifies each expected run "
            "without mutating anything."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Default: inferred from the script location.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Root of the phi4 runtime tree. Default: <repo-root>/runs/phi4. "
            "Use this for local fixture tests."
        ),
    )
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        choices=[
            "canonical_point_scan",
            "hmc_refined",
            "hmc_l512_tuning",
            "hmc_extra_stats",
            "canonical_level_analysis_monitor",
            "per_level_analysis_monitor",
        ],
        help="Restrict output to one or more campaign ids.",
    )
    parser.add_argument(
        "--slurm-user",
        default="",
        help="Slurm user to query. Default: the current local username.",
    )
    parser.add_argument(
        "--sacct-days",
        type=int,
        default=30,
        help="Lookback window for sacct in days. Default: 30.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=6,
        help="Maximum incomplete examples per campaign in text mode. Default: 6.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON with per-run details.",
    )
    parser.add_argument(
        "--no-slurm",
        action="store_true",
        help="Skip both squeue and sacct queries.",
    )
    parser.add_argument(
        "--no-squeue",
        action="store_true",
        help="Skip squeue, but still query sacct unless --no-slurm is set.",
    )
    parser.add_argument(
        "--no-sacct",
        action="store_true",
        help="Skip sacct, but still query squeue unless --no-slurm is set.",
    )
    return parser.parse_args()


def iter_fields(path: Path) -> Iterable[list[str]]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield line.split()


def load_numeric_column(path: Path) -> list[str]:
    values: list[str] = []
    for fields in iter_fields(path):
        token = fields[0]
        if NUMERIC_RE.fullmatch(token):
            values.append(token)
    return values


def load_point_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for fields in iter_fields(path):
        if fields:
            labels.append(fields[0])
    return labels


def parse_schedule_final_epoch(config_path: Path) -> int:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    schedule = cfg.get("schedule", {})
    final_epoch = 0

    explicit = schedule.get("stage", [])
    if isinstance(explicit, dict):
        explicit = [explicit]
    if explicit:
        return max(int(stage["epoch_end"]) for stage in explicit)

    ramp = schedule.get("ramp")
    if ramp:
        initial = int(ramp["epochs_per_stage"])
        num_doubles = int(ramp["num_doubles"])
        final_epoch += initial * (num_doubles + 1)

    anneal = schedule.get("anneal")
    if anneal:
        epoch_ends = [int(value) for value in anneal.get("epoch_ends", [])]
        if epoch_ends:
            final_epoch = max(final_epoch, max(epoch_ends))

    if final_epoch <= 0 and "train" in cfg and "epochs" in cfg["train"]:
        final_epoch = int(cfg["train"]["epochs"])

    if final_epoch <= 0:
        raise ValueError(f"Could not determine final epoch from {config_path}")
    return final_epoch


def tag_mass(mass: str) -> str:
    return mass.replace("-", "m").replace("+", "p").replace(".", "p")


def build_expected_runs(repo_root: Path, runs_root: Path) -> list[ExpectedRun]:
    expected: list[ExpectedRun] = []

    point_labels = load_point_labels(
        repo_root / "configs/phi4/paper-2/canonical-point-scan/points.tsv"
    )
    widths = [64, 48]
    volumes = [16, 32, 64, 128]
    seeds = [0, 1, 2, 3]
    final_epoch_by_volume = {
        volume: parse_schedule_final_epoch(
            repo_root / f"configs/phi4/paper-2/canonical-scaling/L{volume}_uniform.toml"
        )
        for volume in volumes
    }
    for label in point_labels:
        for width in widths:
            for volume in volumes:
                for seed in seeds:
                    run_dir = (
                        runs_root
                        / "canonical-point-scan"
                        / label
                        / f"w{width}"
                        / f"L{volume}"
                        / f"s{seed}"
                    )
                    expected.append(
                        ExpectedRun(
                            campaign_id="canonical_point_scan",
                            title="Canonical Point-Scan Flow Training",
                            mode="orchestrate",
                            run_key=f"{label}/w{width}/L{volume}/s{seed}",
                            job_name=f"phi4-{label}-w{width}-L{volume}-s{seed}",
                            run_dir=run_dir,
                            completion_kind="flow_checkpoint",
                            source_files=(
                                "configs/phi4/paper-2/canonical-point-scan/points.tsv",
                                "scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_campaign_nersc.sh",
                            ),
                            expected_epoch=final_epoch_by_volume[volume],
                        )
                    )

    canonical_level_seeds = {
        16: [0, 1, 2, 3],
        32: [0, 1, 2, 3],
        64: [0, 1, 2, 3],
        128: [0, 1, 3, 4],
    }
    for volume, seeds_for_volume in canonical_level_seeds.items():
        for seed in seeds_for_volume:
            run_dir = (
                runs_root / "canonical-level-analysis-multimetric" / f"L{volume}" / f"s{seed}"
            )
            expected.append(
                ExpectedRun(
                    campaign_id="canonical_level_analysis_monitor",
                    title="Canonical Continued Multimetric Flow Analysis",
                    mode="monitor_only",
                    run_key=f"L{volume}/s{seed}",
                    job_name=f"phi4-L{volume}-levels-s{seed}",
                    run_dir=run_dir,
                    completion_kind="files",
                    source_files=(
                        "scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_campaign_nersc.sh",
                        "scripts/phi4/analysis/analyze_rg_coarse_eta_gaussian_levels.py",
                    ),
                    expected_files=("flow_levels.json",),
                )
            )

    for tag in ["L16", "L32", "L64", "L128"]:
        run_dir = runs_root / "per-level-level-analysis-multimetric" / tag
        expected.append(
            ExpectedRun(
                campaign_id="per_level_analysis_monitor",
                title="Tuned Per-Level Multimetric Flow Analysis",
                mode="monitor_only",
                run_key=tag,
                job_name=f"phi4-perlvl-{tag}",
                run_dir=run_dir,
                completion_kind="files",
                source_files=(
                    "scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_perlevel_campaign_nersc.sh",
                    "scripts/phi4/analysis/analyze_rg_coarse_eta_gaussian_levels.py",
                ),
                expected_files=("flow_levels.json",),
            )
        )

    refined_masses = load_numeric_column(
        repo_root / "configs/phi4/paper-2/hmc-g2-scan/refined_g2_points.tsv"
    )
    for volume in [16, 32, 64, 128, 256]:
        for mass in refined_masses:
            run_dir = runs_root / "hmc-g2-scan" / "production-refined" / f"L{volume}" / f"g2_{mass}"
            expected.append(
                ExpectedRun(
                    campaign_id="hmc_refined",
                    title="HMC Refined Near-Critical Scan",
                    mode="orchestrate",
                    run_key=f"L{volume}/g2_{mass}",
                    job_name=f"phi4-g2-L{volume}-{tag_mass(mass)}",
                    run_dir=run_dir,
                    completion_kind="files",
                    source_files=(
                        "configs/phi4/paper-2/hmc-g2-scan/refined_g2_points.tsv",
                        "configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv",
                        "scripts/phi4/submit_hmc_phi4_refined_mass_campaign_nersc.sh",
                    ),
                    expected_files=("hmc_phi4.json", "hmc_phi4.npz"),
                )
            )

    for fields in iter_fields(
        repo_root / "configs/phi4/paper-2/hmc-g2-scan/tuning_grid_L512.tsv"
    ):
        if len(fields) < 5 or fields[0] == "L":
            continue
        volume, batch, nmd, integrator, _ = fields[:5]
        run_dir = (
            runs_root
            / "hmc-g2-scan"
            / "tuning-L512"
            / f"L{volume}"
            / integrator
            / f"b{batch}_nmd{nmd}"
        )
        expected.append(
            ExpectedRun(
                campaign_id="hmc_l512_tuning",
                title="HMC L=512 Tuning",
                mode="orchestrate",
                run_key=f"L{volume}/{integrator}/b{batch}_nmd{nmd}",
                job_name=f"phi4-hmc-L{volume}-{integrator}-b{batch}-nmd{nmd}",
                run_dir=run_dir,
                completion_kind="files",
                source_files=(
                    "configs/phi4/paper-2/hmc-g2-scan/tuning_grid_L512.tsv",
                    "scripts/phi4/submit_hmc_phi4_l512_tuning_campaign_nersc.sh",
                    "scripts/phi4/submit_hmc_phi4_tuning_campaign_nersc.sh",
                ),
                expected_files=("hmc_phi4.json", "hmc_phi4.npz"),
            )
        )

    extra_masses = load_numeric_column(
        repo_root / "configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv"
    )
    for volume, num_replicas in [(128, 3), (256, 7)]:
        nested_root = runs_root / "hmc-g2-scan" / "production-extra" / f"L{volume}" / f"L{volume}"
        for mass in extra_masses:
            for replica in range(1, num_replicas + 1):
                run_dir = nested_root / f"g2_{mass}" / f"rep{replica}"
                expected.append(
                    ExpectedRun(
                        campaign_id="hmc_extra_stats",
                        title="HMC Extra-Statistics Replicas",
                        mode="orchestrate",
                        run_key=f"L{volume}/g2_{mass}/rep{replica}",
                        job_name=f"phi4-g2-L{volume}-{tag_mass(mass)}-rep{replica}",
                        run_dir=run_dir,
                        completion_kind="files",
                        source_files=(
                            "configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv",
                            "configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv",
                            "scripts/phi4/submit_hmc_phi4_extra_stats_campaign_nersc.sh",
                        ),
                        expected_files=("hmc_phi4.json", "hmc_phi4.npz"),
                    )
                )

    return expected


def current_username() -> str:
    for key in ("USER", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    import getpass

    return getpass.getuser()


def run_command(command: list[str]) -> tuple[list[str], str | None]:
    try:
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [], f"Command not found: {command[0]}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout or f"exit code {exc.returncode}"
        return [], f"{' '.join(command)} failed: {detail}"
    return proc.stdout.splitlines(), None


def query_squeue(user: str) -> tuple[dict[str, list[SlurmRecord]], list[str]]:
    rows, warning = run_command(
        ["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%M|%l|%R"]
    )
    live_by_name: dict[str, list[SlurmRecord]] = defaultdict(list)
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
        return live_by_name, warnings
    for row in rows:
        fields = row.split("|", 5)
        if len(fields) != 6:
            continue
        job_id, job_name, state, elapsed, limit, reason = fields
        if not job_name.startswith("phi4-"):
            continue
        live_by_name[job_name].append(
            SlurmRecord(
                job_id=job_id.strip(),
                job_name=job_name.strip(),
                state=normalize_state(state),
                source="squeue",
                elapsed=elapsed.strip(),
                reason=reason.strip(),
                exit_code="",
                start="",
                end="",
            )
        )
    return live_by_name, warnings


def query_sacct(user: str, days: int) -> tuple[dict[str, SlurmRecord], list[str]]:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    rows, warning = run_command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-u",
            user,
            "-S",
            start,
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
        ]
    )
    latest_by_name: dict[str, SlurmRecord] = {}
    warnings: list[str] = []
    if warning:
        warnings.append(warning)
        return latest_by_name, warnings

    for row in rows:
        fields = row.split("|")
        if len(fields) < 7:
            continue
        job_id, job_name, state, exit_code, elapsed, start_time, end_time = fields[:7]
        if "." in job_id or not job_name.startswith("phi4-"):
            continue
        record = SlurmRecord(
            job_id=job_id.strip(),
            job_name=job_name.strip(),
            state=normalize_state(state),
            source="sacct",
            exit_code=exit_code.strip(),
            elapsed=elapsed.strip(),
            reason="",
            start=start_time.strip(),
            end=end_time.strip(),
        )
        current = latest_by_name.get(record.job_name)
        if current is None:
            latest_by_name[record.job_name] = record
            continue
        current_key = (
            parse_timestamp(current.end),
            parse_timestamp(current.start),
            int(current.job_id) if current.job_id.isdigit() else -1,
        )
        record_key = (
            parse_timestamp(record.end),
            parse_timestamp(record.start),
            int(record.job_id) if record.job_id.isdigit() else -1,
        )
        if record_key >= current_key:
            latest_by_name[record.job_name] = record
    return latest_by_name, warnings


def checkpoint_epoch(path: Path) -> tuple[int | None, str | None]:
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - exercised in integration only
        return None, type(exc).__name__
    epoch = payload.get("epoch")
    if epoch is None:
        return None, "MissingEpoch"
    return int(epoch), None


def inspect_artifacts(run: ExpectedRun) -> tuple[bool, list[str], list[str], int | None]:
    missing: list[str] = []
    notes: list[str] = []
    epoch: int | None = None

    if run.completion_kind == "flow_checkpoint":
        checkpoint = run.run_dir / "checkpoint.pkl"
        if not checkpoint.exists():
            missing.append("checkpoint.pkl")
            return False, missing, notes, epoch
        epoch, error = checkpoint_epoch(checkpoint)
        if error:
            missing.append("checkpoint.pkl")
            notes.append(f"checkpoint_error={error}")
            return False, missing, notes, epoch
        notes.append(f"checkpoint_epoch={epoch}")
        if run.expected_epoch is None or epoch < run.expected_epoch:
            missing.append("checkpoint_final_epoch")
            return False, missing, notes, epoch
        return True, missing, notes, epoch

    for relative in run.expected_files:
        if not (run.run_dir / relative).exists():
            missing.append(relative)
    return not missing, missing, notes, epoch


def classify_run(
    run: ExpectedRun,
    live_by_name: dict[str, list[SlurmRecord]],
    latest_acct: dict[str, SlurmRecord],
) -> dict[str, object]:
    complete, missing_artifacts, notes, epoch = inspect_artifacts(run)
    live_records = live_by_name.get(run.job_name, [])
    live_states = sorted({record.state for record in live_records})
    live_job_ids = [record.job_id for record in live_records]
    acct_record = latest_acct.get(run.job_name)
    acct_state = acct_record.state if acct_record else ""

    if complete:
        status = "done"
        if live_states:
            notes.append("complete_artifacts_but_live_job_present")
        elif acct_state in FAILED_STATES:
            notes.append(f"latest_sacct={acct_state}")
    elif any(state in RUNNING_STATES for state in live_states):
        status = "running"
    elif any(state in PENDING_STATES for state in live_states):
        status = "pending"
    elif acct_state in FAILED_STATES:
        status = "failed"
    elif acct_state in DONE_STATES:
        status = "missing_output"
    elif acct_state in RUNNING_STATES:
        status = "running"
    elif acct_state in PENDING_STATES:
        status = "pending"
    else:
        status = "missing_output"

    return {
        "campaign_id": run.campaign_id,
        "title": run.title,
        "mode": run.mode,
        "run_key": run.run_key,
        "job_name": run.job_name,
        "run_dir": str(run.run_dir),
        "status": status,
        "missing_artifacts": missing_artifacts,
        "checkpoint_epoch": epoch,
        "expected_epoch": run.expected_epoch,
        "live_job_ids": live_job_ids,
        "live_states": live_states,
        "latest_sacct": (
            {
                "job_id": acct_record.job_id,
                "state": acct_record.state,
                "exit_code": acct_record.exit_code,
                "elapsed": acct_record.elapsed,
                "start": acct_record.start,
                "end": acct_record.end,
            }
            if acct_record
            else None
        ),
        "notes": notes,
        "source_files": list(run.source_files),
    }


def summarize_campaign(
    campaign_id: str,
    entries: list[dict[str, object]],
    max_examples: int,
) -> dict[str, object]:
    counts = Counter(entry["status"] for entry in entries)
    missing_artifact_counts = Counter()
    for entry in entries:
        for artifact in entry["missing_artifacts"]:
            missing_artifact_counts[artifact] += 1
    examples = sorted(
        (entry for entry in entries if entry["status"] != "done"),
        key=lambda item: (
            STATUS_ORDER[item["status"]],
            item["run_dir"],
        ),
    )[:max_examples]
    return {
        "campaign_id": campaign_id,
        "title": entries[0]["title"] if entries else campaign_id,
        "mode": entries[0]["mode"] if entries else "unknown",
        "expected_runs": len(entries),
        "counts": {
            status: counts.get(status, 0)
            for status in ["done", "running", "pending", "failed", "missing_output"]
        },
        "missing_artifacts": dict(sorted(missing_artifact_counts.items())),
        "examples": examples,
        "entries": entries,
    }


def render_text(report: dict[str, object], max_examples: int) -> str:
    lines: list[str] = []
    lines.append(f"Repo root: {report['repo_root']}")
    lines.append(f"Runs root: {report['runs_root']}")
    lines.append(f"Slurm user: {report['slurm_user']}")
    lines.append(f"Generated: {report['generated_at']}")
    warnings = report["warnings"]
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    for campaign in report["campaigns"]:
        counts = campaign["counts"]
        lines.append("")
        lines.append(f"{campaign['title']} [{campaign['campaign_id']}] ({campaign['mode']})")
        lines.append(
            "  counts: "
            f"done={counts['done']} "
            f"running={counts['running']} "
            f"pending={counts['pending']} "
            f"failed={counts['failed']} "
            f"missing_output={counts['missing_output']}"
        )
        if campaign["missing_artifacts"]:
            missing_text = ", ".join(
                f"{name}={count}" for name, count in campaign["missing_artifacts"].items()
            )
            lines.append(f"  missing_artifacts: {missing_text}")
        examples = campaign["examples"][:max_examples]
        if examples:
            lines.append("  examples:")
            for example in examples:
                lines.append(f"    - {example['status']}: {example['run_dir']}")
                if example["missing_artifacts"]:
                    lines.append(
                        f"      missing: {', '.join(example['missing_artifacts'])}"
                    )
                if example["checkpoint_epoch"] is not None and example["expected_epoch"] is not None:
                    lines.append(
                        f"      checkpoint_epoch: {example['checkpoint_epoch']}/{example['expected_epoch']}"
                    )
                if example["live_states"]:
                    lines.append(
                        f"      live: {', '.join(example['live_states'])}"
                    )
                if example["latest_sacct"]:
                    lines.append(
                        "      latest_sacct: "
                        f"{example['latest_sacct']['state']} "
                        f"(job {example['latest_sacct']['job_id']})"
                    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    runs_root = (args.runs_root.resolve() if args.runs_root else (repo_root / "runs" / "phi4").resolve())
    slurm_user = args.slurm_user or current_username()

    expected_runs = build_expected_runs(repo_root, runs_root)
    if args.campaign:
        allowed = set(args.campaign)
        expected_runs = [run for run in expected_runs if run.campaign_id in allowed]

    warnings: list[str] = []
    live_by_name: dict[str, list[SlurmRecord]] = {}
    latest_acct: dict[str, SlurmRecord] = {}
    if not args.no_slurm and not args.no_squeue:
        live_by_name, queue_warnings = query_squeue(slurm_user)
        warnings.extend(queue_warnings)
    if not args.no_slurm and not args.no_sacct:
        latest_acct, acct_warnings = query_sacct(slurm_user, args.sacct_days)
        warnings.extend(acct_warnings)

    entries_by_campaign: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in expected_runs:
        entries_by_campaign[run.campaign_id].append(classify_run(run, live_by_name, latest_acct))

    campaign_ids_in_order = [
        "canonical_point_scan",
        "hmc_refined",
        "hmc_l512_tuning",
        "hmc_extra_stats",
        "canonical_level_analysis_monitor",
        "per_level_analysis_monitor",
    ]
    campaigns = []
    for campaign_id in campaign_ids_in_order:
        entries = entries_by_campaign.get(campaign_id, [])
        if not entries:
            continue
        campaigns.append(summarize_campaign(campaign_id, entries, args.max_examples))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "runs_root": str(runs_root),
        "slurm_user": slurm_user,
        "warnings": warnings,
        "campaigns": campaigns,
    }

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(render_text(report, args.max_examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
