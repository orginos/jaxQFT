#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


class SlurmRecord(object):
    def __init__(
        self,
        job_id,
        job_name,
        state,
        source,
        exit_code="",
        elapsed="",
        reason="",
        start="",
        end="",
    ):
        self.job_id = job_id
        self.job_name = job_name
        self.state = state
        self.source = source
        self.exit_code = exit_code
        self.elapsed = elapsed
        self.reason = reason
        self.start = start
        self.end = end


class ExpectedRun(object):
    def __init__(
        self,
        campaign_id,
        title,
        mode,
        run_key,
        job_name,
        run_dir,
        completion_kind,
        source_files,
        expected_files=(),
        expected_epoch=None,
        logical_run_key="",
        logical_job_name="",
    ):
        self.campaign_id = campaign_id
        self.title = title
        self.mode = mode
        self.run_key = run_key
        self.job_name = job_name
        self.run_dir = run_dir
        self.completion_kind = completion_kind
        self.source_files = source_files
        self.expected_files = expected_files
        self.expected_epoch = expected_epoch
        self.logical_run_key = logical_run_key or run_key
        self.logical_job_name = logical_job_name or job_name


class BundleTaskRecord(object):
    def __init__(self, bundle_root, bundle_job_name, task_name, run_dir):
        self.bundle_root = bundle_root
        self.bundle_job_name = bundle_job_name
        self.task_name = task_name
        self.run_dir = run_dir


class _MissingPickleGlobal(object):
    def __init__(self, *args, **kwargs):
        self._pickle_stub_args = args
        self._pickle_stub_kwargs = kwargs

    def __setstate__(self, state):
        self._pickle_stub_state = state


class _TolerantCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return pickle.Unpickler.find_class(self, module, name)
        except Exception:
            return _MissingPickleGlobal


def normalize_state(state: str) -> str:
    normalized = state.strip().upper()
    if not normalized:
        return ""
    normalized = normalized.split()[0]
    normalized = normalized.split("+", 1)[0]
    return normalized


def parse_timestamp(value):
    if not value or value in {"Unknown", "None"}:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def slurm_record_sort_key(record):
    return (
        parse_timestamp(record.end),
        parse_timestamp(record.start),
        int(record.job_id) if record.job_id.isdigit() else -1,
    )


def choose_latest_record(records):
    filtered = [record for record in records if record is not None]
    if not filtered:
        return None
    return max(filtered, key=slurm_record_sort_key)


def parse_args():
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


def iter_fields(path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield line.split()


def load_numeric_column(path):
    values = []
    for fields in iter_fields(path):
        token = fields[0]
        if NUMERIC_RE.fullmatch(token):
            values.append(token)
    return values


def load_point_labels(path):
    labels = []
    for fields in iter_fields(path):
        if fields:
            labels.append(fields[0])
    return labels


def normalize_canonical_arch_tag(value):
    token = str(value).strip()
    if not token:
        raise ValueError("Empty canonical-point architecture tag")
    if token.startswith("w"):
        return token
    return "w{}".format(token)


def load_canonical_point_seed_overrides(path):
    overrides = {}
    if not path.exists():
        return overrides
    for fields in iter_fields(path):
        if len(fields) < 5:
            raise ValueError(
                "Expected at least 5 columns in {}: label arch volume logical_seed active_seed".format(
                    path
                )
            )
        label = fields[0]
        arch_tag = normalize_canonical_arch_tag(fields[1])
        volume = int(fields[2])
        logical_seed = int(fields[3])
        active_seed = int(fields[4])
        overrides[(label, arch_tag, volume, logical_seed)] = active_seed
    return overrides


def parse_schedule_final_epoch(config_path):
    final_epoch = 0
    ramp_epochs = None
    ramp_doubles = None
    section = ""
    explicit_stage_epoch_ends = []

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            section = line[2:-2].strip()
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if section == "schedule.ramp":
            if key == "epochs_per_stage":
                ramp_epochs = int(value)
            elif key == "num_doubles":
                ramp_doubles = int(value)
        elif section == "schedule.anneal" and key == "epoch_ends":
            explicit_stage_epoch_ends.extend(int(token) for token in re.findall(r"-?\d+", value))
        elif section == "schedule.stage" and key == "epoch_end":
            explicit_stage_epoch_ends.append(int(value))
        elif section == "train" and key == "epochs":
            final_epoch = max(final_epoch, int(value))

    if explicit_stage_epoch_ends:
        final_epoch = max(final_epoch, max(explicit_stage_epoch_ends))
    if ramp_epochs is not None and ramp_doubles is not None:
        final_epoch = max(final_epoch, ramp_epochs * (ramp_doubles + 1))
    if final_epoch <= 0:
        raise ValueError("Could not determine final epoch from {}".format(config_path))
    return final_epoch


def tag_mass(mass: str) -> str:
    return mass.replace("-", "m").replace("+", "p").replace(".", "p")


def build_expected_runs(repo_root, runs_root):
    expected = []

    canonical_seed_override_path = (
        repo_root / "configs/phi4/paper-2/canonical-point-scan/replacement_seeds.tsv"
    )
    default_points_path = repo_root / "configs/phi4/paper-2/canonical-point-scan/points.tsv"
    w64c3_points_path = repo_root / "configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv"
    canonical_seed_overrides = load_canonical_point_seed_overrides(canonical_seed_override_path)
    volumes = [16, 32, 64, 128]
    seeds = [0, 1, 2, 3]
    schedule_epoch_cache = {}

    def final_epoch_for_config(config_path):
        key = str(config_path)
        if key not in schedule_epoch_cache:
            schedule_epoch_cache[key] = parse_schedule_final_epoch(config_path)
        return schedule_epoch_cache[key]

    canonical_specs = [
        {
            "arch_tag": "w64",
            "point_labels": load_point_labels(default_points_path),
            "config_for": lambda _label, volume: repo_root
            / "configs/phi4/paper-2/canonical-scaling/L{}_uniform.toml".format(volume),
            "source_files": (
                "configs/phi4/paper-2/canonical-point-scan/points.tsv",
                "configs/phi4/paper-2/canonical-point-scan/replacement_seeds.tsv",
                "scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_campaign_nersc.sh",
            ),
        },
        {
            "arch_tag": "w48",
            "point_labels": load_point_labels(default_points_path),
            "config_for": lambda _label, volume: repo_root
            / "configs/phi4/paper-2/canonical-scaling/L{}_uniform.toml".format(volume),
            "source_files": (
                "configs/phi4/paper-2/canonical-point-scan/points.tsv",
                "configs/phi4/paper-2/canonical-point-scan/replacement_seeds.tsv",
                "scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_campaign_nersc.sh",
            ),
        },
    ]
    if w64c3_points_path.exists():
        canonical_specs.append(
            {
                "arch_tag": "w64c3",
                "point_labels": load_point_labels(w64c3_points_path),
                "config_for": lambda label, volume: (
                    repo_root
                    / "configs/phi4/paper-2/canonical-point-scan/L128_uniform_c3_batch64_then_anneal.toml"
                    if label == "canonical3" and volume == 128
                    else repo_root
                    / "configs/phi4/paper-2/canonical-scaling/L{}_uniform_c3.toml".format(volume)
                ),
                "source_files": (
                    "configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv",
                    "configs/phi4/paper-2/canonical-point-scan/replacement_seeds.tsv",
                    "scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_bundles_nersc.sh",
                ),
            }
        )

    for spec in canonical_specs:
        arch_tag = spec["arch_tag"]
        for label in spec["point_labels"]:
            for volume in volumes:
                config_path = spec["config_for"](label, volume)
                expected_epoch = final_epoch_for_config(config_path)
                for logical_seed in seeds:
                    active_seed = canonical_seed_overrides.get(
                        (label, arch_tag, volume, logical_seed),
                        logical_seed,
                    )
                    run_dir = (
                        runs_root
                        / "canonical-point-scan"
                        / label
                        / arch_tag
                        / "L{}".format(volume)
                        / "s{}".format(active_seed)
                    )
                    run_key = "{}/{}/L{}/s{}".format(label, arch_tag, volume, active_seed)
                    logical_run_key = "{}/{}/L{}/s{}".format(label, arch_tag, volume, logical_seed)
                    job_name = "phi4-{}-{}-L{}-s{}".format(label, arch_tag, volume, active_seed)
                    logical_job_name = "phi4-{}-{}-L{}-s{}".format(
                        label, arch_tag, volume, logical_seed
                    )
                    expected.append(
                        ExpectedRun(
                            campaign_id="canonical_point_scan",
                            title="Canonical Point-Scan Flow Training",
                            mode="orchestrate",
                            run_key=run_key,
                            job_name=job_name,
                            run_dir=run_dir,
                            completion_kind="flow_checkpoint",
                            source_files=spec["source_files"],
                            expected_epoch=expected_epoch,
                            logical_run_key=logical_run_key,
                            logical_job_name=logical_job_name,
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


def run_command(command):
    try:
        proc = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError:
        return [], f"Command not found: {command[0]}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout or f"exit code {exc.returncode}"
        return [], f"{' '.join(command)} failed: {detail}"
    return proc.stdout.splitlines(), None


def query_squeue(user):
    rows, warning = run_command(
        ["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%M|%l|%R"]
    )
    live_by_name = defaultdict(list)
    warnings = []
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


def query_sacct(user, days):
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
    latest_by_name = {}
    warnings = []
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
        current_key = slurm_record_sort_key(current)
        record_key = slurm_record_sort_key(record)
        if record_key >= current_key:
            latest_by_name[record.job_name] = record
    return latest_by_name, warnings


def parse_bundle_job_name(sbatch_path):
    if not sbatch_path.exists():
        return ""
    for raw_line in sbatch_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#SBATCH --job-name="):
            return line.split("=", 1)[1].strip()
    return ""


def parse_bundle_run_dir(fields):
    if len(fields) < 2:
        return None
    if len(fields) >= 3 and fields[1].endswith(".toml"):
        return fields[2]
    return fields[1]


def load_bundle_task_index(runs_root):
    bundle_tasks_by_run_dir = defaultdict(list)
    for tasks_file in runs_root.rglob("tasks.tsv"):
        if not any(part.startswith("_bundles") for part in tasks_file.parts):
            continue
        bundle_root = tasks_file.parent
        bundle_job_name = parse_bundle_job_name(bundle_root / "job.sbatch")
        if not bundle_job_name:
            continue
        for raw_line in tasks_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            run_dir = parse_bundle_run_dir(fields)
            if not run_dir:
                continue
            task_name = fields[0]
            bundle_tasks_by_run_dir[run_dir].append(
                BundleTaskRecord(
                    bundle_root=str(bundle_root),
                    bundle_job_name=bundle_job_name,
                    task_name=task_name,
                    run_dir=run_dir,
                )
            )
    return bundle_tasks_by_run_dir


def checkpoint_epoch(path):
    try:
        with open(path, "rb") as handle:
            payload = _TolerantCheckpointUnpickler(handle).load()
    except Exception as exc:  # pragma: no cover - exercised in integration only
        return None, type(exc).__name__
    epoch = payload.get("epoch")
    if epoch is None:
        return None, "MissingEpoch"
    return int(epoch), None


def inspect_artifacts(run):
    missing = []
    notes = []
    epoch = None

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
    run,
    live_by_name,
    latest_acct,
    bundle_tasks_by_run_dir,
):
    complete, missing_artifacts, notes, epoch = inspect_artifacts(run)
    if run.logical_run_key != run.run_key:
        notes.append(
            "replacement_seed={} active_seed={}".format(
                run.logical_run_key.rsplit("/s", 1)[-1],
                run.run_key.rsplit("/s", 1)[-1],
            )
        )
    bundle_tasks = bundle_tasks_by_run_dir.get(str(run.run_dir), [])
    bundle_job_names = sorted({task.bundle_job_name for task in bundle_tasks})
    bundle_task_names = sorted({task.task_name for task in bundle_tasks})
    bundle_live_records = []
    bundle_acct_records = []
    for bundle_job_name in bundle_job_names:
        bundle_live_records.extend(live_by_name.get(bundle_job_name, []))
        acct_record = latest_acct.get(bundle_job_name)
        if acct_record is not None:
            bundle_acct_records.append(acct_record)

    live_records = list(live_by_name.get(run.job_name, []))
    live_records.extend(bundle_live_records)
    dedup_live = {}
    for record in live_records:
        dedup_live[(record.job_id, record.job_name, record.state, record.source)] = record
    live_records = sorted(dedup_live.values(), key=lambda record: (record.job_name, record.job_id))
    live_states = sorted({record.state for record in live_records})
    live_job_ids = [record.job_id for record in live_records]
    original_acct_record = latest_acct.get(run.job_name)
    if bundle_job_names:
        if bundle_live_records:
            acct_record = choose_latest_record(bundle_acct_records)
        elif bundle_acct_records:
            acct_record = choose_latest_record(bundle_acct_records)
        else:
            acct_record = original_acct_record
    else:
        acct_record = original_acct_record
    acct_state = acct_record.state if acct_record else ""

    if bundle_job_names:
        notes.append("bundle_jobs={}".format(",".join(bundle_job_names)))
    if bundle_task_names:
        notes.append("bundle_task_names={}".format(",".join(bundle_task_names)))
    if bundle_job_names and live_states:
        acct_record = None
        acct_state = ""

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
        "logical_run_key": run.logical_run_key,
        "job_name": run.job_name,
        "logical_job_name": run.logical_job_name,
        "run_dir": str(run.run_dir),
        "status": status,
        "missing_artifacts": missing_artifacts,
        "checkpoint_epoch": epoch,
        "expected_epoch": run.expected_epoch,
        "live_job_ids": live_job_ids,
        "live_states": live_states,
        "bundle_job_names": bundle_job_names,
        "bundle_task_names": bundle_task_names,
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
    campaign_id,
    entries,
    max_examples,
):
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


def render_text(report, max_examples):
    lines = []
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
                if example["logical_run_key"] != example["run_key"]:
                    lines.append(
                        "      replacement: "
                        f"{example['logical_run_key']} -> {example['run_key']}"
                    )
                if example.get("bundle_job_names"):
                    lines.append(
                        "      bundle_jobs: "
                        f"{', '.join(example['bundle_job_names'])}"
                    )
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


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    runs_root = (args.runs_root.resolve() if args.runs_root else (repo_root / "runs" / "phi4").resolve())
    slurm_user = args.slurm_user or current_username()

    expected_runs = build_expected_runs(repo_root, runs_root)
    if args.campaign:
        allowed = set(args.campaign)
        expected_runs = [run for run in expected_runs if run.campaign_id in allowed]

    warnings = []
    live_by_name = {}
    latest_acct = {}
    if not args.no_slurm and not args.no_squeue:
        live_by_name, queue_warnings = query_squeue(slurm_user)
        warnings.extend(queue_warnings)
    if not args.no_slurm and not args.no_sacct:
        latest_acct, acct_warnings = query_sacct(slurm_user, args.sacct_days)
        warnings.extend(acct_warnings)
    bundle_tasks_by_run_dir = load_bundle_task_index(runs_root)

    entries_by_campaign = defaultdict(list)
    for run in expected_runs:
        entries_by_campaign[run.campaign_id].append(
            classify_run(run, live_by_name, latest_acct, bundle_tasks_by_run_dir)
        )

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
