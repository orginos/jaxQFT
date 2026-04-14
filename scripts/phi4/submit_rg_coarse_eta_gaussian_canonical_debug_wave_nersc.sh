#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_debug_wave_nersc.sh [options]

Submit the first bundled canonical-flow debug wave on Perlmutter.

This wave targets 7 debug-only runs under:
  /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-debug

The production trainer remains untouched. The bundle uses:
  scripts/phi4/run_rg_coarse_eta_gaussian_debug.sh
which dispatches:
  scripts/phi4/train_rg_coarse_eta_gaussian_flow_debug.py

Each task writes:
  debug_checkpoint.pkl
  debug_checkpoint.pkl.lastfinite.pkl   (on non-finite failure)
  debug_checkpoint.pkl.debug.nonfinite.json
  slurm/train.out
  slurm/train.err

Optional:
  --repo-root PATH      Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH       Debug run root. Default:
                        /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-debug
  --bundle-root PATH    Bundle root. Default:
                        <run-root>/_bundles/debug_wave_20260414
  --tasks PATH          Task manifest. Default:
                        configs/phi4/paper-2/canonical-point-scan/debug-wave-20260414/tasks.tsv
  --job-name NAME       Slurm job name. Default: phi4-can-debug-wave0
  --time HH:MM:SS       Walltime. Default: 02:00:00
  --account NAME        Slurm account. Default: hadron
  --qos NAME            Slurm QOS. Default: regular
  --print-only          Print the generated submit command and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-debug"
bundle_root=""
tasks_file="configs/phi4/paper-2/canonical-point-scan/debug-wave-20260414/tasks.tsv"
job_name="phi4-can-debug-wave0"
time_limit="02:00:00"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --tasks) tasks_file="${2:-}"; shift 2 ;;
    --job-name) job_name="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

bundle_submit="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"
debug_wrapper="${repo_root}/scripts/phi4/run_rg_coarse_eta_gaussian_debug.sh"

if [[ ! -x "${bundle_submit}" ]]; then
  echo "Bundle submit helper not executable: ${bundle_submit}" >&2
  exit 2
fi
if [[ ! -x "${debug_wrapper}" ]]; then
  echo "Debug wrapper not executable: ${debug_wrapper}" >&2
  exit 2
fi

if [[ "${tasks_file}" != /* ]]; then
  tasks_file="${repo_root}/${tasks_file}"
fi
if [[ ! -f "${tasks_file}" ]]; then
  echo "Tasks file not found: ${tasks_file}" >&2
  exit 2
fi

mkdir -p "${run_root}"
if [[ -z "${bundle_root}" ]]; then
  bundle_root="${run_root}/_bundles/debug_wave_20260414"
elif [[ "${bundle_root}" != /* ]]; then
  bundle_root="${repo_root}/${bundle_root}"
fi

cmd=(
  "${bundle_submit}"
  --repo-root "${repo_root}"
  --tasks "${tasks_file}"
  --bundle-root "${bundle_root}"
  --job-name "${job_name}"
  --time "${time_limit}"
  --account "${account}"
  --qos "${qos}"
  --wrapper "${debug_wrapper}"
)

if [[ ${print_only} -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
