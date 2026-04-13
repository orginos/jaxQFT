#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_bundle_nersc.sh [options]

Submit one or more independent phi^4 HMC tasks as a regular-qos Perlmutter
bundle, with one task per GPU.

Required:
  --tasks PATH           TSV manifest with at least one non-comment task row.
  --bundle-root PATH     Bundle directory. The script writes:
                         - tasks.tsv (normalized absolute manifest)
                         - job.sbatch
                         - slurm/*.out,*.err,job.log

Behavior:
  - tasks are packed at up to 4 tasks per node
  - node count is computed automatically as ceil(task_count / 4)
  - the final node may be partially filled

Manifest format:
  Tab-separated fields. The first two columns are required:
    1. task_name
    2. run_dir path
  Any additional tab-separated columns are passed verbatim to
  scripts/phi4/hmc_phi4.py after --json-out and --hist-out.
  This is the intended place for point-specific arguments such as:
    --shape 512,512 --lam 2.4 --mass -0.595 --seed 51200000 ...

Optional:
  --repo-root PATH       Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --job-name NAME        Slurm job name. Default: basename of bundle root.
  --time HH:MM:SS        Walltime. Default: 02:00:00.
  --account NAME         Slurm account. Default: $NERSC_ACCOUNT or hadron.
  --qos NAME             Slurm QOS. Default: regular.
  --constraint NAME      Slurm constraint. Default: gpu.
  --cpus-per-task INT    CPUs per task. Default: 32.
  --wrapper PATH         Wrapper script. Default: scripts/phi4/run_hmc_phi4.sh
  --launcher PATH        Slurm launcher. Default: scripts/phi4/hmc_phi4_bundle_perlmutter.slurm
  --json-name NAME       JSON file name inside each run dir. Default: hmc_phi4.json
  --hist-name NAME       NPZ file name inside each run dir. Default: hmc_phi4.npz
  --print-only           Print the generated sbatch command and exit.
EOF
}

resolve_input_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  elif [[ -e "${value}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${value}")" && pwd)/$(basename "${value}")"
  else
    printf '%s\n' "${repo_root}/${value}"
  fi
}

contains_exact() {
  local needle="$1"
  shift
  local item=""
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

tasks_file=""
bundle_root=""
repo_root="/global/cfs/cdirs/hadron/jaxQFT"
job_name=""
time_limit="02:00:00"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
constraint="gpu"
cpus_per_task="32"
wrapper="scripts/phi4/run_hmc_phi4.sh"
launcher="scripts/phi4/hmc_phi4_bundle_perlmutter.slurm"
json_name="hmc_phi4.json"
hist_name="hmc_phi4.npz"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) tasks_file="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --job-name) job_name="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --constraint) constraint="${2:-}"; shift 2 ;;
    --cpus-per-task) cpus_per_task="${2:-}"; shift 2 ;;
    --wrapper) wrapper="${2:-}"; shift 2 ;;
    --launcher) launcher="${2:-}"; shift 2 ;;
    --json-name) json_name="${2:-}"; shift 2 ;;
    --hist-name) hist_name="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${tasks_file}" || -z "${bundle_root}" ]]; then
  usage >&2
  exit 2
fi

tasks_file="$(resolve_input_path "${tasks_file}")"
if [[ ! -f "${tasks_file}" ]]; then
  echo "Tasks file not found: ${tasks_file}" >&2
  exit 2
fi

if [[ "${bundle_root}" != /* ]]; then
  bundle_root="${repo_root}/${bundle_root}"
fi
mkdir -p "${bundle_root}/slurm"

if [[ "${wrapper}" != /* ]]; then
  wrapper="${repo_root}/${wrapper}"
fi
if [[ ! -x "${wrapper}" ]]; then
  echo "Wrapper not executable: ${wrapper}" >&2
  exit 2
fi

if [[ "${launcher}" != /* ]]; then
  launcher="${repo_root}/${launcher}"
fi
if [[ ! -f "${launcher}" ]]; then
  echo "Launcher not found: ${launcher}" >&2
  exit 2
fi

if [[ -z "${job_name}" ]]; then
  job_name="$(basename "${bundle_root}")"
fi

normalized_tasks="${bundle_root}/tasks.tsv"
sbatch_file="${bundle_root}/job.sbatch"

declare -a seen_names=()
declare -a seen_run_dirs=()
task_count=0
: > "${normalized_tasks}"

while IFS=$'\t' read -r -a fields || [[ ${#fields[@]} -gt 0 ]]; do
  if [[ ${#fields[@]} -eq 0 ]]; then
    continue
  fi
  if [[ "${fields[0]}" == \#* ]]; then
    continue
  fi
  if [[ ${#fields[@]} -lt 2 ]]; then
    echo "Expected at least 2 TSV columns per task in ${tasks_file}" >&2
    exit 2
  fi

  task_name="${fields[0]}"
  run_dir="${fields[1]}"

  if contains_exact "${task_name}" "${seen_names[@]-}"; then
    echo "Duplicate task name in ${tasks_file}: ${task_name}" >&2
    exit 2
  fi
  seen_names+=("${task_name}")

  if [[ "${run_dir}" != /* ]]; then
    run_dir="${repo_root}/${run_dir}"
  fi
  if contains_exact "${run_dir}" "${seen_run_dirs[@]-}"; then
    echo "Duplicate run_dir in ${tasks_file}: ${run_dir}" >&2
    exit 2
  fi
  seen_run_dirs+=("${run_dir}")

  {
    printf "%s\t%s" "${task_name}" "${run_dir}"
    if [[ ${#fields[@]} -gt 2 ]]; then
      for arg in "${fields[@]:2}"; do
        printf "\t%s" "${arg}"
      done
    fi
    printf "\n"
  } >> "${normalized_tasks}"
  task_count=$((task_count + 1))
done < "${tasks_file}"

if [[ ${task_count} -eq 0 ]]; then
  echo "This submit helper expects at least one non-comment task. Got 0." >&2
  exit 2
fi

node_count=$(( (task_count + 3) / 4 ))

{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=${job_name}"
  echo "#SBATCH --time=${time_limit}"
  echo "#SBATCH --constraint=${constraint}"
  echo "#SBATCH --nodes=${node_count}"
  echo "#SBATCH --ntasks-per-node=4"
  echo "#SBATCH --gpus-per-node=4"
  echo "#SBATCH --cpus-per-task=${cpus_per_task}"
  echo "#SBATCH --output=${bundle_root}/slurm/%x-%j.out"
  echo "#SBATCH --error=${bundle_root}/slurm/%x-%j.err"
  if [[ -n "${account}" ]]; then
    echo "#SBATCH --account=${account}"
  fi
  if [[ -n "${qos}" ]]; then
    echo "#SBATCH --qos=${qos}"
  fi
  echo
  echo "set -euo pipefail"
  printf "export REPO_ROOT=%q\n" "${repo_root}"
  printf "export TASKS_FILE=%q\n" "${normalized_tasks}"
  printf "export BUNDLE_ROOT=%q\n" "${bundle_root}"
  printf "export WRAPPER=%q\n" "${wrapper}"
  printf "export JSON_NAME=%q\n" "${json_name}"
  printf "export HIST_NAME=%q\n" "${hist_name}"
  printf "exec /bin/bash %q\n" "${launcher}"
} > "${sbatch_file}"

chmod +x "${sbatch_file}"

echo "Prepared HMC bundle:"
echo "  repo_root: ${repo_root}"
echo "  bundle_root: ${bundle_root}"
echo "  normalized_tasks: ${normalized_tasks}"
echo "  sbatch_file: ${sbatch_file}"
echo "  task_count: ${task_count}"
echo "  node_count: ${node_count}"

if [[ ${print_only} -eq 1 ]]; then
  echo "sbatch ${sbatch_file}"
  exit 0
fi

sbatch "${sbatch_file}"
