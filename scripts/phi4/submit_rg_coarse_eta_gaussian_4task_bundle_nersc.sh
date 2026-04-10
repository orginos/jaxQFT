#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh [options]

Submit a nonzero multiple of 4 RG coarse-eta Gaussian flow tasks as one
regular-qos Perlmutter bundle, with one independent task per GPU.

Required:
  --tasks PATH           TSV manifest with a nonzero multiple of 4
                         non-comment task rows.
  --bundle-root PATH     Bundle directory. The script writes:
                         - tasks.tsv (normalized absolute manifest)
                         - job.sbatch
                         - slurm/*.out,*.err,job.log

Behavior:
  - tasks are packed at 4 tasks per node
  - node count is computed automatically as task_count / 4

Manifest format:
  Tab-separated fields. The first four columns are required:
    1. task_name
    2. config path
    3. run_dir path
    4. seed
  Any additional tab-separated columns are passed verbatim to the trainer
  after --save checkpoint.pkl. This is the intended place for
  point-specific overrides such as --lam, --mass, and --width.

Optional:
  --repo-root PATH       Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --job-name NAME        Slurm job name. Default: basename of bundle root.
  --time HH:MM:SS        Walltime. Default: 08:00:00.
  --account NAME         Slurm account. Default: $NERSC_ACCOUNT or hadron.
  --qos NAME             Slurm QOS. Default: regular.
  --constraint NAME      Slurm constraint. Default: gpu.
  --cpus-per-task INT    CPUs per task. Default: 32.
  --mode NAME            Wrapper mode. Default: interactive.
  --wrapper PATH         Wrapper script. Default: scripts/phi4/run_rg_coarse_eta_gaussian.sh
  --launcher PATH        Slurm launcher. Default: scripts/phi4/rg_coarse_eta_gaussian_4task_perlmutter.slurm
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
time_limit="08:00:00"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
constraint="gpu"
cpus_per_task="32"
mode="interactive"
wrapper="scripts/phi4/run_rg_coarse_eta_gaussian.sh"
launcher="scripts/phi4/rg_coarse_eta_gaussian_4task_perlmutter.slurm"
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
    --mode) mode="${2:-}"; shift 2 ;;
    --wrapper) wrapper="${2:-}"; shift 2 ;;
    --launcher) launcher="${2:-}"; shift 2 ;;
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
  if [[ ${#fields[@]} -lt 4 ]]; then
    echo "Expected at least 4 TSV columns per task in ${tasks_file}" >&2
    exit 2
  fi

  task_name="${fields[0]}"
  config="${fields[1]}"
  run_dir="${fields[2]}"
  seed="${fields[3]}"

  if contains_exact "${task_name}" "${seen_names[@]-}"; then
    echo "Duplicate task name in ${tasks_file}: ${task_name}" >&2
    exit 2
  fi
  seen_names+=("${task_name}")

  if [[ "${config}" != /* ]]; then
    config="${repo_root}/${config}"
  fi
  if [[ ! -f "${config}" ]]; then
    echo "Config not found for task ${task_name}: ${config}" >&2
    exit 2
  fi

  if [[ "${run_dir}" != /* ]]; then
    run_dir="${repo_root}/${run_dir}"
  fi
  if contains_exact "${run_dir}" "${seen_run_dirs[@]-}"; then
    echo "Duplicate run_dir in ${tasks_file}: ${run_dir}" >&2
    exit 2
  fi
  seen_run_dirs+=("${run_dir}")

  {
    printf "%s\t%s\t%s\t%s" "${task_name}" "${config}" "${run_dir}" "${seed}"
    if [[ ${#fields[@]} -gt 4 ]]; then
      for arg in "${fields[@]:4}"; do
        printf "\t%s" "${arg}"
      done
    fi
    printf "\n"
  } >> "${normalized_tasks}"
  task_count=$((task_count + 1))
done < "${tasks_file}"

if [[ ${task_count} -eq 0 || $(( task_count % 4 )) -ne 0 ]]; then
  echo "This submit helper expects a nonzero multiple of 4 non-comment tasks. Got ${task_count}." >&2
  exit 2
fi

node_count=$(( task_count / 4 ))

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
  printf "export MODE=%q\n" "${mode}"
  printf "export WRAPPER=%q\n" "${wrapper}"
  printf "exec /bin/bash %q\n" "${launcher}"
} > "${sbatch_file}"

chmod +x "${sbatch_file}"

echo "Prepared bundle:"
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
