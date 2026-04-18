#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TASKS_FILE="${TASKS_FILE:-}"
BUNDLE_ROOT="${BUNDLE_ROOT:-}"
MODE="${MODE:-interactive}"
WRAPPER="${WRAPPER:-${REPO_ROOT}/scripts/phi4/run_rg_coarse_eta_gaussian.sh}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint.pkl}"
TASK_STATUS_DIR="${TASK_STATUS_DIR:-${BUNDLE_ROOT}/slurm/task_status}"

if [[ -z "${TASKS_FILE}" || -z "${BUNDLE_ROOT}" ]]; then
  echo "TASKS_FILE and BUNDLE_ROOT are required." >&2
  exit 2
fi

if [[ "${TASKS_FILE}" != /* ]]; then
  TASKS_FILE="${REPO_ROOT}/${TASKS_FILE}"
fi
if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "Tasks file not found: ${TASKS_FILE}" >&2
  exit 2
fi

if [[ "${WRAPPER}" != /* ]]; then
  WRAPPER="${REPO_ROOT}/${WRAPPER}"
fi
if [[ ! -x "${WRAPPER}" ]]; then
  echo "Wrapper not executable: ${WRAPPER}" >&2
  exit 2
fi

task_index="${SLURM_PROCID:-}"
if ! [[ "${task_index}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_PROCID must be set to a nonnegative integer." >&2
  exit 2
fi

mkdir -p "${TASK_STATUS_DIR}"

declare -a fields=()
current_index=0
found=0

while IFS=$'\t' read -r -a fields || [[ ${#fields[@]} -gt 0 ]]; do
  if [[ ${#fields[@]} -eq 0 ]]; then
    continue
  fi
  if [[ "${fields[0]}" == \#* ]]; then
    continue
  fi
  if [[ ${#fields[@]} -lt 4 ]]; then
    echo "Expected at least 4 TSV columns per task in ${TASKS_FILE}" >&2
    exit 2
  fi
  if [[ "${current_index}" -eq "${task_index}" ]]; then
    found=1
    break
  fi
  current_index=$((current_index + 1))
done < "${TASKS_FILE}"

if [[ "${found}" -ne 1 ]]; then
  echo "No task found for SLURM_PROCID=${task_index} in ${TASKS_FILE}" >&2
  exit 2
fi

task_name="${fields[0]}"
config="${fields[1]}"
run_dir="${fields[2]}"
seed="${fields[3]}"
status_file="${TASK_STATUS_DIR}/${task_name}.status"

if [[ "${config}" != /* ]]; then
  config="${REPO_ROOT}/${config}"
fi
if [[ ! -f "${config}" ]]; then
  printf '2\n' > "${status_file}"
  echo "Config not found for task ${task_name}: ${config}" >&2
  exit 2
fi

if [[ "${run_dir}" != /* ]]; then
  run_dir="${REPO_ROOT}/${run_dir}"
fi

mkdir -p "${run_dir}/slurm"
config_copy="${run_dir}/input.toml"
if [[ "${config}" != "${config_copy}" ]]; then
  cp "${config}" "${config_copy}"
fi

task_record="${run_dir}/bundle_task.tsv"
{
  printf "%s\t%s\t%s\t%s" "${task_name}" "${config}" "${run_dir}" "${seed}"
  if [[ ${#fields[@]} -gt 4 ]]; then
    for arg in "${fields[@]:4}"; do
      printf "\t%s" "${arg}"
    done
  fi
  printf "\n"
} > "${task_record}"

declare -a cmd=(
  "${WRAPPER}"
  "--mode" "${MODE}"
  "--config" "${config_copy}"
  "--workdir" "${run_dir}"
  "--gpu" "all"
  "--"
  "--seed" "${seed}"
  "--save" "${CHECKPOINT_NAME}"
)
if [[ ${#fields[@]} -gt 4 ]]; then
  cmd+=("${fields[@]:4}")
fi

set +e
{
  echo "Launching task ${task_name}"
  echo "  slurm_procid=${task_index}"
  echo "  run_dir=${run_dir}"
  echo "  config=${config}"
  echo "  seed=${seed}"
  if [[ ${#fields[@]} -gt 4 ]]; then
    echo "  extra_args=${fields[*]:4}"
  else
    echo "  extra_args=<none>"
  fi
  "${cmd[@]}"
} > "${run_dir}/slurm/train.out" 2> "${run_dir}/slurm/train.err"
status=$?
set -e

printf '%s\n' "${status}" > "${status_file}"
exit "${status}"
