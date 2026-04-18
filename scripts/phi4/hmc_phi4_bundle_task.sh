#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TASKS_FILE="${TASKS_FILE:-}"
BUNDLE_ROOT="${BUNDLE_ROOT:-}"
WRAPPER="${WRAPPER:-${REPO_ROOT}/scripts/phi4/run_hmc_phi4.sh}"
JSON_NAME="${JSON_NAME:-hmc_phi4.json}"
HIST_NAME="${HIST_NAME:-hmc_phi4.npz}"
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
  if [[ ${#fields[@]} -lt 2 ]]; then
    echo "Expected at least 2 TSV columns per task in ${TASKS_FILE}" >&2
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
run_dir="${fields[1]}"
status_file="${TASK_STATUS_DIR}/${task_name}.status"

if [[ "${run_dir}" != /* ]]; then
  run_dir="${REPO_ROOT}/${run_dir}"
fi

mkdir -p "${run_dir}/slurm"

task_record="${run_dir}/bundle_task.tsv"
{
  printf "%s\t%s" "${task_name}" "${run_dir}"
  if [[ ${#fields[@]} -gt 2 ]]; then
    for arg in "${fields[@]:2}"; do
      printf "\t%s" "${arg}"
    done
  fi
  printf "\n"
} > "${task_record}"

json_out="${run_dir}/${JSON_NAME}"
hist_out="${run_dir}/${HIST_NAME}"

declare -a cmd=(
  "${WRAPPER}"
  "--workdir" "${run_dir}"
  "--gpu" "all"
  "--"
  "--json-out" "${json_out}"
  "--hist-out" "${hist_out}"
)
if [[ ${#fields[@]} -gt 2 ]]; then
  cmd+=("${fields[@]:2}")
fi

set +e
{
  echo "Launching task ${task_name}"
  echo "  slurm_procid=${task_index}"
  echo "  run_dir=${run_dir}"
  if [[ ${#fields[@]} -gt 2 ]]; then
    echo "  extra_args=${fields[*]:2}"
  else
    echo "  extra_args=<none>"
  fi
  "${cmd[@]}"
} > "${run_dir}/slurm/hmc.out" 2> "${run_dir}/slurm/hmc.err"
status=$?
set -e

printf '%s\n' "${status}" > "${status_file}"
exit "${status}"
