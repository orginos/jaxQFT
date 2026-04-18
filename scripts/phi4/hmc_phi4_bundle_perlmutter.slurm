#!/bin/bash
#SBATCH --job-name=phi4-hmc-bundle
#SBATCH --account=hadron
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TASKS_FILE="${TASKS_FILE:-}"
BUNDLE_ROOT="${BUNDLE_ROOT:-}"
WRAPPER="${WRAPPER:-${REPO_ROOT}/scripts/phi4/run_hmc_phi4.sh}"
JSON_NAME="${JSON_NAME:-hmc_phi4.json}"
HIST_NAME="${HIST_NAME:-hmc_phi4.npz}"
TASK_RUNNER="${TASK_RUNNER:-${SCRIPT_DIR}/hmc_phi4_bundle_task.sh}"

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

if [[ -z "${TASKS_FILE}" ]]; then
  echo "TASKS_FILE is required." >&2
  echo "Set TASKS_FILE to a TSV with at least one non-comment task row." >&2
  exit 2
fi
if [[ "${TASKS_FILE}" != /* ]]; then
  TASKS_FILE="${REPO_ROOT}/${TASKS_FILE}"
fi
if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "Tasks file not found: ${TASKS_FILE}" >&2
  exit 2
fi

if [[ -z "${BUNDLE_ROOT}" ]]; then
  BUNDLE_ROOT="$(cd "$(dirname "${TASKS_FILE}")" && pwd)"
elif [[ "${BUNDLE_ROOT}" != /* ]]; then
  BUNDLE_ROOT="${REPO_ROOT}/${BUNDLE_ROOT}"
fi

if [[ "${WRAPPER}" != /* ]]; then
  WRAPPER="${REPO_ROOT}/${WRAPPER}"
fi
if [[ ! -x "${WRAPPER}" ]]; then
  echo "Wrapper not executable: ${WRAPPER}" >&2
  exit 2
fi
if [[ ! -x "${TASK_RUNNER}" ]]; then
  echo "Task runner not executable: ${TASK_RUNNER}" >&2
  exit 2
fi

mkdir -p "${BUNDLE_ROOT}/slurm"
exec >> "${BUNDLE_ROOT}/slurm/job.log" 2>&1

echo "Perlmutter HMC bundle launcher"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  TASKS_FILE=${TASKS_FILE}"
echo "  BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "  WRAPPER=${WRAPPER}"
echo "  JSON_NAME=${JSON_NAME}"
echo "  HIST_NAME=${HIST_NAME}"
echo "  TASK_RUNNER=${TASK_RUNNER}"

declare -a TASK_NAMES=()
declare -a TASK_RUN_DIRS=()
declare -a TASK_ARG_COUNTS=()
declare -a TASK_EXTRA_ARGS=()

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

  task_name="${fields[0]}"
  run_dir="${fields[1]}"

  if contains_exact "${task_name}" "${TASK_NAMES[@]-}"; then
    echo "Duplicate task name in ${TASKS_FILE}: ${task_name}" >&2
    exit 2
  fi

  if [[ "${run_dir}" != /* ]]; then
    run_dir="${REPO_ROOT}/${run_dir}"
  fi
  if contains_exact "${run_dir}" "${TASK_RUN_DIRS[@]-}"; then
    echo "Duplicate run_dir in ${TASKS_FILE}: ${run_dir}" >&2
    exit 2
  fi

  TASK_NAMES+=("${task_name}")
  TASK_RUN_DIRS+=("${run_dir}")
  extra_count=$(( ${#fields[@]} - 2 ))
  TASK_ARG_COUNTS+=("${extra_count}")
  if [[ ${extra_count} -gt 0 ]]; then
    TASK_EXTRA_ARGS+=("${fields[@]:2}")
  fi
done < "${TASKS_FILE}"

task_count="${#TASK_NAMES[@]}"
if [[ "${task_count}" -eq 0 ]]; then
  echo "This launcher expects at least one non-comment task. Got 0." >&2
  exit 2
fi

node_count=$(( (task_count + 3) / 4 ))

echo "  task_count=${task_count}"
echo "  node_count=${node_count}"
TASK_STATUS_DIR="${BUNDLE_ROOT}/slurm/task_status_${SLURM_JOB_ID}"
mkdir -p "${TASK_STATUS_DIR}"
export REPO_ROOT TASKS_FILE BUNDLE_ROOT WRAPPER JSON_NAME HIST_NAME TASK_STATUS_DIR

if srun \
  --kill-on-bad-exit=0 \
  --ntasks="${task_count}" \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  "${TASK_RUNNER}"; then
  srun_status=0
else
  srun_status=$?
fi

fail_count=0
: > "${BUNDLE_ROOT}/slurm/success_tasks.txt"
: > "${BUNDLE_ROOT}/slurm/failed_tasks.txt"
: > "${BUNDLE_ROOT}/slurm/failed_task_status.tsv"

for task_name in "${TASK_NAMES[@]}"; do
  status_file="${TASK_STATUS_DIR}/${task_name}.status"
  if [[ -f "${status_file}" ]]; then
    status="$(cat "${status_file}")"
  else
    status=99
  fi
  if [[ "${status}" == "0" ]]; then
    echo "${task_name}" >> "${BUNDLE_ROOT}/slurm/success_tasks.txt"
  else
    echo "${task_name}" >> "${BUNDLE_ROOT}/slurm/failed_tasks.txt"
    printf "%s\t%s\n" "${task_name}" "${status}" >> "${BUNDLE_ROOT}/slurm/failed_task_status.tsv"
    echo "Task ${task_name} failed with exit status ${status}"
    fail_count=$((fail_count + 1))
  fi
done

if [[ ${fail_count} -gt 0 ]]; then
  echo "${fail_count} task(s) failed; see ${BUNDLE_ROOT}/slurm/failed_tasks.txt"
  exit 1
fi
if [[ "${srun_status}" -ne 0 ]]; then
  echo "srun returned nonzero status ${srun_status} despite zero recorded task failures" >&2
  exit "${srun_status}"
fi
