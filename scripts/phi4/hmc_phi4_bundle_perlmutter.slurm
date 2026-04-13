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

mkdir -p "${BUNDLE_ROOT}/slurm"
exec >> "${BUNDLE_ROOT}/slurm/job.log" 2>&1

echo "Perlmutter HMC bundle launcher"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  TASKS_FILE=${TASKS_FILE}"
echo "  BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "  WRAPPER=${WRAPPER}"
echo "  JSON_NAME=${JSON_NAME}"
echo "  HIST_NAME=${HIST_NAME}"

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

launch_one() {
  local idx="$1"
  local task_name="${TASK_NAMES[$idx]}"
  local run_dir="${TASK_RUN_DIRS[$idx]}"
  local extra_count="${TASK_ARG_COUNTS[$idx]}"
  local arg_offset=0
  local extra=()
  local j=0
  local task_record=""
  local json_out=""
  local hist_out=""
  local arg=""

  for ((j = 0; j < idx; j++)); do
    arg_offset=$(( arg_offset + TASK_ARG_COUNTS[j] ))
  done
  for ((j = 0; j < extra_count; j++)); do
    extra+=("${TASK_EXTRA_ARGS[$((arg_offset + j))]}")
  done

  mkdir -p "${run_dir}/slurm"
  task_record="${run_dir}/bundle_task.tsv"
  {
    printf "%s\t%s" "${task_name}" "${run_dir}"
    for arg in "${extra[@]}"; do
      printf "\t%s" "${arg}"
    done
    printf "\n"
  } > "${task_record}"

  json_out="${run_dir}/${JSON_NAME}"
  hist_out="${run_dir}/${HIST_NAME}"

  local cmd=(
    "${WRAPPER}"
    "--workdir" "${run_dir}"
    "--gpu" "all"
    "--"
    "--json-out" "${json_out}"
    "--hist-out" "${hist_out}"
  )
  if [[ ${#extra[@]} -gt 0 ]]; then
    cmd+=("${extra[@]}")
  fi

  echo "Launching task ${task_name}"
  echo "  run_dir=${run_dir}"
  echo "  extra_args=${extra[*]:-<none>}"

  srun \
    --exact \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${cmd[@]}" \
    > "${run_dir}/slurm/hmc.out" \
    2> "${run_dir}/slurm/hmc.err" &
  local pid=$!
  PIDS+=("${pid}")
  PID_TASKS+=("${task_name}")
}

declare -a PIDS=()
declare -a PID_TASKS=()

for idx in "${!TASK_NAMES[@]}"; do
  launch_one "${idx}"
done

fail_count=0
: > "${BUNDLE_ROOT}/slurm/success_tasks.txt"
: > "${BUNDLE_ROOT}/slurm/failed_tasks.txt"
: > "${BUNDLE_ROOT}/slurm/failed_task_status.tsv"

for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  task_name="${PID_TASKS[$idx]}"
  if wait "${pid}"; then
    echo "${task_name}" >> "${BUNDLE_ROOT}/slurm/success_tasks.txt"
  else
    status=$?
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
