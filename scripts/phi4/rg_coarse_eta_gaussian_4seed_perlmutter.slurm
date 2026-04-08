#!/bin/bash
#SBATCH --job-name=phi4-rg4
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

# Edit these variables before submission or override them with:
# sbatch --export=ALL,CONFIG=...,RUN_ROOT=...,SEEDS=0,1,2,3 scripts/phi4/rg_coarse_eta_gaussian_4seed_perlmutter.slurm

REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG="${CONFIG:-configs/phi4/canonical-scaling/L128_uniform.toml}"
RUN_ROOT="${RUN_ROOT:-runs/phi4/canonical-scaling/L128}"
SEEDS_CSV="${SEEDS:-0,1,2,3}"
MODE="${MODE:-interactive}"
WRAPPER="${WRAPPER:-$REPO_ROOT/scripts/phi4/run_rg_coarse_eta_gaussian.sh}"

# Extra trainer args, space separated. Example:
# EXTRA_ARGS="--validate-batch 8 --validate-super 2"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
if [[ "${#SEEDS[@]}" -ne 4 ]]; then
  echo "This script expects exactly 4 seeds, one per GPU. Got: ${SEEDS_CSV}" >&2
  exit 2
fi

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}"
fi

mkdir -p "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/slurm"
exec >> "${RUN_ROOT}/slurm/job.log" 2>&1

echo "Perlmutter 4-seed launcher"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  CONFIG=${CONFIG}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  SEEDS=${SEEDS[*]}"
echo "  MODE=${MODE}"

launch_one() {
  local seed="$1"
  local run_dir="${RUN_ROOT}/s${seed}"
  mkdir -p "${run_dir}/slurm"
  cp "${CONFIG}" "${run_dir}/input.toml"
  local cmd=(
    "${WRAPPER}"
    "--mode" "${MODE}"
    "--config" "${run_dir}/input.toml"
    "--workdir" "${run_dir}"
    "--gpu" "all"
    "--"
    "--seed" "${seed}"
    "--save" "checkpoint.pkl"
  )
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_ARGS} )
    cmd+=("${extra[@]}")
  fi

  echo "Launching seed ${seed} in ${run_dir}"
  srun \
    --exact \
    --exclusive \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${cmd[@]}" \
    > "${run_dir}/slurm/train.out" \
    2> "${run_dir}/slurm/train.err" &
}

for seed in "${SEEDS[@]}"; do
  launch_one "${seed}"
done

wait
