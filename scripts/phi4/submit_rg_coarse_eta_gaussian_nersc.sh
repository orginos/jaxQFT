#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_nersc.sh [options]

Required:
  --config PATH          Repo-relative or absolute TOML config.
  --run-dir PATH         Run directory. The script copies the config here,
                         writes the Slurm file here, and localizes outputs here.

Optional:
  --seed INT             Training seed override. Default: 0.
  --job-name NAME        Slurm job name. Default: basename of run dir.
  --time HH:MM:SS        Walltime. Default: 06:00:00.
  --gpus INT             GPUs per node. Default: 1.
  --constraint NAME      Slurm constraint. Default: gpu.
  --account NAME         Slurm account. Default: $NERSC_ACCOUNT or hadron.
  --qos NAME             Optional Slurm QOS.
  --partition NAME       Optional Slurm partition.
  --gpu ID|all           Value passed to the wrapper. Default: 0.
  --mode NAME            Wrapper mode. Default: interactive.
  --checkpoint NAME      Checkpoint file name inside run dir. Default: checkpoint.pkl.
  --wrapper PATH         Wrapper script to call. Default: scripts/phi4/run_rg_coarse_eta_gaussian.sh
  --print-only           Print the generated sbatch command and exit.
  --                     Remaining args are passed to the trainer after the config.

Example:
  scripts/phi4/submit_rg_coarse_eta_gaussian_nersc.sh \
    --config configs/phi4/paper-2/canonical-scaling/L128_uniform.toml \
    --run-dir runs/phi4/paper-2/canonical-scaling/L128/s0 \
    --seed 0 --time 08:00:00
EOF
}

config=""
run_dir=""
seed="0"
job_name=""
time_limit="06:00:00"
gpus="1"
constraint="gpu"
account="${NERSC_ACCOUNT:-hadron}"
qos=""
partition=""
gpu="0"
mode="interactive"
checkpoint_name="checkpoint.pkl"
wrapper="scripts/phi4/run_rg_coarse_eta_gaussian.sh"
print_only=0
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:-}"; shift 2 ;;
    --run-dir) run_dir="${2:-}"; shift 2 ;;
    --seed) seed="${2:-}"; shift 2 ;;
    --job-name) job_name="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --gpus) gpus="${2:-}"; shift 2 ;;
    --constraint) constraint="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --partition) partition="${2:-}"; shift 2 ;;
    --gpu) gpu="${2:-}"; shift 2 ;;
    --mode) mode="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint_name="${2:-}"; shift 2 ;;
    --wrapper) wrapper="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; passthrough=("$@"); break ;;
    *) passthrough+=("$1"); shift ;;
  esac
done

if [[ -z "${config}" || -z "${run_dir}" ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [[ "${config}" != /* ]]; then
  config="${repo_root}/${config}"
fi
if [[ ! -f "${config}" ]]; then
  echo "Config not found: ${config}" >&2
  exit 2
fi

if [[ "${run_dir}" != /* ]]; then
  run_dir="${repo_root}/${run_dir}"
fi
mkdir -p "${run_dir}/slurm"

if [[ "${wrapper}" != /* ]]; then
  wrapper="${repo_root}/${wrapper}"
fi
if [[ ! -x "${wrapper}" ]]; then
  echo "Wrapper not executable: ${wrapper}" >&2
  exit 2
fi

if [[ -z "${job_name}" ]]; then
  job_name="$(basename "${run_dir}")"
fi

config_copy="${run_dir}/input.toml"
cp "${config}" "${config_copy}"

checkpoint_path="${run_dir}/${checkpoint_name}"
sbatch_file="${run_dir}/job.sbatch"

{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=${job_name}"
  echo "#SBATCH --time=${time_limit}"
  echo "#SBATCH --constraint=${constraint}"
  echo "#SBATCH --gpus=${gpus}"
  echo "#SBATCH --output=${run_dir}/slurm/%x-%j.out"
  echo "#SBATCH --error=${run_dir}/slurm/%x-%j.err"
  if [[ -n "${account}" ]]; then
    echo "#SBATCH --account=${account}"
  fi
  if [[ -n "${qos}" ]]; then
    echo "#SBATCH --qos=${qos}"
  fi
  if [[ -n "${partition}" ]]; then
    echo "#SBATCH --partition=${partition}"
  fi
  echo
  echo "set -euo pipefail"
  printf "%q " "${wrapper}"
  printf -- "--mode %q " "${mode}"
  printf -- "--config %q " "${config_copy}"
  printf -- "--workdir %q " "${run_dir}"
  printf -- "--gpu %q " "${gpu}"
  printf -- "-- "
  printf -- "--seed %q " "${seed}"
  printf -- "--save %q " "${checkpoint_path}"
  for arg in "${passthrough[@]}"; do
    printf "%q " "${arg}"
  done
  printf "\n"
} > "${sbatch_file}"

chmod +x "${sbatch_file}"

echo "Prepared NERSC run:"
echo "  repo_root: ${repo_root}"
echo "  run_dir: ${run_dir}"
echo "  config_copy: ${config_copy}"
echo "  checkpoint: ${checkpoint_path}"
echo "  sbatch_file: ${sbatch_file}"

if [[ ${print_only} -eq 1 ]]; then
  echo "sbatch ${sbatch_file}"
  exit 0
fi

sbatch "${sbatch_file}"
