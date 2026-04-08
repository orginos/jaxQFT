#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_nersc.sh [options]

Required:
  --checkpoint PATH       Repo-relative or absolute checkpoint path.
  --run-dir PATH          Run directory. Slurm logs and analysis JSON stay here.

Optional:
  --job-name NAME         Slurm job name. Default: basename of run dir.
  --time HH:MM:SS         Walltime. Default: 02:00:00.
  --gpus INT              GPUs per node. Default: 1.
  --constraint NAME       Slurm constraint. Default: gpu.
  --account NAME          Slurm account. Default: $NERSC_ACCOUNT or hadron.
  --qos NAME              Optional Slurm QOS. Default: shared
  --partition NAME        Optional Slurm partition.
  --gpu ID|all            Value passed to the wrapper. Default: 0.
  --json-name NAME        JSON file name inside run dir. Default: flow_levels.json.
  --wrapper PATH          Wrapper script to call. Default:
                          scripts/phi4/run_rg_coarse_eta_gaussian_level_analysis.sh
  --print-only            Print the generated sbatch command and exit.
  --                      Remaining args are passed to the analysis script.
EOF
}

checkpoint=""
run_dir=""
job_name=""
time_limit="02:00:00"
gpus="1"
constraint="gpu"
account="${NERSC_ACCOUNT:-hadron}"
qos="shared"
partition=""
gpu="0"
json_name="flow_levels.json"
wrapper="scripts/phi4/run_rg_coarse_eta_gaussian_level_analysis.sh"
print_only=0
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --run-dir) run_dir="${2:-}"; shift 2 ;;
    --job-name) job_name="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --gpus) gpus="${2:-}"; shift 2 ;;
    --constraint) constraint="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --partition) partition="${2:-}"; shift 2 ;;
    --gpu) gpu="${2:-}"; shift 2 ;;
    --json-name) json_name="${2:-}"; shift 2 ;;
    --wrapper) wrapper="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; passthrough=("$@"); break ;;
    *) passthrough+=("$1"); shift ;;
  esac
done

if [[ -z "${checkpoint}" || -z "${run_dir}" ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [[ "${checkpoint}" != /* ]]; then
  checkpoint="${repo_root}/${checkpoint}"
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
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

sbatch_file="${run_dir}/job.sbatch"
json_out="${run_dir}/${json_name}"

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
  printf -- "--checkpoint %q " "${checkpoint}"
  printf -- "--workdir %q " "${run_dir}"
  printf -- "--json-out %q " "${json_out}"
  printf -- "--gpu %q " "${gpu}"
  printf -- "-- "
  for arg in "${passthrough[@]}"; do
    printf "%q " "${arg}"
  done
  printf "\n"
} > "${sbatch_file}"

chmod +x "${sbatch_file}"

echo "Prepared NERSC level-analysis run:"
echo "  repo_root: ${repo_root}"
echo "  run_dir: ${run_dir}"
echo "  checkpoint: ${checkpoint}"
echo "  json_out: ${json_out}"
echo "  sbatch_file: ${sbatch_file}"

if [[ ${print_only} -eq 1 ]]; then
  echo "sbatch ${sbatch_file}"
  exit 0
fi

sbatch "${sbatch_file}"
