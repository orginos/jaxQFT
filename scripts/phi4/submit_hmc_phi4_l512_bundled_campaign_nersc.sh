#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_l512_bundled_campaign_nersc.sh [options]

Build and submit bundled L=512 phi^4 HMC production jobs on Perlmutter using
the tuned batch/integrator choice.

Behavior:
  - one task = one independent HMC stream on one GPU
  - tasks are grouped into bundles and submitted with qos=regular by default
  - each task uses the same tuned L=512 production settings

Optional:
  --repo-root PATH         Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH          Run root. Default:
                           /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production-L512
  --bundle-root PATH       Bundle root. Default: <run-root>/_bundles
  --g2-file PATH           Tracked g2 list. Default:
                           configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv
  --lam FLOAT              Default: 2.4
  --replicas INT           Independent replicas per g2 point. Default: 8
  --replica-start INT      Starting replica index. Default: 0
  --nwarm INT              Default: 2000
  --nmeas INT              Default: 10000
  --nskip INT              Default: 20
  --tau FLOAT              Default: 1.0
  --k-max INT              Default: 8
  --batch-size INT         Default: 16
  --nmd INT                Default: 10
  --integrator NAME        Default: forcegrad
  --time HH:MM:SS          Default: 02:00:00
  --tasks-per-bundle INT   Max tasks per submitted bundle. Default: 16
  --qos NAME               Default: regular
  --account NAME           Default: $NERSC_ACCOUNT or hadron
  --constraint NAME        Default: gpu
  --cpus-per-task INT      Default: 32
  --print-only             Print generated bundle submissions and exit.

Notes:
  - The default g2 list is the same 13-point broad scan used at `L=256`.
  - With the defaults, the helper emits `13 * 8 = 104` independent streams.
  - If replicas=1, each run lives in <run-root>/L512/g2_<mass>.
  - If replicas>1 or replica-start!=0, each run lives in
    <run-root>/L512/g2_<mass>/rep<rep>.
EOF
}

tag_mass() {
  local mass="$1"
  local tag="${mass}"
  tag="${tag//-/m}"
  tag="${tag//+/p}"
  tag="${tag//./p}"
  printf '%s\n' "${tag}"
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

flush_bundle() {
  local bundle_index="$1"
  local bundle_task_count="$2"
  local bundle_tasks_file="$3"
  local bundle_dir="$4"
  local submit_out=""
  local cmd=(
    "${submit_bundle}"
    --repo-root "${repo_root}"
    --tasks "${bundle_tasks_file}"
    --bundle-root "${bundle_dir}"
    --job-name "$(printf 'phi4-hmc-L512-b%03d' "${bundle_index}")"
    --time "${time_limit}"
    --account "${account}"
    --qos "${qos}"
    --constraint "${constraint}"
    --cpus-per-task "${cpus_per_task}"
  )
  if [[ ${print_only} -eq 1 ]]; then
    cmd+=(--print-only)
  fi
  printf 'Bundle %03d: %d tasks\n' "${bundle_index}" "${bundle_task_count}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ ${print_only} -eq 0 ]]; then
    submit_out="$("${cmd[@]}")"
    printf '%s\n' "${submit_out}"
  fi
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production-L512"
bundle_root=""
g2_file="configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv"
lam="2.4"
replicas="8"
replica_start="0"
nwarm="2000"
nmeas="10000"
nskip="20"
tau="1.0"
k_max="8"
batch_size="16"
nmd="10"
integrator="forcegrad"
time_limit="02:00:00"
tasks_per_bundle="16"
qos="regular"
account="${NERSC_ACCOUNT:-hadron}"
constraint="gpu"
cpus_per_task="32"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --g2-file) g2_file="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
    --replicas) replicas="${2:-}"; shift 2 ;;
    --replica-start) replica_start="${2:-}"; shift 2 ;;
    --nwarm) nwarm="${2:-}"; shift 2 ;;
    --nmeas) nmeas="${2:-}"; shift 2 ;;
    --nskip) nskip="${2:-}"; shift 2 ;;
    --tau) tau="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
    --batch-size) batch_size="${2:-}"; shift 2 ;;
    --nmd) nmd="${2:-}"; shift 2 ;;
    --integrator) integrator="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --tasks-per-bundle) tasks_per_bundle="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --constraint) constraint="${2:-}"; shift 2 ;;
    --cpus-per-task) cpus_per_task="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${run_root}" != /* ]]; then
  run_root="${repo_root}/${run_root}"
fi
if [[ -z "${bundle_root}" ]]; then
  bundle_root="${run_root}/_bundles"
elif [[ "${bundle_root}" != /* ]]; then
  bundle_root="${repo_root}/${bundle_root}"
fi

g2_file="$(resolve_input_path "${g2_file}")"
if [[ ! -f "${g2_file}" ]]; then
  echo "g2 file not found: ${g2_file}" >&2
  exit 2
fi

submit_bundle="${repo_root}/scripts/phi4/submit_hmc_phi4_bundle_nersc.sh"
if [[ ! -x "${submit_bundle}" ]]; then
  echo "Submit helper not executable: ${submit_bundle}" >&2
  exit 2
fi

if ! [[ "${replicas}" =~ ^[0-9]+$ ]] || [[ "${replicas}" -lt 1 ]]; then
  echo "replicas must be a positive integer" >&2
  exit 2
fi
if ! [[ "${replica_start}" =~ ^[0-9]+$ ]]; then
  echo "replica-start must be a nonnegative integer" >&2
  exit 2
fi
if ! [[ "${tasks_per_bundle}" =~ ^[0-9]+$ ]] || [[ "${tasks_per_bundle}" -lt 1 ]]; then
  echo "tasks-per-bundle must be a positive integer" >&2
  exit 2
fi

mkdir -p "${run_root}/L512" "${bundle_root}"

bundle_index=0
bundle_task_count=0
current_tasks_file="${bundle_root}/bundle_$(printf '%03d' "${bundle_index}").tsv"
current_bundle_dir="${bundle_root}/bundle_$(printf '%03d' "${bundle_index}")"
: > "${current_tasks_file}"

mass_index=0
total_tasks=0
while read -r mass; do
  [[ -z "${mass}" || "${mass}" == \#* ]] && continue
  if [[ ! "${mass}" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]]; then
    continue
  fi
  mass_tag="$(tag_mass "${mass}")"
  for ((rep=replica_start; rep<replica_start+replicas; rep++)); do
    seed=$(( 512 * 100000 + mass_index * 100 + rep ))
    if [[ "${replicas}" -gt 1 || "${replica_start}" -ne 0 ]]; then
      run_dir="${run_root}/L512/g2_${mass}/rep${rep}"
      task_name="phi4-hmc-L512-${mass_tag}-rep${rep}"
    else
      run_dir="${run_root}/L512/g2_${mass}"
      task_name="phi4-hmc-L512-${mass_tag}"
    fi

    {
      printf "%s\t%s" "${task_name}" "${run_dir}"
      printf "\t%s\t%s" "--shape" "512,512"
      printf "\t%s\t%s" "--lam" "${lam}"
      printf "\t%s\t%s" "--mass" "${mass}"
      printf "\t%s\t%s" "--nwarm" "${nwarm}"
      printf "\t%s\t%s" "--nmeas" "${nmeas}"
      printf "\t%s\t%s" "--nskip" "${nskip}"
      printf "\t%s\t%s" "--batch-size" "${batch_size}"
      printf "\t%s\t%s" "--nmd" "${nmd}"
      printf "\t%s\t%s" "--tau" "${tau}"
      printf "\t%s\t%s" "--integrator" "${integrator}"
      printf "\t%s\t%s" "--seed" "${seed}"
      printf "\t%s\t%s" "--k-max" "${k_max}"
      printf "\n"
    } >> "${current_tasks_file}"

    bundle_task_count=$((bundle_task_count + 1))
    total_tasks=$((total_tasks + 1))

    if [[ "${bundle_task_count}" -ge "${tasks_per_bundle}" ]]; then
      mkdir -p "${current_bundle_dir}"
      flush_bundle "${bundle_index}" "${bundle_task_count}" "${current_tasks_file}" "${current_bundle_dir}"
      bundle_index=$((bundle_index + 1))
      bundle_task_count=0
      current_tasks_file="${bundle_root}/bundle_$(printf '%03d' "${bundle_index}").tsv"
      current_bundle_dir="${bundle_root}/bundle_$(printf '%03d' "${bundle_index}")"
      : > "${current_tasks_file}"
    fi
  done
  mass_index=$((mass_index + 1))
done < "${g2_file}"

if [[ ${bundle_task_count} -gt 0 ]]; then
  mkdir -p "${current_bundle_dir}"
  flush_bundle "${bundle_index}" "${bundle_task_count}" "${current_tasks_file}" "${current_bundle_dir}"
fi

echo "Prepared ${total_tasks} L512 HMC task(s) under ${bundle_root}"
