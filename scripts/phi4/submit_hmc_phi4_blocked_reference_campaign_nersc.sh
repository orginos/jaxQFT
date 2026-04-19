#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_blocked_reference_campaign_nersc.sh [options]

Submit the blocked-reference phi^4 HMC campaign on NERSC. This is the same
production HMC scan machinery, but with blocked-level measurements enabled and
defaults chosen for the RT-reference program.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Production run root. Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/blocked-reference
  --lam FLOAT               Fixed g4=lambda. Default: 2.4
  --volumes LIST            Comma-separated volumes. Default: 64,128
  --g2-file PATH            Tracked g2 scan file. Default:
                            configs/phi4/paper-2/hmc-g2-scan/blocked_core_points.tsv
  --settings-file PATH      Tracked tuned production settings. Default:
                            configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv
  --replicas INT            Number of independent replicas per point. Default: 1
  --replica-start INT       Starting replica index. Default: 0
  --nwarm INT               Default: 2000
  --nmeas INT               Default: 10000
  --nskip INT               Default: 20
  --tau FLOAT               Default: 1.0
  --k-max INT               Default: 8
  --blocked-rg-mode NAME    Default: average
  --qos NAME                Slurm QOS. Default: shared
  --account NAME            Slurm account. Default: hadron
  --print-only              Print sbatch commands and exit.

Notes:
  - The HMC runner does not currently checkpoint or resume.
  - This campaign always enables --measure-blocked-levels.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/blocked-reference"
lam="2.4"
volumes_csv="64,128"
g2_file="configs/phi4/paper-2/hmc-g2-scan/blocked_core_points.tsv"
settings_file="configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv"
replicas="1"
replica_start="0"
nwarm="2000"
nmeas="10000"
nskip="20"
tau="1.0"
k_max="8"
blocked_rg_mode="average"
qos="shared"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
    --volumes) volumes_csv="${2:-}"; shift 2 ;;
    --g2-file) g2_file="${2:-}"; shift 2 ;;
    --settings-file) settings_file="${2:-}"; shift 2 ;;
    --replicas) replicas="${2:-}"; shift 2 ;;
    --replica-start) replica_start="${2:-}"; shift 2 ;;
    --nwarm) nwarm="${2:-}"; shift 2 ;;
    --nmeas) nmeas="${2:-}"; shift 2 ;;
    --nskip) nskip="${2:-}"; shift 2 ;;
    --tau) tau="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
    --blocked-rg-mode) blocked_rg_mode="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

submit="${repo_root}/scripts/phi4/submit_hmc_phi4_nersc.sh"
if [[ ! -x "${submit}" ]]; then
  echo "Submit helper not executable: ${submit}" >&2
  exit 2
fi

if [[ "${g2_file}" != /* ]]; then
  g2_file="${repo_root}/${g2_file}"
fi
if [[ "${settings_file}" != /* ]]; then
  settings_file="${repo_root}/${settings_file}"
fi
if [[ ! -f "${g2_file}" ]]; then
  echo "g2 file not found: ${g2_file}" >&2
  exit 2
fi
if [[ ! -f "${settings_file}" ]]; then
  echo "settings file not found: ${settings_file}" >&2
  exit 2
fi

tag_mass() {
  local mass="$1"
  local tag="${mass}"
  tag="${tag//-/m}"
  tag="${tag//+/p}"
  tag="${tag//./p}"
  echo "${tag}"
}

declare -A batch_by_L=()
declare -A nmd_by_L=()
declare -A time_by_L=()
declare -A integrator_by_L=()

while read -r L batch nmd walltime integrator _; do
  [[ -z "${L}" || "${L}" == \#* ]] && continue
  if [[ "${L}" == "L" || "${L}" == "volume" ]]; then
    continue
  fi
  batch_by_L["${L}"]="${batch}"
  nmd_by_L["${L}"]="${nmd}"
  time_by_L["${L}"]="${walltime}"
  integrator_by_L["${L}"]="${integrator:-minnorm2}"
done < "${settings_file}"

IFS=',' read -r -a volumes <<< "${volumes_csv}"
submitted=0
mass_index=0
while read -r mass; do
  [[ -z "${mass}" || "${mass}" == \#* ]] && continue
  if [[ ! "${mass}" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]]; then
    continue
  fi
  for L in "${volumes[@]}"; do
    if [[ -z "${batch_by_L[${L}]:-}" || -z "${nmd_by_L[${L}]:-}" || -z "${time_by_L[${L}]:-}" ]]; then
      echo "Missing tuned settings for L=${L}" >&2
      exit 2
    fi
    batch="${batch_by_L[${L}]}"
    nmd="${nmd_by_L[${L}]}"
    time_limit="${time_by_L[${L}]}"
    integrator="${integrator_by_L[${L}]:-minnorm2}"
    mass_tag="$(tag_mass "${mass}")"
    for ((rep=replica_start; rep<replica_start+replicas; rep++)); do
      seed=$(( L * 100000 + mass_index * 100 + rep ))
      if [[ ${replicas} -gt 1 || ${replica_start} -ne 0 ]]; then
        run_dir="${run_root}/L${L}/g2_${mass}/rep${rep}"
        job_name="phi4-g2-blocked-L${L}-${mass_tag}-rep${rep}"
      else
        run_dir="${run_root}/L${L}/g2_${mass}"
        job_name="phi4-g2-blocked-L${L}-${mass_tag}"
      fi
      cmd=(
        "${submit}"
        --run-dir "${run_dir}"
        --job-name "${job_name}"
        --time "${time_limit}"
        --account "${account}"
        --qos "${qos}"
        --
        --shape "${L},${L}"
        --lam "${lam}"
        --mass "${mass}"
        --nwarm "${nwarm}"
        --nmeas "${nmeas}"
        --nskip "${nskip}"
        --batch-size "${batch}"
        --nmd "${nmd}"
        --tau "${tau}"
        --integrator "${integrator}"
        --seed "${seed}"
        --k-max "${k_max}"
        --measure-blocked-levels
        --blocked-rg-mode "${blocked_rg_mode}"
      )
      if [[ ${print_only} -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
      else
        out="$("${cmd[@]}")"
        printf '%s\n' "${out}"
        submitted=$((submitted + 1))
      fi
    done
  done
  mass_index=$((mass_index + 1))
done < "${g2_file}"

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${submitted} blocked-reference HMC jobs."
fi
