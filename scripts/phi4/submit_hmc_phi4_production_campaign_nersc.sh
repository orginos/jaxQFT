#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_production_campaign_nersc.sh [options]

Submit the production phi^4 HMC g2 scan on NERSC using tuned `(batch_size, nmd)`
choices by volume.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Production run root. Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production
  --lam FLOAT               Fixed g4=lambda. Default: 2.4
  --g2-file PATH            Tracked g2 scan file. Default:
                            configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv
  --settings-file PATH      Tracked tuned production settings. Default:
                            configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv
  --nwarm INT               Default: 2000
  --nmeas INT               Default: 10000
  --nskip INT               Default: 20
  --tau FLOAT               Default: 1.0
  --k-max INT               Default: 8
  --qos NAME                Slurm QOS. Default: shared
  --account NAME            Slurm account. Default: hadron
  --print-only              Print sbatch commands and exit.

Notes:
  - The HMC runner does not currently checkpoint or resume.
  - Walltimes come from the tuned settings file and are intentionally conservative.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production"
lam="2.4"
g2_file="configs/phi4/paper-2/hmc-g2-scan/g2_points.tsv"
settings_file="configs/phi4/paper-2/hmc-g2-scan/production_tuned.tsv"
nwarm="2000"
nmeas="10000"
nskip="20"
tau="1.0"
k_max="8"
qos="shared"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
    --g2-file) g2_file="${2:-}"; shift 2 ;;
    --settings-file) settings_file="${2:-}"; shift 2 ;;
    --nwarm) nwarm="${2:-}"; shift 2 ;;
    --nmeas) nmeas="${2:-}"; shift 2 ;;
    --nskip) nskip="${2:-}"; shift 2 ;;
    --tau) tau="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
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

while read -r L batch nmd walltime; do
  [[ -z "${L}" || "${L}" == \#* ]] && continue
  batch_by_L["${L}"]="${batch}"
  nmd_by_L["${L}"]="${nmd}"
  time_by_L["${L}"]="${walltime}"
done < "${settings_file}"

submitted=0
while read -r mass; do
  [[ -z "${mass}" || "${mass}" == \#* ]] && continue
  for L in 16 32 64 128 256; do
    if [[ -z "${batch_by_L[${L}]:-}" || -z "${nmd_by_L[${L}]:-}" || -z "${time_by_L[${L}]:-}" ]]; then
      echo "Missing tuned settings for L=${L}" >&2
      exit 2
    fi
    batch="${batch_by_L[${L}]}"
    nmd="${nmd_by_L[${L}]}"
    time_limit="${time_by_L[${L}]}"
    mass_tag="$(tag_mass "${mass}")"
    run_dir="${run_root}/L${L}/g2_${mass}"
    job_name="phi4-g2-L${L}-${mass_tag}"
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
      --k-max "${k_max}"
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
done < "${g2_file}"

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${submitted} HMC production jobs."
fi
