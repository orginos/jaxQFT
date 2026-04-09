#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_tuning_campaign_nersc.sh [options]

Submit the canonical phi^4 HMC tuning campaign on NERSC.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Tuning run root. Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/tuning
  --grid-file PATH          Tuning grid. Default:
                            configs/phi4/paper-2/hmc-g2-scan/tuning_grid.tsv
  --lam FLOAT               Fixed g4=lambda. Default: 2.4
  --mass FLOAT              Canonical g2=m^2 point for tuning. Default: -0.4
  --nwarm INT               Default: 1000
  --nmeas INT               Default: 1000
  --nskip INT               Default: 20
  --tau FLOAT               Default: 1.0
  --integrator NAME         Default integrator if grid omits one. Default: minnorm2
  --k-max INT               Default: 4
  --time HH:MM:SS           Walltime for all jobs. Default: 01:00:00
  --qos NAME                Slurm QOS. Default: shared
  --account NAME            Slurm account. Default: hadron
  --print-only              Print sbatch commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/tuning"
grid_file="configs/phi4/paper-2/hmc-g2-scan/tuning_grid.tsv"
lam="2.4"
mass="-0.4"
nwarm="1000"
nmeas="1000"
nskip="20"
tau="1.0"
integrator="minnorm2"
k_max="4"
time_limit="01:00:00"
qos="shared"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --grid-file) grid_file="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
    --mass) mass="${2:-}"; shift 2 ;;
    --nwarm) nwarm="${2:-}"; shift 2 ;;
    --nmeas) nmeas="${2:-}"; shift 2 ;;
    --nskip) nskip="${2:-}"; shift 2 ;;
    --tau) tau="${2:-}"; shift 2 ;;
    --integrator) integrator="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
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

if [[ "${grid_file}" != /* ]]; then
  grid_file="${repo_root}/${grid_file}"
fi
if [[ ! -f "${grid_file}" ]]; then
  echo "Grid file not found: ${grid_file}" >&2
  exit 2
fi

submitted=0
while read -r L batch nmd grid_integrator grid_time _; do
  [[ -z "${L}" || "${L}" == \#* ]] && continue
  if [[ "${L}" == "L" || "${L}" == "volume" ]]; then
    continue
  fi
  use_integrator="${grid_integrator:-${integrator}}"
  use_time="${grid_time:-${time_limit}}"
  run_dir="${run_root}/L${L}/${use_integrator}/b${batch}_nmd${nmd}"
  job_name="phi4-hmc-L${L}-${use_integrator}-b${batch}-nmd${nmd}"
  cmd=(
    "${submit}"
    --run-dir "${run_dir}"
    --job-name "${job_name}"
    --time "${use_time}"
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
    --integrator "${use_integrator}"
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
done < "${grid_file}"

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${submitted} HMC tuning jobs."
fi
