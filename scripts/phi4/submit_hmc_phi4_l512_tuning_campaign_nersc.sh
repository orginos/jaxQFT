#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_l512_tuning_campaign_nersc.sh [options]

Submit the dedicated L=512 HMC tuning campaign, including minnorm2 and
force-gradient integrators.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/tuning-L512
  --grid-file PATH          Default:
                            configs/phi4/paper-2/hmc-g2-scan/tuning_grid_L512.tsv
  --mass FLOAT              Default: -0.595
  --lam FLOAT               Default: 2.4
  --nwarm INT               Default: 1000
  --nmeas INT               Default: 1000
  --nskip INT               Default: 20
  --tau FLOAT               Default: 1.0
  --k-max INT               Default: 8
  --qos NAME                Default: shared
  --account NAME            Default: hadron
  --print-only              Print commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/tuning-L512"
grid_file="configs/phi4/paper-2/hmc-g2-scan/tuning_grid_L512.tsv"
mass="-0.595"
lam="2.4"
nwarm="1000"
nmeas="1000"
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
    --grid-file) grid_file="${2:-}"; shift 2 ;;
    --mass) mass="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
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

submit="${repo_root}/scripts/phi4/submit_hmc_phi4_tuning_campaign_nersc.sh"
cmd=(
  "${submit}"
  --repo-root "${repo_root}"
  --run-root "${run_root}"
  --grid-file "${grid_file}"
  --mass "${mass}"
  --lam "${lam}"
  --nwarm "${nwarm}"
  --nmeas "${nmeas}"
  --nskip "${nskip}"
  --tau "${tau}"
  --k-max "${k_max}"
  --qos "${qos}"
  --account "${account}"
)

if [[ ${print_only} -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
