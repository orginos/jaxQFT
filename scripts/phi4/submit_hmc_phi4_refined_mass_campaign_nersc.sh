#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_hmc_phi4_refined_mass_campaign_nersc.sh [options]

Submit the refined near-critical HMC mass scan using the current tuned
production settings for L = 16,32,64,128,256.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production-refined
  --g2-file PATH            Default:
                            configs/phi4/paper-2/hmc-g2-scan/refined_g2_points.tsv
  --lam FLOAT               Default: 2.4
  --qos NAME                Default: shared
  --account NAME            Default: hadron
  --print-only              Print commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/hmc-g2-scan/production-refined"
g2_file="configs/phi4/paper-2/hmc-g2-scan/refined_g2_points.tsv"
lam="2.4"
qos="shared"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --g2-file) g2_file="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

submit="${repo_root}/scripts/phi4/submit_hmc_phi4_production_campaign_nersc.sh"
cmd=(
  "${submit}"
  --repo-root "${repo_root}"
  --run-root "${run_root}"
  --g2-file "${g2_file}"
  --volumes "16,32,64,128,256"
  --lam "${lam}"
  --qos "${qos}"
  --account "${account}"
)

if [[ ${print_only} -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
