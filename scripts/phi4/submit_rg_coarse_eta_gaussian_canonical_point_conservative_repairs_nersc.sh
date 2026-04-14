#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_conservative_repairs_nersc.sh [options]

Submit the April 14, 2026 conservative canonical-point repair wave.

The wave contains:
  - a 3-task L128 tail bundle for the remaining canonical2 repairs
  - a 12-task canonical3 L128 conservative bundle
  - a 12-task canonical4 L32 conservative bundle
  - a 12-task mixed L64 conservative bundle
  - a 12-task canonical4 L128 conservative bundle

Optional:
  --repo-root PATH      Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --bundle-root PATH    Bundle root. Default:
                        /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan/_bundles_conservative_repair_20260414
  --account NAME        Slurm account. Default: hadron
  --qos NAME            Slurm QOS. Default: regular
  --print-only          Print the generated submit commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan/_bundles_conservative_repair_20260414"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

bundle_submit="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"
if [[ ! -x "${bundle_submit}" ]]; then
  echo "Bundle submit helper not executable: ${bundle_submit}" >&2
  exit 2
fi

declare -a manifests=(
  "configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/canonical2_l128_tail.tsv"
  "configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/canonical3_l128_conservative.tsv"
  "configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/canonical4_l32_conservative.tsv"
  "configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/canonical4plus_canonical3_l64_conservative.tsv"
  "configs/phi4/paper-2/canonical-point-scan/repair-wave-20260414/canonical4_l128_conservative.tsv"
)
declare -a bundle_names=(
  "phi4-can-repair-c2-L128"
  "phi4-can-repair-c3-L128"
  "phi4-can-repair-c4-L32"
  "phi4-can-repair-L64"
  "phi4-can-repair-c4-L128"
)
declare -a time_limits=(
  "00:45:00"
  "00:45:00"
  "00:30:00"
  "00:35:00"
  "00:45:00"
)

declare -a submitted=()

for idx in "${!manifests[@]}"; do
  manifest="${manifests[$idx]}"
  name="${bundle_names[$idx]}"
  time_limit="${time_limits[$idx]}"
  cmd=(
    "${bundle_submit}"
    --repo-root "${repo_root}"
    --tasks "${repo_root}/${manifest}"
    --bundle-root "${bundle_root}/$(basename "${manifest}" .tsv)"
    --job-name "${name}"
    --time "${time_limit}"
    --account "${account}"
    --qos "${qos}"
  )
  if [[ ${print_only} -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    out="$("${cmd[@]}")"
    printf '%s\n' "${out}"
    submitted+=("${name}")
  fi
done

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} conservative canonical repair bundles."
fi
