#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_forward_accum4_warmup_lr3e5_canonical3_repairs_nersc.sh [options]

Submit the unresolved canonical3 w64c3 forward L128 repairs as fresh
replacement runs, using the batch32 + grad_accum_steps=4 + warmup/anneal card
with peak lr reduced to 3e-5.

Optional:
  --repo-root PATH       Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --bundle-root PATH     Parent bundle root. Default:
                         /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward/w64c3-accum4-warmup-lr3e5-canonical3-repairs-20260418
  --time HH:MM:SS        Walltime for the bundle. Default: 01:15:00
  --qos NAME             Slurm QOS. Default: regular
  --account NAME         Slurm account. Default: $NERSC_ACCOUNT or hadron
  --print-only           Print submit command and exit
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward/w64c3-accum4-warmup-lr3e5-canonical3-repairs-20260418"
time_limit="01:15:00"
qos="regular"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${repo_root}" != /* ]]; then
  repo_root="$(cd "${repo_root}" && pwd)"
fi
if [[ "${bundle_root}" != /* ]]; then
  bundle_root="${repo_root}/${bundle_root}"
fi

tasks_file="${repo_root}/configs/phi4/paper-2/canonical-point-scan/forward-repair-wave-20260418-accum4-warmup-lr3e5-canonical3/w64c3_bundle.tsv"
submit_helper="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"

for path in "${tasks_file}" "${submit_helper}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path missing: ${path}" >&2
    exit 2
  fi
done

while IFS=$'\t' read -r -a fields || [[ ${#fields[@]} -gt 0 ]]; do
  if [[ ${#fields[@]} -eq 0 ]]; then
    continue
  fi
  if [[ "${fields[0]}" == \#* ]]; then
    continue
  fi
  if [[ ${#fields[@]} -lt 3 ]]; then
    echo "Malformed task row in ${tasks_file}" >&2
    exit 2
  fi
  run_dir="${fields[2]}"
  if [[ "${run_dir}" != /* ]]; then
    run_dir="${repo_root}/${run_dir}"
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run dir: ${run_dir}" >&2
    exit 2
  fi
done < "${tasks_file}"

declare -a cmd=(
  /bin/bash "${submit_helper}"
  --repo-root "${repo_root}"
  --tasks "${tasks_file}"
  --bundle-root "${bundle_root}/bundle"
  --job-name "phi4-can3-w64c3-fwd-accum4-warmup-lr3e5"
  --time "${time_limit}"
  --account "${account}"
  --qos "${qos}"
)

if [[ ${print_only} -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
