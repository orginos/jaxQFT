#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_forward_accum4_lr5e5_repairs_nersc.sh [options]

Submit the unresolved w64c3 forward L128 repairs as fresh replacement bundles,
using the batch32 + grad_accum_steps=4 + ramp-lr=5e-5 repair card.

Optional:
  --repo-root PATH       Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --bundle-root PATH     Parent bundle root. Default:
                         /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward/w64c3-accum4-lr5e5-repairs-20260418
  --time-a HH:MM:SS      Walltime for bundle A. Default: 00:55:00
  --time-b HH:MM:SS      Walltime for bundle B. Default: 00:55:00
  --qos NAME             Slurm QOS. Default: regular
  --account NAME         Slurm account. Default: $NERSC_ACCOUNT or hadron
  --print-only           Print submit commands and exit
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward/w64c3-accum4-lr5e5-repairs-20260418"
time_a="00:55:00"
time_b="00:55:00"
qos="regular"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --time-a) time_a="${2:-}"; shift 2 ;;
    --time-b) time_b="${2:-}"; shift 2 ;;
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

bundle_a_tasks="${repo_root}/configs/phi4/paper-2/canonical-point-scan/forward-repair-wave-20260418-lr5e5/w64c3_bundle_a.tsv"
bundle_b_tasks="${repo_root}/configs/phi4/paper-2/canonical-point-scan/forward-repair-wave-20260418-lr5e5/w64c3_bundle_b.tsv"
submit_helper="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"

for path in "${bundle_a_tasks}" "${bundle_b_tasks}" "${submit_helper}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path missing: ${path}" >&2
    exit 2
  fi
done

assert_fresh_runs() {
  local tasks_file="$1"
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
    local run_dir="${fields[2]}"
    if [[ "${run_dir}" != /* ]]; then
      run_dir="${repo_root}/${run_dir}"
    fi
    if [[ -e "${run_dir}" ]]; then
      echo "Refusing to overwrite existing run dir: ${run_dir}" >&2
      exit 2
    fi
  done < "${tasks_file}"
}

assert_fresh_runs "${bundle_a_tasks}"
assert_fresh_runs "${bundle_b_tasks}"

declare -a cmd_a=(
  /bin/bash "${submit_helper}"
  --repo-root "${repo_root}"
  --tasks "${bundle_a_tasks}"
  --bundle-root "${bundle_root}/bundle-a"
  --job-name "phi4-can-w64c3-fwd-accum4-lr5e5-a"
  --time "${time_a}"
  --account "${account}"
  --qos "${qos}"
)
declare -a cmd_b=(
  /bin/bash "${submit_helper}"
  --repo-root "${repo_root}"
  --tasks "${bundle_b_tasks}"
  --bundle-root "${bundle_root}/bundle-b"
  --job-name "phi4-can-w64c3-fwd-accum4-lr5e5-b"
  --time "${time_b}"
  --account "${account}"
  --qos "${qos}"
)

if [[ ${print_only} -eq 1 ]]; then
  printf '%q ' "${cmd_a[@]}"
  printf '\n'
  printf '%q ' "${cmd_b[@]}"
  printf '\n'
  exit 0
fi

"${cmd_a[@]}"
"${cmd_b[@]}"
