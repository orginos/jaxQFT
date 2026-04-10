#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_w64c3_bundles_nersc.sh [options]

Submit the canonical-point w64c3 campaign on NERSC as regular-qos bundles,
grouped by common volume. Each volume bundle contains one task per
point/seed combination and therefore uses 4 tasks per node and task_count/4
nodes overall.

This campaign scans:
  - points from configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv
  - architecture tag w64c3
  - volumes L = 16, 32, 64, 128
  - seeds 0,1,2,3

Special case:
  - canonical3/L128 uses the batch-64-from-start rescue card
    L128_uniform_c3_batch64_then_anneal.toml

Optional:
  --repo-root PATH      Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH       Runtime root. Default:
                        /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan
  --bundle-root PATH    Bundle root. Default:
                        /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan/w64c3
  --points PATH         Point TSV. Default:
                        configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv
  --volumes LIST        Comma-separated volumes. Default: 16,32,64,128
  --seeds LIST          Comma-separated seeds. Default: 0,1,2,3
  --lam FLOAT           Quartic coupling. Default: 2.4
  --account NAME        Slurm account. Default: hadron
  --qos NAME            Slurm QOS. Default: regular
  --print-only          Print the generated submit commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan/w64c3"
points_file="configs/phi4/paper-2/canonical-point-scan/w64c3_points.tsv"
volumes_csv="16,32,64,128"
seeds_csv="0,1,2,3"
lam="2.4"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --points) points_file="${2:-}"; shift 2 ;;
    --volumes) volumes_csv="${2:-}"; shift 2 ;;
    --seeds) seeds_csv="${2:-}"; shift 2 ;;
    --lam) lam="${2:-}"; shift 2 ;;
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

if [[ "${points_file}" != /* ]]; then
  points_file="${repo_root}/${points_file}"
fi
if [[ ! -f "${points_file}" ]]; then
  echo "Points file not found: ${points_file}" >&2
  exit 2
fi

time_for_L() {
  case "$1" in
    16) echo "04:00:00" ;;
    32) echo "04:00:00" ;;
    64) echo "06:00:00" ;;
    128) echo "08:00:00" ;;
    *)
      echo "Unsupported volume: $1" >&2
      return 2
      ;;
  esac
}

config_for_point_and_L() {
  local label="$1"
  local L="$2"
  if [[ "${label}" == "canonical3" && "${L}" == "128" ]]; then
    echo "${repo_root}/configs/phi4/paper-2/canonical-point-scan/L128_uniform_c3_batch64_then_anneal.toml"
    return 0
  fi
  case "${L}" in
    16|32|64|128) echo "${repo_root}/configs/phi4/paper-2/canonical-scaling/L${L}_uniform_c3.toml" ;;
    *)
      echo "Unsupported volume: ${L}" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a volumes <<< "${volumes_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

declare -a submitted=()

for L in "${volumes[@]}"; do
  bundle_dir="${bundle_root}/L${L}"
  tasks_in="${bundle_dir}/tasks_in.tsv"
  walltime="$(time_for_L "${L}")"
  mkdir -p "${bundle_dir}"
  : > "${tasks_in}"

  while IFS=$'\t' read -r label mass note; do
    [[ -z "${label}" ]] && continue
    [[ "${label}" == \#* ]] && continue
    config="$(config_for_point_and_L "${label}" "${L}")"
    for seed in "${seeds[@]}"; do
      run_dir="${run_root}/${label}/w64c3/L${L}/s${seed}"
      task_name="phi4-${label}-w64c3-L${L}-s${seed}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${task_name}" \
        "${config}" \
        "${run_dir}" \
        "${seed}" \
        "--lam" "${lam}" \
        "--mass" "${mass}" \
        "--width" "64" \
        "--n-cycles" "3" \
        >> "${tasks_in}"
    done
  done < "${points_file}"

  cmd=(
    "${bundle_submit}"
    --repo-root "${repo_root}"
    --tasks "${tasks_in}"
    --bundle-root "${bundle_dir}"
    --job-name "phi4-can-w64c3-L${L}"
    --time "${walltime}"
    --account "${account}"
    --qos "${qos}"
  )
  if [[ ${print_only} -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    out="$("${cmd[@]}")"
    printf '%s\n' "${out}"
    submitted+=("L${L}")
  fi
done

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} bundled w64c3 canonical-point jobs."
fi
