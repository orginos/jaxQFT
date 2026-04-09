#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_campaign_nersc.sh [options]

Submit the canonical-point uniform-model training campaign on NERSC.

This campaign scans:
  - points from configs/phi4/paper-2/canonical-point-scan/points.tsv
  - widths 64 and 48
  - volumes L = 16, 32, 64, 128
  - seeds 0,1,2,3

Optional:
  --repo-root PATH      Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH       Runtime root. Default:
                        /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan
  --points PATH         Point TSV. Default:
                        configs/phi4/paper-2/canonical-point-scan/points.tsv
  --widths LIST         Comma-separated widths. Default: 64,48
  --volumes LIST        Comma-separated volumes. Default: 16,32,64,128
  --seeds LIST          Comma-separated seeds. Default: 0,1,2,3
  --lam FLOAT           Quartic coupling. Default: 2.4
  --account NAME        Slurm account. Default: hadron
  --qos NAME            Slurm QOS. Default: shared
  --print-only          Print the generated submit commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan"
points_file="configs/phi4/paper-2/canonical-point-scan/points.tsv"
widths_csv="64,48"
volumes_csv="16,32,64,128"
seeds_csv="0,1,2,3"
lam="2.4"
account="${NERSC_ACCOUNT:-hadron}"
qos="shared"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --points) points_file="${2:-}"; shift 2 ;;
    --widths) widths_csv="${2:-}"; shift 2 ;;
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

submit="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_nersc.sh"
if [[ ! -x "${submit}" ]]; then
  echo "Submit helper not executable: ${submit}" >&2
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

base_config_for_L() {
  case "$1" in
    16|32|64|128) echo "${repo_root}/configs/phi4/paper-2/canonical-scaling/L${1}_uniform.toml" ;;
    *)
      echo "Unsupported volume: $1" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a widths <<< "${widths_csv}"
IFS=',' read -r -a volumes <<< "${volumes_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

declare -a submitted=()

while IFS=$'\t' read -r label mass note; do
  [[ -z "${label}" ]] && continue
  [[ "${label}" == \#* ]] && continue
  for width in "${widths[@]}"; do
    for L in "${volumes[@]}"; do
      config="$(base_config_for_L "${L}")"
      walltime="$(time_for_L "${L}")"
      for seed in "${seeds[@]}"; do
        run_dir="${run_root}/${label}/w${width}/L${L}/s${seed}"
        job_name="phi4-${label}-w${width}-L${L}-s${seed}"
        cmd=(
          "${submit}"
          --config "${config}"
          --run-dir "${run_dir}"
          --seed "${seed}"
          --job-name "${job_name}"
          --time "${walltime}"
          --account "${account}"
          --qos "${qos}"
          --
          --lam "${lam}"
          --mass "${mass}"
          --width "${width}"
        )
        if [[ ${print_only} -eq 1 ]]; then
          printf '%q ' "${cmd[@]}"
          printf '\n'
        else
          out="$("${cmd[@]}")"
          printf '%s\n' "${out}"
          submitted+=("${job_name}")
        fi
      done
    done
  done
done < "${points_file}"

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} canonical-point training jobs."
fi
