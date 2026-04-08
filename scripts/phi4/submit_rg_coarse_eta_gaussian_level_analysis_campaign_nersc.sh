#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_campaign_nersc.sh [options]

Submit the canonical-model level-analysis campaign on NERSC.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Analysis run root. Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-level-analysis
  --nsamples INT            Total iid model samples per checkpoint. Default: 65536
  --jk-bin-size INT         Jackknife bin size. Default: 256
  --k-max INT               Momentum scan depth. Default: 4
  --locality-nsamples INT   Number of full fields per level for locality. Default: 16
  --locality-nsources INT   Number of probe sources per field. Default: 4
  --reweight-ess-min FLOAT  Reportability threshold for reweighted observables. Default: 0.05
  --time HH:MM:SS           Walltime for all jobs. Default: 03:00:00
  --qos NAME                Slurm QOS. Default: shared
  --account NAME            Slurm account. Default: hadron
  --print-only              Print sbatch commands and exit.

Checkpoint set:
  L16:  s0 s1 s2 s3
  L32:  s0 s1 s2 s3
  L64:  s0 s1 s2 s3
  L128: s0 s1 s3 s4
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-level-analysis"
nsamples="65536"
jk_bin_size="256"
k_max="4"
locality_nsamples="16"
locality_nsources="4"
reweight_ess_min="0.05"
time_limit="03:00:00"
qos="shared"
account="${NERSC_ACCOUNT:-hadron}"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --nsamples) nsamples="${2:-}"; shift 2 ;;
    --jk-bin-size) jk_bin_size="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
    --locality-nsamples) locality_nsamples="${2:-}"; shift 2 ;;
    --locality-nsources) locality_nsources="${2:-}"; shift 2 ;;
    --reweight-ess-min) reweight_ess_min="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

submit="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_nersc.sh"
if [[ ! -x "${submit}" ]]; then
  echo "Submit helper not executable: ${submit}" >&2
  exit 2
fi

batch_for_L() {
  case "$1" in
    16) echo "4096" ;;
    32) echo "2048" ;;
    64) echo "512" ;;
    128) echo "128" ;;
    *)
      echo "Unsupported L: $1" >&2
      return 2
      ;;
  esac
}

seeds_for_L() {
  case "$1" in
    16|32|64) echo "0 1 2 3" ;;
    128) echo "0 1 3 4" ;;
    *)
      echo "Unsupported L: $1" >&2
      return 2
      ;;
  esac
}

declare -a submitted=()

for L in 16 32 64 128; do
  batch_size="$(batch_for_L "${L}")"
  for seed in $(seeds_for_L "${L}"); do
    checkpoint="${repo_root}/runs/phi4/canonical-scaling-continuation/L${L}/s${seed}/checkpoint.pkl"
    if [[ ${L} -eq 128 && ${seed} -eq 4 ]]; then
      checkpoint="${repo_root}/runs/phi4/canonical-scaling-continuation/L128/s4/checkpoint.pkl"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint: ${checkpoint}" >&2
      continue
    fi
    run_dir="${run_root}/L${L}/s${seed}"
    job_name="phi4-L${L}-levels-s${seed}"
    cmd=(
      "${submit}"
      --checkpoint "${checkpoint}"
      --run-dir "${run_dir}"
      --job-name "${job_name}"
      --time "${time_limit}"
      --account "${account}"
      --qos "${qos}"
      --
      --nsamples "${nsamples}"
      --batch-size "${batch_size}"
      --jk-bin-size "${jk_bin_size}"
      --k-max "${k_max}"
      --reweight-ess-min "${reweight_ess_min}"
      --locality
      --locality-nsamples "${locality_nsamples}"
      --locality-nsources "${locality_nsources}"
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

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} canonical level-analysis jobs."
fi
