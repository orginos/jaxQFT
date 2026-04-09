#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_level_analysis_perlevel_campaign_nersc.sh [options]

Submit level-analysis jobs for the tuned per-level checkpoint set on NERSC.

Optional:
  --repo-root PATH          Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --run-root PATH           Analysis run root. Default:
                            /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/per-level-level-analysis
  --nsamples INT            Total iid model samples per checkpoint. Default: 65536
  --jk-bin-size INT         Jackknife bin size. Default: 256
  --k-max INT               Momentum scan depth. Default: 4
  --locality-nsamples INT   Number of full fields per level for locality. Default: 16
  --locality-nsources INT   Number of probe sources per field. Default: 4
  --locality-metrics LIST   Comma-separated metrics. Default: manhattan
  --locality-fit-rmax FLOAT Legacy/user fit upper bound. Default: 0
  --reweight-ess-min FLOAT  Reportability threshold for reweighted observables. Default: 0.05
  --time HH:MM:SS           Walltime for all jobs. Default: 03:00:00
  --qos NAME                Slurm QOS. Default: shared
  --account NAME            Slurm account. Default: hadron
  --print-only              Print sbatch commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/per-level-level-analysis"
nsamples="65536"
jk_bin_size="256"
k_max="4"
locality_nsamples="16"
locality_nsources="4"
locality_metrics="manhattan"
locality_fit_rmax="0"
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
    --locality-metrics) locality_metrics="${2:-}"; shift 2 ;;
    --locality-fit-rmax) locality_fit_rmax="${2:-}"; shift 2 ;;
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

declare -a specs=(
  "L16|rg_coarse_eta_gauss_fresh_example.pkl|4096"
  "L32|rg_coarse_eta_gauss_L32_m-0.4_l2.4_perlevel_w128-96-64-96_nc3-3-2-3_r1-1-1-2_eglevel_tglearned.pkl|2048"
  "L64|rg_coarse_eta_gauss_L64_m-0.4_l2.4_perlevel_w160-128-96-64-96_nc3-3-3-2-3_r1-1-1-1-2_eglevel_tglearned.pkl|512"
  "L128|rg_coarse_eta_gauss_L128_m-0.4_l2.4_perlevel_w192-160-128-96-64-96_nc3-3-3-3-2-3_r1-1-1-1-1-2_eglevel_tglearned.pkl|128"
)

declare -a submitted=()

for spec in "${specs[@]}"; do
  IFS='|' read -r tag ckpt_name batch_size <<< "${spec}"
  checkpoint="${repo_root}/runs/phi4/per-level-run-0/${ckpt_name}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    continue
  fi
  run_dir="${run_root}/${tag}"
  job_name="phi4-perlvl-${tag}"
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
    --locality-metric "${locality_metrics}"
    --locality-fit-rmax "${locality_fit_rmax}"
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

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} per-level level-analysis jobs."
fi
