#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_forward_level_analysis_bundles_nersc.sh [options]

Submit bundled level-analysis jobs for the completed forward-loss canonical-point
training runs. Tasks are grouped by architecture and volume, with up to 4
independent checkpoint analyses per node.

Default scope:
  - points: canonical, canonical2, canonical3, canonical4
  - arches: w64, w48
  - volumes: 16,32,64,128
  - seeds: 0,1,2,3
  - shards: 4 independent analysis calls per checkpoint

Optional:
  --repo-root PATH            Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --checkpoint-root PATH      Source checkpoint root. Default:
                              /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward
  --run-root PATH             Analysis output root. Default:
                              /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-level-analysis/before_refine4096
  --bundle-root PATH          Bundle root. Default:
                              /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward-level-analysis/before_refine4096
  --points PATH               Point TSV. Default:
                              configs/phi4/paper-2/canonical-point-scan/forward_points.tsv
  --arches LIST               Comma-separated architectures. Default: w64,w48
  --volumes LIST              Comma-separated volumes. Default: 16,32,64,128
  --seeds LIST                Comma-separated seeds. Default: 0,1,2,3
  --checkpoint-name NAME      Checkpoint file name inside each run dir. Default: checkpoint.pkl
  --shards INT                Number of independent analysis calls per checkpoint. Default: 4
  --nsamples INT              Total i.i.d. model samples per checkpoint. Default: 65536
  --jk-bin-size INT           Jackknife bin size. Default: 256
  --k-max INT                 Momentum depth. Default: 4
  --locality-nsamples INT     Locality sample count. Default: 16
  --locality-nsources INT     Sources per locality field. Default: 4
  --locality-metrics LIST     Comma-separated metrics. Default: manhattan,euclidean2
  --locality-fit-rmin FLOAT   Default: 1.0
  --locality-fit-rmax FLOAT   Default: 0.0
  --reweight-ess-min FLOAT    Default: 0.05
  --account NAME              Slurm account. Default: $NERSC_ACCOUNT or hadron
  --qos NAME                  Slurm QOS. Default: regular
  --print-only                Print submit commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
checkpoint_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-level-analysis/before_refine4096"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward-level-analysis/before_refine4096"
points_file="configs/phi4/paper-2/canonical-point-scan/forward_points.tsv"
arches_csv="w64,w48"
volumes_csv="16,32,64,128"
seeds_csv="0,1,2,3"
checkpoint_name="checkpoint.pkl"
shards="4"
nsamples="65536"
jk_bin_size="256"
k_max="4"
locality_nsamples="16"
locality_nsources="4"
locality_metrics="manhattan,euclidean2"
locality_fit_rmin="1.0"
locality_fit_rmax="0.0"
reweight_ess_min="0.05"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --points) points_file="${2:-}"; shift 2 ;;
    --arches) arches_csv="${2:-}"; shift 2 ;;
    --volumes) volumes_csv="${2:-}"; shift 2 ;;
    --seeds) seeds_csv="${2:-}"; shift 2 ;;
    --checkpoint-name) checkpoint_name="${2:-}"; shift 2 ;;
    --shards) shards="${2:-}"; shift 2 ;;
    --nsamples) nsamples="${2:-}"; shift 2 ;;
    --jk-bin-size) jk_bin_size="${2:-}"; shift 2 ;;
    --k-max) k_max="${2:-}"; shift 2 ;;
    --locality-nsamples) locality_nsamples="${2:-}"; shift 2 ;;
    --locality-nsources) locality_nsources="${2:-}"; shift 2 ;;
    --locality-metrics) locality_metrics="${2:-}"; shift 2 ;;
    --locality-fit-rmin) locality_fit_rmin="${2:-}"; shift 2 ;;
    --locality-fit-rmax) locality_fit_rmax="${2:-}"; shift 2 ;;
    --reweight-ess-min) reweight_ess_min="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

submit_helper="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"
task_runner="${repo_root}/scripts/phi4/rg_coarse_eta_gaussian_level_analysis_bundle_task.sh"
wrapper="${repo_root}/scripts/phi4/run_rg_coarse_eta_gaussian_level_analysis.sh"

for path in "${submit_helper}" "${task_runner}" "${wrapper}"; do
  if [[ ! -x "${path}" ]]; then
    echo "Required executable missing: ${path}" >&2
    exit 2
  fi
done

if [[ "${points_file}" != /* ]]; then
  points_file="${repo_root}/${points_file}"
fi
if [[ ! -f "${points_file}" ]]; then
  echo "Points file not found: ${points_file}" >&2
  exit 2
fi

time_for_L() {
  case "$1" in
    16|32) echo "04:00:00" ;;
    64) echo "06:00:00" ;;
    128) echo "08:00:00" ;;
    *)
      echo "Unsupported volume: $1" >&2
      return 2
      ;;
  esac
}

batch_for_L() {
  case "$1" in
    16) echo "4096" ;;
    32) echo "2048" ;;
    64) echo "512" ;;
    128) echo "128" ;;
    *)
      echo "Unsupported volume: $1" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a arches <<< "${arches_csv}"
IFS=',' read -r -a volumes <<< "${volumes_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

declare -a submitted=()

for arch in "${arches[@]}"; do
  for L in "${volumes[@]}"; do
    bundle_dir="${bundle_root}/${arch}/L${L}"
    tasks_in="${bundle_dir}/tasks_in.tsv"
    walltime="$(time_for_L "${L}")"
    batch_size="$(batch_for_L "${L}")"
    mkdir -p "${bundle_dir}"
    : > "${tasks_in}"

    while IFS=$'\t' read -r label mass note; do
      [[ -z "${label}" ]] && continue
      [[ "${label}" == \#* ]] && continue
      for seed in "${seeds[@]}"; do
        checkpoint="${checkpoint_root}/${label}/${arch}/L${L}/s${seed}/${checkpoint_name}"
        if [[ ! -f "${checkpoint}" ]]; then
          echo "Skipping missing checkpoint: ${checkpoint}" >&2
          continue
        fi
        for ((shard = 0; shard < shards; ++shard)); do
          analysis_seed=$(( seed * 1000 + shard ))
          run_dir="${run_root}/${label}/${arch}/L${L}/s${seed}/shard$(printf '%02d' "${shard}")"
          task_name="phi4-${label}-${arch}-L${L}-s${seed}-levels-sh$(printf '%02d' "${shard}")"
          row=(
            "${task_name}"
            "${checkpoint}"
            "${run_dir}"
            "${analysis_seed}"
            "--nsamples" "${nsamples}"
            "--batch-size" "${batch_size}"
            "--jk-bin-size" "${jk_bin_size}"
            "--k-max" "${k_max}"
            "--reweight-ess-min" "${reweight_ess_min}"
            "--locality"
            "--locality-nsamples" "${locality_nsamples}"
            "--locality-nsources" "${locality_nsources}"
            "--locality-metric" "${locality_metrics}"
            "--locality-fit-rmin" "${locality_fit_rmin}"
            "--locality-fit-rmax" "${locality_fit_rmax}"
          )
          {
            printf "%s" "${row[0]}"
            for field in "${row[@]:1}"; do
              printf "\t%s" "${field}"
            done
            printf "\n"
          } >> "${tasks_in}"
        done
      done
    done < "${points_file}"

    if [[ ! -s "${tasks_in}" ]]; then
      echo "No tasks for ${arch} L${L}; skipping bundle." >&2
      continue
    fi

    cmd=(
      /bin/bash "${submit_helper}"
      --repo-root "${repo_root}"
      --tasks "${tasks_in}"
      --bundle-root "${bundle_dir}/bundle"
      --job-name "phi4-can-levels-${arch}-L${L}"
      --time "${walltime}"
      --account "${account}"
      --qos "${qos}"
      --wrapper "${wrapper}"
      --task-runner "${task_runner}"
    )
    if [[ ${print_only} -eq 1 ]]; then
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      out="$("${cmd[@]}")"
      printf '%s\n' "${out}"
      submitted+=("${arch}:L${L}")
    fi
  done
done

if [[ ${print_only} -eq 0 ]]; then
  echo "Submitted ${#submitted[@]} bundled forward level-analysis jobs."
fi
