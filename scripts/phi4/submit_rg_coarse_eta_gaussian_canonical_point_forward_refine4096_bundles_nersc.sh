#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/submit_rg_coarse_eta_gaussian_canonical_point_forward_refine4096_bundles_nersc.sh [options]

Submit bundled high-effective-batch refinement jobs for the stable forward-loss
canonical-point families. These are resume jobs: they preserve the original
forward checkpoints and write new refinement checkpoints into a separate run
root using checkpoint_refine4096.pkl.

Default scope:
  - points: canonical, canonical2, canonical3, canonical4
  - arches: w64, w48
  - volumes: 16,32,64,128
  - seeds: 0,1,2,3
  - effective batch: 4096 via gradient accumulation
  - refinement length: +10000 epochs (target epoch 21000)
  - lr: 5e-6

Optional:
  --repo-root PATH         Repo root on NERSC. Default: /global/cfs/cdirs/hadron/jaxQFT
  --source-root PATH       Existing forward checkpoint root. Default:
                           /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward
  --run-root PATH          Refinement run root. Default:
                           /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-refine4096
  --bundle-root PATH       Bundle root. Default:
                           /global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward-refine4096
  --points PATH            Point TSV. Default:
                           configs/phi4/paper-2/canonical-point-scan/forward_points.tsv
  --arches LIST            Comma-separated architectures. Default: w64,w48
  --volumes LIST           Comma-separated volumes. Default: 16,32,64,128
  --seeds LIST             Comma-separated seeds. Default: 0,1,2,3
  --account NAME           Slurm account. Default: $NERSC_ACCOUNT or hadron
  --qos NAME               Slurm QOS. Default: regular
  --print-only             Print submit commands and exit.
EOF
}

repo_root="/global/cfs/cdirs/hadron/jaxQFT"
source_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward"
run_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-refine4096"
bundle_root="/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward-refine4096"
points_file="configs/phi4/paper-2/canonical-point-scan/forward_points.tsv"
arches_csv="w64,w48"
volumes_csv="16,32,64,128"
seeds_csv="0,1,2,3"
account="${NERSC_ACCOUNT:-hadron}"
qos="regular"
print_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:-}"; shift 2 ;;
    --source-root) source_root="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --bundle-root) bundle_root="${2:-}"; shift 2 ;;
    --points) points_file="${2:-}"; shift 2 ;;
    --arches) arches_csv="${2:-}"; shift 2 ;;
    --volumes) volumes_csv="${2:-}"; shift 2 ;;
    --seeds) seeds_csv="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --print-only) print_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

submit_helper="${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_4task_bundle_nersc.sh"
if [[ ! -x "${submit_helper}" ]]; then
  echo "Bundle submit helper not executable: ${submit_helper}" >&2
  exit 2
fi

if [[ "${points_file}" != /* ]]; then
  points_file="${repo_root}/${points_file}"
fi
if [[ ! -f "${points_file}" ]]; then
  echo "Points file not found: ${points_file}" >&2
  exit 2
fi

config_for_L() {
  case "$1" in
    16|32|64|128)
      echo "${repo_root}/configs/phi4/paper-2/canonical-point-scan/L${1}_uniform_refine4096_resume.toml"
      ;;
    *)
      echo "Unsupported volume: $1" >&2
      return 2
      ;;
  esac
}

# Walltimes are based on measured forward baseline task timings with additional
# headroom for the 10k-epoch low-lr resume phase. The earlier pure sample-work
# scaling heuristic was too conservative for queue placement.
time_for_L() {
  case "$1" in
    16) echo "05:00:00" ;;
    32) echo "06:30:00" ;;
    64) echo "09:00:00" ;;
    128) echo "14:00:00" ;;
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
    config="$(config_for_L "${L}")"
    walltime="$(time_for_L "${L}")"
    bundle_dir="${bundle_root}/${arch}/L${L}"
    tasks_in="${bundle_dir}/tasks_in.tsv"
    mkdir -p "${bundle_dir}"
    : > "${tasks_in}"

    while IFS=$'\t' read -r label mass note; do
      [[ -z "${label}" ]] && continue
      [[ "${label}" == \#* ]] && continue
      for seed in "${seeds[@]}"; do
        resume_ckpt="${source_root}/${label}/${arch}/L${L}/s${seed}/checkpoint.pkl"
        if [[ ! -f "${resume_ckpt}" ]]; then
          echo "Skipping missing source checkpoint: ${resume_ckpt}" >&2
          continue
        fi
        run_dir="${run_root}/${label}/${arch}/L${L}/s${seed}"
        if [[ -e "${run_dir}" ]]; then
          echo "Refusing to overwrite existing refinement run dir: ${run_dir}" >&2
          exit 2
        fi
        task_name="phi4-${label}-${arch}-L${L}-s${seed}-refine4096"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${task_name}" \
          "${config}" \
          "${run_dir}" \
          "${seed}" \
          "--resume" "${resume_ckpt}" \
          >> "${tasks_in}"
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
      --job-name "phi4-can-refine4096-${arch}-L${L}"
      --time "${walltime}"
      --account "${account}"
      --qos "${qos}"
      --checkpoint-name "checkpoint_refine4096.pkl"
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
  echo "Submitted ${#submitted[@]} bundled refine4096 jobs."
fi
