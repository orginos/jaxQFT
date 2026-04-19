#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

exec "${repo_root}/scripts/phi4/submit_rg_coarse_eta_gaussian_forward_level_analysis_bundles_nersc.sh" \
  --checkpoint-root "/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-refine4096" \
  --run-root "/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/canonical-point-scan-forward-level-analysis/after_refine4096" \
  --bundle-root "/global/cfs/cdirs/hadron/jaxQFT/runs/phi4/bundles/canonical-point-scan-forward-level-analysis/after_refine4096" \
  --checkpoint-name "checkpoint_refine4096.pkl" \
  "$@"
