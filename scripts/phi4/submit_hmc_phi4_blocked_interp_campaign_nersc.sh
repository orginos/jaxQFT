#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

exec "${repo_root}/scripts/phi4/submit_hmc_phi4_blocked_reference_campaign_nersc.sh" \
  --g2-file "${repo_root}/configs/phi4/paper-2/hmc-g2-scan/blocked_interp_points.tsv" \
  --replicas 2 \
  "$@"
