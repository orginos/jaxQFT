#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/run_rg_coarse_eta_gaussian_level_analysis.sh [wrapper options] [-- analysis args...]

Required:
  --checkpoint PATH          Checkpoint to analyze.

Wrapper options:
  --workdir PATH             Directory to run from. Outputs stay here. Default: repo root.
  --json-out PATH            JSON output path. Relative paths are resolved from workdir.
                             Default: flow_levels.json
  --gpu ID|all               GPU to expose via CUDA_VISIBLE_DEVICES. Default: 0.
  --python EXE               Python executable to use. Default: python.
  --no-activate              Do not load the NERSC python module or source ~/venv/jax.
  --print-only               Print the resolved command/environment and exit.
  -h, --help                 Show this help.

Examples:
  scripts/phi4/run_rg_coarse_eta_gaussian_level_analysis.sh \
    --checkpoint runs/phi4/canonical-scaling-continuation/L32/s0/checkpoint.pkl \
    --workdir runs/phi4/canonical-level-analysis/L32/s0 \
    -- --nsamples 65536 --batch-size 2048 --k-max 4 --locality
EOF
}

append_xla_flag() {
  local flag="$1"
  local cur="${XLA_FLAGS:-}"
  if [[ " ${cur} " != *" ${flag} "* ]]; then
    if [[ -n "${cur}" ]]; then
      export XLA_FLAGS="${cur} ${flag}"
    else
      export XLA_FLAGS="${flag}"
    fi
  fi
}

ensure_module_cmd() {
  if declare -F module >/dev/null 2>&1; then
    return 0
  fi
  local init_script=""
  local had_u=0
  if [[ $- == *u* ]]; then
    had_u=1
    set +u
  fi
  if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
    init_script="${MODULESHOME}/init/bash"
  elif [[ -f /opt/cray/pe/lmod/lmod/init/bash ]]; then
    init_script="/opt/cray/pe/lmod/lmod/init/bash"
  fi
  if [[ -n "${init_script}" ]]; then
    # shellcheck disable=SC1090
    source "${init_script}"
  fi
  if [[ ${had_u} -eq 1 ]]; then
    set -u
  fi
  declare -F module >/dev/null 2>&1
}

load_nersc_python_module() {
  local lmod_output=""
  if [[ -n "${LMOD_CMD:-}" && -x "${LMOD_CMD}" ]]; then
    lmod_output="$("${LMOD_CMD}" shell load python)"
    lmod_output="$(printf '%s\n' "${lmod_output}" | sed \
      -e '/^[[:space:]]*ERROR: auth\.munge: Failed to encode MUNGE\. Socket communication error$/d' \
      -e "/^source '.*\\/conda\\.sh';$/d" \
      -e "/^source '.*\\/mamba\\.sh';$/d" \
      -e '/^declare -fx conda;$/d' \
      -e '/^declare -fx __conda_.*;$/d' \
      -e '/^declare -fx mamba;$/d' \
      -e '/^declare -fx __mamba_.*;$/d' \
      -e '/^conda activate .*;$/d')"
    if printf '%s\n' "${lmod_output}" | grep -q '^[[:space:]]*ERROR:'; then
      printf '%s\n' "${lmod_output}" >&2
      return 1
    fi
    eval "${lmod_output}"
    return 0
  fi
  if ensure_module_cmd; then
    module load python
    return 0
  fi
  return 1
}

checkpoint=""
workdir=""
json_out="flow_levels.json"
gpu="0"
python_exe="python"
activate=1
print_only=0
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      checkpoint="${2:-}"
      shift 2
      ;;
    --workdir)
      workdir="${2:-}"
      shift 2
      ;;
    --json-out)
      json_out="${2:-}"
      shift 2
      ;;
    --gpu)
      gpu="${2:-}"
      shift 2
      ;;
    --python)
      python_exe="${2:-}"
      shift 2
      ;;
    --no-activate)
      activate=0
      shift
      ;;
    --print-only)
      print_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      passthrough=("$@")
      break
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${checkpoint}" ]]; then
  echo "Missing --checkpoint" >&2
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [[ -z "${workdir}" ]]; then
  workdir="${repo_root}"
elif [[ "${workdir}" != /* ]]; then
  workdir="${repo_root}/${workdir}"
fi

if [[ "${checkpoint}" != /* ]]; then
  checkpoint="${repo_root}/${checkpoint}"
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi

mkdir -p "${workdir}"

if [[ "${json_out}" != /* ]]; then
  json_out="${workdir}/${json_out}"
fi

if [[ ${activate} -eq 1 ]]; then
  if ! load_nersc_python_module; then
    echo "Failed to load the NERSC python module" >&2
    exit 2
  fi
  activate_path=""
  if [[ -f "${HOME}/venv/jax/bin/activate" ]]; then
    activate_path="${HOME}/venv/jax/bin/activate"
  elif [[ -f "${HOME}/venvs/jax/bin/activate" ]]; then
    activate_path="${HOME}/venvs/jax/bin/activate"
  else
    echo "Expected venv not found: ${HOME}/venv/jax/bin/activate (or fallback ${HOME}/venvs/jax/bin/activate)" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${activate_path}"
fi

cd "${workdir}"

if [[ "${gpu}" != "all" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi

unset JAX_PLATFORM_NAME || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"
autotune_root="${XLA_AUTOTUNE_CACHE_ROOT:-/tmp/xla-autotune}"
cache_dir="${autotune_root}/phi4_rg_coarse_eta_gaussian_level_analysis"
mkdir -p "${MPLCONFIGDIR}" "${cache_dir}"
append_xla_flag "--xla_gpu_per_fusion_autotune_cache_dir=${cache_dir}"

cmd=(
  "${python_exe}"
  "${repo_root}/scripts/phi4/analysis/analyze_rg_coarse_eta_gaussian_levels.py"
  "--resume" "${checkpoint}"
  "--json-out" "${json_out}"
)
if [[ ${#passthrough[@]} -gt 0 ]]; then
  cmd+=("${passthrough[@]}")
fi

echo "phi4 level-analysis launcher"
echo "  repo_root: ${repo_root}"
echo "  workdir: ${workdir}"
echo "  checkpoint: ${checkpoint}"
echo "  json_out: ${json_out}"
echo "  python: ${python_exe}"
echo "  VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "  MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "  XLA_FLAGS: ${XLA_FLAGS:-<unset>}"
echo "  command: ${cmd[*]}"

if [[ ${print_only} -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}"
