#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/phi4/run_rg_coarse_eta_gaussian.sh [wrapper options] [-- trainer args...]

Wrapper options:
  --mode interactive|debug   Run the production L32 card or a short debug card.
  --config PATH              Override the default config for the selected mode.
  --workdir PATH             Directory to run from. Outputs stay here. Default: repo root.
  --gpu ID|all               GPU to expose via CUDA_VISIBLE_DEVICES. Default: 0.
  --no-triton                Disable Triton GEMM kernels for this launch.
  --python EXE               Python executable to use. Default: python.
  --no-activate              Do not load the NERSC python module or source ~/venvs/jax.
  --print-only               Print the resolved command/environment and exit.
  -h, --help                 Show this help.

Modes:
  interactive
    Uses configs/phi4/rg_coarse_eta_gaussian_L32_perlevel.toml

  debug
    Uses configs/phi4/rg_coarse_eta_gaussian_L32_perlevel_debug.toml
    and disables Triton GEMM autotuning for a quieter first-pass debug run.

Examples:
  scripts/phi4/run_rg_coarse_eta_gaussian.sh
  scripts/phi4/run_rg_coarse_eta_gaussian.sh --mode debug
  scripts/phi4/run_rg_coarse_eta_gaussian.sh --mode interactive -- --save my_run.pkl
  scripts/phi4/run_rg_coarse_eta_gaussian.sh --workdir runs/phi4/jobA -- --save checkpoint.pkl
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

mode="interactive"
config=""
gpu="0"
workdir=""
no_triton=0
python_exe="python"
activate=1
print_only=0
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
      shift 2
      ;;
    --gpu)
      gpu="${2:-}"
      shift 2
      ;;
    --workdir)
      workdir="${2:-}"
      shift 2
      ;;
    --no-triton)
      no_triton=1
      shift
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

case "${mode}" in
  interactive)
    default_config="configs/phi4/rg_coarse_eta_gaussian_L32_perlevel.toml"
    ;;
  debug)
    default_config="configs/phi4/rg_coarse_eta_gaussian_L32_perlevel_debug.toml"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${config}" ]]; then
  config="${default_config}"
fi

if [[ -z "${workdir}" ]]; then
  workdir="${repo_root}"
elif [[ "${workdir}" != /* ]]; then
  workdir="${repo_root}/${workdir}"
fi

if [[ "${config}" != /* ]]; then
  config="${repo_root}/${config}"
fi

if [[ ! -f "${config}" ]]; then
  echo "Config not found: ${config}" >&2
  exit 2
fi

mkdir -p "${workdir}"

if [[ ${activate} -eq 1 ]]; then
  if ! load_nersc_python_module; then
    echo "Failed to load the NERSC python module" >&2
    exit 2
  fi
  if [[ ! -f "${HOME}/venvs/jax/bin/activate" ]]; then
    echo "Expected venv not found: ${HOME}/venvs/jax/bin/activate" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${HOME}/venvs/jax/bin/activate"
fi

cd "${workdir}"

if [[ "${gpu}" != "all" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi

unset JAX_PLATFORM_NAME || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"
autotune_root="${XLA_AUTOTUNE_CACHE_ROOT:-/tmp/xla-autotune}"
cache_dir="${autotune_root}/phi4_rg_coarse_eta_gaussian_${mode}"
mkdir -p "${MPLCONFIGDIR}" "${cache_dir}"

append_xla_flag "--xla_gpu_per_fusion_autotune_cache_dir=${cache_dir}"
if [[ "${mode}" == "debug" || ${no_triton} -eq 1 ]]; then
  append_xla_flag "--xla_gpu_enable_triton_gemm=false"
fi
if [[ "${mode}" == "debug" ]]; then
  export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"
fi

cmd=(
  "${python_exe}"
  "${repo_root}/scripts/phi4/train_rg_coarse_eta_gaussian_flow.py"
  "--config" "${config}"
)
if [[ ${#passthrough[@]} -gt 0 ]]; then
  cmd+=("${passthrough[@]}")
fi

echo "phi4 launcher"
echo "  repo_root: ${repo_root}"
echo "  workdir: ${workdir}"
echo "  mode: ${mode}"
echo "  config: ${config}"
echo "  python: ${python_exe}"
echo "  VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "  MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "  no_triton: ${no_triton}"
echo "  XLA_FLAGS: ${XLA_FLAGS:-<unset>}"
echo "  command: ${cmd[*]}"

if [[ ${print_only} -eq 1 ]]; then
  exit 0
fi

exec "${cmd[@]}"
