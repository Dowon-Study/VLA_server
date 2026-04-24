#!/usr/bin/env bash

# Source this file:
#   source scripts/activate_local_env.sh

OPENVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_PATH="${OPENVLA_ROOT}/.conda"

if [[ ! -f "${CONDA_ENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] OpenVLA env not found at ${CONDA_ENV_PATH}"
    echo "        Run: bash scripts/setup_local_env.sh"
    return 1 2>/dev/null || exit 1
fi

export HF_HOME="${HF_HOME:-${OPENVLA_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

# shellcheck disable=SC1091
source "${CONDA_ENV_PATH}/bin/activate"

echo "Activated OpenVLA env: ${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-${CONDA_ENV_PATH}}}"
