#!/usr/bin/env bash

# Source this file:
#   source scripts/activate_local_env.sh

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${OPENPI_ROOT}/.venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] openpi virtualenv not found at ${VENV_PATH}"
    echo "        Create it first with the local setup steps in README."
    return 1 2>/dev/null || exit 1
fi

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/media/idna/Data/Server/Libeoro_dataset}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_ROOT}/.cache/openpi}"
export HF_HOME="${HF_HOME:-${OPENPI_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "Activated openpi env: ${VIRTUAL_ENV}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
