#!/usr/bin/env bash

# Create a local OpenVLA-only environment under this repo.
#
# Usage:
#   bash scripts/setup_local_env.sh
#   bash scripts/setup_local_env.sh --dev

set -euo pipefail

OPENVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_PATH="${OPENVLA_ROOT}/.conda"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) DEV_MODE=true; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found."
    exit 1
fi

if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
    echo "[1/4] Creating OpenVLA env at ${CONDA_ENV_PATH}"
    conda create -y -p "${CONDA_ENV_PATH}" python="${PYTHON_VERSION}" pip
else
    echo "[1/4] Reusing existing OpenVLA env at ${CONDA_ENV_PATH}"
fi

PYTHON="${CONDA_ENV_PATH}/bin/python"
PIP="${CONDA_ENV_PATH}/bin/pip"

echo "[2/4] Installing PyTorch/cu128"
"${PIP}" install --index-url "${PYTORCH_INDEX}" --no-cache-dir torch torchvision

echo "[3/4] Installing OpenVLA requirements"
"${PIP}" install --no-cache-dir -r "${OPENVLA_ROOT}/requirements.txt"

echo "[4/4] Installing flash-attn (best effort)"
if "${PIP}" install flash-attn --no-build-isolation >/dev/null 2>&1; then
    echo "flash-attn installed"
else
    echo "[WARN] flash-attn install failed; keep attn_implementation=eager if needed"
fi

if [[ "${DEV_MODE}" == "true" ]]; then
    "${PIP}" install jupyter ipdb rich
fi

mkdir -p "${OPENVLA_ROOT}/.cache/huggingface"

echo
echo "OpenVLA env ready:"
echo "  source ${OPENVLA_ROOT}/scripts/activate_local_env.sh"
