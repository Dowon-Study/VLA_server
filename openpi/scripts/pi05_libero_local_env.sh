#!/usr/bin/env bash

# Source this file before running local LIBERO fine-tuning commands.
#
# Example:
#   source scripts/pi05_libero_local_env.sh

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/media/idna/Data/Server/Libeoro_dataset}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_ROOT}/.cache/openpi}"
export HF_HOME="${HF_HOME:-${OPENPI_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "${OPENPI_DATA_HOME}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo
echo "Available local openpi LIBERO configs:"
echo "  pi05_libero_local_hfvla    -> ${HF_LEROBOT_HOME}/HuggingFaceVLA_libero_lerobot"
echo "  pi05_libero_local_gt       -> ${HF_LEROBOT_HOME}/dataset_gt_prepared"
echo "  pi05_libero_local_gt_aug   -> ${HF_LEROBOT_HOME}/libero_gt_aug_balanced"
