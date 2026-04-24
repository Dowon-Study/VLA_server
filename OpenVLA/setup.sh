#!/bin/bash
# =============================================================================
# setup.sh  —  OpenVLA fine-tuning environment setup
#
# Tested on:  Ubuntu 22.04 / CUDA 12.8 / B200 (Blackwell sm_100)
#
# Usage:
#   bash setup.sh            # full install (conda + packages)
#   bash setup.sh --no-conda # pip only (inside existing env)
#   bash setup.sh --dev      # + dev tools (jupyter, ipdb)
# =============================================================================

set -e

CONDA_ENV="openvla"
PYTHON_VERSION="3.11"
# For B200 (Blackwell / sm_100): CUDA 12.8+, PyTorch >= 2.6
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

NO_CONDA=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-conda) NO_CONDA=true; shift ;;
        --dev)      DEV_MODE=true; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# ── Step 1: Create conda environment ─────────────────────────────────────────
if [ "$NO_CONDA" = "false" ]; then
    if ! command -v conda &>/dev/null; then
        echo "[ERROR] conda not found. Install Miniconda or use --no-conda."
        exit 1
    fi

    echo "[1/4] Creating conda env '$CONDA_ENV' (Python $PYTHON_VERSION) ..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
    PYTHON="$(conda run -n $CONDA_ENV which python)"
    PIP="$(conda run -n $CONDA_ENV which pip)"
else
    echo "[1/4] Skipping conda — using current environment."
    PYTHON="$(which python)"
    PIP="$(which pip)"
fi

echo "  Python: $PYTHON"
echo "  Pip:    $PIP"

# ── Step 2: Install PyTorch (CUDA 12.8 for B200) ─────────────────────────────
echo ""
echo "[2/4] Installing PyTorch (cu128) ..."
$PIP install torch torchvision \
    --index-url "$PYTORCH_INDEX" \
    --no-cache-dir

# Verify CUDA
$PYTHON -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU[{i}]: {torch.cuda.get_device_name(i)}')
"

# ── Step 3: Install Python packages ──────────────────────────────────────────
echo ""
echo "[3/4] Installing Python packages ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PIP install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir

# ── Step 4: Install flash-attn (optional, speeds up attention on Ampere+) ────
echo ""
echo "[4/4] Installing flash-attention-2 (optional, may take ~5 min) ..."
# Build from source — required for Blackwell (B200) until pre-built wheels ship
if $PIP install flash-attn --no-build-isolation 2>/dev/null; then
    echo "  flash-attn installed."
else
    echo "  [WARN] flash-attn build failed — falling back to 'eager' attention."
    echo "         Set attn_implementation: eager in configs/libero_full.yaml"
fi

# ── Dev tools (optional) ─────────────────────────────────────────────────────
if [ "$DEV_MODE" = "true" ]; then
    echo ""
    echo "[DEV] Installing dev tools ..."
    $PIP install jupyter ipdb rich
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
if [ "$NO_CONDA" = "false" ]; then
echo "    conda activate $CONDA_ENV"
fi
echo ""
echo "  Download the OpenVLA base model (one-time, ~14 GB):"
echo "    python -c \"from transformers import AutoProcessor, AutoModelForVision2Seq;"
echo "      AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True);"
echo "      AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)\""
echo ""
echo "  Run fine-tuning:"
echo "    bash scripts/run_train.sh --dataset_root /path/to/dataset_hf_aug_mix"
echo ""
