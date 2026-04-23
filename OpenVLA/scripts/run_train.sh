#!/bin/bash
# =============================================================================
# run_train.sh  —  OpenVLA LoRA fine-tuning launcher for B200 server
#
# Usage:
#   bash scripts/run_train.sh                                # defaults
#   bash scripts/run_train.sh --dataset_root /data/libero   # custom path
#   bash scripts/run_train.sh --max_steps 100000 --wandb    # more steps
#   bash scripts/run_train.sh --ngpu 2                      # 2 GPUs
# =============================================================================

set -e

# ── Python environment ────────────────────────────────────────────────────────
# Adjust CONDA_ENV if your env name differs
CONDA_ENV="openvla"
PYTHON="$(conda run -n $CONDA_ENV which python 2>/dev/null || echo python)"

# ── Default paths (override via CLI) ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT=""           # Required — set to lerobot-format merged dataset path
OUTPUT_ROOT="$SCRIPT_DIR/runs"
CONFIG="$SCRIPT_DIR/configs/libero_lora.yaml"

# ── Training overrides (leave empty to use config file values) ────────────────
MAX_STEPS=""
BATCH_SIZE=""
LR=""
LORA_RANK=""
SAVE_STEPS=""
CAMERA_KEY=""
WANDB_ENABLE=false
RUN_NAME=""
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_root)   DATASET_ROOT="$2"; shift 2 ;;
        --dataset_root=*) DATASET_ROOT="${1#*=}"; shift ;;
        --output_dir)     OUTPUT_ROOT="$2"; shift 2 ;;
        --max_steps)      MAX_STEPS="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --lora_rank)      LORA_RANK="$2"; shift 2 ;;
        --save_steps)     SAVE_STEPS="$2"; shift 2 ;;
        --camera_key)     CAMERA_KEY="$2"; shift 2 ;;
        --run_name)       RUN_NAME="$2"; shift 2 ;;
        --ngpu)           NGPU="$2"; shift 2 ;;
        --wandb)          WANDB_ENABLE=true; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$DATASET_ROOT" ]; then
    echo "[ERROR] --dataset_root is required."
    echo "  Example: bash scripts/run_train.sh --dataset_root /data/dataset_hf_aug_mix"
    exit 1
fi
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "[ERROR] $DATASET_ROOT does not look like a lerobot dataset (no meta/ dir)."
    exit 1
fi

# ── Output directory with timestamp ──────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME:-openvla_libero_lora_${TIMESTAMP}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ── Build optional CLI args ───────────────────────────────────────────────────
EXTRA_ARGS=""
[ -n "$MAX_STEPS"   ] && EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
[ -n "$BATCH_SIZE"  ] && EXTRA_ARGS="$EXTRA_ARGS --batch_size $BATCH_SIZE"
[ -n "$LR"          ] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"
[ -n "$LORA_RANK"   ] && EXTRA_ARGS="$EXTRA_ARGS --lora_rank $LORA_RANK"
[ -n "$SAVE_STEPS"  ] && EXTRA_ARGS="$EXTRA_ARGS --save_steps $SAVE_STEPS"
[ -n "$CAMERA_KEY"  ] && EXTRA_ARGS="$EXTRA_ARGS --camera_key $CAMERA_KEY"
[ -n "$RUN_NAME"    ] && EXTRA_ARGS="$EXTRA_ARGS --run_name $RUN_NAME"
[ "$WANDB_ENABLE" = "true" ] && EXTRA_ARGS="$EXTRA_ARGS --wandb"

# ── Info ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  OpenVLA LoRA fine-tuning — LIBERO"
echo "============================================================"
echo "  Dataset root : $DATASET_ROOT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Config       : $CONFIG"
echo "  GPUs         : $NGPU"
echo "  WandB        : $WANDB_ENABLE"
echo "============================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
# torchrun for multi-GPU DDP; falls back to direct python for single GPU
if [ "$NGPU" -gt 1 ]; then
    echo "Launching with torchrun ($NGPU GPUs) ..."
    conda run -n "$CONDA_ENV" \
        torchrun \
            --standalone \
            --nproc_per_node="$NGPU" \
            "$SCRIPT_DIR/train.py" \
            --config "$CONFIG" \
            --dataset_root "$DATASET_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS \
        2>&1 | tee "$OUTPUT_DIR/train.log"
else
    echo "Launching on single GPU ..."
    conda run -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/train.py" \
            --config "$CONFIG" \
            --dataset_root "$DATASET_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS \
        2>&1 | tee "$OUTPUT_DIR/train.log"
fi

echo ""
echo "Training complete. Checkpoints → $OUTPUT_DIR"
