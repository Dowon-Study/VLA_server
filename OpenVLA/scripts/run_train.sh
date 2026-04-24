#!/bin/bash
# =============================================================================
# run_train.sh  —  OpenVLA full fine-tuning launcher (B200 × 16 GPU, FSDP)
#
# Usage:
#   bash scripts/run_train.sh --dataset_root /data/dataset_hf_aug_mix
#   bash scripts/run_train.sh --dataset_root /data --ngpu 2
#   bash scripts/run_train.sh --dataset_root /data --max_steps 20000 --wandb
#   bash scripts/run_train.sh --dataset_root /data --no_image_aug   # aug 없는 baseline
# =============================================================================

set -euo pipefail

CONDA_ENV="openvla"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 기본값 ────────────────────────────────────────────────────────────────────
DATASET_ROOT=""
OUTPUT_ROOT="$SCRIPT_DIR/runs"
CONFIG="$SCRIPT_DIR/configs/libero_full.yaml"
FSDP_CONFIG="$SCRIPT_DIR/configs/fsdp_config.yaml"

MAX_STEPS=""
BATCH_SIZE=""
LR=""
SAVE_STEPS=""
CAMERA_KEY=""
UNNORM_KEY=""
IMAGE_AUG_FLAG=""
NOOP_FLAG=""
WANDB_ENABLE=false
RUN_NAME=""
DEFAULT_BATCH_SIZE=65
DEFAULT_NGPU=2
NGPU="$DEFAULT_NGPU"

# ── CLI 파싱 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_root)   DATASET_ROOT="$2"; shift 2 ;;
        --dataset_root=*) DATASET_ROOT="${1#*=}"; shift ;;
        --output_dir)     OUTPUT_ROOT="$2"; shift 2 ;;
        --max_steps)      MAX_STEPS="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --save_steps)     SAVE_STEPS="$2"; shift 2 ;;
        --camera_key)     CAMERA_KEY="$2"; shift 2 ;;
        --unnorm_key)     UNNORM_KEY="$2"; shift 2 ;;
        --image_aug)      IMAGE_AUG_FLAG="--image_aug"; shift ;;
        --no_image_aug)   IMAGE_AUG_FLAG="--no_image_aug"; shift ;;
        --skip_noops)     NOOP_FLAG="--skip_noops"; shift ;;
        --keep_noops)     NOOP_FLAG="--keep_noops"; shift ;;
        --run_name)       RUN_NAME="$2"; shift 2 ;;
        --ngpu)           NGPU="$2"; shift 2 ;;
        --wandb)          WANDB_ENABLE=true; shift ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

# ── 검증 ──────────────────────────────────────────────────────────────────────
if [ -z "$DATASET_ROOT" ]; then
    echo "[ERROR] --dataset_root is required."
    echo "  Example: bash scripts/run_train.sh --dataset_root /data/dataset_hf_aug_mix"
    exit 1
fi
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "[ERROR] $DATASET_ROOT 은 lerobot 데이터셋이 아닙니다 (meta/ 없음)."
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "[ERROR] $DATASET_ROOT/meta/stats.json 이 없습니다."
    echo "        먼저 scripts/prepare_libero_dataset.py 로 GT-only/병합 데이터셋을 준비하세요."
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/tasks.parquet" ] && [ ! -f "$DATASET_ROOT/meta/tasks.jsonl" ] && [ ! -f "$DATASET_ROOT/meta/episodes.jsonl" ]; then
    echo "[ERROR] $DATASET_ROOT/meta/tasks.parquet (또는 tasks.jsonl / episodes.jsonl) 이 없습니다."
    echo "        먼저 scripts/prepare_libero_dataset.py 로 task metadata를 생성하세요."
    exit 1
fi

# ── FSDP config: GPU 수에 따라 num_processes 업데이트 ─────────────────────────
TMP_FSDP_CONFIG="$SCRIPT_DIR/configs/fsdp_config_runtime.yaml"
sed "s/num_processes: 16/num_processes: $NGPU/" "$FSDP_CONFIG" > "$TMP_FSDP_CONFIG"

# ── 출력 디렉터리 ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${RUN_NAME:-openvla_full_${TIMESTAMP}}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ── 추가 인자 조립 ────────────────────────────────────────────────────────────
EXTRA_ARGS="--run_name $RUN_NAME"
[ -n "$MAX_STEPS"  ] && EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
[ -n "$BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS --batch_size $BATCH_SIZE"
[ -n "$LR"         ] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"
[ -n "$SAVE_STEPS" ] && EXTRA_ARGS="$EXTRA_ARGS --save_steps $SAVE_STEPS"
[ -n "$CAMERA_KEY" ] && EXTRA_ARGS="$EXTRA_ARGS --camera_key $CAMERA_KEY"
[ -n "$UNNORM_KEY" ] && EXTRA_ARGS="$EXTRA_ARGS --unnorm_key $UNNORM_KEY"
[ -n "$IMAGE_AUG_FLAG" ] && EXTRA_ARGS="$EXTRA_ARGS $IMAGE_AUG_FLAG"
[ -n "$NOOP_FLAG" ] && EXTRA_ARGS="$EXTRA_ARGS $NOOP_FLAG"
[ "$WANDB_ENABLE" = "true" ] && EXTRA_ARGS="$EXTRA_ARGS --wandb"

# ── 정보 출력 ─────────────────────────────────────────────────────────────────
EFFECTIVE_BATCH=$((DEFAULT_BATCH_SIZE * NGPU))
[ -n "$BATCH_SIZE" ] && EFFECTIVE_BATCH=$((BATCH_SIZE * NGPU))
echo ""
echo "============================================================"
echo "  OpenVLA Full Fine-tuning — LIBERO"
echo "============================================================"
echo "  Dataset root   : $DATASET_ROOT"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Config         : $CONFIG"
echo "  GPUs           : $NGPU"
echo "  Effective batch: $EFFECTIVE_BATCH  (${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}/GPU × $NGPU)"
echo "  FSDP           : ZeRO-3 (full_shard)"
echo "  Unnorm key     : ${UNNORM_KEY:-config default}"
echo "  Image aug flag : ${IMAGE_AUG_FLAG:-config default}"
echo "  No-op flag     : ${NOOP_FLAG:-config default}"
echo "  WandB          : $WANDB_ENABLE"
echo "============================================================"
echo ""

# ── 학습 실행 ─────────────────────────────────────────────────────────────────
LAUNCH_PREFIX=()
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    LAUNCH_PREFIX=(conda run -n "$CONDA_ENV")
fi

"${LAUNCH_PREFIX[@]}" \
accelerate launch \
    --config_file "$TMP_FSDP_CONFIG" \
    "$SCRIPT_DIR/train.py" \
        --config   "$CONFIG" \
        --dataset_root "$DATASET_ROOT" \
        --output_dir   "$OUTPUT_DIR" \
        $EXTRA_ARGS \
2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "학습 완료. 체크포인트 → $OUTPUT_DIR"
