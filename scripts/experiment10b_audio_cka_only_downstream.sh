#!/bin/bash
# Experiment 10b: Audio CKA-Only Downstream Evaluation
#
# Evaluates CKA-only distilled checkpoints on ESC-50 (5-fold CV):
#   G. CLIP text CKA-only distilled
#   H. Sentence-BERT CKA-only distilled
#
# Each condition tested at 3 label fractions (1%, 10%, 100%)
# in 2 modes (fine-tune, linear probe).
#
# All 12 runs launched in parallel (B200 has 183GB, each run ~1-2GB).
#
# Usage:
#   ./scripts/experiment10b_audio_cka_only_downstream.sh
#   FOLD=1 ./scripts/experiment10b_audio_cka_only_downstream.sh  # single fold

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.01}"
FOLD="${FOLD:-}"

# CKA-only checkpoint paths
CLIP_CKA_CKPT="${CLIP_CKA_CKPT:-$OUTPUT_DIR/audiocaps_clip_text_cka_only_distilled_best.pth}"
SBERT_CKA_CKPT="${SBERT_CKA_CKPT:-$OUTPUT_DIR/audiocaps_sentence_bert_cka_only_distilled_best.pth}"

echo "===== Experiment 10b: Audio CKA-Only Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "CLIP CKA-only checkpoint: $CLIP_CKA_CKPT"
echo "SBERT CKA-only checkpoint: $SBERT_CKA_CKPT"

if [ -n "$FOLD" ]; then
    echo "Running single fold: $FOLD"
    FOLD_ARG="--fold $FOLD"
else
    echo "Running ALL folds (full CV)"
    FOLD_ARG=""
fi
echo ""

# Verify checkpoints exist
if [ ! -f "$CLIP_CKA_CKPT" ]; then
    echo "ERROR: CLIP CKA-only checkpoint not found: $CLIP_CKA_CKPT"
    exit 1
fi
if [ ! -f "$SBERT_CKA_CKPT" ]; then
    echo "ERROR: SBERT CKA-only checkpoint not found: $SBERT_CKA_CKPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

DATASET="esc50"
PIDS=()
LOGS=()

run_downstream() {
    local CKPT=$1
    local TEACHER=$2
    local FRAC=$3
    local FREEZE=$4  # "" or "--freeze_backbone"
    local MODE="finetune"
    if [ -n "$FREEZE" ]; then
        MODE="linprobe"
    fi

    local LOGFILE="$OUTPUT_DIR/log_exp10b_cka_only_${TEACHER}_${DATASET}_f${FRAC}_${MODE}.txt"

    echo "  Launching: $DATASET / $TEACHER CKA-only / frac=$FRAC / $MODE"

    $PYTHON src/train_audio_downstream.py \
        --dataset "$DATASET" \
        --data_root "$DATA_ROOT" \
        --label_fraction $FRAC \
        --init distilled \
        --checkpoint "$CKPT" \
        --teacher_name "$TEACHER" \
        $FREEZE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED \
        --no_wandb \
        $FOLD_ARG \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
    LOGS+=("$LOGFILE")
}

echo "Launching all 12 downstream runs in parallel..."
echo ""

for FRAC in 1.0 0.1 0.01; do
    # G: CLIP text CKA-only — fine-tune + linprobe
    run_downstream "$CLIP_CKA_CKPT" "clip_text" "$FRAC" ""
    run_downstream "$CLIP_CKA_CKPT" "clip_text" "$FRAC" "--freeze_backbone"

    # H: SBERT CKA-only — fine-tune + linprobe
    run_downstream "$SBERT_CKA_CKPT" "sentence_bert" "$FRAC" ""
    run_downstream "$SBERT_CKA_CKPT" "sentence_bert" "$FRAC" "--freeze_backbone"
done

echo ""
echo "All 12 runs launched. Waiting for completion..."
echo ""

# Wait for all and track failures
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${LOGS[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "===== Experiment 10b: Downstream Evaluation Complete ====="
echo "Succeeded: $((${#PIDS[@]} - FAILED))/${#PIDS[@]}"
if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED (check log files above)"
fi
echo ""
echo "Results saved to: $OUTPUT_DIR/results_audio_*.csv"
echo ""
echo "Key comparisons:"
echo "  G vs B: CKA-only vs embedding loss for CLIP (does CKA bypass cone?)"
echo "  G vs D: CKA-only vs whitened embedding loss for CLIP"
echo "  H vs E: CKA-only vs embedding loss for SBERT"
