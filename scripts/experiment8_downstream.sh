#!/bin/bash
# Experiment 8: Downstream Evaluation — ViT-L Teacher Distillation
#
# Evaluates 3 ViT-L CKA-distilled checkpoints across 6 datasets,
# 3 label fractions, and 2 modes (fine-tune + linear probe).
#
# No projector modes — Exp 7 showed they don't help.
# Baselines (random, imagenet pretrained) already exist from Exp 7.
#
# Total: 3 teachers × 6 datasets × 3 fractions × 2 modes = 108 runs
#
# Parallelized with xargs -P (default 4 concurrent GPU jobs).
# Each downstream run uses ~2-3GB VRAM on a RegNetY-400MF.
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment8_downstream.sh
#   MAX_PARALLEL=6 ./scripts/experiment8_downstream.sh   # more parallelism
#   SKIP_EXISTING=1 ./scripts/experiment8_downstream.sh  # skip completed runs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LAMBDA_CKA="${LAMBDA_CKA:-0.1}"

ALL_DATASETS="voc pets eurosat dtd flowers102 imagenette"
ALL_FRACTIONS="0.01 0.1 1.0"
TEACHERS="supervised_l clip_l dinov2_l"

echo "===== Experiment 8: ViT-L Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo "TEACHERS: $TEACHERS"
echo ""

mkdir -p "$OUTPUT_DIR"
JOBFILE=$(mktemp /tmp/exp8_jobs_XXXXXX.txt)
FAIL_LOG="$OUTPUT_DIR/experiment8_failures.log"
> "$FAIL_LOG"

# ============================================================
# Helper: add a job to the job file
# ============================================================
add_job() {
    local DESC="$1"
    shift
    echo "$DESC|||$*" >> "$JOBFILE"
}

# ============================================================
# Helper: check if result CSV already exists (for SKIP_EXISTING)
# ============================================================
result_exists() {
    local DATASET=$1 TEACHER_NAME=$2 FRAC=$3 MODE_SUFFIX=$4
    local CSV="$OUTPUT_DIR/results_${DATASET}_distilled_${TEACHER_NAME}${MODE_SUFFIX}_frac${FRAC}_s${SEED}.csv"
    [ -f "$CSV" ]
}

# ============================================================
# ImageNet ViT-L CKA (λ=0.1) on ALL datasets
# 2 modes: fine-tune + linear probe
# ============================================================
echo "--- Generating: ImageNet ViT-L CKA λ=$LAMBDA_CKA on all datasets ---"
COUNT=0
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/imagenet_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping $TEACHER"
        continue
    fi
    for DATASET in $ALL_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            TNAME="imagenet_${TEACHER}_cka_l${LAMBDA_CKA}"

            # Fine-tune (drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT=$((COUNT + 1))
                add_job "vitl_cka/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe (frozen backbone, drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT=$((COUNT + 1))
                add_job "vitl_cka/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  ViT-L CKA (2 modes): $COUNT jobs"

# ============================================================
# Execute all jobs
# ============================================================
TOTAL=$(wc -l < "$JOBFILE")
echo ""
echo "=============================================="
echo "Total jobs to run: $TOTAL"
echo "Parallelism: $MAX_PARALLEL concurrent"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No jobs to run. All results may already exist (SKIP_EXISTING=$SKIP_EXISTING)."
    rm -f "$JOBFILE"
    exit 0
fi

# Run jobs with xargs parallelism
# Each line: "description|||command args..."
run_single_job() {
    local LINE="$1"
    local DESC="${LINE%%|||*}"
    local CMD="${LINE#*|||}"
    local LOG_NAME=$(echo "$DESC" | tr '/' '_')
    local LOG="$OUTPUT_DIR/log_exp8_${LOG_NAME}.txt"

    echo "[START] $DESC"
    if eval $CMD > "$LOG" 2>&1; then
        echo "[DONE]  $DESC"
    else
        echo "[FAIL]  $DESC (see $LOG)"
        echo "$DESC" >> "$FAIL_LOG"
    fi
}
export -f run_single_job
export OUTPUT_DIR FAIL_LOG

cat "$JOBFILE" | xargs -P $MAX_PARALLEL -I {} bash -c 'run_single_job "$@"' _ {}

# ============================================================
# Summary
# ============================================================
FAIL_COUNT=0
if [ -f "$FAIL_LOG" ]; then
    FAIL_COUNT=$(wc -l < "$FAIL_LOG")
fi

echo ""
echo "===== Experiment 8: Downstream Complete ====="
echo "Total jobs: $TOTAL"
echo "Failed: $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    cat "$FAIL_LOG"
    echo ""
    echo "Check logs: $OUTPUT_DIR/log_exp8_*.txt"
fi
echo ""
echo "Results saved to $OUTPUT_DIR/results_*.csv"
echo ""
echo "Key comparisons to check:"
echo "  1. ViT-L backbone CKA vs ViT-B (0.31 sup, 0.79 clip)"
echo "  2. VOC/Pets best accuracy: ViT-L vs ViT-B vs ImageNet pretrained"
echo "  3. DINOv2 vs CLIP vs supervised on all downstream tasks"
echo "  4. 1% label results: does ViT-L distilled beat ViT-B distilled?"
echo "  5. Does backbone CKA predict downstream performance across ViT-L teachers?"

rm -f "$JOBFILE"
