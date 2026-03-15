#!/bin/bash
# Experiment 9: Downstream Evaluation — CKA-Only vs Cosine-Only
#
# Evaluates 4 new checkpoints (2 teachers × 2 loss types) across 6 datasets,
# 3 label fractions, and 2 modes (fine-tune + linear probe).
#
# New checkpoints:
#   imagenet_clip_l_cka_only_distilled_best.pth       (CKA-only)
#   imagenet_supervised_l_cka_only_distilled_best.pth  (CKA-only)
#   imagenet_clip_l_distilled_best.pth                 (cosine-only)
#   imagenet_supervised_l_distilled_best.pth           (cosine-only)
#
# CKA+cosine combined results already exist from Exp 8 (not re-run).
# Baselines (random, imagenet pretrained) already exist from Exp 7.
#
# Total: 4 checkpoints × 6 datasets × 3 fractions × 2 modes = 144 runs
#
# Parallelized with xargs -P (default 4 concurrent GPU jobs).
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment9_downstream.sh
#   MAX_PARALLEL=6 ./scripts/experiment9_downstream.sh   # more parallelism
#   SKIP_EXISTING=1 ./scripts/experiment9_downstream.sh  # skip completed runs

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

ALL_DATASETS="voc pets eurosat dtd flowers102 imagenette"
ALL_FRACTIONS="0.01 0.1 1.0"

echo "===== Experiment 9: CKA-Only vs Cosine-Only Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo ""

mkdir -p "$OUTPUT_DIR"
JOBFILE=$(mktemp /tmp/exp9_jobs_XXXXXX.txt)
FAIL_LOG="$OUTPUT_DIR/experiment9_failures.log"
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
# CKA-only checkpoints (2 teachers × 6 datasets × 3 fracs × 2 modes)
# ============================================================
echo "--- Generating: CKA-only runs ---"
COUNT_CKA=0
for TEACHER in clip_l supervised_l; do
    CKPT="$OUTPUT_DIR/imagenet_${TEACHER}_cka_only_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping $TEACHER cka_only"
        continue
    fi
    TNAME="imagenet_${TEACHER}_cka_only"
    for DATASET in $ALL_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            # Fine-tune
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT_CKA=$((COUNT_CKA + 1))
                add_job "cka_only/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_CKA=$((COUNT_CKA + 1))
                add_job "cka_only/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
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
echo "  CKA-only (2 modes): $COUNT_CKA jobs"

# ============================================================
# Cosine-only (embedding) checkpoints (2 teachers × 6 datasets × 3 fracs × 2 modes)
# ============================================================
echo "--- Generating: Cosine-only (embedding) runs ---"
COUNT_COS=0
for TEACHER in clip_l supervised_l; do
    CKPT="$OUTPUT_DIR/imagenet_${TEACHER}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping $TEACHER embedding"
        continue
    fi
    TNAME="imagenet_${TEACHER}"
    for DATASET in $ALL_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            # Fine-tune
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT_COS=$((COUNT_COS + 1))
                add_job "cosine/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_COS=$((COUNT_COS + 1))
                add_job "cosine/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
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
echo "  Cosine-only (2 modes): $COUNT_COS jobs"

# ============================================================
# Execute all jobs
# ============================================================
TOTAL=$(wc -l < "$JOBFILE")
echo ""
echo "=============================================="
echo "Total jobs to run: $TOTAL"
echo "  CKA-only: $COUNT_CKA"
echo "  Cosine-only: $COUNT_COS"
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
    local LOG="$OUTPUT_DIR/log_exp9_${LOG_NAME}.txt"

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
echo "===== Experiment 9: Downstream Complete ====="
echo "Total jobs: $TOTAL"
echo "Failed: $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    cat "$FAIL_LOG"
    echo ""
    echo "Check logs: $OUTPUT_DIR/log_exp9_*.txt"
fi
echo ""
echo "Results saved to $OUTPUT_DIR/results_*.csv"
echo ""
echo "Key analysis (3×2 comparison matrix):"
echo "  Rows: clip_l, supervised_l"
echo "  Cols: CKA-only, Cosine-only, CKA+cosine (Exp 8)"
echo ""
echo "Questions to answer:"
echo "  1. CKA-only vs Cosine-only accuracy → structural alignment vs embedding content?"
echo "  2. clip_l CKA-only vs sup_l CKA-only → is CLIP's advantage structural?"
echo "  3. CKA+cosine vs best of {CKA-only, cosine-only} → does combining help?"

rm -f "$JOBFILE"
