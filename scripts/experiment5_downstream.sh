#!/bin/bash
# Experiment 5: CKA Distillation - Downstream Phase
# 42 runs: linear probing (18) + fine-tuning (24)
# Uses CKA-distilled checkpoints from experiment5_cka_distill.sh.
# Parallelized 6-at-a-time (downstream runs are lightweight).
#
# Set LAMBDA env var to pick which lambda to evaluate (default: 0.5).
# Example: LAMBDA=1.0 ./scripts/experiment5_downstream.sh

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
LAMBDA="${LAMBDA:-0.5}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"

TEACHERS="supervised clip_768 clip_512"
DATASETS="pets eurosat"

echo "===== Experiment 5: CKA Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "LAMBDA: $LAMBDA"
echo "EPOCHS: $EPOCHS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo ""

mkdir -p "$OUTPUT_DIR"

# Verify checkpoints exist
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_cka_l${LAMBDA}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        echo "Run experiment5_cka_distill.sh first."
        exit 1
    fi
done

FAIL_COUNT=0
run_job() {
    local MODE=$1 TEACHER=$2 DATASET=$3 FRAC=$4 CKPT=$5
    shift 5
    local EXTRA_ARGS="$@"
    local TAG="${MODE}_${TEACHER}_cka_l${LAMBDA}_${DATASET}_f${FRAC}"
    local LOG="$OUTPUT_DIR/log_downstream_${TAG}.txt"

    echo "  Starting: $TAG"
    $PYTHON src/train_downstream.py \
        --dataset "$DATASET" \
        --data_root "$DATA_ROOT" \
        --init distilled \
        --checkpoint "$CKPT" \
        --teacher_name "$TEACHER" \
        --label_fraction "$FRAC" \
        $EXTRA_ARGS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_dir "$OUTPUT_DIR" \
        --no_wandb \
        > "$LOG" 2>&1 || {
            echo "  FAILED: $TAG (see $LOG)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        }
}

# ============================================================
# Part 1: Linear Probing (18 runs)
# 3 teachers x 2 datasets x 3 fractions, frozen backbone
# ============================================================
echo ""
echo "--- Part 1: Linear Probing (18 runs) ---"

RUN=0
TOTAL_LP=18
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_cka_l${LAMBDA}_distilled_best.pth"
    for DATASET in $DATASETS; do
        for FRAC in 0.01 0.1 1.0; do
            RUN=$((RUN + 1))
            echo "$RUN/$TOTAL_LP: linprobe / $TEACHER / $DATASET / frac=$FRAC"
            run_job linprobe "$TEACHER" "$DATASET" "$FRAC" "$CKPT" \
                --freeze_backbone --lr 0.1 &

            # Limit parallelism
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 5
            done
        done
    done
done
wait

echo ""
echo "--- Linear probing complete ---"

# ============================================================
# Part 2: Fine-tuning (24 runs)
# 3 teachers x 2 datasets x 2 fractions x 2 projector modes
# ============================================================
echo ""
echo "--- Part 2: Fine-tuning (24 runs) ---"

RUN=0
TOTAL_FT=24
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_cka_l${LAMBDA}_distilled_best.pth"
    for DATASET in $DATASETS; do
        for FRAC in 0.01 1.0; do
            # Run 1: drop projector (standard)
            RUN=$((RUN + 1))
            echo "$RUN/$TOTAL_FT: finetune / $TEACHER / $DATASET / frac=$FRAC / drop projector"
            run_job finetune "$TEACHER" "$DATASET" "$FRAC" "$CKPT" &

            # Limit parallelism
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 5
            done

            # Run 2: trainable projector
            RUN=$((RUN + 1))
            echo "$RUN/$TOTAL_FT: finetune / $TEACHER / $DATASET / frac=$FRAC / trainable projector"
            run_job finetune_proj "$TEACHER" "$DATASET" "$FRAC" "$CKPT" \
                --keep_projector --train_projector &

            # Limit parallelism
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 5
            done
        done
    done
done
wait

echo ""
echo "===== Experiment 5: Downstream Complete ====="
if [ $FAIL_COUNT -gt 0 ]; then
    echo "WARNING: $FAIL_COUNT runs failed. Check logs in $OUTPUT_DIR/log_downstream_*.txt"
fi
echo "Results saved to $OUTPUT_DIR/results_*_distilled_*.csv"
