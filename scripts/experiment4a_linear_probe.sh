#!/bin/bash
# Experiment 4A: Linear Probing — Freeze backbone, train only classifier head
# 30 runs: (3 distilled + 2 baselines) x 2 datasets x 3 label fractions
# Tests representation quality in isolation from fine-tuning dynamics.

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

echo "===== Experiment 4A: Linear Probing ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "LR: 0.1 (fixed for linear probing)"
echo ""

mkdir -p "$OUTPUT_DIR"

RUN=0
TOTAL=30

# --- Distilled students (18 runs) ---
for TEACHER in supervised clip_768 clip_512; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_distilled_best.pth"

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        echo "Run experiment4_platonic.sh first."
        exit 1
    fi

    for DATASET in pets eurosat; do
        for FRAC in 0.01 0.1 1.0; do
            RUN=$((RUN + 1))
            echo ""
            echo "$RUN/$TOTAL: distilled($TEACHER) / $DATASET / frac=$FRAC / LINEAR PROBE"
            echo "=============================================="
            $PYTHON src/train_downstream.py \
                --dataset "$DATASET" \
                --data_root "$DATA_ROOT" \
                --init distilled \
                --checkpoint "$CKPT" \
                --teacher_name "$TEACHER" \
                --label_fraction "$FRAC" \
                --freeze_backbone \
                --lr 0.1 \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_dir "$OUTPUT_DIR" \
                --no_wandb
        done
    done
done

# --- Baselines (12 runs) ---
for INIT in random imagenet; do
    for DATASET in pets eurosat; do
        for FRAC in 0.01 0.1 1.0; do
            RUN=$((RUN + 1))
            echo ""
            echo "$RUN/$TOTAL: $INIT / $DATASET / frac=$FRAC / LINEAR PROBE"
            echo "=============================================="
            $PYTHON src/train_downstream.py \
                --dataset "$DATASET" \
                --data_root "$DATA_ROOT" \
                --init "$INIT" \
                --label_fraction "$FRAC" \
                --freeze_backbone \
                --lr 0.1 \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_dir "$OUTPUT_DIR" \
                --no_wandb
        done
    done
done

echo ""
echo "===== Experiment 4A: Linear Probing Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_linprobe_*.csv"
