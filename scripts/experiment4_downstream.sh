#!/bin/bash
# Experiment 4: Platonic Representation - Downstream Phase
# 24 runs: 3 teachers x 2 projector modes x 2 datasets x 2 fractions

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

echo "===== Experiment 4: Platonic Representation - Downstream ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo ""

mkdir -p "$OUTPUT_DIR"

RUN=0
TOTAL=24

for TEACHER in supervised clip_768 clip_512; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_distilled_best.pth"

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        echo "Run experiment4_platonic.sh first."
        exit 1
    fi

    for DATASET in pets eurosat; do
        for FRAC in 0.01 1.0; do
            # Run 1: drop projector (standard)
            RUN=$((RUN + 1))
            echo ""
            echo "$RUN/$TOTAL: $TEACHER / $DATASET / frac=$FRAC / drop projector"
            echo "=============================================="
            $PYTHON src/train_downstream.py \
                --dataset "$DATASET" \
                --data_root "$DATA_ROOT" \
                --init distilled \
                --checkpoint "$CKPT" \
                --label_fraction "$FRAC" \
                --teacher_name "$TEACHER" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_dir "$OUTPUT_DIR" \
                --no_wandb

            # Run 2: trainable projector
            RUN=$((RUN + 1))
            echo ""
            echo "$RUN/$TOTAL: $TEACHER / $DATASET / frac=$FRAC / trainable projector"
            echo "=============================================="
            $PYTHON src/train_downstream.py \
                --dataset "$DATASET" \
                --data_root "$DATA_ROOT" \
                --init distilled \
                --checkpoint "$CKPT" \
                --label_fraction "$FRAC" \
                --teacher_name "$TEACHER" \
                --keep_projector \
                --train_projector \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --seed $SEED \
                --output_dir "$OUTPUT_DIR" \
                --no_wandb
        done
    done
done

echo ""
echo "===== Experiment 4: Downstream Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_distilled_*.csv"
