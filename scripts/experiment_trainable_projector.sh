#!/bin/bash
# Experiment: Trainable Projector Ablation
# Tests keeping the projector trainable (not frozen) during fine-tuning
# Compares: drop projector vs frozen projector vs trainable projector

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Default values
DATA_ROOT="${DATA_ROOT:-./data}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/coco_distilled_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"

echo "=============================================="
echo "Trainable Projector Ablation Experiment"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "1/4: Pets, 1% labels, trainable projector"
echo "=============================================="
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 0.01 \
    --keep_projector \
    --train_projector \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "2/4: Pets, 100% labels, trainable projector"
echo "=============================================="
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 1.0 \
    --keep_projector \
    --train_projector \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "3/4: EuroSAT, 1% labels, trainable projector"
echo "=============================================="
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 0.01 \
    --keep_projector \
    --train_projector \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "4/4: EuroSAT, 100% labels, trainable projector"
echo "=============================================="
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 1.0 \
    --keep_projector \
    --train_projector \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "===== Trainable Projector Experiment Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_trainproj_*.csv"
