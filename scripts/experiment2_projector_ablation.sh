#!/bin/bash
# Experiment 2: Keep vs Drop Projector Ablation
# Compare downstream performance with frozen COCO-distilled projector vs without

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR="./checkpoints"
CHECKPOINT="./checkpoints/coco_distilled_best.pth"
EPOCHS=50
BATCH_SIZE=64
LR=0.01
SEED=42

echo "===== Experiment 2: Projector Ablation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CHECKPOINT: $CHECKPOINT"
echo "PYTHON: $PYTHON"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: COCO distilled checkpoint not found at $CHECKPOINT"
    echo "Please run distillation first: ./scripts/phase1_distill.sh"
    exit 1
fi

# Run with --keep_projector (frozen projector + classifier on top)
# Baseline (drop projector) results already exist from previous experiments

echo "Running with KEEP_PROJECTOR (frozen projector from COCO distillation)"
echo ""

echo "1/4: Pets, 1% labels, keep projector"
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --keep_projector \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "2/4: Pets, 100% labels, keep projector"
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --keep_projector \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "3/4: EuroSAT, 1% labels, keep projector"
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --keep_projector \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "4/4: EuroSAT, 100% labels, keep projector"
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$CHECKPOINT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --keep_projector \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "===== Experiment 2 Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_distilled_keepproj_*.csv"
echo ""
echo "Compare with baseline (drop projector) results:"
echo "  $OUTPUT_DIR/results_*_distilled_frac*.csv (without keepproj suffix)"
