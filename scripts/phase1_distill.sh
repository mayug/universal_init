#!/bin/bash
# Phase 1: Main distillation on COCO
# Purpose: Train student to match ImageBind embeddings on larger corpus

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-universal_init}"

echo "========================================"
echo "Phase 1: COCO Distillation"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Main distillation run (combined loss)
echo "Running COCO distillation with combined loss..."
python src/train_distill.py \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --loss combined \
    --lambda_rel 0.5 \
    --projector linear \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --save_every 20 \
    --seed 42

echo ""
echo "========================================"
echo "COCO Distillation Complete!"
echo "========================================"
echo "Checkpoint: $OUTPUT_DIR/coco_distilled_best.pth"
echo "Next: Run phase1_downstream.sh"
