#!/bin/bash
# Phase 0: Sanity check with Imagenette
# Purpose: Verify pipeline works end-to-end before running expensive experiments

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-universal_init}"

echo "========================================"
echo "Phase 0: Sanity Check (Imagenette)"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Distillation on Imagenette
echo "Step 1: Distillation training..."
python src/train_distill.py \
    --dataset imagenette \
    --data_root "$DATA_ROOT" \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-3 \
    --loss combined \
    --lambda_rel 0.5 \
    --projector linear \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

echo ""
echo "Step 2: Quick downstream check with all three inits..."

# Random init
echo "  - Random init..."
python src/train_downstream.py \
    --dataset imagenette \
    --data_root "$DATA_ROOT" \
    --init random \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.01 \
    --label_fraction 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

# ImageNet init
echo "  - ImageNet init..."
python src/train_downstream.py \
    --dataset imagenette \
    --data_root "$DATA_ROOT" \
    --init imagenet \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.01 \
    --label_fraction 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

# Distilled init
echo "  - Distilled init..."
python src/train_downstream.py \
    --dataset imagenette \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$OUTPUT_DIR/imagenette_distilled_best.pth" \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.01 \
    --label_fraction 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

echo ""
echo "========================================"
echo "Phase 0 Complete!"
echo "========================================"
echo "Check W&B dashboard for results comparison."
echo "Success criteria:"
echo "  - Distillation loss decreased smoothly"
echo "  - Cosine similarity > 0.7 by end of distillation"
echo "  - All downstream runs completed without errors"
echo "  - Distilled init shows faster early convergence than random init"
