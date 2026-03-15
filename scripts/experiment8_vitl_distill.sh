#!/bin/bash
# Experiment 8: ViT-L Teacher Distillation on ImageNet
#
# Tests whether ViT-L teachers (304M params, ~3.5x larger than ViT-B) produce
# qualitatively different results that support the platonic representation hypothesis.
#
# 3 distillation runs (CKA combined loss only, consistent with Exp 7 methodology):
#   1. supervised_l  (vit_large_patch16_224.augreg_in21k_ft_in1k)  — 1024-dim
#   2. clip_l        (vit_large_patch14_clip_224.openai)            — 1024-dim
#   3. dinov2_l      (vit_large_patch14_dinov2.lvd142m)             — 1024-dim (518→224px)
#
# Checkpoint naming:
#   imagenet_{teacher}_cka_l0.1_distilled_best.pth
#
# Hardware: B200 (183GB). Batch size 512 (reduced from 1024 in Exp 7 due to
# ViT-L ~3.5x memory). LR scaled proportionally: 4e-3 → 2e-3 (linear scaling).
#
# Estimated time: ~10-12h per run × 3 = ~30-36h total.
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment8_vitl_distill.sh
#   BATCH_SIZE=768 ./scripts/experiment8_vitl_distill.sh  # if GPU mem allows

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-2e-3}"
LAMBDA_CKA="${LAMBDA_CKA:-0.1}"

TEACHERS="supervised_l clip_l dinov2_l"

echo "===== Experiment 8: ViT-L Teacher Distillation on ImageNet ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "LAMBDA_CKA: $LAMBDA_CKA"
echo "TEACHERS: $TEACHERS"
echo ""

# Verify ImageNet exists
IMAGENET_TRAIN="$DATA_ROOT/imagenet/train"
if [ ! -d "$IMAGENET_TRAIN" ]; then
    echo "ERROR: ImageNet train directory not found at $IMAGENET_TRAIN"
    echo "Expected structure: $DATA_ROOT/imagenet/train/n01440764/..."
    exit 1
fi

NUM_CLASSES=$(ls -d "$IMAGENET_TRAIN"/*/ 2>/dev/null | wc -l)
echo "ImageNet train classes found: $NUM_CLASSES"
if [ "$NUM_CLASSES" -lt 1000 ]; then
    echo "WARNING: Expected 1000 classes, found $NUM_CLASSES"
fi
echo ""

mkdir -p "$OUTPUT_DIR"

# ============================================================
# CKA combined loss (3 runs, sequential)
# ============================================================
RUN=0
for TEACHER in $TEACHERS; do
    RUN=$((RUN + 1))
    echo ""
    echo "=============================================="
    echo "Run ${RUN}/3: $TEACHER / cka_combined (lambda_cka=$LAMBDA_CKA)"
    echo "=============================================="
    $PYTHON src/train_distill.py \
        --teacher "$TEACHER" \
        --dataset imagenet \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss cka_combined \
        --lambda_cka $LAMBDA_CKA \
        --warmup_epochs 5 \
        --output_dir "$OUTPUT_DIR" \
        --save_every 10 \
        --val_every 1 \
        --val_fraction 0.1 \
        --probe_every 0 \
        --seed $SEED \
        --amp \
        --no_wandb

    echo ""
    echo "Run ${RUN} complete: $OUTPUT_DIR/imagenet_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
done

echo ""
echo "===== Experiment 8: ViT-L Distillation Complete ====="
echo ""
echo "Checkpoints:"
for TEACHER in $TEACHERS; do
    echo "  $OUTPUT_DIR/imagenet_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
done
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment8_downstream.sh"
