#!/bin/bash
# Experiment 7: ImageNet Distillation — Does 15x more data close the gap?
#
# 6 distillation runs: 3 teachers × 2 loss types
#   Part 1 (combined loss):     supervised, clip_768, clip_512
#   Part 2 (CKA λ=0.1):        supervised, clip_768, clip_512
#
# Checkpoint naming:
#   imagenet_{teacher}_distilled_best.pth            (combined)
#   imagenet_{teacher}_cka_l0.1_distilled_best.pth   (CKA)
#
# Hardware: B200 (183GB). Batch size 1024 fits comfortably.
# Estimated time: ~2.5h per run × 6 = ~15h total.
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment7_imagenet_distill.sh
#   BATCH_SIZE=2048 ./scripts/experiment7_imagenet_distill.sh  # if GPU mem allows

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-4e-3}"
LAMBDA_REL="${LAMBDA_REL:-0.5}"
LAMBDA_CKA="${LAMBDA_CKA:-0.1}"

TEACHERS="supervised clip_768 clip_512"

echo "===== Experiment 7: ImageNet Distillation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "TEACHERS: $TEACHERS"
echo "LAMBDA_REL: $LAMBDA_REL (combined loss)"
echo "LAMBDA_CKA: $LAMBDA_CKA (CKA loss)"
echo ""

# Verify ImageNet exists
IMAGENET_TRAIN="$DATA_ROOT/imagenet/train"
if [ ! -d "$IMAGENET_TRAIN" ]; then
    echo "ERROR: ImageNet train directory not found at $IMAGENET_TRAIN"
    echo "Expected structure: $DATA_ROOT/imagenet/train/n01440764/..."
    echo "Run: bash scripts/extract_imagenet_fast.sh $DATA_ROOT/imagenet 16"
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
# Part 1: Combined loss (3 runs)
# ============================================================
echo "=============================================="
echo "Part 1/2: Combined loss (embedding + relational)"
echo "=============================================="

RUN=0
for TEACHER in $TEACHERS; do
    RUN=$((RUN + 1))
    echo ""
    echo "D${RUN}/3: $TEACHER / combined (lambda_rel=$LAMBDA_REL)"
    echo "=============================================="
    $PYTHON src/train_distill.py \
        --teacher "$TEACHER" \
        --dataset imagenet \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss combined \
        --lambda_rel $LAMBDA_REL \
        --warmup_epochs 5 \
        --output_dir "$OUTPUT_DIR" \
        --save_every 10 \
        --val_every 1 \
        --val_fraction 0.1 \
        --probe_every 0 \
        --seed $SEED \
        --amp \
        --no_wandb
done

echo ""
echo "Part 1 complete. Checkpoints:"
for TEACHER in $TEACHERS; do
    echo "  $OUTPUT_DIR/imagenet_${TEACHER}_distilled_best.pth"
done

# ============================================================
# Part 2: CKA combined loss (3 runs)
# ============================================================
echo ""
echo "=============================================="
echo "Part 2/2: CKA combined loss (lambda_cka=$LAMBDA_CKA)"
echo "=============================================="

RUN=0
for TEACHER in $TEACHERS; do
    RUN=$((RUN + 1))
    echo ""
    echo "D${RUN}/3: $TEACHER / cka_combined (lambda_cka=$LAMBDA_CKA)"
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
done

echo ""
echo "===== Experiment 7: Distillation Complete ====="
echo ""
echo "All checkpoints:"
for TEACHER in $TEACHERS; do
    echo "  $OUTPUT_DIR/imagenet_${TEACHER}_distilled_best.pth"
    echo "  $OUTPUT_DIR/imagenet_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
done
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment7_downstream.sh"
