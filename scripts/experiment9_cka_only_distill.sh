#!/bin/bash
# Experiment 9: CKA-Only Distillation — Isolating the Platonic Claim
#
# Tests whether structural alignment alone (CKA loss, no point-wise cosine)
# drives downstream utility, or whether embedding content matters more.
#
# 4 distillation runs:
#   1. clip_l       / cka_only   → imagenet_clip_l_cka_only_distilled_best.pth
#   2. supervised_l / cka_only   → imagenet_supervised_l_cka_only_distilled_best.pth
#   3. clip_l       / embedding  → imagenet_clip_l_distilled_best.pth
#   4. supervised_l / embedding  → imagenet_supervised_l_distilled_best.pth
#
# Existing from Exp 8 (NOT re-run):
#   imagenet_clip_l_cka_l0.1_distilled_best.pth        (CKA+cosine combined)
#   imagenet_supervised_l_cka_l0.1_distilled_best.pth   (CKA+cosine combined)
#
# Config: Same as Exp 8 — AMP, 5-epoch warmup, cosine LR, grad clipping.
# Hardware: B200 (183GB). BS=512, LR=2e-3.
#
# Estimated time: ~10-12h per run × 4 = ~40-48h total (sequential).
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment9_cka_only_distill.sh
#   BATCH_SIZE=768 ./scripts/experiment9_cka_only_distill.sh  # if GPU mem allows

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

echo "===== Experiment 9: CKA-Only Distillation (Isolating the Platonic Claim) ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
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
# CKA-only loss (2 runs)
# ============================================================
RUN=0
for TEACHER in clip_l supervised_l; do
    RUN=$((RUN + 1))
    echo ""
    echo "=============================================="
    echo "Run ${RUN}/4: $TEACHER / cka_only"
    echo "=============================================="
    $PYTHON src/train_distill.py \
        --teacher "$TEACHER" \
        --dataset imagenet \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss cka_only \
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
    echo "Run ${RUN} complete: $OUTPUT_DIR/imagenet_${TEACHER}_cka_only_distilled_best.pth"
done

# ============================================================
# Cosine-only (embedding) loss (2 runs)
# ============================================================
for TEACHER in clip_l supervised_l; do
    RUN=$((RUN + 1))
    echo ""
    echo "=============================================="
    echo "Run ${RUN}/4: $TEACHER / embedding (cosine-only)"
    echo "=============================================="
    $PYTHON src/train_distill.py \
        --teacher "$TEACHER" \
        --dataset imagenet \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss embedding \
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
    echo "Run ${RUN} complete: $OUTPUT_DIR/imagenet_${TEACHER}_distilled_best.pth"
done

echo ""
echo "===== Experiment 9: Distillation Complete ====="
echo ""
echo "New checkpoints:"
echo "  $OUTPUT_DIR/imagenet_clip_l_cka_only_distilled_best.pth"
echo "  $OUTPUT_DIR/imagenet_supervised_l_cka_only_distilled_best.pth"
echo "  $OUTPUT_DIR/imagenet_clip_l_distilled_best.pth"
echo "  $OUTPUT_DIR/imagenet_supervised_l_distilled_best.pth"
echo ""
echo "Existing from Exp 8 (reuse):"
echo "  $OUTPUT_DIR/imagenet_clip_l_cka_l0.1_distilled_best.pth"
echo "  $OUTPUT_DIR/imagenet_supervised_l_cka_l0.1_distilled_best.pth"
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment9_downstream.sh"
