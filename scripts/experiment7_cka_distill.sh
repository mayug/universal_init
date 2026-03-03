#!/bin/bash
# Experiment 7: CKA Distillation on ImageNet — 2 runs
#
# Tests whether CKA structural alignment + 15x more data finally closes
# the gap with ImageNet pretraining. Prioritized over combined loss runs
# because CKA directly optimizes the platonic structure matching objective.
#
# Runs:
#   1. supervised + cka_combined (λ=0.1)
#   2. clip_768 + cka_combined (λ=0.1)
#
# clip_512 dropped: worse downstream, unstable with CKA λ=0.1 (backbone CKA=0.038).

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
LAMBDA_CKA="${LAMBDA_CKA:-0.1}"

echo "===== Experiment 7: CKA Distillation on ImageNet ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "LAMBDA_CKA: $LAMBDA_CKA"
echo ""

# Verify ImageNet exists
IMAGENET_TRAIN="$DATA_ROOT/imagenet/train"
if [ ! -d "$IMAGENET_TRAIN" ]; then
    echo "ERROR: ImageNet train directory not found at $IMAGENET_TRAIN"
    exit 1
fi

NUM_CLASSES=$(ls -d "$IMAGENET_TRAIN"/*/ 2>/dev/null | wc -l)
echo "ImageNet train classes found: $NUM_CLASSES"
echo ""

mkdir -p "$OUTPUT_DIR"

# Run 1: Supervised ViT-B/16 + CKA
echo "=============================================="
echo "Run 1/2: supervised + cka_combined (lambda=$LAMBDA_CKA)"
echo "=============================================="
$PYTHON src/train_distill.py \
    --teacher supervised \
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
echo "Run 1 complete: $OUTPUT_DIR/imagenet_supervised_cka_l${LAMBDA_CKA}_distilled_best.pth"
echo ""

# Run 2: CLIP ViT-B/16 pre-projection + CKA
echo "=============================================="
echo "Run 2/2: clip_768 + cka_combined (lambda=$LAMBDA_CKA)"
echo "=============================================="
$PYTHON src/train_distill.py \
    --teacher clip_768 \
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
echo "===== Experiment 7: CKA Distillation Complete ====="
echo "Checkpoints:"
echo "  $OUTPUT_DIR/imagenet_supervised_cka_l${LAMBDA_CKA}_distilled_best.pth"
echo "  $OUTPUT_DIR/imagenet_clip_768_cka_l${LAMBDA_CKA}_distilled_best.pth"
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment7_downstream.sh"
