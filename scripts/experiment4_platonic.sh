#!/bin/bash
# Experiment 4: Platonic Representation - Distillation Phase
# Distill with 3 different teachers on COCO to compare representation quality:
#   D1: Supervised ViT-B/16 (768-dim) - trained with labels
#   D2: CLIP ViT-B/16 pre-projection (768-dim) - self-supervised, raw features
#   D3: CLIP ViT-B/16 with projection (512-dim) - self-supervised, projected features

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
EPOCHS=30
BATCH_SIZE=256
LR=1e-3
LOSS="combined"
LAMBDA_REL=0.5
SEED=42

echo "===== Experiment 4: Platonic Representation - Distillation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo ""

# Verify COCO exists
COCO_DIR="$DATA_ROOT/coco/train2017"
if [ ! -d "$COCO_DIR" ]; then
    echo "ERROR: COCO train2017 not found at $COCO_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# D1: Supervised ViT-B/16 (768-dim)
echo ""
echo "D1/3: Supervised ViT-B/16 teacher"
echo "=============================================="
$PYTHON src/train_distill.py \
    --teacher supervised \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
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

# D2: CLIP ViT-B/16 pre-projection (768-dim)
echo ""
echo "D2/3: CLIP ViT-B/16 pre-projection teacher (768-dim)"
echo "=============================================="
$PYTHON src/train_distill.py \
    --teacher clip_768 \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
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

# D3: CLIP ViT-B/16 with projection (512-dim)
echo ""
echo "D3/3: CLIP ViT-B/16 with projection teacher (512-dim)"
echo "=============================================="
$PYTHON src/train_distill.py \
    --teacher clip_512 \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
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

echo ""
echo "===== Experiment 4: Distillation Complete ====="
echo "Checkpoints:"
echo "  $OUTPUT_DIR/coco_supervised_distilled_best.pth"
echo "  $OUTPUT_DIR/coco_clip_768_distilled_best.pth"
echo "  $OUTPUT_DIR/coco_clip_512_distilled_best.pth"
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment4_downstream.sh"
