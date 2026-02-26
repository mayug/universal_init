#!/bin/bash
# Experiment 3: Distill on ImageNet Train
# Run distillation on ImageNet train split (unlabeled)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Add ImageBind to PYTHONPATH
export PYTHONPATH="/home/ubuntu/projects/ImageBind:$PYTHONPATH"

# Configuration - optimized for H200 (144GB)
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR="./checkpoints"
EPOCHS=20
BATCH_SIZE=1024
LR=4e-3
LOSS="combined"
LAMBDA_REL=0.5
SEED=42

echo "===== Experiment 3: ImageNet Distillation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo ""

# Verify ImageNet exists
IMAGENET_TRAIN="$DATA_ROOT/imagenet/train"
if [ ! -d "$IMAGENET_TRAIN" ]; then
    echo "ERROR: ImageNet train directory not found at $IMAGENET_TRAIN"
    echo "Expected structure: $DATA_ROOT/imagenet/train/n01440764/..."
    exit 1
fi

echo "Starting ImageNet distillation..."
$PYTHON src/train_distill.py \
    --dataset imagenet \
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
echo "===== Experiment 3: Distillation Complete ====="
echo "Checkpoint saved to $OUTPUT_DIR/imagenet_distilled_best.pth"
echo ""
echo "Expected validation metrics:"
echo "  - Cosine similarity: ~0.65-0.75"
echo "  - Loss: ~0.3-0.5"
echo ""
echo "Next step: Run downstream comparison"
echo "  ./scripts/experiment3_downstream.sh"
