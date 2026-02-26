#!/bin/bash
# Experiment 3: Compare ImageNet-distilled vs COCO-distilled
# Run downstream tasks with both distillation sources

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR="./checkpoints"
IMAGENET_CHECKPOINT="./checkpoints/imagenet_distilled_best.pth"
COCO_CHECKPOINT="./checkpoints/coco_distilled_best.pth"
EPOCHS=50
BATCH_SIZE=64
LR=0.01
SEED=42

echo "===== Experiment 3: Downstream Comparison ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Verify checkpoints exist
if [ ! -f "$IMAGENET_CHECKPOINT" ]; then
    echo "ERROR: ImageNet checkpoint not found at $IMAGENET_CHECKPOINT"
    echo "Please run distillation first: ./scripts/experiment3_imagenet_distill.sh"
    exit 1
fi

if [ ! -f "$COCO_CHECKPOINT" ]; then
    echo "ERROR: COCO checkpoint not found at $COCO_CHECKPOINT"
    echo "Please run distillation first: ./scripts/phase1_distill.sh"
    exit 1
fi

echo "Comparing ImageNet-distilled vs COCO-distilled on Pets and EuroSAT"
echo ""

# Run with ImageNet-distilled checkpoint
# Results will be saved with init="distilled" and checkpoint path will distinguish them

echo "===== ImageNet-Distilled Init ====="
echo ""

echo "1/8: Pets, 1% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

# Rename to distinguish from COCO-distilled
mv "$OUTPUT_DIR/results_pets_distilled_frac0.01_s${SEED}.csv" \
   "$OUTPUT_DIR/results_pets_distilled_imagenet_frac0.01_s${SEED}.csv"

echo ""
echo "2/8: Pets, 10% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 0.1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

mv "$OUTPUT_DIR/results_pets_distilled_frac0.1_s${SEED}.csv" \
   "$OUTPUT_DIR/results_pets_distilled_imagenet_frac0.1_s${SEED}.csv"

echo ""
echo "3/8: Pets, 100% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

mv "$OUTPUT_DIR/results_pets_distilled_frac1.0_s${SEED}.csv" \
   "$OUTPUT_DIR/results_pets_distilled_imagenet_frac1.0_s${SEED}.csv"

echo ""
echo "4/8: EuroSAT, 1% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

mv "$OUTPUT_DIR/results_eurosat_distilled_frac0.01_s${SEED}.csv" \
   "$OUTPUT_DIR/results_eurosat_distilled_imagenet_frac0.01_s${SEED}.csv"

echo ""
echo "5/8: EuroSAT, 10% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 0.1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

mv "$OUTPUT_DIR/results_eurosat_distilled_frac0.1_s${SEED}.csv" \
   "$OUTPUT_DIR/results_eurosat_distilled_imagenet_frac0.1_s${SEED}.csv"

echo ""
echo "6/8: EuroSAT, 100% labels, ImageNet-distilled"
$PYTHON src/train_downstream.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --init distilled \
    --checkpoint "$IMAGENET_CHECKPOINT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

mv "$OUTPUT_DIR/results_eurosat_distilled_frac1.0_s${SEED}.csv" \
   "$OUTPUT_DIR/results_eurosat_distilled_imagenet_frac1.0_s${SEED}.csv"

echo ""
echo "===== COCO-Distilled Init (for comparison) ====="
echo ""

# Note: If COCO-distilled results already exist, skip these runs
if [ -f "$OUTPUT_DIR/results_pets_distilled_coco_frac0.01_s${SEED}.csv" ]; then
    echo "COCO-distilled results already exist, skipping..."
else
    echo "7/8: Running COCO-distilled comparisons..."

    # Run same experiments with COCO checkpoint
    $PYTHON src/train_downstream.py \
        --dataset pets \
        --data_root "$DATA_ROOT" \
        --init distilled \
        --checkpoint "$COCO_CHECKPOINT" \
        --label_fraction 0.01 \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --output_dir "$OUTPUT_DIR" \
        --no_wandb

    mv "$OUTPUT_DIR/results_pets_distilled_frac0.01_s${SEED}.csv" \
       "$OUTPUT_DIR/results_pets_distilled_coco_frac0.01_s${SEED}.csv"

    # Add similar runs for other configs if needed
    echo "Note: Add remaining COCO-distilled runs as needed"
fi

echo ""
echo "===== Experiment 3 Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_distilled_imagenet_*.csv"
echo ""
echo "Compare with COCO-distilled results:"
echo "  $OUTPUT_DIR/results_*_distilled_coco_*.csv"
echo ""
echo "Analyze results:"
echo "  $PYTHON src/evaluate.py --results_dir ./checkpoints"
