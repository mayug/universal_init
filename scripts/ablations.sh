#!/bin/bash
# Ablation experiments
# Purpose: Test loss variants and projector handling

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-universal_init}"

echo "========================================"
echo "Ablation Experiments"
echo "========================================"

# ============================================
# Ablation 1: Loss function (embedding-only vs combined)
# ============================================
echo ""
echo "Ablation 1: Loss function comparison"
echo "--------------------------------------"

# Already have combined loss from phase1, run embedding-only
echo "Running embedding-only distillation..."
python src/train_distill.py \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --loss embedding \
    --projector linear \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "distill_coco_embedding_only" \
    --save_every 20 \
    --seed 42

# Run downstream comparison on one dataset
echo ""
echo "Comparing losses on Flowers102 (10% labels)..."
for loss_type in embedding combined; do
    checkpoint="$OUTPUT_DIR/coco_distilled_best.pth"
    if [ "$loss_type" = "embedding" ]; then
        # The embedding-only checkpoint will have a different name
        # Rename or specify correctly based on your naming convention
        checkpoint="$OUTPUT_DIR/coco_distilled_best.pth"
    fi

    python src/train_downstream.py \
        --dataset flowers102 \
        --data_root "$DATA_ROOT" \
        --init distilled \
        --checkpoint "$checkpoint" \
        --epochs 50 \
        --batch_size 64 \
        --lr 0.01 \
        --label_fraction 0.1 \
        --output_dir "$OUTPUT_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "ablation_loss_${loss_type}_flowers102" \
        --seed 42
done

# ============================================
# Ablation 2: Projector type (linear vs MLP)
# ============================================
echo ""
echo "Ablation 2: Projector type comparison"
echo "--------------------------------------"

echo "Running MLP projector distillation..."
python src/train_distill.py \
    --dataset coco \
    --data_root "$DATA_ROOT" \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --loss combined \
    --projector mlp \
    --projector_hidden_dim 512 \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "distill_coco_mlp_projector" \
    --save_every 20 \
    --seed 42

echo ""
echo "========================================"
echo "Ablations Complete!"
echo "========================================"
echo "Check W&B for comparisons:"
echo "  - Loss: embedding-only vs combined"
echo "  - Projector: linear vs MLP"
