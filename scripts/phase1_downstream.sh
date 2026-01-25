#!/bin/bash
# Phase 1: Downstream experiments on all datasets
# Purpose: Compare random vs ImageNet vs distilled init across datasets and label fractions

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-universal_init}"
DISTILL_CHECKPOINT="${DISTILL_CHECKPOINT:-$OUTPUT_DIR/coco_distilled_best.pth}"

# Configuration
DATASETS="pets flowers102 dtd eurosat"
INITS="random imagenet distilled"
FRACTIONS="0.01 0.1 1.0"
SEEDS="1 2 3"
EPOCHS=50

echo "========================================"
echo "Phase 1: Downstream Experiments"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Distilled checkpoint: $DISTILL_CHECKPOINT"
echo "Datasets: $DATASETS"
echo "Init methods: $INITS"
echo "Label fractions: $FRACTIONS"
echo "Seeds: $SEEDS"
echo "========================================"
echo ""

# Check checkpoint exists
if [ ! -f "$DISTILL_CHECKPOINT" ]; then
    echo "ERROR: Distilled checkpoint not found at $DISTILL_CHECKPOINT"
    echo "Run phase1_distill.sh first."
    exit 1
fi

# Total experiments
total_runs=$(echo "$DATASETS" | wc -w)
total_runs=$((total_runs * 3 * 3 * 3))  # 4 datasets * 3 inits * 3 fractions * 3 seeds
current_run=0

for dataset in $DATASETS; do
    for init in $INITS; do
        for frac in $FRACTIONS; do
            for seed in $SEEDS; do
                current_run=$((current_run + 1))
                echo ""
                echo "[$current_run/$total_runs] $dataset | $init | frac=$frac | seed=$seed"

                # Build command
                cmd="python src/train_downstream.py \
                    --dataset $dataset \
                    --data_root $DATA_ROOT \
                    --init $init \
                    --epochs $EPOCHS \
                    --batch_size 64 \
                    --lr 0.01 \
                    --label_fraction $frac \
                    --output_dir $OUTPUT_DIR \
                    --wandb_project $WANDB_PROJECT \
                    --seed $seed"

                # Add checkpoint for distilled init
                if [ "$init" = "distilled" ]; then
                    cmd="$cmd --checkpoint $DISTILL_CHECKPOINT"
                fi

                # Run
                eval $cmd
            done
        done
    done
done

echo ""
echo "========================================"
echo "All Downstream Experiments Complete!"
echo "========================================"
echo "Total runs: $total_runs"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Check W&B dashboard for learning curves"
echo "2. Run python src/evaluate.py to generate summary tables"
