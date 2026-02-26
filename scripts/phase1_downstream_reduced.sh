#!/bin/bash
# Phase 1: Reduced downstream experiments
# 2 datasets, 2 label fractions, 3 inits, 1 seed = 12 experiments

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-universal_init}"
DISTILL_CHECKPOINT="${DISTILL_CHECKPOINT:-$OUTPUT_DIR/coco_distilled_best.pth}"

# Reduced configuration
DATASETS="pets eurosat"
INITS="random imagenet distilled"
FRACTIONS="0.01 1.0"
SEED=1
EPOCHS=50

echo "========================================"
echo "Phase 1: Reduced Downstream Experiments"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Distilled checkpoint: $DISTILL_CHECKPOINT"
echo "Datasets: $DATASETS"
echo "Init methods: $INITS"
echo "Label fractions: $FRACTIONS"
echo "Seed: $SEED"
echo "Total runs: 12 (2 datasets × 3 inits × 2 fractions)"
echo "========================================"
echo ""

# Check checkpoint exists
if [ ! -f "$DISTILL_CHECKPOINT" ]; then
    echo "ERROR: Distilled checkpoint not found at $DISTILL_CHECKPOINT"
    exit 1
fi

current_run=0
total_runs=12

for dataset in $DATASETS; do
    for init in $INITS; do
        for frac in $FRACTIONS; do
            current_run=$((current_run + 1))
            echo ""
            echo "[$current_run/$total_runs] $dataset | $init | frac=$frac"

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
                --seed $SEED"

            # Add checkpoint for distilled init
            if [ "$init" = "distilled" ]; then
                cmd="$cmd --checkpoint $DISTILL_CHECKPOINT"
            fi

            # Run
            eval $cmd
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
