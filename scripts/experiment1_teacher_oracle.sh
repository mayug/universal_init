#!/bin/bash
# Experiment 1: Teacher Oracle Probes
# Train linear probes on frozen ImageBind embeddings to establish performance ceiling

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Add ImageBind to PYTHONPATH
export PYTHONPATH="/home/ubuntu/projects/ImageBind:$PYTHONPATH"

# Configuration
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR="./checkpoints"
EPOCHS=50
BATCH_SIZE=64
LR=0.01
SEED=42

echo "===== Experiment 1: Teacher Oracle Probes ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PYTHON: $PYTHON"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run on Pets and EuroSAT at 1% and 100% labels
# 4 total runs

echo "1/4: Pets, 1% labels"
$PYTHON src/train_teacher_probe.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "2/4: Pets, 100% labels"
$PYTHON src/train_teacher_probe.py \
    --dataset pets \
    --data_root "$DATA_ROOT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "3/4: EuroSAT, 1% labels"
$PYTHON src/train_teacher_probe.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --label_fraction 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "4/4: EuroSAT, 100% labels"
$PYTHON src/train_teacher_probe.py \
    --dataset eurosat \
    --data_root "$DATA_ROOT" \
    --label_fraction 1.0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --no_wandb

echo ""
echo "===== Experiment 1 Complete ====="
echo "Results saved to $OUTPUT_DIR/results_*_teacher_oracle_*.csv"
