#!/bin/bash
# Experiment 13: OOD Vision Downstream — Medical Imaging
#
# Tests whether CLIP-distilled initialization beats ImageNet pretraining
# on out-of-distribution medical imaging datasets (MedMNIST).
#
# 4 init modes × 3 label fractions × 2 eval modes = 24 runs per dataset
#
# Datasets: PathMNIST (9-class histopathology), DermaMNIST (7-class skin lesion),
#           BloodMNIST (8-class blood cell microscopy)
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment13_ood_vision.sh
#   DATASET=pathmnist DATA_ROOT=./data ./scripts/experiment13_ood_vision.sh  # single dataset

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_FT="${LR_FT:-0.01}"       # Learning rate for fine-tuning
LR_LP="${LR_LP:-0.1}"        # Learning rate for linear probe
DATASET="${DATASET:-}"        # Empty = all datasets
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Checkpoint paths
CLIP_CKPT="${CLIP_CKPT:-$OUTPUT_DIR/imagenet_clip_768_cka_l0.1_distilled_best.pth}"
SUPERVISED_CKPT="${SUPERVISED_CKPT:-$OUTPUT_DIR/imagenet_supervised_cka_l0.1_distilled_best.pth}"

echo "===== Experiment 13: OOD Vision Downstream (Medical Imaging) ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "CLIP checkpoint: $CLIP_CKPT"
echo "Supervised checkpoint: $SUPERVISED_CKPT"
echo ""

# Verify checkpoints exist
for ckpt in "$CLIP_CKPT" "$SUPERVISED_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        exit 1
    fi
done

# Determine datasets to run
if [ -n "$DATASET" ]; then
    DATASETS=("$DATASET")
else
    DATASETS=(pathmnist dermamnist bloodmnist)
fi

PIDS=()
LOG_DIR="$OUTPUT_DIR/logs_exp13"
mkdir -p "$LOG_DIR"

run_one() {
    local dataset=$1
    local init=$2
    local frac=$3
    local mode=$4  # finetune or linprobe

    local run_name="exp13_${dataset}_${init}_frac${frac}_${mode}"
    local log_file="$LOG_DIR/${run_name}.log"

    # Build command
    local cmd="$PYTHON src/train_downstream.py \
        --dataset $dataset \
        --data_root $DATA_ROOT \
        --label_fraction $frac \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --wandb_run_name $run_name \
        --no_wandb"

    # Init mode
    case $init in
        random)
            cmd="$cmd --init random"
            ;;
        imagenet)
            cmd="$cmd --init imagenet"
            ;;
        clip_distilled)
            cmd="$cmd --init distilled --checkpoint $CLIP_CKPT"
            ;;
        supervised_distilled)
            cmd="$cmd --init distilled --checkpoint $SUPERVISED_CKPT"
            ;;
    esac

    # Learning rate and mode
    if [ "$mode" == "linprobe" ]; then
        cmd="$cmd --freeze_backbone --lr $LR_LP"
    else
        cmd="$cmd --lr $LR_FT"
    fi

    echo "  Starting: $run_name"
    $cmd > "$log_file" 2>&1 &
    PIDS+=($!)
}

# Throttle: wait if too many parallel jobs
wait_for_slot() {
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done
}

# Launch all runs
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: $dataset ---"
    for init in random imagenet clip_distilled supervised_distilled; do
        for frac in 0.01 0.1 1.0; do
            for mode in finetune linprobe; do
                wait_for_slot
                run_one "$dataset" "$init" "$frac" "$mode"
            done
        done
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for all jobs to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: Job $pid exited with non-zero status"
done

echo ""
echo "===== Experiment 13 complete! ====="
echo "Logs saved to: $LOG_DIR"
echo "Results CSVs saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results:"
echo "  ls $OUTPUT_DIR/results_{pathmnist,dermamnist,bloodmnist}_*.csv"
