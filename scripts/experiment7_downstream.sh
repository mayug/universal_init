#!/bin/bash
# Experiment 7: Downstream Evaluation — ImageNet distillation + COCO gap-fill
#
# Runs downstream jobs across 6 datasets, 3 label fractions,
# for ImageNet-distilled, COCO-distilled, and baseline checkpoints.
#
# CKA checkpoints get 4 modes: fine-tune (drop proj), linear probe,
# frozen projector, trainable projector — because CKA alignment lives
# mostly in the projector (projected CKA >> backbone CKA).
#
# Non-CKA checkpoints get 2 modes: fine-tune + linear probe (standard).
#
# Parallelized with xargs -P (default 4 concurrent GPU jobs).
# Each downstream run uses ~2-3GB VRAM on a RegNetY-400MF.
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment7_downstream.sh
#   MAX_PARALLEL=6 ./scripts/experiment7_downstream.sh   # more parallelism
#   SKIP_EXISTING=1 ./scripts/experiment7_downstream.sh  # skip completed runs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LAMBDA_CKA="${LAMBDA_CKA:-0.1}"

ALL_DATASETS="voc pets eurosat dtd flowers102 imagenette"
ALL_FRACTIONS="0.01 0.1 1.0"
# clip_512 dropped: worse downstream than clip_768 and unstable with CKA λ=0.1
TEACHERS="supervised clip_768"

# Datasets that need baselines run (not yet evaluated in earlier experiments)
NEW_DATASETS="dtd flowers102 imagenette"
# Datasets already evaluated in earlier experiments
OLD_DATASETS="voc pets eurosat"

echo "===== Experiment 7: Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo "TEACHERS: $TEACHERS"
echo ""

mkdir -p "$OUTPUT_DIR"
JOBFILE=$(mktemp /tmp/exp7_jobs_XXXXXX.txt)
FAIL_LOG="$OUTPUT_DIR/experiment7_failures.log"
> "$FAIL_LOG"

# ============================================================
# Helper: add a job to the job file
# ============================================================
add_job() {
    local DESC="$1"
    shift
    echo "$DESC|||$*" >> "$JOBFILE"
}

# ============================================================
# Helper: check if result CSV already exists (for SKIP_EXISTING)
# Suffix convention from train_downstream.py:
#   (default)      → results_{dataset}_{init}_frac{f}_s{s}.csv
#   _linprobe      → results_{dataset}_{init}_linprobe_frac{f}_s{s}.csv
#   _keepproj      → results_{dataset}_{init}_keepproj_frac{f}_s{s}.csv
#   _trainproj     → results_{dataset}_{init}_trainproj_frac{f}_s{s}.csv
# For distilled: init part is "distilled_{teacher_name}"
# ============================================================
result_exists() {
    local DATASET=$1 INIT=$2 TEACHER_NAME=$3 FRAC=$4 MODE_SUFFIX=$5
    # MODE_SUFFIX: "" (finetune), "_linprobe", "_keepproj", "_trainproj"
    if [ "$INIT" = "distilled" ]; then
        local CSV="$OUTPUT_DIR/results_${DATASET}_distilled_${TEACHER_NAME}${MODE_SUFFIX}_frac${FRAC}_s${SEED}.csv"
    else
        local CSV="$OUTPUT_DIR/results_${DATASET}_${INIT}${MODE_SUFFIX}_frac${FRAC}_s${SEED}.csv"
    fi
    [ -f "$CSV" ]
}

# ============================================================
# Part 1: Baselines (random + imagenet) for NEW datasets
# 36 runs: 2 inits × 3 datasets × 3 fractions × 2 modes
# ============================================================
echo "--- Generating: Baselines for new datasets ---"
COUNT_BASELINES=0
for INIT in random imagenet; do
    for DATASET in $NEW_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            # Fine-tune
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$INIT" "" "$FRAC" ""; then
                true  # skip
            else
                COUNT_BASELINES=$((COUNT_BASELINES + 1))
                add_job "baseline/${INIT}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init "$INIT" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "$INIT" "" "$FRAC" "_linprobe"; then
                true  # skip
            else
                COUNT_BASELINES=$((COUNT_BASELINES + 1))
                add_job "baseline/${INIT}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init "$INIT" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  Baselines: $COUNT_BASELINES jobs"

# ============================================================
# Part 2: COCO-distilled (no CKA) on NEW datasets
# 2 modes: fine-tune + linear probe
# ============================================================
echo "--- Generating: COCO-distilled (no CKA) on new datasets ---"
COUNT_COCO=0
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping COCO $TEACHER (no CKA)"
        continue
    fi
    for DATASET in $NEW_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            # Fine-tune
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TEACHER" "$FRAC" ""; then
                true
            else
                COUNT_COCO=$((COUNT_COCO + 1))
                add_job "coco/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TEACHER" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TEACHER" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_COCO=$((COUNT_COCO + 1))
                add_job "coco/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TEACHER" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  COCO (no CKA): $COUNT_COCO jobs"

# ============================================================
# Part 3: COCO CKA λ=0.1 on NEW datasets
# 4 modes: fine-tune, linear probe, frozen projector, trainable projector
# ============================================================
echo "--- Generating: COCO CKA λ=$LAMBDA_CKA on new datasets (4 modes) ---"
COUNT_COCO_CKA=0
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/coco_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping COCO $TEACHER CKA"
        continue
    fi
    for DATASET in $NEW_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            TNAME="${TEACHER}_cka_l${LAMBDA_CKA}"

            # Fine-tune (drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT_COCO_CKA=$((COUNT_COCO_CKA + 1))
                add_job "coco_cka/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe (frozen backbone, drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_COCO_CKA=$((COUNT_COCO_CKA + 1))
                add_job "coco_cka/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Frozen projector
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_keepproj"; then
                true
            else
                COUNT_COCO_CKA=$((COUNT_COCO_CKA + 1))
                add_job "coco_cka/${TEACHER}/${DATASET}/f${FRAC}/keepproj" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --keep_projector \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Trainable projector
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_trainproj"; then
                true
            else
                COUNT_COCO_CKA=$((COUNT_COCO_CKA + 1))
                add_job "coco_cka/${TEACHER}/${DATASET}/f${FRAC}/trainproj" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --keep_projector --train_projector \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  COCO CKA (4 modes): $COUNT_COCO_CKA jobs"

# ============================================================
# Part 4: ImageNet-distilled (no CKA) on ALL datasets
# 2 modes: fine-tune + linear probe
# ============================================================
echo "--- Generating: ImageNet-distilled (no CKA) on all datasets ---"
COUNT_IN=0
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/imagenet_${TEACHER}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping ImageNet $TEACHER (no CKA)"
        continue
    fi
    for DATASET in $ALL_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            TNAME="imagenet_${TEACHER}"
            # Fine-tune
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT_IN=$((COUNT_IN + 1))
                add_job "imagenet/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_IN=$((COUNT_IN + 1))
                add_job "imagenet/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  ImageNet (no CKA): $COUNT_IN jobs"

# ============================================================
# Part 5: ImageNet CKA λ=0.1 on ALL datasets
# 4 modes: fine-tune, linear probe, frozen projector, trainable projector
# This is the key experiment — CKA alignment lives mostly in the projector,
# so projector modes test whether preserving it helps downstream.
# ============================================================
echo "--- Generating: ImageNet CKA λ=$LAMBDA_CKA on all datasets (4 modes) ---"
COUNT_IN_CKA=0
for TEACHER in $TEACHERS; do
    CKPT="$OUTPUT_DIR/imagenet_${TEACHER}_cka_l${LAMBDA_CKA}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: $CKPT not found, skipping ImageNet $TEACHER CKA"
        continue
    fi
    for DATASET in $ALL_DATASETS; do
        for FRAC in $ALL_FRACTIONS; do
            TNAME="imagenet_${TEACHER}_cka_l${LAMBDA_CKA}"

            # Fine-tune (drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" ""; then
                true
            else
                COUNT_IN_CKA=$((COUNT_IN_CKA + 1))
                add_job "imagenet_cka/${TEACHER}/${DATASET}/f${FRAC}/finetune" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Linear probe (frozen backbone, drop projector)
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_linprobe"; then
                true
            else
                COUNT_IN_CKA=$((COUNT_IN_CKA + 1))
                add_job "imagenet_cka/${TEACHER}/${DATASET}/f${FRAC}/linprobe" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --freeze_backbone --lr 0.1 \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Frozen projector
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_keepproj"; then
                true
            else
                COUNT_IN_CKA=$((COUNT_IN_CKA + 1))
                add_job "imagenet_cka/${TEACHER}/${DATASET}/f${FRAC}/keepproj" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --keep_projector \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi

            # Trainable projector
            if [ "$SKIP_EXISTING" = "1" ] && result_exists "$DATASET" "distilled" "$TNAME" "$FRAC" "_trainproj"; then
                true
            else
                COUNT_IN_CKA=$((COUNT_IN_CKA + 1))
                add_job "imagenet_cka/${TEACHER}/${DATASET}/f${FRAC}/trainproj" \
                    $PYTHON src/train_downstream.py \
                    --dataset "$DATASET" --data_root "$DATA_ROOT" \
                    --init distilled --checkpoint "$CKPT" \
                    --teacher_name "$TNAME" --label_fraction "$FRAC" \
                    --keep_projector --train_projector \
                    --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED \
                    --output_dir "$OUTPUT_DIR" --no_wandb
            fi
        done
    done
done
echo "  ImageNet CKA (4 modes): $COUNT_IN_CKA jobs"

# ============================================================
# Execute all jobs
# ============================================================
TOTAL=$(wc -l < "$JOBFILE")
echo ""
echo "=============================================="
echo "Total jobs to run: $TOTAL"
echo "Parallelism: $MAX_PARALLEL concurrent"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No jobs to run. All results may already exist (SKIP_EXISTING=$SKIP_EXISTING)."
    rm -f "$JOBFILE"
    exit 0
fi

# Run jobs with xargs parallelism
# Each line: "description|||command args..."
run_single_job() {
    local LINE="$1"
    local DESC="${LINE%%|||*}"
    local CMD="${LINE#*|||}"
    local LOG_NAME=$(echo "$DESC" | tr '/' '_')
    local LOG="$OUTPUT_DIR/log_exp7_${LOG_NAME}.txt"

    echo "[START] $DESC"
    if eval $CMD > "$LOG" 2>&1; then
        echo "[DONE]  $DESC"
    else
        echo "[FAIL]  $DESC (see $LOG)"
        echo "$DESC" >> "$FAIL_LOG"
    fi
}
export -f run_single_job
export OUTPUT_DIR FAIL_LOG

cat "$JOBFILE" | xargs -P $MAX_PARALLEL -I {} bash -c 'run_single_job "$@"' _ {}

# ============================================================
# Summary
# ============================================================
FAIL_COUNT=0
if [ -f "$FAIL_LOG" ]; then
    FAIL_COUNT=$(wc -l < "$FAIL_LOG")
fi

echo ""
echo "===== Experiment 7: Downstream Complete ====="
echo "Total jobs: $TOTAL"
echo "Failed: $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    cat "$FAIL_LOG"
    echo ""
    echo "Check logs: $OUTPUT_DIR/log_exp7_*.txt"
fi
echo ""
echo "Results saved to $OUTPUT_DIR/results_*.csv"
echo ""
echo "Key comparisons to check:"
echo "  1. CKA projector modes: does frozen/trainable projector beat drop projector?"
echo "  2. VOC 100% linear probe: ImageNet CKA vs ImageNet pretrained (gap closure?)"
echo "  3. ImageNet-distilled vs COCO-distilled across all datasets"

rm -f "$JOBFILE"
