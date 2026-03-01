#!/bin/bash
# Experiment 5: CKA Distillation Loss
# 9 distillation runs: 3 teachers x 3 lambda_cka values
# Tests whether CKA loss improves structural alignment vs relational loss.
# Parallelized 3-at-a-time (one per lambda, grouped by teacher).

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

TEACHERS="supervised clip_768 clip_512"
LAMBDAS="0.1 0.5 1.0"

echo "===== Experiment 5: CKA Distillation Loss ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "TEACHERS: $TEACHERS"
echo "LAMBDAS: $LAMBDAS"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo ""

# Verify COCO exists
COCO_DIR="$DATA_ROOT/coco/train2017"
if [ ! -d "$COCO_DIR" ]; then
    echo "ERROR: COCO train2017 not found at $COCO_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_distill() {
    local TEACHER=$1
    local LAMBDA=$2
    local LOG="$OUTPUT_DIR/log_${TEACHER}_cka_l${LAMBDA}.txt"

    echo "Starting: teacher=$TEACHER lambda_cka=$LAMBDA (log: $LOG)"
    $PYTHON src/train_distill.py \
        --teacher "$TEACHER" \
        --dataset coco \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss cka_combined \
        --lambda_cka "$LAMBDA" \
        --warmup_epochs 5 \
        --output_dir "$OUTPUT_DIR" \
        --save_every 10 \
        --val_every 1 \
        --val_fraction 0.1 \
        --probe_every 0 \
        --seed $SEED \
        --amp \
        --no_wandb \
        > "$LOG" 2>&1

    echo "Finished: teacher=$TEACHER lambda_cka=$LAMBDA"
}

# Run grouped by teacher (3 concurrent lambdas per teacher)
TOTAL=9
RUN=0
for TEACHER in $TEACHERS; do
    echo ""
    echo "===== Teacher: $TEACHER ====="
    for LAMBDA in $LAMBDAS; do
        RUN=$((RUN + 1))
        echo "$RUN/$TOTAL: $TEACHER / lambda_cka=$LAMBDA"
        run_distill "$TEACHER" "$LAMBDA" &

        # Limit parallelism
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 10
        done
    done
    wait  # finish all for this teacher before next
done

echo ""
echo "===== Experiment 5: CKA Distillation Complete ====="
echo "Checkpoints:"
for TEACHER in $TEACHERS; do
    for LAMBDA in $LAMBDAS; do
        echo "  $OUTPUT_DIR/coco_${TEACHER}_cka_l${LAMBDA}_distilled_best.pth"
    done
done
echo ""
echo "Next steps:"
echo "  1. Analyze CKA: python src/analyze_cka.py --dataset pets --include_teachers --include_baselines"
echo "  2. Pick best lambda per teacher from val/cka_projected metrics"
echo "  3. Run downstream: LAMBDA=X ./scripts/experiment5_downstream.sh"
