#!/bin/bash
# Experiment 10: Audio Downstream Evaluation
#
# Evaluates 4 conditions on ESC-50 (5-fold CV) and UrbanSound8K (10-fold CV):
#   A. Random init (floor)
#   B. CLIP text distilled (platonic test)
#   C. AudioSet pretrained mn10_as (ceiling)
#   E. Sentence-BERT distilled (unimodal text baseline)
#
# Each condition is tested at 3 label fractions (1%, 10%, 100%)
# in 2 modes (fine-tune, linear probe).
#
# Total runs per dataset:
#   4 conditions × 3 fractions × 2 modes × N folds = many runs
#
# To do a quick sanity check with single fold first:
#   FOLD=1 ./scripts/experiment10_audio_downstream.sh
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment10_audio_downstream.sh
#   FOLD=1 DATA_ROOT=./data ./scripts/experiment10_audio_downstream.sh  # single fold

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
LR="${LR:-0.01}"
FOLD="${FOLD:-}"  # Empty = all folds

# Checkpoint paths (from distillation phase)
CLIP_CKPT="${CLIP_CKPT:-$OUTPUT_DIR/audiocaps_clip_text_distilled_best.pth}"
SBERT_CKPT="${SBERT_CKPT:-$OUTPUT_DIR/audiocaps_sentence_bert_distilled_best.pth}"

echo "===== Experiment 10: Audio Downstream Evaluation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "CLIP checkpoint: $CLIP_CKPT"
echo "SBERT checkpoint: $SBERT_CKPT"
if [ -n "$FOLD" ]; then
    echo "Running single fold: $FOLD"
    FOLD_ARG="--fold $FOLD"
else
    echo "Running ALL folds (full CV)"
    FOLD_ARG=""
fi
echo ""

mkdir -p "$OUTPUT_DIR"

DATASETS="esc50 urbansound8k"
FRACTIONS="1.0 0.1 0.01"

# Helper function
run_downstream() {
    local DATASET=$1
    local INIT=$2
    local FRAC=$3
    local FREEZE=$4  # "" or "--freeze_backbone"
    local CKPT_ARG=$5  # "" or "--checkpoint path"
    local TEACHER_ARG=$6  # "" or "--teacher_name name"

    local MODE="finetune"
    if [ -n "$FREEZE" ]; then
        MODE="linprobe"
    fi

    echo ""
    echo "  >> $DATASET / $INIT / frac=$FRAC / $MODE"

    $PYTHON src/train_audio_downstream.py \
        --dataset "$DATASET" \
        --data_root "$DATA_ROOT" \
        --label_fraction $FRAC \
        --init "$INIT" \
        $CKPT_ARG \
        $TEACHER_ARG \
        $FREEZE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED \
        --no_wandb \
        $FOLD_ARG
}

RUN=0
TOTAL_RUNS=0

# Count total runs
for DATASET in $DATASETS; do
    for FRAC in $FRACTIONS; do
        for MODE in finetune linprobe; do
            TOTAL_RUNS=$((TOTAL_RUNS + 4))  # 4 conditions
        done
    done
done

echo "Total experiment configurations: $TOTAL_RUNS"
echo ""

for DATASET in $DATASETS; do
    echo "=============================================="
    echo "Dataset: $DATASET"
    echo "=============================================="

    for FRAC in $FRACTIONS; do
        echo ""
        echo "--- Label fraction: $FRAC ---"

        # ---- Fine-tuning mode ----
        echo ""
        echo "  [Fine-tuning]"

        # A: Random init
        RUN=$((RUN + 1))
        echo "  ($RUN/$TOTAL_RUNS)"
        run_downstream "$DATASET" "random" "$FRAC" "" "" ""

        # B: CLIP text distilled
        if [ -f "$CLIP_CKPT" ]; then
            RUN=$((RUN + 1))
            echo "  ($RUN/$TOTAL_RUNS)"
            run_downstream "$DATASET" "distilled" "$FRAC" "" \
                "--checkpoint $CLIP_CKPT" "--teacher_name clip_text"
        else
            echo "  SKIPPING CLIP distilled (checkpoint not found: $CLIP_CKPT)"
            RUN=$((RUN + 1))
        fi

        # C: AudioSet pretrained
        RUN=$((RUN + 1))
        echo "  ($RUN/$TOTAL_RUNS)"
        run_downstream "$DATASET" "audioset_pretrained" "$FRAC" "" "" ""

        # E: Sentence-BERT distilled
        if [ -f "$SBERT_CKPT" ]; then
            RUN=$((RUN + 1))
            echo "  ($RUN/$TOTAL_RUNS)"
            run_downstream "$DATASET" "distilled" "$FRAC" "" \
                "--checkpoint $SBERT_CKPT" "--teacher_name sentence_bert"
        else
            echo "  SKIPPING SBERT distilled (checkpoint not found: $SBERT_CKPT)"
            RUN=$((RUN + 1))
        fi

        # ---- Linear probe mode ----
        echo ""
        echo "  [Linear probe]"

        # A: Random init
        RUN=$((RUN + 1))
        echo "  ($RUN/$TOTAL_RUNS)"
        run_downstream "$DATASET" "random" "$FRAC" "--freeze_backbone" "" ""

        # B: CLIP text distilled
        if [ -f "$CLIP_CKPT" ]; then
            RUN=$((RUN + 1))
            echo "  ($RUN/$TOTAL_RUNS)"
            run_downstream "$DATASET" "distilled" "$FRAC" "--freeze_backbone" \
                "--checkpoint $CLIP_CKPT" "--teacher_name clip_text"
        else
            echo "  SKIPPING CLIP distilled linear probe"
            RUN=$((RUN + 1))
        fi

        # C: AudioSet pretrained
        RUN=$((RUN + 1))
        echo "  ($RUN/$TOTAL_RUNS)"
        run_downstream "$DATASET" "audioset_pretrained" "$FRAC" "--freeze_backbone" "" ""

        # E: Sentence-BERT distilled
        if [ -f "$SBERT_CKPT" ]; then
            RUN=$((RUN + 1))
            echo "  ($RUN/$TOTAL_RUNS)"
            run_downstream "$DATASET" "distilled" "$FRAC" "--freeze_backbone" \
                "--checkpoint $SBERT_CKPT" "--teacher_name sentence_bert"
        else
            echo "  SKIPPING SBERT distilled linear probe"
            RUN=$((RUN + 1))
        fi
    done
done

echo ""
echo "===== Experiment 10: Downstream Evaluation Complete ====="
echo ""
echo "Results saved to: $OUTPUT_DIR/results_audio_*.csv"
echo ""
echo "Key comparisons:"
echo "  B vs E: CLIP text vs Sentence-BERT (platonic test)"
echo "  B vs A: Cross-modal distillation vs random init"
echo "  B vs C: Gap closure relative to AudioSet pretrained ceiling"
