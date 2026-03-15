#!/bin/bash
# Experiment 10: Cross-Modal Audio Distillation via CLIP Text Geometry
#
# Distills text teacher embeddings (from audio captions) into an audio student
# (MobileNetV3). The teacher has never heard audio — testing whether its
# geometric structure transfers cross-modally.
#
# 2 distillation runs:
#   1. CLIP text (ViT-L/14, 768-dim) — cross-modal platonic geometry
#   2. Sentence-BERT (all-mpnet-base-v2, 768-dim) — unimodal text baseline
#
# Both produce 768-dim teacher embeddings → same Linear(960,768) projector.
#
# Distillation data: AudioCaps (~46K audio-caption pairs, 10s clips)
#
# Estimated time: ~40-60 min per run (46K samples, BS=256, 20 epochs).
#
# Usage:
#   DATA_ROOT=./data ./scripts/experiment10_audio_distill.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
LOSS="${LOSS:-embedding}"

echo "===== Experiment 10: Cross-Modal Audio Distillation ====="
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "LOSS: $LOSS"
echo ""

# Verify AudioCaps exists
AUDIOCAPS_DIR="$DATA_ROOT/audiocaps"
if [ ! -d "$AUDIOCAPS_DIR" ]; then
    echo "ERROR: AudioCaps directory not found at $AUDIOCAPS_DIR"
    echo "Expected structure:"
    echo "  $AUDIOCAPS_DIR/train/       (audio files)"
    echo "  $AUDIOCAPS_DIR/val/         (audio files)"
    echo "  $AUDIOCAPS_DIR/dataset/     (metadata CSVs)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Run 1: CLIP text teacher
# ============================================================
echo ""
echo "=============================================="
echo "Run 1/2: CLIP text teacher (ViT-L/14, 768-dim)"
echo "=============================================="
$PYTHON src/train_audio_distill.py \
    --teacher clip_text \
    --dataset audiocaps \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
    --warmup_epochs 2 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 5 \
    --seed $SEED \
    --amp \
    --no_wandb

echo ""
echo "Run 1 complete: $OUTPUT_DIR/audiocaps_clip_text_distilled_best.pth"

# ============================================================
# Run 2: Sentence-BERT teacher
# ============================================================
echo ""
echo "=============================================="
echo "Run 2/2: Sentence-BERT teacher (all-mpnet-base-v2, 768-dim)"
echo "=============================================="
$PYTHON src/train_audio_distill.py \
    --teacher sentence_bert \
    --dataset audiocaps \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
    --warmup_epochs 2 \
    --output_dir "$OUTPUT_DIR" \
    --save_every 5 \
    --seed $SEED \
    --amp \
    --no_wandb

echo ""
echo "Run 2 complete: $OUTPUT_DIR/audiocaps_sentence_bert_distilled_best.pth"

echo ""
echo "===== Experiment 10: Distillation Complete ====="
echo ""
echo "Checkpoints:"
echo "  $OUTPUT_DIR/audiocaps_clip_text_distilled_best.pth     (CLIP text)"
echo "  $OUTPUT_DIR/audiocaps_sentence_bert_distilled_best.pth (Sentence-BERT)"
echo ""
echo "Next step: Run downstream evaluation"
echo "  ./scripts/experiment10_audio_downstream.sh"
