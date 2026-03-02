#!/bin/bash
set -e

PYTHON=/home/ubuntu/projects/universal_init/venv/bin/python
DATA_ROOT=./data
EPOCHS=50
BATCH=64
COMMON="--data_root $DATA_ROOT --epochs $EPOCHS --batch_size $BATCH --no_wandb --device cuda"

echo "========================================"
echo "VOC Label Fraction Experiments (1%, 10%)"
echo "========================================"

# Focus on the most informative configs:
# - Baselines: random, imagenet
# - Non-CKA distilled: supervised, clip_768, clip_512
# - CKA λ=0.1 distilled: supervised, clip_768, clip_512
# Each: fine-tune + linear probe × 2 fractions = 32 runs

for FRAC in 0.01 0.1; do
    echo ""
    echo "========================================"
    echo "Label fraction: ${FRAC}"
    echo "========================================"

    # --- Baselines ---
    echo -e "\n>>> Baseline: random — fine-tune (frac=${FRAC})"
    $PYTHON src/train_downstream.py --dataset voc --init random \
        --label_fraction $FRAC $COMMON

    echo -e "\n>>> Baseline: random — linear probe (frac=${FRAC})"
    $PYTHON src/train_downstream.py --dataset voc --init random \
        --freeze_backbone --lr 0.1 --label_fraction $FRAC $COMMON

    echo -e "\n>>> Baseline: imagenet — fine-tune (frac=${FRAC})"
    $PYTHON src/train_downstream.py --dataset voc --init imagenet \
        --label_fraction $FRAC $COMMON

    echo -e "\n>>> Baseline: imagenet — linear probe (frac=${FRAC})"
    $PYTHON src/train_downstream.py --dataset voc --init imagenet \
        --freeze_backbone --lr 0.1 --label_fraction $FRAC $COMMON

    # --- Non-CKA COCO distilled ---
    for TEACHER in supervised clip_768 clip_512; do
        CKPT="checkpoints/coco_${TEACHER}_distilled_best.pth"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP (not found): $CKPT"
            continue
        fi

        echo -e "\n>>> Non-CKA distilled: teacher=${TEACHER} — fine-tune (frac=${FRAC})"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}" \
            --label_fraction $FRAC $COMMON

        echo -e "\n>>> Non-CKA distilled: teacher=${TEACHER} — linear probe (frac=${FRAC})"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}" \
            --freeze_backbone --lr 0.1 --label_fraction $FRAC $COMMON
    done

    # --- CKA λ=0.1 distilled ---
    for TEACHER in supervised clip_768 clip_512; do
        CKPT="checkpoints/coco_${TEACHER}_cka_l0.1_distilled_best.pth"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP (not found): $CKPT"
            continue
        fi

        echo -e "\n>>> CKA λ=0.1 distilled: teacher=${TEACHER} — fine-tune (frac=${FRAC})"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}_cka_l0.1" \
            --label_fraction $FRAC $COMMON

        echo -e "\n>>> CKA λ=0.1 distilled: teacher=${TEACHER} — linear probe (frac=${FRAC})"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}_cka_l0.1" \
            --freeze_backbone --lr 0.1 --label_fraction $FRAC $COMMON
    done
done

echo -e "\n========================================"
echo "All VOC label fraction runs complete!"
echo "========================================"
