#!/bin/bash
set -e

PYTHON=/home/ubuntu/projects/universal_init/venv/bin/python
DATA_ROOT=./data
EPOCHS=50
BATCH=64
COMMON="--data_root $DATA_ROOT --epochs $EPOCHS --batch_size $BATCH --no_wandb --device cuda"

echo "========================================"
echo "VOC Downstream Evaluation — Full Suite"
echo "========================================"

# --- Baselines ---
echo -e "\n>>> Baseline: random — fine-tune"
$PYTHON src/train_downstream.py --dataset voc --init random $COMMON

echo -e "\n>>> Baseline: random — linear probe"
$PYTHON src/train_downstream.py --dataset voc --init random --freeze_backbone --lr 0.1 $COMMON

echo -e "\n>>> Baseline: imagenet — fine-tune"
$PYTHON src/train_downstream.py --dataset voc --init imagenet $COMMON

echo -e "\n>>> Baseline: imagenet — linear probe"
$PYTHON src/train_downstream.py --dataset voc --init imagenet --freeze_backbone --lr 0.1 $COMMON

# --- CKA-distilled checkpoints ---
for TEACHER in supervised clip_768 clip_512; do
    for LAMBDA in 0.1 0.5 1.0; do
        CKPT="checkpoints/coco_${TEACHER}_cka_l${LAMBDA}_distilled_best.pth"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP (not found): $CKPT"
            continue
        fi

        echo -e "\n>>> CKA distilled: teacher=${TEACHER}, lambda=${LAMBDA} — fine-tune"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}_cka_l${LAMBDA}" $COMMON

        echo -e "\n>>> CKA distilled: teacher=${TEACHER}, lambda=${LAMBDA} — linear probe"
        $PYTHON src/train_downstream.py --dataset voc --init distilled \
            --checkpoint "$CKPT" --teacher_name "${TEACHER}_cka_l${LAMBDA}" \
            --freeze_backbone --lr 0.1 $COMMON
    done
done

# --- Non-CKA COCO distilled checkpoints ---
for TEACHER in supervised clip_768 clip_512; do
    CKPT="checkpoints/coco_${TEACHER}_distilled_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP (not found): $CKPT"
        continue
    fi

    echo -e "\n>>> COCO distilled (no CKA): teacher=${TEACHER} — fine-tune"
    $PYTHON src/train_downstream.py --dataset voc --init distilled \
        --checkpoint "$CKPT" --teacher_name "${TEACHER}" $COMMON

    echo -e "\n>>> COCO distilled (no CKA): teacher=${TEACHER} — linear probe"
    $PYTHON src/train_downstream.py --dataset voc --init distilled \
        --checkpoint "$CKPT" --teacher_name "${TEACHER}" \
        --freeze_backbone --lr 0.1 $COMMON
done

echo -e "\n========================================"
echo "All VOC downstream runs complete!"
echo "========================================"
