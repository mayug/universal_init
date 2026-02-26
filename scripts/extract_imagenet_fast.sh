#!/bin/bash
# Fast parallel ImageNet extraction
set -e

DATA_DIR="${1:-./data/imagenet}"
# Convert to absolute path
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
TRAIN_TAR="$DATA_DIR/ILSVRC2012_img_train.tar"
TRAIN_DIR="$DATA_DIR/train"
NUM_WORKERS="${2:-8}"

echo "=============================================="
echo "Fast ImageNet Extraction (parallel)"
echo "=============================================="
echo "Source: $TRAIN_TAR"
echo "Target: $TRAIN_DIR"
echo "Workers: $NUM_WORKERS"
echo "=============================================="

# Check tar exists
if [ ! -f "$TRAIN_TAR" ]; then
    echo "ERROR: $TRAIN_TAR not found"
    exit 1
fi

mkdir -p "$TRAIN_DIR"
cd "$TRAIN_DIR"

# Step 1: Extract main tar (get all class tars)
echo ""
echo "Step 1/2: Extracting main tar..."
tar -xf "$TRAIN_TAR"
echo "Main tar extracted. Found $(ls *.tar 2>/dev/null | wc -l) class tars."

# Step 2: Extract class tars in parallel
echo ""
echo "Step 2/2: Extracting class tars in parallel..."
ls *.tar | xargs -P $NUM_WORKERS -I {} bash -c '
    dir="${1%.tar}"
    mkdir -p "$dir"
    tar -xf "$1" -C "$dir"
    rm -f "$1"
    echo "Extracted: $dir"
' _ {}

echo ""
echo "=============================================="
echo "ImageNet extraction complete!"
echo "Train classes: $(ls -d */ 2>/dev/null | wc -l)"
echo "Train images: $(find . -name '*.JPEG' | wc -l)"
echo "=============================================="
