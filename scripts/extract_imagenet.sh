#!/bin/bash
# Extract ImageNet dataset from tar files

set -e

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

DATA_DIR="${DATA_ROOT:-$PROJECT_ROOT/data/imagenet}"
# Convert to absolute path
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
TRAIN_TAR="$DATA_DIR/ILSVRC2012_img_train.tar"
VAL_TAR="$DATA_DIR/ILSVRC2012_img_val.tar"

echo "=============================================="
echo "Extracting ImageNet dataset"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# Extract validation set (simple - just images)
if [ ! -d "$DATA_DIR/val/n01440764" ]; then
    echo "Extracting validation set..."
    mkdir -p "$DATA_DIR/val"
    tar -xf "$VAL_TAR" -C "$DATA_DIR/val"

    # Organize val images into class folders using the devkit
    echo "Organizing validation images into class folders..."
    cd "$DATA_DIR/val"

    # Download the script to organize val images if needed
    if [ ! -f "valprep.sh" ]; then
        wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    fi
    bash valprep.sh
    cd -
    echo "Validation set extracted!"
else
    echo "Validation set already extracted, skipping..."
fi

# Extract training set (nested tars - one per class)
if [ ! -d "$DATA_DIR/train/n01440764" ]; then
    echo "Extracting training set (this takes a while)..."
    mkdir -p "$DATA_DIR/train"
    cd "$DATA_DIR/train"

    # Extract main tar
    tar -xf "$TRAIN_TAR"

    # Extract each class tar
    echo "Extracting class tars..."
    for f in *.tar; do
        if [ -f "$f" ]; then
            dir="${f%.tar}"
            mkdir -p "$dir"
            tar -xf "$f" -C "$dir"
            rm "$f"
        fi
    done

    cd -
    echo "Training set extracted!"
else
    echo "Training set already extracted, skipping..."
fi

echo ""
echo "=============================================="
echo "ImageNet extraction complete!"
echo "Train images: $(find $DATA_DIR/train -name '*.JPEG' 2>/dev/null | wc -l)"
echo "Val images: $(find $DATA_DIR/val -name '*.JPEG' 2>/dev/null | wc -l)"
echo "=============================================="
