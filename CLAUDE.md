# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project tests whether initializing a small vision model (RegNetY-400MF) by distilling ImageBind vision embeddings yields more sample-efficient downstream learning compared to random or ImageNet-pretrained initialization. The hypothesis is that ImageBind's embedding space approximates a "universal/platonic" latent space that transfers efficiently across diverse tasks.

## Commands

### Environment Setup
```bash
pip install -e .
# ImageBind requires separate installation:
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind && pip install -e .
```

### Running Experiments

**Phase 0 (Sanity Check):**
```bash
DATA_ROOT=/path/to/data ./scripts/phase0_sanity.sh
```

**Phase 1 (Full Experiments):**
```bash
# Step 1: Distill on COCO
DATA_ROOT=/path/to/data ./scripts/phase1_distill.sh

# Step 2: Run all downstream experiments
DATA_ROOT=/path/to/data ./scripts/phase1_downstream.sh
```

**Individual Training Runs:**
```bash
# Distillation
python src/train_distill.py --dataset coco --epochs 100 --batch_size 256 --loss combined

# Downstream fine-tuning
python src/train_downstream.py --dataset pets --init distilled --checkpoint checkpoints/coco_distilled_best.pth --label_fraction 0.1
```

**Analyze Results:**
```bash
python src/evaluate.py --results_dir ./checkpoints --output_dir ./results
```

## Architecture

```
src/
├── models/
│   ├── teacher.py      # ImageBindTeacher: frozen ImageBind vision encoder (1024-dim output)
│   └── student.py      # StudentModel: RegNetY-400MF backbone (440-dim) + projector/classifier head
├── losses/
│   └── distillation.py # embedding_loss (cosine), relational_loss (similarity matrix MSE), combined_loss
├── data/
│   ├── distill_datasets.py    # Imagenette, COCO loaders (images only, no labels)
│   └── downstream_datasets.py # Pets, Flowers102, DTD, EuroSAT with stratified label subsampling
├── train_distill.py    # Distillation training: student matches teacher embeddings
├── train_downstream.py # Classification fine-tuning with three init modes
└── evaluate.py         # Generate summary tables, plots, win-rate analysis
```

## Key Design Decisions

- **Student architecture**: RegNetY-400MF chosen for fast iteration and available ImageNet weights for baseline comparison
- **Projector**: Linear(440→1024) mapping backbone features to ImageBind's embedding dimension; MLP variant available for ablation
- **Loss function**: Combined loss = cosine embedding loss + λ×relational loss (λ=0.5 default). Relational loss preserves within-batch similarity structure
- **Downstream protocol**: SGD with momentum for fine-tuning (standard for transfer learning), AdamW for distillation
- **Label fractions**: 1%, 10%, 100% with stratified sampling to maintain class balance
- **Metrics**: Best accuracy, AULC (Area Under Learning Curve), accuracy at epochs 5/10/20

## Data Requirements

Datasets should be organized under `DATA_ROOT`:
```
DATA_ROOT/
├── imagenette2/          # Download: wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
├── coco/train2017/       # COCO images from https://cocodataset.org
├── oxford-iiit-pet/      # Auto-downloaded by torchvision
├── flowers-102/          # Auto-downloaded by torchvision
├── dtd/                  # Auto-downloaded by torchvision
└── eurosat/              # Auto-downloaded by torchvision
```

## W&B Logging

All experiments log to Weights & Biases project `universal_init`. Key metrics:
- Distillation: `train/loss`, `train/cosine_sim_mean`
- Downstream: `train/acc`, `val/acc`, `val/best_acc`, `val/aulc`

Disable with `--no_wandb` flag.
