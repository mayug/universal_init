# Three Experiments Implementation Summary

This document summarizes the implementation of three minimal next experiments to isolate the value of different components in the universal initialization project.

## Implementation Status: ✅ Complete

All three experiments have been fully implemented and are ready to run.

---

## Experiment 1: Teacher Oracle Probes

**Goal:** Establish performance ceiling by training linear probes directly on frozen ImageBind embeddings.

### New Files Created

1. **`src/train_teacher_probe.py`** (344 lines)
   - Loads frozen ImageBind teacher model
   - Extracts embeddings once for train/val (memory cached)
   - Trains linear probe: `Linear(1024 → num_classes)` with SGD
   - Saves results in same CSV format as downstream experiments

2. **`scripts/experiment1_teacher_oracle.sh`**
   - Runs on Pets and EuroSAT at 1% and 100% labels
   - 4 total runs, ~2-4 hours estimated

### Usage

```bash
# Single run
python src/train_teacher_probe.py \
    --dataset pets \
    --label_fraction 0.01 \
    --epochs 50 \
    --no_wandb

# Full experiment
DATA_ROOT=./data ./scripts/experiment1_teacher_oracle.sh
```

### Expected Output

- `checkpoints/results_pets_teacher_oracle_frac0.01_s42.csv`
- `checkpoints/results_pets_teacher_oracle_frac1.0_s42.csv`
- `checkpoints/results_eurosat_teacher_oracle_frac0.01_s42.csv`
- `checkpoints/results_eurosat_teacher_oracle_frac1.0_s42.csv`

### Key Insight

If `distilled ≈ teacher_oracle`, distillation is highly effective.
If `distilled << teacher_oracle`, there's room for improvement in student capacity or training.

---

## Experiment 2: Keep vs Drop Projector Ablation

**Goal:** Compare downstream performance with frozen COCO-distilled projector vs without projector.

### Architecture Comparison

**Baseline (drop projector):**
```
Backbone (fine-tuned, from COCO) → Classifier (440 → num_classes)
```

**Ablation (keep frozen projector):**
```
Backbone (fine-tuned, from COCO) → Projector (FROZEN, 440→1024) → Classifier (1024 → num_classes)
```

### Modified Files

1. **`src/models/student.py`**
   - Added `keep_projector` parameter to `__init__()` and `for_downstream()`
   - If `keep_projector=True`:
     - Loads projector from checkpoint: `Linear(440 → 1024)`
     - Freezes projector weights (no gradients)
     - Attaches classifier to projector output: `Linear(1024 → num_classes)`
     - Mode: `"classifier_with_projector"`
   - Updated `load_distilled_weights()` to optionally load and freeze projector
   - Updated `forward()` to handle frozen projector path

2. **`src/train_downstream.py`**
   - Added `--keep_projector` flag
   - Pass to model creation
   - Results CSV includes `keep_projector` column
   - Filename includes `_keepproj` suffix when enabled

3. **`scripts/experiment2_projector_ablation.sh`**
   - Runs on Pets and EuroSAT at 1% and 100% with `--keep_projector`
   - 4 NEW runs (baseline results already exist from previous experiments)

### Usage

```bash
# Keep projector (ablation)
python src/train_downstream.py \
    --dataset pets \
    --init distilled \
    --checkpoint ./checkpoints/coco_distilled_best.pth \
    --label_fraction 0.01 \
    --keep_projector

# Full experiment
./scripts/experiment2_projector_ablation.sh
```

### Expected Output

**New files (keep_projector=True):**
- `checkpoints/results_pets_distilled_keepproj_frac0.01_s42.csv`
- `checkpoints/results_pets_distilled_keepproj_frac1.0_s42.csv`
- `checkpoints/results_eurosat_distilled_keepproj_frac0.01_s42.csv`
- `checkpoints/results_eurosat_distilled_keepproj_frac1.0_s42.csv`

**Existing baseline files (keep_projector=False):**
- `checkpoints/results_pets_distilled_frac0.01_s42.csv`
- `checkpoints/results_pets_distilled_frac1.0_s42.csv`
- `checkpoints/results_eurosat_distilled_frac0.01_s42.csv`
- `checkpoints/results_eurosat_distilled_frac1.0_s42.csv`

### Key Insight

Tests if frozen projector provides better feature space for downstream tasks.
Trade-off: Higher-dim features (1024) vs lower-dim (440), fewer trainable params with frozen projector.

---

## Experiment 3: Distill on ImageNet Train

**Goal:** Compare ImageNet-distilled vs COCO-distilled checkpoints on downstream tasks.

### Modified Files

1. **`src/data/distill_datasets.py`**
   - Added `ImageNetDataset` class (similar to `COCOImagesDataset`)
   - Uses `ImageFolder` to handle class subdirectories
   - Returns images only (no labels for distillation)
   - Updated `get_distill_dataloader()` to handle `"imagenet"`
   - Updated `get_distill_dataloaders_with_val()` to handle `"imagenet"`

2. **`src/train_distill.py`**
   - Updated dataset choices: `["imagenette", "coco", "imagenet"]`

3. **`scripts/experiment3_imagenet_distill.sh`**
   - Runs distillation on ImageNet train (50 epochs recommended)
   - ~6-8 hours estimated

4. **`scripts/experiment3_downstream.sh`**
   - Compares ImageNet-distilled vs COCO-distilled on downstream tasks
   - Runs Pets and EuroSAT at 1%, 10%, 100% labels
   - Results saved with `_imagenet` and `_coco` suffixes

### Expected Directory Structure

```
DATA_ROOT/imagenet/
├── train/
│   ├── n01440764/  (class folders with images)
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### Usage

```bash
# Step 1: Distill on ImageNet
DATA_ROOT=./data ./scripts/experiment3_imagenet_distill.sh

# Step 2: Compare downstream performance
DATA_ROOT=./data ./scripts/experiment3_downstream.sh
```

### Expected Output

**Distillation checkpoint:**
- `checkpoints/imagenet_distilled_best.pth`

**Downstream results (ImageNet-distilled):**
- `checkpoints/results_pets_distilled_imagenet_frac0.01_s42.csv`
- `checkpoints/results_pets_distilled_imagenet_frac0.1_s42.csv`
- `checkpoints/results_pets_distilled_imagenet_frac1.0_s42.csv`
- `checkpoints/results_eurosat_distilled_imagenet_frac0.01_s42.csv`
- `checkpoints/results_eurosat_distilled_imagenet_frac0.1_s42.csv`
- `checkpoints/results_eurosat_distilled_imagenet_frac1.0_s42.csv`

**Comparison baselines (COCO-distilled):**
- `checkpoints/results_*_distilled_coco_*.csv`

### Key Insight

- ImageNet: More images (1.28M vs 118K), natural images
- COCO: Diverse scenes and objects
- Which transfers better to Pets (fine-grained) and EuroSAT (domain shift)?

---

## Execution Order

1. **Experiments 1 & 2 First** (quick, use existing COCO checkpoint)
   - Run in parallel if desired
   - Combined time: ~6-10 hours

2. **Experiment 3 Last** (requires new ImageNet distillation)
   - Distillation: ~6-8 hours
   - Downstream: ~4-6 hours
   - Total: ~10-14 hours

---

## Verification Commands

### Check All Scripts Are Executable

```bash
ls -lh scripts/experiment*.sh
# All should have 'x' permission
```

### Verify COCO Checkpoint Exists (for Experiments 1 & 2)

```bash
ls -lh checkpoints/coco_distilled_best.pth
# Should exist from previous experiments
```

### Verify ImageNet Directory (for Experiment 3)

```bash
ls -d $DATA_ROOT/imagenet/train/n*/  | head -5
# Should show class directories
```

### Run Analysis

```bash
python src/evaluate.py --results_dir ./checkpoints
# Generates summary tables and plots comparing all experiments
```

---

## Files Summary

### New Files (5)

1. `src/train_teacher_probe.py` - Teacher oracle probe training
2. `scripts/experiment1_teacher_oracle.sh` - Run teacher probes
3. `scripts/experiment2_projector_ablation.sh` - Projector ablation
4. `scripts/experiment3_imagenet_distill.sh` - ImageNet distillation
5. `scripts/experiment3_downstream.sh` - Compare distillation sources

### Modified Files (4)

1. `src/data/distill_datasets.py` - Added `ImageNetDataset` class
2. `src/train_distill.py` - Added "imagenet" to dataset choices
3. `src/models/student.py` - Added `keep_projector` functionality
4. `src/train_downstream.py` - Added `--keep_projector` flag

### Documentation (1)

1. `EXPERIMENTS_IMPLEMENTATION.md` - This file

---

## Expected Insights

**Experiment 1:** Teacher oracle accuracy sets upper bound
- Quantifies how much performance is lost in distillation

**Experiment 2:** Frozen projector utility during fine-tuning
- Tests if projector provides better feature space for downstream tasks
- Identifies if projector should be kept or dropped

**Experiment 3:** ImageNet vs COCO distillation source
- Identifies which pretraining dataset transfers better
- May inform future distillation strategy

---

## Troubleshooting

### Issue: COCO checkpoint not found

```bash
# Run COCO distillation first
DATA_ROOT=./data ./scripts/phase1_distill.sh
```

### Issue: ImageNet directory not found

Ensure ImageNet is organized as:
```
$DATA_ROOT/imagenet/train/n01440764/*.JPEG
```

### Issue: Out of memory during teacher probe

Reduce batch size in embedding extraction:
```python
# In train_teacher_probe.py, line ~80
batch_size = 32  # Reduce from 64
```

### Issue: ImageNet distillation too slow

Reduce epochs or use smaller subset:
```bash
# In experiment3_imagenet_distill.sh
EPOCHS=30  # Reduce from 50
```

---

## Notes

- All experiments use `--no_wandb` by default to avoid requiring W&B login
- Remove `--no_wandb` flag if you want to log to Weights & Biases
- All scripts use `set -e` to exit on error
- Random seed is fixed at 42 for reproducibility
- Results are saved as CSV files for easy analysis
