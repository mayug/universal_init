# Experiments Quick Start Guide

Three minimal experiments to isolate component value in universal initialization.

## ✅ Implementation Complete

All code implemented and syntax-verified. Ready to run.

---

## Quick Commands

### Experiment 1: Teacher Oracle Probes (4 runs, ~2-4 hours)
```bash
DATA_ROOT=./data ./scripts/experiment1_teacher_oracle.sh
```

**What it does:** Trains linear probes on frozen ImageBind embeddings (performance ceiling).

**Requirements:** ImageBind installed, downstream datasets (Pets, EuroSAT).

---

### Experiment 2: Projector Ablation (4 runs, ~3-4 hours)
```bash
./scripts/experiment2_projector_ablation.sh
```

**What it does:** Compares frozen projector (440→1024→classes) vs no projector (440→classes).

**Requirements:** COCO distilled checkpoint at `./checkpoints/coco_distilled_best.pth`.

---

### Experiment 3: ImageNet Distillation + Downstream (12 runs, ~10-14 hours)

**Step 1: Distill on ImageNet (6-8 hours)**
```bash
DATA_ROOT=./data ./scripts/experiment3_imagenet_distill.sh
```

**Step 2: Compare downstream (4-6 hours)**
```bash
DATA_ROOT=./data ./scripts/experiment3_downstream.sh
```

**What it does:** Compares ImageNet-distilled (1.28M images) vs COCO-distilled (118K images).

**Requirements:** ImageNet organized as `$DATA_ROOT/imagenet/train/n01440764/*.JPEG`.

---

## Individual Run Examples

### Teacher Oracle Probe
```bash
python src/train_teacher_probe.py \
    --dataset pets \
    --label_fraction 0.01 \
    --epochs 50 \
    --no_wandb
```

### Keep Frozen Projector
```bash
python src/train_downstream.py \
    --dataset eurosat \
    --init distilled \
    --checkpoint ./checkpoints/coco_distilled_best.pth \
    --label_fraction 1.0 \
    --keep_projector \
    --no_wandb
```

### ImageNet Distillation
```bash
python src/train_distill.py \
    --dataset imagenet \
    --epochs 50 \
    --batch_size 256 \
    --loss combined \
    --amp \
    --no_wandb
```

---

## Results Analysis

After running experiments:

```bash
python src/evaluate.py --results_dir ./checkpoints
```

This generates summary tables comparing:
- Teacher oracle vs distilled vs imagenet vs random
- Keep projector vs drop projector
- ImageNet-distilled vs COCO-distilled

---

## Expected Results Files

### Experiment 1
- `checkpoints/results_pets_teacher_oracle_frac0.01_s42.csv`
- `checkpoints/results_pets_teacher_oracle_frac1.0_s42.csv`
- `checkpoints/results_eurosat_teacher_oracle_frac0.01_s42.csv`
- `checkpoints/results_eurosat_teacher_oracle_frac1.0_s42.csv`

### Experiment 2
- `checkpoints/results_*_distilled_keepproj_*.csv` (new)
- `checkpoints/results_*_distilled_frac*.csv` (baseline, already exists)

### Experiment 3
- `checkpoints/imagenet_distilled_best.pth` (checkpoint)
- `checkpoints/results_*_distilled_imagenet_*.csv` (6 files)
- `checkpoints/results_*_distilled_coco_*.csv` (comparison baseline)

---

## Verification Checklist

Before running:

- [ ] ImageBind installed: `pip install git+https://github.com/facebookresearch/ImageBind.git`
- [ ] COCO checkpoint exists (Exp 1 & 2): `ls checkpoints/coco_distilled_best.pth`
- [ ] Downstream datasets available: `ls $DATA_ROOT/oxford-iiit-pet` or similar
- [ ] ImageNet available (Exp 3): `ls $DATA_ROOT/imagenet/train/n*/` shows class folders

After running:

- [ ] Results CSVs created in `checkpoints/`
- [ ] No errors in experiment output
- [ ] Run `python src/evaluate.py --results_dir ./checkpoints` for analysis

---

## Troubleshooting

**COCO checkpoint not found:**
```bash
DATA_ROOT=./data ./scripts/phase1_distill.sh
```

**ImageNet not found:**
Ensure structure is `$DATA_ROOT/imagenet/train/n01440764/*.JPEG`

**Out of memory:**
Reduce batch size in scripts or use `--amp` flag for mixed precision.

**Want to use W&B logging:**
Remove `--no_wandb` flags from scripts.

---

## Execution Time Estimates

| Experiment | Runs | Est. Time | Can Run in Parallel |
|------------|------|-----------|---------------------|
| 1: Teacher Oracle | 4 | 2-4 hours | Yes (with Exp 2) |
| 2: Projector Ablation | 4 | 3-4 hours | Yes (with Exp 1) |
| 3: ImageNet Distill | 1 | 6-8 hours | No (requires checkpoint) |
| 3: Downstream | 6 | 4-6 hours | After distillation |
| **Total** | **15** | **15-22 hours** | |

**Recommended order:**
1. Run Experiments 1 & 2 in parallel (use existing COCO checkpoint)
2. Run Experiment 3 distillation (produces ImageNet checkpoint)
3. Run Experiment 3 downstream (compares checkpoints)

---

## Key Questions Answered

**Experiment 1:** How close does distillation get to the teacher's performance ceiling?

**Experiment 2:** Should we keep the frozen projector during fine-tuning?

**Experiment 3:** Does ImageNet or COCO transfer better to downstream tasks?

---

## Full Documentation

See `EXPERIMENTS_IMPLEMENTATION.md` for:
- Detailed implementation notes
- Architecture diagrams
- File-by-file changes
- Expected insights
- Extended troubleshooting

---

## Quick Status Check

```bash
# Check all experiment scripts exist
ls -1 scripts/experiment*.sh

# Check all Python files compile
python3 -m py_compile src/train_teacher_probe.py
python3 -m py_compile src/models/student.py
python3 -m py_compile src/train_downstream.py
python3 -m py_compile src/data/distill_datasets.py
python3 -m py_compile src/train_distill.py
```

All should pass without errors. ✅
