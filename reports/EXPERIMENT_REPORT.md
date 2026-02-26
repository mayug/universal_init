# Experiment Report: Universal Initialization via ImageBind Distillation

**Date:** January 26, 2026
**Status:** In Progress

---

## Executive Summary

This report documents findings from experiments testing whether initializing a small vision model (RegNetY-400MF) by distilling ImageBind embeddings yields more sample-efficient downstream learning compared to random or ImageNet-pretrained initialization.

### Key Findings

1. **ImageNet Pretrained Significantly Outperforms COCO-Distilled** across all tested conditions
2. **COCO Distillation Shows Poor Transfer on Fine-Grained Tasks** (Pets dataset)
3. **COCO Distillation Performs Better on Satellite Imagery** (EuroSAT dataset)
4. **Sample Efficiency Varies Dramatically by Task** and initialization method

---

## Experiments Conducted

### Completed Experiments

1. **Baseline Comparisons** (34 runs)
   - Three initialization methods: Random, ImageNet, COCO-Distilled
   - Three datasets: Pets, EuroSAT, Imagenette
   - Multiple label fractions: 1%, 10%, 100%
   - Multiple seeds for robustness

### In Progress

2. **Experiment 2: Projector Ablation** (Running)
   - Testing whether keeping frozen COCO-distilled projector improves downstream performance
   - Architecture: Backbone → Frozen Projector (440→1024) → Classifier
   - Status: 1/4 runs complete (Pets 1% labels)

### Planned

3. **Experiment 1: Teacher Oracle Probes**
   - Requires ImageBind installation
   - Would establish performance ceiling with frozen teacher embeddings

4. **Experiment 3: ImageNet Distillation**
   - Requires ImageNet dataset (~147GB)
   - Would compare ImageNet-distilled vs COCO-distilled

---

## Detailed Results

### 1. Pets Dataset (Fine-Grained Classification, 37 classes)

**Challenge:** Fine-grained dog and cat breed classification with limited data

| Init Method | 1% Labels (37 samples) | 10% Labels (370 samples) | 100% Labels (3,680 samples) |
|-------------|------------------------|--------------------------|------------------------------|
| **COCO-Distilled** | 5.78% ± 0.81% | 17.31% ± 2.32% | 54.16% |
| **ImageNet** | **42.75% ± 2.85%** | **82.01% ± 0.05%** | **91.16% ± 0.16%** |
| **Random** | 3.81% ± 0.35% | 6.68% ± 0.38% | 29.41% ± 0.52% |

**Observations:**
- ImageNet initialization provides **7.4× better accuracy** than COCO-distilled at 1% labels
- COCO-distilled performs only marginally better than random init at low label fractions
- Even at 100% labels, COCO-distilled achieves only 54.16% vs ImageNet's 91.16%

**AULC (Area Under Learning Curve):**
- COCO-Distilled: 4.64 (1%), 13.63 (10%), 47.27 (100%)
- ImageNet: 34.76 (1%), 76.66 (10%), 89.65 (100%)
- Random: 3.08 (1%), 4.99 (10%), 21.32 (100%)

---

### 2. EuroSAT Dataset (Satellite Image Classification, 10 classes)

**Challenge:** Domain shift from natural images to satellite imagery

| Init Method | 1% Labels (270 samples) | 100% Labels (27,000 samples) |
|-------------|-------------------------|------------------------------|
| **COCO-Distilled** | 74.74% | 97.67% |
| **ImageNet** | **86.28%** | **98.94%** |
| **Random** | 64.04% | 97.20% |

**Observations:**
- All methods perform well on EuroSAT (simpler task, 10 classes vs 37)
- COCO-distilled shows **much better relative performance** compared to Pets
- At 100% labels, all three methods converge (97-99% accuracy)
- ImageNet still leads at 1% labels with 11.5% absolute improvement over COCO-distilled

**AULC (Area Under Learning Curve):**
- COCO-Distilled: 58.39 (1%), 95.21 (100%)
- ImageNet: 73.04 (1%), 98.32 (100%)
- Random: 46.09 (1%), 92.46 (100%)

---

### 3. Imagenette Dataset (10-class ImageNet Subset)

**Challenge:** Simplified ImageNet classification

| Init Method | 100% Labels |
|-------------|-------------|
| **COCO-Distilled** | 80.54% |
| **ImageNet** | **99.16%** |
| **Random** | 77.17% |

**Observations:**
- ImageNet initialization has massive advantage on ImageNet-derived data
- COCO-distilled barely outperforms random (80.54% vs 77.17%)
- This suggests COCO distillation didn't transfer well even to natural images

---

## Analysis and Insights

### Why Does COCO Distillation Fail on Pets?

**Hypothesis 1: Domain Mismatch**
- COCO contains general object detection scenes, not fine-grained breed distinctions
- ImageBind embeddings trained on COCO may not capture subtle visual differences needed for breeds
- Pets requires fine-grained features (whisker patterns, ear shapes, face structure)

**Hypothesis 2: Insufficient Distillation**
- Student model (RegNetY-400MF, 440-dim) may lack capacity to capture ImageBind's 1024-dim embeddings
- Linear projector may be too simple to bridge the representation gap
- COCO training set (118K images) may be insufficient compared to ImageNet (1.28M images)

**Hypothesis 3: Optimization Issues**
- Distillation hyperparameters (learning rate, loss weights, epochs) may not be optimal
- Combined loss (embedding + relational) may not be the right objective
- Student may have overfit to COCO's embedding space without learning transferable features

### Why Does EuroSAT Work Better?

**Hypothesis:**
- Satellite imagery is far from both COCO and ImageNet distributions
- This levels the playing field - no initialization has strong prior knowledge
- COCO's broader object recognition may be more useful than ImageNet's natural image bias
- EuroSAT is a simpler task (10 classes vs 37) where even weak features suffice

---

## Ongoing: Experiment 2 - Projector Ablation

### Research Question
Should we keep the frozen projector (440→1024) from COCO distillation during fine-tuning, or drop it and train a classifier directly on the 440-dim backbone features?

### Architectures Being Compared

**Baseline (Drop Projector):**
```
Backbone (fine-tuned) → Classifier (440 → num_classes)
```

**Ablation (Keep Frozen Projector):**
```
Backbone (fine-tuned) → Projector (FROZEN 440→1024) → Classifier (1024 → num_classes)
```

### Status: 1/4 Runs Complete

**Current run:** Pets, 1% labels, keep_projector=True
- Epoch 20/50
- Training accuracy: 100.00%
- Validation accuracy: 6.43%
- This is showing signs of overfitting (100% train, 6.43% val)

### Preliminary Observations

The model is using:
- Total parameters: 4,392,653
- Trainable parameters: 3,941,069 (frozen projector reduces trainable params)
- Classifier input: 1024-dim (higher dimensional features)

---

## Statistical Summary

### Sample Efficiency Metrics

**Sample Efficiency Ratio (ImageNet / COCO-Distilled):**

| Dataset | 1% Labels | 10% Labels | 100% Labels |
|---------|-----------|------------|-------------|
| Pets | 7.39× | 4.74× | 1.68× |
| EuroSAT | 1.15× | - | 1.01× |

**Interpretation:**
- ImageNet is 7× more sample-efficient than COCO-distilled on fine-grained tasks at 1% labels
- This gap closes as more data becomes available
- On domain-shifted tasks (EuroSAT), the gap is much smaller

### Learning Efficiency (AULC Comparison)

**AULC Improvement Ratio (ImageNet / COCO-Distilled):**

| Dataset | 1% Labels | 100% Labels |
|---------|-----------|-------------|
| Pets | 7.49× | 1.90× |
| EuroSAT | 1.25× | 1.03× |

**Interpretation:**
- AULC captures learning speed across all epochs
- ImageNet learns much faster on Pets, reaching high accuracy earlier
- On EuroSAT, all methods converge at similar speeds

---

## Milestone Accuracy Tracking

### Pets Dataset - Learning Trajectories

**COCO-Distilled (1% labels):**
- Epoch 5: Not improving significantly
- Epoch 10: Still below 10%
- Epoch 20: ~5.21%

**ImageNet (1% labels):**
- Epoch 5: Already ~30-35%
- Epoch 10: ~40%
- Epoch 20: ~41.66%

**Interpretation:** ImageNet initialization starts strong and maintains lead throughout training.

---

## Recommendations

### Based on Current Findings

1. **For Production Use:**
   - Use ImageNet-pretrained initialization for most computer vision tasks
   - COCO distillation does not provide competitive performance on tested tasks
   - Only consider COCO distillation for highly domain-shifted tasks where ImageNet bias is detrimental

2. **For Research:**
   - Investigate why COCO distillation fails so dramatically
   - Test alternative distillation datasets (ImageNet itself?)
   - Experiment with larger student models and MLP projectors
   - Try different loss functions (contrastive loss, knowledge distillation variants)

3. **Next Experiments:**
   - Complete Experiment 2 (projector ablation) to understand if the issue is in the projector or backbone
   - If possible, run Experiment 1 (teacher oracle) to establish upper bound on distillation performance
   - If possible, run Experiment 3 (ImageNet distillation) to test if distillation source is the key factor

---

## Experimental Setup

### Model Architecture
- **Backbone:** RegNetY-400MF (440-dim features)
- **Projector:** Linear(440 → 1024) for distillation
- **Teacher:** ImageBind vision encoder (frozen, 1024-dim embeddings)

### Training Configuration

**Distillation (Phase 1):**
- Dataset: COCO train2017 (118K images, unlabeled)
- Epochs: 30
- Batch size: 256
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: Combined (embedding + 0.5×relational)

**Downstream Fine-tuning (Phase 2):**
- Epochs: 50
- Batch size: 64
- Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
- LR Schedule: Linear warmup (5 epochs) + cosine decay

### Datasets
- **Pets:** Oxford-IIIT Pet (37 classes, 3,680 train / 3,669 val)
- **EuroSAT:** Satellite imagery (10 classes, 27,000 total)
- **Imagenette:** ImageNet subset (10 classes, 12,894 total)

---

## Limitations

1. **Single Distillation Source:** Only tested COCO, not ImageNet or other datasets
2. **Single Student Architecture:** Only tested RegNetY-400MF
3. **Limited Projector Exploration:** Only tested linear projector
4. **No Teacher Oracle Baseline:** Cannot measure distillation efficiency without comparing to frozen teacher embeddings
5. **Limited Hyperparameter Search:** Used default settings from similar papers

---

## Conclusions

### Main Findings

1. **COCO distillation is not competitive with ImageNet pretraining** for the tested downstream tasks
2. **Fine-grained tasks (Pets) show dramatically worse performance** with COCO distillation
3. **Domain-shifted tasks (EuroSAT) show relatively better performance** but ImageNet still leads
4. **Sample efficiency gap is largest at very low label fractions** (1-10%)

### Hypothesis on Root Cause

The poor performance likely stems from:
- **Domain mismatch:** COCO's general scenes don't transfer to fine-grained classification
- **Insufficient training data:** 118K COCO images vs 1.28M ImageNet images
- **Student capacity limitations:** 440-dim RegNetY may not capture 1024-dim ImageBind embeddings well

### Path Forward

To make distillation-based initialization competitive:
1. **Try distilling from ImageNet** instead of COCO (Experiment 3)
2. **Use larger student models** to match teacher capacity
3. **Optimize distillation hyperparameters** systematically
4. **Explore alternative loss functions** beyond embedding + relational
5. **Test MLP projectors** instead of linear ones

---

## Appendix: Result Files

All experimental results are saved in `checkpoints/results_*.csv`:

- `results_pets_distilled_frac0.01_s1.csv` (and s2, s3)
- `results_pets_imagenet_frac0.01_s1.csv` (and s2, s3)
- `results_pets_random_frac0.01_s1.csv` (and s2, s3)
- `results_eurosat_distilled_frac0.01_s1.csv`
- `results_eurosat_imagenet_frac0.01_s1.csv`
- `results_eurosat_random_frac0.01_s1.csv`
- ... (34 total files)

Combined results: `reports/combined_results.csv`

---

## Experiment Timeline

- **Phase 0:** Sanity check on Imagenette (✅ Complete)
- **Phase 1:** Full COCO distillation (✅ Complete)
- **Phase 1 Downstream:** Baseline comparisons (✅ Complete, 34 runs)
- **Experiment 2:** Projector ablation (⏳ In Progress, 1/4 runs)
- **Experiment 1:** Teacher oracle probes (⏸️ Blocked: requires ImageBind)
- **Experiment 3:** ImageNet distillation (⏸️ Blocked: requires ImageNet dataset)

---

*Report generated automatically from experimental results.*
*Last updated: January 26, 2026*
