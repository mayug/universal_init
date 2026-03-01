# Final Experiment Report: Universal Initialization via ImageBind Distillation

## Executive Summary

This report presents results from experiments testing whether distilling ImageBind embeddings into a small vision model (RegNetY-400MF) yields more sample-efficient downstream learning. We compared multiple initialization strategies across Oxford-IIIT Pets and EuroSAT datasets at various label fractions.

**Key Finding: ImageNet-pretrained initialization significantly outperforms ImageBind distillation for fine-grained classification tasks, while distillation provides modest benefits for satellite imagery.**

---

## Experiment Overview

### Initialization Methods Compared

| Init Method | Description |
|-------------|-------------|
| `random` | Random weight initialization |
| `imagenet` | ImageNet-1k pretrained RegNetY-400MF |
| `distilled` (COCO) | Backbone distilled from ImageBind on COCO images |
| `distilled` (ImageNet) | Backbone distilled from ImageBind on ImageNet train (4 epochs) |
| `teacher_oracle` | Linear probe on frozen ImageBind embeddings (1024-dim) |

### Projector Ablations

| Configuration | Description |
|---------------|-------------|
| Drop projector | Classifier directly on backbone (440-dim) - **default** |
| Keep frozen projector | Backbone -> Frozen Projector -> Classifier (1024-dim) |
| Trainable projector | Backbone -> Trainable Projector -> Classifier (1024-dim) |

---

## Results Summary

### Pets Dataset (Fine-grained Classification)

| Initialization | 1% Labels | 100% Labels |
|----------------|-----------|-------------|
| Random | 3.81% | 29.41% |
| **ImageNet-pretrained** | **42.75%** | **91.16%** |
| COCO-distilled | 5.78% | 54.16% |
| ImageNet-distilled | 8.61% | 72.20% |
| Teacher Oracle | 15.32% | 88.36% |

**Key Observations (Pets):**
- ImageNet-pretrained is the clear winner, outperforming all other methods
- ImageNet-distilled (72.2%) significantly outperforms COCO-distilled (54.2%) at 100% labels
- Distillation underperforms the teacher oracle, indicating capacity/training limitations
- Gap between teacher oracle (88.4%) and ImageNet-pretrained (91.2%) shows ImageNet features are well-suited for pet classification

### EuroSAT Dataset (Satellite Imagery)

| Initialization | 1% Labels | 100% Labels |
|----------------|-----------|-------------|
| Random | 64.04% | 97.20% |
| **ImageNet-pretrained** | **86.28%** | **98.94%** |
| COCO-distilled | 74.74% | 97.67% |
| ImageNet-distilled | 75.87% | 97.94% |
| Teacher Oracle | 28.89% | 89.13% |

**Key Observations (EuroSAT):**
- ImageNet-pretrained again dominates
- Distillation (75-76%) significantly outperforms teacher oracle (29%) at 1% labels
- At 100% labels, all pretrained methods converge (~97-99%)
- COCO and ImageNet distillation perform similarly

### Teacher Oracle Anomaly

The teacher oracle (linear probe on frozen ImageBind embeddings) underperforms distilled student models on EuroSAT:
- Teacher Oracle 1%: 28.89% vs Distilled: 74-76%
- Teacher Oracle 100%: 89.13% vs Distilled: 97-98%

This suggests ImageBind's embedding space is not optimal for satellite imagery classification, but the distillation process allows the student to adapt better to this domain.

---

## Projector Ablation Results

### Pets 100% Labels

| Configuration | Best Accuracy |
|---------------|---------------|
| Drop projector | 54.16% |
| Keep frozen projector | 33.31% |
| Trainable projector | 53.26% |

### EuroSAT 100% Labels

| Configuration | Best Accuracy |
|---------------|---------------|
| Drop projector | 97.67% |
| Keep frozen projector | 84.72% |
| Trainable projector | 97.41% |

**Key Finding:** Keeping a frozen projector significantly hurts performance (~20% drop). Making the projector trainable recovers most of this performance, matching "drop projector" results. The 1024-dim projector output provides no benefit over the 440-dim backbone features.

---

## ImageNet vs COCO Distillation Source

| Dataset | Fraction | COCO-distilled | ImageNet-distilled | Difference |
|---------|----------|----------------|-------------------|------------|
| Pets | 1% | 5.78% | 8.61% | **+2.83%** |
| Pets | 10% | 17.27% | 34.70% | **+17.43%** |
| Pets | 100% | 54.16% | 72.20% | **+18.04%** |
| EuroSAT | 1% | 74.74% | 75.87% | +1.13% |
| EuroSAT | 10% | - | 93.54% | - |
| EuroSAT | 100% | 97.67% | 97.94% | +0.27% |

**Key Finding:** ImageNet-distilled checkpoint significantly outperforms COCO-distilled on Pets (+18% at 100% labels), likely because:
1. ImageNet contains animal categories similar to pet breeds
2. ImageNet has 10x more training images (1.28M vs 118K)

Note: ImageNet distillation ran for only 4 epochs due to computational constraints.

---

## Distillation Efficiency Analysis

**Definition:** Efficiency = Student Accuracy / Teacher Oracle Accuracy

| Dataset | Fraction | Efficiency (COCO) | Efficiency (ImageNet) |
|---------|----------|-------------------|----------------------|
| Pets | 1% | 37.7% | 56.2% |
| Pets | 100% | 61.3% | 81.7% |
| EuroSAT | 1% | 258.7%* | 262.6%* |
| EuroSAT | 100% | 109.6%* | 109.9%* |

*Values >100% indicate student outperforms teacher oracle (domain shift scenario)

---

## Conclusions

### 1. ImageNet Pretraining Remains Superior
For the tested tasks, ImageNet-pretrained weights consistently outperform ImageBind distillation. The "universal" embedding hypothesis does not hold for these specific downstream tasks.

### 2. Distillation Source Matters
ImageNet-distilled significantly outperforms COCO-distilled on Pets, suggesting the distillation source domain should align with downstream tasks.

### 3. Projector is Best Dropped or Made Trainable
Keeping a frozen projector hurts performance. Either drop it entirely or make it trainable.

### 4. Domain-Specific Anomalies
On EuroSAT (domain shift from natural images), the student outperforms teacher oracle, suggesting distillation provides a form of domain adaptation.

### 5. Sample Efficiency Not Achieved
The primary hypothesis that distillation yields better sample efficiency was not confirmed. ImageNet pretraining provides better low-data performance on both tasks.

### 6. Structural Alignment Does Not Close the Gap (Experiment 5)
Replacing relational loss with a differentiable CKA loss improved student-teacher backbone CKA from 0.01–0.26 to 0.52–0.68 (3-50x improvement). However, downstream accuracy was unchanged within noise (< 1 pp). This disproves the hypothesis that structural misalignment was the performance bottleneck. The gap is more fundamental: insufficient training data diversity (82K COCO vs 1.2M ImageNet) and information loss through the distillation pipeline.

---

## Recommendations

1. **Use ImageNet pretraining** for vision tasks when available
2. **If distilling**, choose a source dataset aligned with target domain
3. **Drop the projector** for downstream classification tasks
4. **Consider longer distillation** - our ImageNet run was only 4 epochs

---

## Experimental Details

### Model Architecture
- **Backbone:** RegNetY-400MF (4.3M parameters)
- **Projector:** Linear (440 -> 1024)
- **Teacher:** ImageBind Vision Encoder (frozen, 1024-dim output)

### Training Configuration
- **Distillation:** AdamW, cosine schedule, batch size 1024
- **Downstream:** SGD with momentum 0.9, 50 epochs, batch size 64

### Datasets
- **Pets:** 37 breeds, 3,680 train images
- **EuroSAT:** 10 land cover classes, 27,000 images (RGB version)

---

## Appendix: Full Results Tables

### All Pets Results

| Init | Projector | Fraction | Best Acc | AULC |
|------|-----------|----------|----------|------|
| random | - | 1% | 3.81% | 3.08 |
| random | - | 100% | 29.41% | 21.32 |
| imagenet | - | 1% | 42.75% | 34.76 |
| imagenet | - | 100% | 91.16% | 89.65 |
| distilled (COCO) | drop | 1% | 5.78% | 4.64 |
| distilled (COCO) | drop | 100% | 54.16% | 47.27 |
| distilled (COCO) | frozen | 1% | 6.57% | 5.61 |
| distilled (COCO) | frozen | 100% | 33.31% | 29.73 |
| distilled (COCO) | trainable | 1% | 6.68% | 5.65 |
| distilled (COCO) | trainable | 100% | 53.26% | 46.61 |
| distilled (ImageNet) | drop | 1% | 8.61% | 6.03 |
| distilled (ImageNet) | drop | 10% | 34.70% | 27.22 |
| distilled (ImageNet) | drop | 100% | 72.20% | 65.90 |
| teacher_oracle | - | 1% | 15.32% | 8.24 |
| teacher_oracle | - | 100% | 88.36% | 71.25 |

### All EuroSAT Results

| Init | Projector | Fraction | Best Acc | AULC |
|------|-----------|----------|----------|------|
| random | - | 1% | 64.04% | 46.09 |
| random | - | 100% | 97.20% | 92.46 |
| imagenet | - | 1% | 86.28% | 73.04 |
| imagenet | - | 100% | 98.94% | 98.32 |
| distilled (COCO) | drop | 1% | 74.74% | 58.39 |
| distilled (COCO) | drop | 100% | 97.67% | 95.21 |
| distilled (COCO) | frozen | 1% | 68.65% | 59.66 |
| distilled (COCO) | frozen | 100% | 84.72% | 82.29 |
| distilled (COCO) | trainable | 1% | 73.46% | 61.48 |
| distilled (COCO) | trainable | 100% | 97.41% | 95.29 |
| distilled (ImageNet) | drop | 1% | 75.87% | 60.36 |
| distilled (ImageNet) | drop | 10% | 93.54% | 85.46 |
| distilled (ImageNet) | drop | 100% | 97.94% | 96.22 |
| teacher_oracle | - | 1% | 28.89% | 22.45 |
| teacher_oracle | - | 100% | 89.13% | 82.98 |

---

---

## Experiments 4 & 5: Multi-Teacher Distillation and CKA Loss

See `reports/experiment4_platonic_representation.md` for the full report covering:

- **Experiment 4:** Multi-teacher distillation (Supervised ViT-B/16, CLIP ViT-B/16 pre-projection, CLIP ViT-B/16 with projection), downstream fine-tuning, linear probing, and CKA similarity analysis.
- **Experiment 5:** Differentiable CKA distillation loss — replacing relational loss with CKA to directly optimize structural alignment. 9 distillation runs (3 teachers x 3 λ_cka values) + 84 downstream evaluation runs.

---

*Report last updated: February 28, 2026*
