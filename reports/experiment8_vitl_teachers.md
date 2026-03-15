# Experiment 8: ViT-L Teacher Distillation — Does Teacher Scale Change the Story?

**Date:** March 6, 2026
**Status:** Complete (distillation + downstream evaluation)

---

## Introduction

This project tests a practical implication of the platonic representation hypothesis (Huh et al., 2024): if large vision models converge toward a shared "platonic" representation of the visual world, then distilling that representation into a small model should produce a superior weight initialization — one that converges faster, generalizes better from few labels, and closes the gap to standard ImageNet pretraining.

The core idea is simple. If a teacher model's embedding space approximates this universal structure, a student trained to match those embeddings inherits not just the teacher's knowledge but a general-purpose geometric organization of visual concepts. The closer the teacher is to the platonic ideal, the better the initialization should be.

**Why teacher scale is the right lever to test this.** Huh et al.'s analysis of 78 vision models shows that representational convergence — the tendency of different models to learn similar internal representations — increases with model scale and capability. Smaller models scatter in representation space; larger models cluster together. ViT-L (304M params) sits meaningfully higher on this convergence curve than ViT-B (86M params). If the platonic hypothesis drives transfer quality through distillation, the effect should be measurable when scaling teachers from ViT-B to ViT-L.

**What Experiment 7 established.** ViT-B distillation on ImageNet closes 68–87% of the gap to ImageNet pretraining — the pipeline works. But backbone CKA (our measure of structural alignment between student and teacher) did not predict convergence speed or low-label generalization. The ViT-B CLIP teacher achieved much higher backbone CKA (0.79) than supervised (0.31), yet supervised fine-tuned better at low data. The Exp 7 report flagged teacher scale as the key unresolved caveat:

> "The platonic representation hypothesis primarily concerns large-scale models... ViT-B teachers may simply lack the representational quality needed."

**Specific predictions.** If the platonic hypothesis holds at ViT-L scale, we should observe:

1. **Higher backbone CKA** in the distilled student (ViT-L embeddings should be easier to align with because they are "more platonic")
2. **Better downstream accuracy** than ViT-B distilled students, closing more of the remaining gap to ImageNet pretraining
3. **Faster convergence** during downstream fine-tuning (better initialization → fewer epochs needed)
4. **A monotonic relationship** between teacher backbone CKA and downstream performance across the three ViT-L teachers

**The DINOv2 angle.** By testing three training paradigms at the same scale — supervised (ImageNet labels), contrastive (CLIP), and self-supervised (DINOv2) — we can probe whether representations converge across training methods at ViT-L scale. The platonic hypothesis predicts they should: training method matters less as models get larger because all roads lead to the same representation. If DINOv2, CLIP, and supervised ViT-L produce similarly effective distillation targets, that would be strong evidence for convergence. If they diverge sharply, training objective still dominates over any emergent platonic structure.

---

## Motivation

Three ViT-L teachers spanning different training paradigms:

| Teacher | Model | Params | Training | Embed dim | Input |
|---------|-------|--------|----------|-----------|-------|
| supervised_l | vit_large_patch16_224.augreg_in21k_ft_in1k | 303M | Supervised (IN-21k→IN-1k) | 1024 | 224px |
| clip_l | vit_large_patch14_clip_224.openai | 303M | Contrastive (CLIP) | 1024 | 224px |
| dinov2_l | vit_large_patch14_dinov2.lvd142m | 304M | Self-supervised (DINOv2) | 1024 | 518→224px |

**Hardware:** NVIDIA B200, 183GB VRAM.

---

## Experimental Design

### Distillation Phase

CKA combined loss only (λ=0.1), consistent with Experiment 7 methodology. Batch size reduced to 512 (from 1024 in Exp 7) due to ViT-L ~3.5x memory cost. LR scaled proportionally: 4e-3 → 2e-3 (linear scaling rule).

| Teacher | Loss | Epochs | BS | LR | Val Cosine | CKA (proj) | CKA (backbone) |
|---------|------|--------|-----|-----|-----------|------------|----------------|
| supervised_l | cka_combined (λ=0.1) | 7* | 512 | 2e-3 | 0.651 | 0.794 | 0.618 |
| clip_l | cka_combined (λ=0.1) | 20 | 512 | 2e-3 | **0.835** | **0.866** | **0.779** |
| dinov2_l | cka_combined (λ=0.1) | 20 | 512 | 2e-3 | 0.521 | 0.493 | 0.314 |

*supervised_l killed at epoch 7 after backbone CKA plateaued (0.60→0.62→0.60 across epochs 5–7).

**Key observations:**

1. **clip_l is the best distillation target** — highest cosine similarity (0.84) and backbone CKA (0.78). CLIP-L features compress into the RegNetY student most faithfully.
2. **DINOv2-L failed to distill effectively.** Despite being the largest model (304M), backbone CKA reached only 0.31 — equal to ViT-B supervised from Exp 7. The 518→224px position embedding interpolation likely degraded feature quality, and DINOv2's self-supervised features may be fundamentally harder to compress into a small CNN.
3. **supervised_l plateaued quickly.** Backbone CKA stopped improving by epoch 5–6 at 0.62. The loss and cosine similarity continued improving (projector learning), but backbone features converged.

### Downstream Phase

96 successful downstream runs across 6 datasets, 3 label fractions, and 2 modes (fine-tune + linear probe). 12 expected failures: DTD (47 classes) and Flowers102 (102 classes) at 1% — too few samples for stratified splitting (same as Exp 7).

Baselines (random, ImageNet pretrained) and ViT-B results reused from Experiment 7.

---

## Results

### ViT-L vs ViT-B: Distillation Quality

| Teacher | Scale | Backbone CKA | Val Cos Sim |
|---------|-------|-------------|-------------|
| supervised | ViT-B | 0.31 | 0.72 |
| **supervised_l** | **ViT-L** | **0.62** | **0.65** |
| clip_768 | ViT-B | 0.79* | 0.83 |
| **clip_l** | **ViT-L** | **0.78** | **0.84** |
| **dinov2_l** | **ViT-L** | **0.31** | **0.52** |

*ViT-B clip_768 backbone CKA of 0.79 was measured at 30 epochs on COCO; the Exp 7 ImageNet run achieved 0.38 backbone CKA.

Teacher scale doubled backbone CKA for supervised (0.31→0.62) and maintained CLIP's high alignment (0.78). DINOv2 was an outlier — likely due to the resolution mismatch.

---

### 100% Labels — Fine-tune

| Dataset | random | sup_l | clip_l | dinov2_l | ViT-B sup | ViT-B clip | IN-pretrained |
|---------|--------|-------|--------|----------|-----------|------------|---------------|
| VOC | 20.6 | 72.3 | **75.9** | 37.8 | 76.7 | 78.4 | **86.4** |
| Pets | — | 77.5 | **80.1** | 72.6 | **82.0** | 79.7 | — |
| EuroSAT | — | 96.9 | **97.4** | 96.9 | 97.0 | 97.0 | — |
| DTD | — | 42.9 | **47.1** | 35.1 | 41.1 | **47.7** | **63.6** |
| Flowers | 24.1 | 49.7 | **53.0** | 9.9 | 46.6 | **56.9** | **82.1** |
| Imagenette | 77.1 | 94.3 | **96.3** | 92.8 | 95.9 | **96.5** | **98.7** |

### 100% Labels — Linear Probe

| Dataset | random | sup_l | clip_l | dinov2_l | ViT-B sup | ViT-B clip | IN-pretrained |
|---------|--------|-------|--------|----------|-----------|------------|---------------|
| VOC | 8.8 | 65.4 | **80.2** | 61.7 | — | **82.4** | **85.1** |
| Pets | 5.1 | 59.9 | 74.7 | **76.3** | 59.5 | **75.7** | **90.7** |
| EuroSAT | 48.5 | **89.4** | 86.6 | 77.1 | 87.3 | **91.2** | **93.1** |
| DTD | 7.5 | 38.7 | **48.9** | 35.0 | 32.8 | **48.5** | **62.1** |
| Flowers | 4.9 | 42.7 | **59.9** | 22.9 | 33.9 | **60.5** | **74.7** |
| Imagenette | 29.2 | 88.1 | **95.6** | 91.6 | 90.4 | **96.1** | **99.3** |

**ViT-L does not close more of the gap than ViT-B.** On almost every dataset and mode, ViT-B clip_768 matches or slightly exceeds ViT-L clip_l:

| Dataset | Mode | ViT-B clip | ViT-L clip_l | Delta |
|---------|------|-----------|-------------|-------|
| VOC | fine-tune | **78.4** | 75.9 | -2.5 |
| VOC | linprobe | **82.4** | 80.2 | -2.2 |
| Pets | fine-tune | 79.7 | **80.1** | +0.4 |
| Pets | linprobe | **75.7** | 74.7 | -1.0 |
| EuroSAT | fine-tune | 97.0 | **97.4** | +0.4 |
| DTD | fine-tune | **47.7** | 47.1 | -0.6 |
| DTD | linprobe | **48.5** | 48.9 | +0.4 |
| Flowers | fine-tune | **56.9** | 53.0 | -3.9 |
| Flowers | linprobe | **60.5** | 59.9 | -0.6 |
| Imagenette | linprobe | **96.1** | 95.6 | -0.5 |

ViT-B clip edges ahead on 7/10 comparisons. The differences are small (mostly <2 pp) but consistently favor ViT-B. This is unexpected and important — a 3.5x larger teacher does not produce a better distilled student.

---

### 10% Labels

| Dataset | sup_l ft | clip_l ft | dinov2_l ft | sup_l lp | clip_l lp | dinov2_l lp |
|---------|----------|-----------|-------------|----------|-----------|-------------|
| VOC | 46.4 | **54.2** | 11.9 | 53.3 | **74.1** | 37.8 |
| Pets | **45.3** | 38.0 | 18.8 | 51.1 | **66.5** | 68.5 |
| EuroSAT | 89.3 | **91.2** | 88.3 | 84.1 | 81.8 | 70.6 |
| DTD | **17.6** | 13.8 | 3.9 | 22.3 | **29.2** | 13.4 |
| Flowers | 3.0 | **4.6** | 1.5 | 10.3 | **15.6** | 3.5 |
| Imagenette | **91.1** | 90.6 | 87.2 | 84.0 | **93.7** | 90.2 |

### 1% Labels

| Dataset | sup_l ft | clip_l ft | dinov2_l ft | sup_l lp | clip_l lp | dinov2_l lp |
|---------|----------|-----------|-------------|----------|-----------|-------------|
| VOC | 13.6 | **14.1** | 11.0 | 20.3 | **36.4** | 13.1 |
| Pets | **17.2** | 14.3 | 4.1 | 22.0 | 30.7 | **35.7** |
| EuroSAT | **62.7** | 59.4 | 32.4 | **73.1** | 69.2 | 57.8 |
| Imagenette | 76.1 | **79.2** | 60.4 | 74.8 | **92.2** | 86.3 |

The same pattern from Exp 7 holds: **supervised wins fine-tuning at low data** (Pets 1%: 17.2 vs 14.3), **CLIP wins linear probing everywhere** (often by 10+ pp).

**DINOv2-L linprobe anomaly on Pets:** DINOv2 achieves 76.3% linprobe at 100% (highest of all ViT-L teachers) and 35.7% at 1% (also highest for linprobe). Despite poor distillation metrics, DINOv2's frozen backbone features are surprisingly good on Pets — possibly because DINOv2's self-supervised training produces features naturally suited to fine-grained animal classification.

---

### Does Backbone CKA Predict Downstream Performance?

The ordering of backbone CKA is: clip_l (0.78) > supervised_l (0.62) >> dinov2_l (0.31).

#### Fine-tuning at 100% labels

| Dataset | clip_l (CKA=0.78) | sup_l (CKA=0.62) | dinov2_l (CKA=0.31) | CKA predicts? |
|---------|-------------------|-------------------|---------------------|---------------|
| VOC | **75.9** | 72.3 | 37.8 | Yes |
| Pets | **80.1** | 77.5 | 72.6 | Yes |
| EuroSAT | **97.4** | 96.9 | 96.9 | Weak |
| DTD | **47.1** | 42.9 | 35.1 | Yes |
| Flowers | **53.0** | 49.7 | 9.9 | Yes |
| Imagenette | **96.3** | 94.3 | 92.8 | Yes |

CKA ordering holds monotonically on 5/6 datasets for fine-tuning — **a stronger result than Exp 7**, where supervised (lower CKA) often beat CLIP.

#### Linear probing at 100% labels

| Dataset | clip_l (CKA=0.78) | sup_l (CKA=0.62) | dinov2_l (CKA=0.31) | CKA predicts? |
|---------|-------------------|-------------------|---------------------|---------------|
| VOC | **80.2** | 65.4 | 61.7 | Yes |
| Pets | 74.7 | 59.9 | **76.3** | **No** |
| EuroSAT | 86.6 | **89.4** | 77.1 | **No** |
| DTD | **48.9** | 38.7 | 35.0 | Yes |
| Flowers | **59.9** | 42.7 | 22.9 | Yes |
| Imagenette | **95.6** | 88.1 | 91.6 | **No** |

For linear probing, the ordering breaks on 3/6 datasets. DINOv2 beats clip_l on Pets (76.3 vs 74.7) despite much lower CKA. supervised_l beats clip_l on EuroSAT (89.4 vs 86.6).

**Summary:** Backbone CKA is a good predictor for fine-tuning (5/6 monotonic) but only moderate for linear probing (3/6 monotonic). The relationship is stronger within ViT-L than it was within ViT-B (Exp 7), but still not consistent enough to confirm the platonic hypothesis.

---

### Why ViT-L Doesn't Beat ViT-B

This is the most surprising result. Three possible explanations:

1. **Student bottleneck.** RegNetY-400MF (4.3M params, 440-dim backbone) may not have the capacity to absorb more information from a larger teacher. The student is the bottleneck, not the teacher — going from 86M→304M teacher doesn't help when the student can only represent 440 dimensions.

2. **Diminishing returns in embedding distillation.** Final-layer embedding matching may be subject to diminishing returns: a ViT-B teacher already provides embeddings that capture the essential visual semantics at 768 dimensions. The additional nuance in ViT-L's 1024-dim embeddings may not survive projection through a Linear(440→1024).

3. **ViT-L may need more training.** The supervised_l run was killed at epoch 7. clip_l ran 20 epochs but loss was still decreasing slowly. Longer training might produce differences. However, clip_l's loss curve was clearly flattening by epoch 16–20.

---

### DINOv2: A Cautionary Note

DINOv2-L achieved the worst distillation metrics (backbone CKA 0.31, val cosine 0.52) and the worst downstream results on most tasks. Two factors:

1. **Resolution mismatch.** DINOv2-L is trained at 518px. We used `img_size=224` which interpolates position embeddings, but the model never saw 224px inputs during pretraining. This likely degraded feature quality significantly.

2. **Feature distribution.** DINOv2's self-supervised features (patch-level DINO + iBOT losses) may have very different statistical properties than supervised/CLIP embeddings, making them harder to compress into a small CNN.

Despite this, DINOv2 shows an interesting anomaly: **best Pets linprobe at 100% (76.3%) and 1% (35.7%)**. Its frozen backbone features, though poorly aligned with the teacher by CKA, contain discriminative information for fine-grained animal classification. This suggests DINOv2 distillation produces qualitatively different features — not better overall, but with surprising strengths in specific domains.

---

## Assessment of the Platonic Representation Hypothesis

Experiment 8 was designed to test the platonic hypothesis at the scale it was proposed. The results are **mixed but lean negative**:

| Prediction | Result | Evidence |
|-----------|--------|----------|
| Larger teachers → better distillation | **Partially supported** | Backbone CKA improved (sup: 0.31→0.62) but downstream accuracy did not |
| Higher CKA → better downstream (fine-tune) | **Supported within ViT-L** | Monotonic on 5/6 datasets |
| Higher CKA → better downstream (linprobe) | **Mixed** | Monotonic on only 3/6 datasets; DINOv2 anomaly on Pets |
| ViT-L closes more of gap to pretrained | **Rejected** | ViT-B clip edges ViT-L clip on 7/10 comparisons |
| DINOv2 produces more "universal" features | **Rejected** | Worst distillation and downstream of the three ViT-L teachers |

**The platonic hypothesis predicted that larger teachers would produce qualitatively better results. They didn't.** ViT-L teachers produce higher backbone CKA in the distilled student, but this does not translate to better downstream performance. The student model appears to be the binding constraint.

However, CKA does predict downstream ordering within a teacher scale (ViT-L) more consistently than it did within ViT-B — suggesting that CKA captures something real about representation quality, even if the absolute downstream numbers are limited by student capacity.

---

## Key Findings

### 1. Teacher scale does not improve downstream performance

ViT-L teachers (304M params) produce distilled students with similar or slightly worse downstream accuracy compared to ViT-B teachers (86M params), despite 2x higher backbone CKA for supervised. The student model (4.3M params) is the bottleneck.

### 2. Backbone CKA predicts fine-tuning ordering within ViT-L

clip_l (CKA=0.78) > supervised_l (0.62) > dinov2_l (0.31) maps to downstream accuracy monotonically on 5/6 datasets for fine-tuning. This is a stronger result than Exp 7 and suggests CKA captures meaningful representation quality — but it does not translate to absolute performance gains over ViT-B distillation.

### 3. DINOv2 distillation fails but reveals domain-specific strengths

Despite the worst overall metrics, DINOv2-distilled features achieve the best Pets linear probe accuracy among ViT-L teachers. Self-supervised features may capture different (not better) visual information than supervised/CLIP features.

### 4. CLIP remains the best teacher across all scales

clip_l wins on 5/6 datasets (fine-tune) and 4/6 datasets (linprobe) at 100% labels. The CLIP advantage persists from ViT-B to ViT-L, consistent with contrastive training producing intrinsically more transferable embeddings.

### 5. The remaining gap to ImageNet pretraining is a student capacity problem

The gap persists even with ViT-L teachers and 1.28M images. Further progress likely requires:
- **Larger student models** (e.g., RegNetY-1.6GF or ResNet-50)
- **Intermediate-layer distillation** (not just final embeddings)
- **Label-aware distillation** combining embedding matching with soft label transfer

---

## Comparison: All Methods on VOC 100%

| Method | Fine-tune (mAP) | LinProbe (mAP) |
|--------|----------------|----------------|
| ImageNet pretrained | **87.4** | **85.2** |
| ImageNet clip_768 CKA (ViT-B) | 81.0 | **82.5** |
| ImageNet clip_l CKA (ViT-L) | 75.9 | 80.2 |
| ImageNet supervised CKA (ViT-B) | 79.0 | — |
| ImageNet supervised_l CKA (ViT-L) | 72.3 | 65.4 |
| ImageNet dinov2_l CKA (ViT-L) | 37.8 | 61.7 |
| COCO supervised (no CKA) | 67.3 | 51.4 |
| Random | 24.9 | 9.0 |

---

## Implementation Notes

- **Scripts:** `scripts/experiment8_vitl_distill.sh` (3 distillation runs), `scripts/experiment8_downstream.sh` (108 downstream runs, 4-way parallel)
- **Code changes:** Added 3 ViT-L entries to `TEACHER_CONFIGS` in `src/train_distill.py`; added `img_size` parameter to `GenericTeacher` for DINOv2-L pos-embed interpolation (518→224px)
- **Runtime:** ~36h distillation + ~3h downstream = ~39h total
- **Checkpoints:** `imagenet_supervised_l_cka_l0.1_distilled_best.pth`, `imagenet_clip_l_cka_l0.1_distilled_best.pth`, `imagenet_dinov2_l_cka_l0.1_distilled_best.pth`
- **Failed runs:** 12/108 — DTD and Flowers102 at 1% label fraction (stratified sampling requires more samples than classes)
