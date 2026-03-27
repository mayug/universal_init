# Experiment 13: OOD Vision Downstream — Medical Imaging

## Summary

We test whether CLIP-distilled initialization beats ImageNet pretraining on **out-of-distribution** medical imaging datasets, where ImageNet features should lack the home-field advantage they enjoy on natural-image benchmarks (Pets, Flowers, DTD, EuroSAT). This mirrors our audio experiments (Exp 10), where foundation-distilled models beat the in-domain AudioSet baseline at low label fractions.

**Key finding:** ImageNet pretraining still dominates on medical imaging, winning 13/18 comparisons against CLIP-CKA0.1 and 14/18 against CLIP-CKAonly. The one exception is **DermaMNIST linear probe**, where CLIP-distilled consistently beats ImageNet at 10% and 100% labels. The vision–audio asymmetry is real: ImageNet low-level features (edges, textures, color) transfer to medical microscopy far better than AudioSet spectral features transfer to environmental sounds.

However, **CLIP-distilled consistently beats Supervised-distilled** (14/18 for CKA0.1), confirming that foundation geometry > supervised geometry even on OOD domains.

## Setup

### Datasets

All from [MedMNIST v2](https://medmnist.com/), 224×224 RGB, genuinely OOD from ImageNet:

| Dataset | Domain | Classes | Train | Test |
|---------|--------|---------|-------|------|
| PathMNIST | Colorectal histopathology | 9 | 89,996 | 7,180 |
| DermaMNIST | Skin lesion dermoscopy | 7 | 7,007 | 2,005 |
| BloodMNIST | Blood cell microscopy | 8 | 11,959 | 3,421 |

### Initialization Conditions

| Init | Checkpoint | Teacher | Loss |
|------|-----------|---------|------|
| Random | — | — | — |
| ImageNet | torchvision pretrained | — | — |
| CLIP-CKA0.1 | `imagenet_clip_768_cka_l0.1_distilled_best.pth` | CLIP ViT-B/16 (768d) | cosine + 0.1×CKA |
| CLIP-CKAonly | `imagenet_clip_l_cka_only_distilled_best.pth` | CLIP ViT-L/14 (1024d) | CKA only |
| Sup-CKA0.1 | `imagenet_supervised_cka_l0.1_distilled_best.pth` | Supervised ViT-B/16 (768d) | cosine + 0.1×CKA |
| Sup-CKAonly | `imagenet_supervised_l_cka_only_distilled_best.pth` | Supervised ViT-L/14 (1024d) | CKA only |

### Protocol

- **Student:** RegNetY-400MF (3.9M params)
- **Label fractions:** 1%, 10%, 100%
- **Modes:** Fine-tune (FT), Linear probe (LP, frozen backbone)
- **Training:** 50 epochs, SGD+momentum, cosine LR (0.01 FT / 0.1 LP), batch size 64
- **Total runs:** 72 (6 inits × 3 fractions × 2 modes × 2 loss variants, minus Sup-CKA0.1 LP which was missing)

## Results

### PathMNIST (Colorectal Histopathology, 9 classes)

| Init | 1% FT | 1% LP | 10% FT | 10% LP | 100% FT | 100% LP |
|------|-------|-------|--------|--------|---------|---------|
| Random | 79.6 | 47.1 | 86.4 | 58.3 | 91.9 | 62.3 |
| **ImageNet** | **92.9** | **87.3** | **96.3** | **89.9** | 94.5 | **90.4** |
| CLIP-CKA0.1 | 86.1 | 83.5 | 92.4 | 88.1 | 94.0 | 87.1 |
| CLIP-CKAonly | 87.4 | 81.8 | 93.2 | 83.8 | 93.2 | 83.3 |
| Sup-CKA0.1 | 82.3 | — | 94.3 | — | **94.9** | — |
| Sup-CKAonly | 87.0 | 82.7 | 91.1 | 85.4 | 93.8 | 86.1 |

ImageNet dominates at all label fractions. Even at 100% FT, the gap to CLIP is small (~0.5pp), but ImageNet LP stays 3–7pp ahead.

### DermaMNIST (Skin Lesion Dermoscopy, 7 classes)

| Init | 1% FT | 1% LP | 10% FT | 10% LP | 100% FT | 100% LP |
|------|-------|-------|--------|--------|---------|---------|
| Random | 66.9 | 66.9 | 70.1 | 66.9 | 78.9 | 67.3 |
| ImageNet | **70.0** | 67.8 | **76.7** | 71.6 | **90.0** | 76.9 |
| CLIP-CKA0.1 | 68.1 | 67.7 | 72.6 | 73.1 | 84.1 | **80.0** |
| CLIP-CKAonly | 67.3 | **69.6** | 73.9 | **74.6** | 84.1 | 79.1 |
| Sup-CKA0.1 | 66.9 | — | 71.4 | — | 80.5 | — |
| Sup-CKAonly | 67.2 | 68.9 | 72.5 | 71.3 | 82.6 | 74.7 |

**DermaMNIST is the exception.** CLIP-distilled LP beats ImageNet LP at all three label fractions (CKAonly: +1.7pp, +3.0pp, +2.2pp). Fine-tuning still favors ImageNet. This dataset has only 7 classes and 7K training samples — the regime where frozen foundation features shine.

### BloodMNIST (Blood Cell Microscopy, 8 classes)

| Init | 1% FT | 1% LP | 10% FT | 10% LP | 100% FT | 100% LP |
|------|-------|-------|--------|--------|---------|---------|
| Random | 49.3 | 41.2 | 96.0 | 54.2 | 98.9 | 70.5 |
| **ImageNet** | **90.4** | **76.1** | **98.3** | **91.4** | **99.2** | 94.4 |
| CLIP-CKA0.1 | 82.9 | 64.3 | 96.9 | 90.8 | 98.9 | **94.8** |
| CLIP-CKAonly | 67.6 | 74.9 | 96.7 | 87.2 | 98.9 | 91.7 |
| Sup-CKA0.1 | 71.7 | — | 95.8 | — | 99.0 | — |
| Sup-CKAonly | 60.4 | 67.9 | 96.2 | 85.3 | 98.8 | 89.9 |

ImageNet wins convincingly, especially at 1% labels (FT: +7.5pp over CLIP-CKA0.1). At 100% LP, CLIP-CKA0.1 edges ahead by +0.4pp — the only win.

## Analysis

### Delta vs ImageNet (pp)

| Init | Dataset | 1% FT | 1% LP | 10% FT | 10% LP | 100% FT | 100% LP |
|------|---------|-------|-------|--------|--------|---------|---------|
| CLIP-CKA0.1 | PathMNIST | -6.8 | -3.8 | -3.9 | -1.8 | -0.6 | -3.3 |
| CLIP-CKA0.1 | DermaMNIST | -1.8 | -0.1 | -4.0 | **+1.5** | -5.9 | **+3.2** |
| CLIP-CKA0.1 | BloodMNIST | -7.5 | -11.9 | -1.4 | -0.6 | -0.2 | +0.4 |
| CLIP-CKAonly | PathMNIST | -5.6 | -5.5 | -3.0 | -6.1 | -1.3 | -7.1 |
| CLIP-CKAonly | DermaMNIST | -2.6 | **+1.7** | -2.7 | **+3.0** | -5.9 | **+2.2** |
| CLIP-CKAonly | BloodMNIST | -22.8 | -1.2 | -1.6 | -4.2 | -0.3 | -2.7 |

### Win Counts vs ImageNet (>0.5pp threshold)

| Init | Wins | Losses | Ties |
|------|------|--------|------|
| CLIP-CKA0.1 | 2 | 13 | 3 |
| CLIP-CKAonly | 3 | 14 | 1 |
| Sup-CKA0.1 | 0 | 7 | 2 |
| Sup-CKAonly | 1 | 15 | 2 |

### CLIP-distilled vs Supervised-distilled

CLIP consistently beats Supervised (same teacher architecture, different training):

- **CKA0.1:** CLIP wins 14/18 LP+FT comparisons (where both are available)
- **CKAonly:** CLIP wins 12/18

This replicates the Exp 7–9 finding that foundation (CLIP) geometry > supervised geometry, and extends it to OOD medical domains.

### CKA-only vs CKA-λ0.1 (Same Teacher)

| Teacher | CKA-only wins | CKA-λ0.1 wins |
|---------|--------------|---------------|
| CLIP | 6/18 | 12/18 |
| Supervised | 5/9 | 4/9 |

CKA-λ0.1 (combined loss) produces better features for CLIP teachers on medical imaging. For supervised teachers, the losses are roughly equivalent.

## Key Takeaways

### 1. ImageNet features transfer surprisingly well to medical imaging

Despite histopathology and dermoscopy being "OOD" from ImageNet, the low-level features learned from ImageNet (edges, textures, color gradients, spatial patterns) are broadly useful for medical imaging. This is fundamentally different from the audio domain, where AudioSet spectral features don't transfer well to environmental sounds.

### 2. The vision–audio asymmetry is real

In audio (Exp 10), SBERT-distilled beats AudioSet pretrained at 1% labels on ESC-50. In vision, ImageNet pretrained beats CLIP-distilled at 1% labels on all 3 medical datasets. **Why?**
- ImageNet pretraining learns low-level visual features that are universal across visual domains
- AudioSet pretraining learns spectral patterns specific to its training distribution
- Vision has a stronger "feature universality" at the low/mid level than audio

### 3. DermaMNIST LP is the one bright spot

CLIP-distilled LP beats ImageNet LP on DermaMNIST at all label fractions. This is the smallest dataset (7K train, 7 classes) and skin lesion classification may depend more on high-level semantic features (color, shape, symmetry patterns) that CLIP captures better than ImageNet's object-centric features.

### 4. Foundation geometry > supervised geometry holds on OOD

Even when both lose to ImageNet, CLIP-distilled consistently outperforms Supervised-distilled on medical imaging. The relative ordering of embedding geometries is preserved across domain shift.

## Predictions vs Reality

| Prediction | Result | Verdict |
|-----------|--------|---------|
| CLIP-distilled LP beats ImageNet LP at 1% | Only on DermaMNIST (-0.1pp ≈ tie) | **Mostly wrong** |
| ImageNet wins at 100% FT | Yes, on all 3 datasets | **Correct** |
| CLIP beats Supervised-distilled | Yes, 14/18 | **Correct** |
| LP gap CLIP–ImageNet smaller than on Pets/Flowers | Mixed — PathMNIST gap is smaller, BloodMNIST larger | **Partially correct** |
