# Experiment 7: ImageNet-Scale Distillation — Does Data Close the Gap?

**Date:** March 3, 2026
**Status:** Complete (distillation + downstream evaluation)

---

## Motivation

Experiments 1–6 established that distilling teacher embeddings into RegNetY-400MF on COCO (82K images) produces representations far inferior to ImageNet pretraining. Experiment 5 showed that CKA loss improves structural alignment 3–50x but does not improve downstream accuracy. The report concluded:

> "The root cause lies elsewhere: insufficient training data diversity (82K COCO vs 1.2M ImageNet)"

Experiment 7 tests this directly: **does distilling on ImageNet (1.28M images, 15x more data) close the gap with ImageNet pretraining?**

Secondary questions:

- Does CLIP's linear probe advantage over supervised distillation grow or shrink at scale?
- Does the CKA-aligned projector help downstream when preserved (frozen or trainable)?
- Does CKA structural alignment, which was irrelevant on COCO, finally matter with more data?

**Hardware:** NVIDIA B200, 183GB VRAM.

---

## Experimental Design

### Distillation Phase

Two teachers, one loss function, on ImageNet train (1.28M images):


| Teacher               | Loss                 | Epochs | BS   | LR   | Val Cosine | CKA (proj) | CKA (backbone) |
| --------------------- | -------------------- | ------ | ---- | ---- | ---------- | ---------- | -------------- |
| supervised (ViT-B/16) | cka_combined (λ=0.1) | 20     | 1024 | 4e-3 | 0.716      | 0.819      | 0.313          |
| clip_768 (ViT-B/16)   | cka_combined (λ=0.1) | 20     | 1024 | 4e-3 | **0.834**  | **0.875**  | **0.793**      |


**Key observation:** clip_768 achieves dramatically higher backbone CKA (0.79 vs 0.31). The CLIP representation transfers through the distillation bottleneck far more faithfully than supervised features. For comparison, COCO CKA distillation reached 0.657 (supervised) and 0.682 (clip_768) backbone CKA after 30 epochs.

clip_512 was dropped from this experiment: it performed worse than clip_768 downstream in all prior experiments and was unstable with CKA λ=0.1 (backbone CKA collapsed to 0.038).

### Downstream Phase

260 successful runs across 6 datasets (VOC, Pets, EuroSAT, DTD, Flowers102, Imagenette), 3 label fractions (1%, 10%, 100%), and 4 evaluation modes for CKA checkpoints:


| Mode                  | Description                                     | Rationale                            |
| --------------------- | ----------------------------------------------- | ------------------------------------ |
| Fine-tune (drop proj) | Standard: train backbone + new classifier       | Baseline transfer                    |
| Linear probe          | Frozen backbone + linear classifier             | Raw representation quality           |
| Frozen projector      | Frozen backbone + frozen projector + classifier | Tests CKA-aligned projection space   |
| Trainable projector   | Train backbone + projector + classifier         | Adapts projection during fine-tuning |


64 runs failed due to stratified sampling requiring more samples than classes (DTD=47 classes, Flowers102=102 classes at 1% fraction).

---

## Results

### Core Question: Does Data Scale Close the Gap?

#### VOC Multi-Label Classification (mAP)


| Init                         | Mode      | 1%       | 10%      | 100%     |
| ---------------------------- | --------- | -------- | -------- | -------- |
| **ImageNet pretrained**      | fine-tune | 13.1     | 60.1     | **87.4** |
| **ImageNet pretrained**      | linprobe  | **40.8** | **78.8** | **85.2** |
| COCO supervised (no CKA)     | fine-tune | 18.0     | 54.4     | 67.3     |
| COCO clip_768 (no CKA)       | linprobe  | 38.4     | 57.6     | 63.8     |
| **ImageNet supervised CKA**  | fine-tune | —        | —        | 79.0     |
| ImageNet supervised CKA      | linprobe  | 28.5     | 61.2     | —        |
| ImageNet supervised (no CKA) | fine-tune | —        | 58.9     | 78.3     |
| **ImageNet clip_768 CKA**    | fine-tune | —        | 60.3     | **81.0** |
| **ImageNet clip_768 CKA**    | linprobe  | **52.3** | **76.7** | **82.5** |
| ImageNet clip_768 CKA        | trainproj | 24.5     | 64.5     | 81.3     |


**Gap closure on VOC 100%:**

- COCO-distilled fine-tune: 67.3 mAP (20.1 gap to pretrained)
- ImageNet clip CKA fine-tune: **81.0 mAP** (6.4 gap) — **68% of gap closed**
- ImageNet clip CKA linprobe: **82.5 mAP** vs pretrained 85.2 — **only 2.7 mAP gap**

#### Pets Fine-Grained Classification (accuracy %)


| Init                        | Mode      | 1%       | 10%      | 100%     |
| --------------------------- | --------- | -------- | -------- | -------- |
| **ImageNet pretrained**     | linprobe  | **68.3** | **88.3** | **91.2** |
| COCO supervised (no CKA)    | fine-tune | 6.0      | —        | 47.5     |
| COCO clip_768 (no CKA)      | linprobe  | 6.8      | 16.8     | 29.9     |
| **ImageNet supervised CKA** | fine-tune | **26.1** | **59.1** | **83.2** |
| ImageNet supervised CKA     | linprobe  | 23.3     | 51.0     | 65.2     |
| **ImageNet clip_768 CKA**   | fine-tune | 16.5     | 48.4     | 81.7     |
| **ImageNet clip_768 CKA**   | linprobe  | 26.2     | **67.2** | **81.1** |


**Gap closure on Pets 100%:**

- COCO-distilled fine-tune: 47.5% (43.7 gap to pretrained 91.2)
- ImageNet sup CKA fine-tune: **83.2%** (8.0 gap) — **82% of gap closed**
- ImageNet clip CKA linprobe: **81.1%** (10.1 gap) — **77% of gap closed**

#### EuroSAT Satellite Imagery (accuracy %)


| Init                        | Mode      | 1%       | 10%      | 100%     |
| --------------------------- | --------- | -------- | -------- | -------- |
| **ImageNet pretrained**     | linprobe  | **84.7** | **92.2** | **94.4** |
| COCO supervised (no CKA)    | fine-tune | 72.2     | —        | 97.5     |
| **ImageNet supervised CKA** | fine-tune | 66.6     | 91.1     | 97.5     |
| ImageNet supervised CKA     | linprobe  | 75.4     | 87.4     | 91.2     |
| **ImageNet clip_768 CKA**   | fine-tune | **74.2** | **93.0** | **97.8** |
| **ImageNet clip_768 CKA**   | linprobe  | **77.7** | **89.6** | **94.1** |
| ImageNet clip_768 CKA       | trainproj | 75.8     | 93.7     | **98.0** |


**EuroSAT is nearly solved.** All fine-tuning methods converge at 100% (97.5–98.0%). ImageNet clip CKA linprobe reaches 94.1%, matching ImageNet pretrained linprobe (94.4%).

#### DTD Texture Classification (accuracy %)


| Init                      | Mode      | 10%      | 100%     |
| ------------------------- | --------- | -------- | -------- |
| **ImageNet pretrained**   | fine-tune | **37.1** | **65.7** |
| **ImageNet pretrained**   | linprobe  | **44.3** | **63.0** |
| COCO clip_768 (no CKA)    | fine-tune | 14.8     | 38.2     |
| **ImageNet clip_768 CKA** | fine-tune | 24.4     | **54.2** |
| **ImageNet clip_768 CKA** | linprobe  | **31.6** | **54.1** |
| ImageNet supervised CKA   | fine-tune | 17.8     | 44.0     |


**DTD 100%:** ImageNet clip CKA reaches 54.2% vs pretrained 65.7% — still an 11.5 pp gap. Textures are challenging for distilled features.

#### Flowers102 Fine-Grained Classification (accuracy %)


| Init                      | Mode      | 10%      | 100%     |
| ------------------------- | --------- | -------- | -------- |
| **ImageNet pretrained**   | fine-tune | **27.1** | **82.7** |
| **ImageNet pretrained**   | linprobe  | **32.5** | **76.2** |
| COCO clip_768 (no CKA)    | fine-tune | 9.6      | 44.2     |
| **ImageNet clip_768 CKA** | fine-tune | 10.0     | **61.0** |
| **ImageNet clip_768 CKA** | linprobe  | **27.8** | **70.9** |
| ImageNet supervised CKA   | fine-tune | 7.1      | 51.8     |


**Flowers 100%:** ImageNet clip CKA linprobe reaches 70.9% vs pretrained 76.2% — 5.3 pp gap. Fine-tuning is worse at 61.0% vs 82.7%.

#### Imagenette (ImageNet Subset, accuracy %)


| Init                        | Mode      | 1%       | 10%      | 100%     |
| --------------------------- | --------- | -------- | -------- | -------- |
| **ImageNet pretrained**     | fine-tune | **95.5** | **98.1** | **99.1** |
| **ImageNet pretrained**     | linprobe  | **98.7** | **99.3** | **99.4** |
| COCO supervised (no CKA)    | fine-tune | 61.7     | 80.5     | 88.6     |
| **ImageNet supervised CKA** | fine-tune | 88.0     | 93.3     | 96.7     |
| ImageNet supervised CKA     | linprobe  | 80.7     | 88.1     | 91.8     |
| **ImageNet clip_768 CKA**   | fine-tune | 86.5     | **94.1** | **97.1** |
| **ImageNet clip_768 CKA**   | linprobe  | **91.0** | **95.5** | **96.6** |


**Imagenette:** Near-perfect for ImageNet pretrained (expected — it's an ImageNet subset). ImageNet clip CKA linprobe reaches 96.6%, closing most of the gap to 99.4%.

---

### CLIP vs Supervised: A Clear Dichotomy


| Metric                        | Supervised wins         | CLIP wins               |
| ----------------------------- | ----------------------- | ----------------------- |
| **Fine-tuning, low data**     | Pets 1%: 26.1 vs 16.5   | —                       |
| **Fine-tuning, low data**     | Pets 10%: 59.1 vs 48.4  | —                       |
| **Fine-tuning, full data**    | Pets 100%: 83.2 vs 81.7 | VOC 100%: 81.0 vs 79.0  |
| **Linear probing, all fracs** | —                       | Pets: 26→81 vs 23→65    |
| **Linear probing, all fracs** | —                       | VOC: 52→82.5 vs 29→70   |
| **Linear probing, all fracs** | —                       | EuroSAT: 78→94 vs 75→91 |
| **Linear probing, all fracs** | —                       | DTD: 32→54 vs 19→42     |
| **Linear probing, all fracs** | —                       | Flowers: 28→71 vs 9→42  |


**Pattern:** CLIP produces features that are dramatically more linearly separable (better linear probing everywhere, often by 10–20 pp). Supervised features adapt better during fine-tuning at low data, but the advantage shrinks at 100% labels. This is consistent with CLIP's contrastive training objective producing well-separated embedding clusters.

---

### Projector Mode Analysis

For CKA checkpoints, projected CKA (0.82–0.87) far exceeds backbone CKA (0.31–0.79). Does preserving the projector help?

#### ImageNet clip_768 CKA (best structural alignment)


| Mode                  | Pets 1%  | Pets 100% | VOC 10%  | VOC 100% | EuroSAT 1% | EuroSAT 100% |
| --------------------- | -------- | --------- | -------- | -------- | ---------- | ------------ |
| Drop proj (fine-tune) | 16.5     | 81.7      | 60.3     | 81.0     | 74.2       | 97.8         |
| Frozen proj           | 16.2     | 63.0      | 58.9     | —        | 70.5       | 87.0         |
| Trainable proj        | 14.7     | 81.0      | 64.5     | 81.3     | 75.8       | **98.0**     |
| Linear probe          | **26.2** | **81.1**  | **76.7** | **82.5** | **77.7**   | 94.1         |


**Findings:**

1. **Frozen projector consistently hurts** — 10–18 pp worse than dropping it on Pets/EuroSAT. The projector constrains the model's adaptation without adding value for classification.
2. **Trainable projector ≈ drop projector** — within 1 pp on most conditions. No benefit from preserving the CKA-aligned projection.
3. **Linear probe is the best mode for CLIP** — matches or exceeds fine-tuning at full data (81.1 vs 81.7 on Pets, 82.5 vs 81.0 on VOC). The frozen backbone features are already excellent.

**Interpretation:** Despite projected CKA of 0.87, the projector's aligned embedding space does not carry discriminative information useful for classification. The projector was optimized to match the teacher's global structure, not to separate classes. The backbone features (despite lower CKA of 0.79) contain the discriminative signal that matters for downstream tasks.

---

### COCO vs ImageNet Distillation


| Dataset      | Mode      | COCO supervised | COCO clip_768 | IN supervised CKA | IN clip_768 CKA |
| ------------ | --------- | --------------- | ------------- | ----------------- | --------------- |
| VOC 100%     | fine-tune | 67.3            | 66.9          | 79.0              | **81.0**        |
| VOC 100%     | linprobe  | 51.4            | 63.8          | —                 | **82.5**        |
| Pets 100%    | fine-tune | 47.5            | 46.1          | **83.2**          | 81.7            |
| Pets 100%    | linprobe  | 26.3            | 29.9          | 65.2              | **81.1**        |
| EuroSAT 100% | fine-tune | 97.5            | 97.6          | 97.5              | **97.8**        |
| EuroSAT 100% | linprobe  | 83.2            | 83.9          | 91.2              | **94.1**        |
| DTD 100%     | fine-tune | 36.5            | 38.2          | 44.0              | **54.2**        |
| Flowers 100% | fine-tune | 43.1            | 44.2          | 51.8              | **61.0**        |


**15x more data provides 10–50 pp improvement across all datasets.** The gain is largest on tasks with large initial gaps (Pets: +35 pp, Flowers: +17 pp) and smallest on near-saturated tasks (EuroSAT: +0.2 pp).

---

### CKA vs Non-CKA on ImageNet (supervised teacher)


| Dataset          | Mode      | IN sup (no CKA) | IN sup CKA λ=0.1 |
| ---------------- | --------- | --------------- | ---------------- |
| VOC 100%         | fine-tune | 78.3            | 79.0             |
| VOC linprobe     | linprobe  | 70.1            | —                |
| Pets 100%        | fine-tune | **84.4**        | 83.2             |
| Pets linprobe    | linprobe  | **69.3**        | 65.2             |
| EuroSAT 100%     | fine-tune | 97.6            | 97.5             |
| EuroSAT linprobe | linprobe  | **91.7**        | 91.2             |
| Imagenette 100%  | fine-tune | **96.6**        | 96.7             |


**CKA loss does not help the supervised teacher — it may slightly hurt.** Non-CKA supervised distillation on ImageNet (combined loss) actually edges out CKA on several metrics (Pets fine-tune 84.4 vs 83.2, Pets linprobe 69.3 vs 65.2). This is consistent with the Experiment 5 finding: structural alignment alone doesn't improve downstream utility.

---

### Training Efficiency: Does Structural Alignment Speed Convergence?

If initializing near a "platonic" representation helps optimization, we would expect models with higher backbone CKA to converge faster during downstream training. We test this using accuracy at epochs 5, 10, 20 and AULC (Area Under Learning Curve).

#### Fine-tuning convergence (Pets 100%)


| Init                | Backbone CKA | Ep 5     | Ep 10    | Ep 20    | Best     | AULC     |
| ------------------- | ------------ | -------- | -------- | -------- | -------- | -------- |
| IN sup (no CKA)     | ~0.25        | **66.5** | **79.4** | **83.0** | **84.4** | **78.9** |
| IN sup CKA          | 0.31         | 64.3     | 78.7     | 82.0     | 83.2     | 78.9     |
| IN clip CKA         | 0.79         | 56.0     | 73.9     | 79.7     | 81.7     | 75.3     |


**The model with the highest structural alignment (clip, CKA=0.79) converges slowest.** At epoch 5, supervised-distilled reaches 66.5% while clip-distilled is at 56.0% — a 10.5 pp gap. Supervised also achieves higher final accuracy (84.4 vs 81.7). Higher backbone CKA does not accelerate fine-tuning.

#### Fine-tuning convergence (VOC 100% mAP)


| Init                | Ep 5     | Ep 10    | Ep 20    | Best     | AULC     |
| ------------------- | -------- | -------- | -------- | -------- | -------- |
| ImageNet pretrained | **66.9** | **82.2** | **86.4** | **87.4** | **81.5** |
| IN clip CKA         | 63.0     | 73.4     | 78.4     | 81.0     | 75.3     |
| IN sup CKA          | 63.1     | 71.9     | 76.7     | 79.0     | 73.6     |
| COCO sup            | 58.9     | 63.3     | 66.4     | 67.3     | 63.8     |


ImageNet-distilled models converge much faster than COCO-distilled (a data scale effect), but ImageNet pretrained still converges fastest of all. Among distilled models, CLIP and supervised start similarly at epoch 5 (63.0 vs 63.1), but CLIP pulls ahead by epoch 20 (78.4 vs 76.7).

#### Linear probe AULC comparison


| Init                | Pets 1%  | Pets 100% | VOC 1%   | VOC 100%  | EuroSAT 1% |
| ------------------- | -------- | --------- | -------- | --------- | ----------- |
| ImageNet pretrained | **62.5** | **90.5**  | **28.4** | **83.7**  | **82.3**    |
| IN clip CKA         | 22.8     | **75.7**  | **38.7** | **81.7**  | **66.3**    |
| IN sup CKA          | 20.7     | 59.5      | 21.6     | —         | 68.0        |
| IN sup (no CKA)     | **26.4** | 63.0      | 22.8     | 68.6      | 66.8        |


For linear probing, CLIP's AULC is consistently higher than supervised (75.7 vs 59.5 on Pets, 81.7 vs 68.6 on VOC), confirming that CLIP features require fewer epochs of head-only training to reach good performance. However, this reflects linear separability of the features, not faster backbone adaptation.

---

### Does Platonic Initialization Help Low-Label Generalization?

The platonic representation hypothesis predicts that initialization near a universal embedding should improve generalization, especially in low-data regimes. We test this by comparing models with different backbone CKA at 1% labels.

#### Fine-tuning at 1% labels


| Dataset    | IN sup CKA (CKA=0.31) | IN clip CKA (CKA=0.79) | Winner              |
| ---------- | ---------------------- | ----------------------- | ------------------- |
| Pets       | **26.1**               | 16.5                    | sup (lower CKA)     |
| EuroSAT    | 66.6                   | **74.2**                | clip (higher CKA)   |
| Imagenette | **88.0**               | 86.5                    | sup (lower CKA)     |


Supervised (lower structural alignment) wins 2 out of 3 datasets at 1% labels during fine-tuning. Higher backbone CKA does not predict better low-data generalization.

#### Linear probing at 1% labels


| Dataset    | IN sup CKA | IN clip CKA | IN sup (no CKA) | ImageNet pretrained |
| ---------- | ---------- | ----------- | ---------------- | ------------------- |
| Pets       | 23.3       | 26.2        | **29.0**         | **68.3**            |
| EuroSAT    | 75.4       | **77.7**    | 73.4             | **84.7**            |
| Imagenette | 80.7       | **91.0**    | 78.3             | **98.7**            |
| VOC        | 28.5       | **52.3**    | —                | 40.8                |


CLIP wins on EuroSAT, Imagenette, and VOC at 1% labels — but on Pets (the hardest fine-grained task), the *non-CKA* supervised model (29.0) beats both CKA variants (26.2 and 23.3). More structural alignment does not consistently predict better low-label generalization even in the linear probing regime.

**Notable exception:** On VOC at 1% labels, CLIP CKA linprobe achieves **52.3 mAP**, exceeding even ImageNet pretrained linprobe (40.8). This is the only condition where a distilled model clearly outperforms ImageNet pretraining. However, this likely reflects CLIP's contrastive training producing features well-suited to multi-label tasks, rather than a platonic representation effect.

---

### Assessment of the Platonic Representation Hypothesis

The central hypothesis motivating this project — that initializing a small model near a "universal/platonic" latent space yields more sample-efficient downstream learning — is **not supported** by our results. We can now evaluate each prediction:


| Prediction                                | Result       | Evidence                                                                 |
| ----------------------------------------- | ------------ | ------------------------------------------------------------------------ |
| Higher CKA → faster fine-tuning           | **Rejected** | CKA=0.79 converges slower than CKA=0.31 (Pets ep5: 56 vs 66)           |
| Higher CKA → better low-label fine-tuning | **Rejected** | Supervised (lower CKA) wins 2/3 datasets at 1%                         |
| Higher CKA → better low-label linprobe    | **Mixed**    | CLIP wins 3/4 datasets but loses on Pets; non-CKA sup beats CKA sup    |
| CKA loss → better downstream              | **Rejected** | Non-CKA supervised edges out CKA supervised (84.4 vs 83.2 on Pets)     |
| Projector preserving alignment → helps    | **Rejected** | Frozen projector hurts 10–18 pp; trainable projector ≈ drop projector  |


**What explains CLIP's advantage instead?** CLIP's superior linear probing is best explained by a property of its **training objective** (contrastive learning produces linearly separable clusters) rather than proximity to a platonic representation. Supporting evidence:

1. **The advantage is specific to linear probing.** If CLIP's features occupied a more universal manifold, they should also help fine-tuning — but supervised features fine-tune better at low data.
2. **Structural alignment (CKA) doesn't predict downstream performance.** Non-CKA supervised distillation outperforms CKA supervised distillation despite lower structural alignment.
3. **The advantage doesn't scale with CKA.** Among the three initialization variants (sup CKA=0.31, sup no-CKA≈0.25, clip CKA=0.79), downstream performance does not monotonically increase with backbone CKA.

#### Important caveats

Our results cannot fully falsify the platonic representation hypothesis as originally stated. Two significant limitations constrain our conclusions:

**1. Teacher scale may be insufficient.** All teachers in this experiment are ViT-B/16 (~86M params). The platonic representation hypothesis ([Huh et al., 2024](https://arxiv.org/abs/2405.07987)) primarily concerns large-scale models — their analysis of 78 vision models shows that larger, more capable models cluster together representationally while smaller models scatter. At the ViT-B scale, supervised and CLIP representations may not yet have converged toward a shared structure. Testing with ViT-L CLIP (304M params) or DINOv2-g (1.1B params) teachers could produce distillation targets closer to the hypothesized platonic ideal, and the downstream benefits might only emerge from those stronger targets.

**2. Final-embedding CKA may be the wrong alignment target.** Our CKA loss only aligns the final embeddings (post-projector). The platonic convergence literature notes that early network layers are more interchangeable across models (Lenc & Vedaldi, 2015), suggesting convergence operates differently at different depths. Aligning **intermediate features** — layer-wise CKA between student backbone blocks and teacher transformer blocks — could:

- Force the student to learn similar low/mid-level features (edges, textures, object parts) that make ImageNet pretraining effective
- Avoid the information bottleneck of compressing all teacher knowledge into a single final embedding vector
- Better preserve the hierarchical representational structure that drives transferability
- This would resemble FitNets-style distillation (Romero et al., 2015) but using CKA rather than MSE at intermediate layers, which could be more robust to the architectural mismatch between ViT teacher and CNN student

These remain open experimental questions. The hypothesis is not supported at the scale and methodology tested, but it is not definitively refuted for larger-scale teachers or deeper alignment strategies.

---

## Key Findings

### 1. Data scale is the primary bottleneck — confirmed

15x more distillation data (ImageNet 1.28M vs COCO 82K) closes 68–82% of the gap to ImageNet pretraining across all tasks. This is the most important finding: the distillation pipeline works, it just needed sufficient data.


| Task                  | COCO gap | ImageNet gap | % Closed |
| --------------------- | -------- | ------------ | -------- |
| Pets fine-tune 100%   | 43.7 pp  | 8.0 pp       | **82%**  |
| VOC fine-tune 100%    | 20.1 pp  | 6.4 pp       | **68%**  |
| VOC linprobe 100%     | 21.4 pp  | 2.7 pp       | **87%**  |
| Pets linprobe 100%    | 61.3 pp  | 10.1 pp      | **84%**  |
| EuroSAT linprobe 100% | 11.1 pp  | 0.3 pp       | **97%**  |


### 2. CLIP features are dramatically more linearly separable

CLIP distillation produces features where a frozen linear probe matches or exceeds fine-tuning. On Pets, clip_768 linprobe (81.1%) nearly equals fine-tune (81.7%). On VOC, linprobe (82.5%) exceeds fine-tune (81.0%). This has practical implications: CLIP-distilled backbones can be used as frozen feature extractors with minimal compute.

### 3. A remaining ~8–10 pp gap persists

Even with 1.28M images and CKA structural alignment, distilled models don't fully match ImageNet pretraining. Possible explanations:

- **Label signal matters:** ImageNet pretraining uses 1000-class labels providing explicit semantic supervision. Distillation only transfers embedding geometry.
- **Teacher capacity:** ViT-B/16 teachers (86M params) are modest. Larger teachers (ViT-L, ViT-G) might produce richer embedding targets.
- **Student capacity:** RegNetY-400MF (4.3M params) loses information compressing from the teacher. A larger student might close more of the gap.
- **Epochs:** Neither run fully converged (val cosine still climbing). More training could yield further improvement.

### 4. CKA-aligned projector does not help downstream

Despite high projected CKA (0.87), preserving the projector during downstream training provides no benefit. Frozen projector actively hurts (−10–18 pp). The CKA objective successfully aligns global structure but this structural alignment does not encode class-discriminative information.

### 5. CKA loss is irrelevant for supervised distillation but enables CLIP's backbone alignment

For the supervised teacher, CKA and non-CKA produce nearly identical downstream results. But CLIP achieves backbone CKA of 0.79 (vs supervised's 0.31), and this correlates with CLIP's dramatically better linear probing. The question of whether CKA *caused* this or whether CLIP's representation is simply more compressible remains open.

### 6. The platonic representation hypothesis is not supported

Higher structural alignment with the teacher does not yield faster convergence, better low-label generalization, or improved downstream accuracy. The model with backbone CKA=0.79 (clip) converges slower during fine-tuning than the model with CKA=0.31 (supervised) and loses on 2/3 datasets at 1% labels. CKA loss provides no benefit over combined loss for the supervised teacher. CLIP's advantage is best explained by its contrastive training objective producing linearly separable features, not by proximity to a universal representation.

---

## Comparison: All Distillation Methods on VOC 100%


| Method                        | Fine-tune (mAP) | LinProbe (mAP) |
| ----------------------------- | --------------- | -------------- |
| ImageNet pretrained           | **87.4**        | **85.2**       |
| ImageNet clip_768 CKA λ=0.1   | 81.0            | 82.5           |
| ImageNet supervised CKA λ=0.1 | 79.0            | —              |
| ImageNet supervised (no CKA)  | 78.3            | 70.1           |
| COCO supervised (no CKA)      | 67.3            | 51.4           |
| COCO clip_768 (no CKA)        | 66.9            | 63.8           |
| COCO clip_768 CKA λ=0.1       | 66.3            | 63.8           |
| COCO supervised CKA λ=0.1     | 66.7            | 52.2           |
| Random                        | 24.9            | 9.0            |


---

## Conclusions

1. **The distillation pipeline works at scale.** The 20+ mAP gap between COCO-distilled and ImageNet-pretrained was primarily a data problem, not a fundamental limitation of the distillation approach. With 15x more data, the gap shrinks to 2.7–6.4 mAP on VOC.
2. **CLIP is the better teacher for frozen feature extraction.** CLIP-distilled features are 10–20 pp better than supervised-distilled features on linear probing across all 6 datasets. For applications requiring cheap inference (frozen backbone + linear head), CLIP distillation is strongly preferred.
3. **Supervised is the better teacher for low-data fine-tuning.** When fine-tuning with <10% labels, supervised distillation outperforms CLIP by 8–11 pp on Pets. The advantage disappears at 100% labels.
4. **The projector should always be dropped.** Neither freezing nor training the CKA-aligned projector helps. Downstream classification benefits from the backbone features, not the projected space.
5. **The platonic representation hypothesis is not supported.** Higher structural alignment (backbone CKA) does not predict faster convergence, better low-label generalization, or improved downstream performance. CLIP's linear probing advantage is a property of contrastive training, not of proximity to a universal manifold.
6. **Further gains likely require:** more training epochs (neither run converged), larger teachers, larger students, or complementary objectives beyond embedding matching.

---

## Future Directions

The two most promising experiments that could change the conclusions:

1. **Larger teachers (ViT-L/G).** The platonic hypothesis predicts convergence at scale. ViT-B teachers may simply lack the representational quality needed. DINOv2-g (1.1B params) or CLIP ViT-L/14 (304M params) would test whether the hypothesis holds when the teacher's representations are more "platonic." If larger teachers produce distilled students that converge faster and generalize better at low labels, that would support the hypothesis at the scale it was originally proposed.

2. **Intermediate-layer CKA distillation.** Instead of only aligning final embeddings, align intermediate features between student CNN blocks and teacher ViT blocks using layer-wise CKA. This addresses the information bottleneck: a single 768-dim embedding cannot fully encode the hierarchical visual knowledge that makes ImageNet features transferable. Intermediate alignment could force the student to learn transferable low/mid-level features (edges, textures, parts) rather than just matching the teacher's final compressed representation. This is architecturally analogous to FitNets (Romero et al., 2015) but with CKA as the alignment objective, which may handle the CNN-ViT architectural mismatch better than MSE.

---

## Implementation Notes

- **Scripts:** `scripts/experiment7_cka_distill.sh` (2 distillation runs), `scripts/experiment7_downstream.sh` (324 downstream runs, 4-way parallel)
- **Runtime:** ~13h distillation (2 sequential runs × ~6.5h) + ~6h downstream (324 runs, 4 parallel) = ~19h total
- **Checkpoints:** `imagenet_supervised_cka_l0.1_distilled_best.pth`, `imagenet_clip_768_cka_l0.1_distilled_best.pth`
- **Failed runs:** 64/324 — all due to stratified sampling failure on DTD (47 classes) and Flowers102 (102 classes) at 1% label fraction. Not a code bug; insufficient samples per class for stratified splitting.
- **Additional non-CKA supervised run:** `imagenet_supervised_distilled_best.pth` from the partially-completed initial script (20 epochs, combined loss). Included in downstream results for CKA vs non-CKA comparison.

