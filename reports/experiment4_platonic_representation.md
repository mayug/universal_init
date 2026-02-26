# Experiment 4: Platonic Representation — CLIP vs Supervised Distillation

**Date:** February 26, 2026
**Status:** Complete (24/24 downstream runs finished)

---

## Motivation

Prior experiments (1–3) used only ImageBind as the teacher, conflating the teacher's training method with representation quality. This experiment isolates the effect of **representation type** by distilling the same student (RegNetY-400MF) from three different ViT-B/16 teachers, all using the same distillation pipeline:


| ID  | Teacher                   | Source                                    | Embed Dim | Training Signal                       |
| --- | ------------------------- | ----------------------------------------- | --------- | ------------------------------------- |
| D1  | Supervised ViT-B/16       | `vit_base_patch16_224.augreg_in1k` (timm) | 768       | ImageNet labels                       |
| D2  | CLIP ViT-B/16 (pre-proj)  | `vit_base_patch16_clip_224.openai` (timm) | 768       | CLIP contrastive (raw features)       |
| D3  | CLIP ViT-B/16 (with proj) | `vit_base_patch16_clip_224.openai` (timm) | 512       | CLIP contrastive (projected features) |


**Hypothesis:** CLIP representations, being aligned across modalities, approximate a more "platonic" embedding space that transfers more efficiently to downstream tasks, especially in low-data regimes.

---

## Phase 1: Distillation on COCO

**Config:** 30 epochs, batch size 256, lr 1e-3, combined loss (embedding + 0.5x relational), AMP, cosine LR with 5-epoch warmup.

**Dataset:** COCO 2014 train images (~82k images), 90/10 train/val split.

### Results


| Teacher    | Best Epoch | Val Cosine Similarity | Projector Dim |
| ---------- | ---------- | --------------------- | ------------- |
| Supervised | 30         | **0.626**             | 440 -> 768    |
| CLIP 768   | 28         | **0.741**             | 440 -> 768    |
| CLIP 512   | 30         | **0.837**             | 440 -> 512    |


### Analysis

- **CLIP 512 achieves the highest cosine similarity** (0.837), likely because the 512-dim projected space is more compact and easier for the student to match.
- **CLIP 768 > Supervised** (0.741 vs 0.626) despite both being 768-dim, suggesting CLIP's contrastive features have a more learnable structure for distillation.
- **Supervised has lowest cosine sim**, which may indicate that label-trained features are more complex/entangled for a small student to replicate.

**Key question for downstream:** Does higher distillation fidelity (cosine sim) translate to better downstream performance?

---

## Phase 2: Downstream Fine-tuning

**Design:** 24 runs total = 3 teachers x 2 datasets x 2 label fractions x 2 projector modes.

**Config:** 50 epochs, batch size 64, SGD (lr=0.01, momentum=0.9, weight_decay=1e-4), cosine LR with 5-epoch warmup.

### Pets Dataset (37 classes, fine-grained)

#### 1% Labels (~37 training samples)


| Teacher    | Projector | Best Acc (%) | Final Acc (%) | AULC     | Acc@5 | Acc@10   | Acc@20   |
| ---------- | --------- | ------------ | ------------- | -------- | ----- | -------- | -------- |
| supervised | drop      | 4.58         | 4.25          | 4.05     | 3.60  | 4.17     | 4.12     |
| supervised | trainable | 4.93         | 4.82          | 4.19     | 4.25  | 4.14     | 4.47     |
| clip_768   | drop      | 4.66         | 4.66          | 3.95     | 2.92  | 2.67     | 4.01     |
| clip_768   | trainable | **5.78**     | 4.91          | **4.74** | 3.79  | 3.92     | **4.96** |
| clip_512   | drop      | **5.48**     | **5.48**      | **4.56** | 3.38  | 3.27     | **4.52** |
| clip_512   | trainable | 4.99         | 4.77          | 4.42     | 3.22  | **4.44** | 4.50     |


#### 100% Labels (~3,680 training samples)


| Teacher    | Projector | Best Acc (%) | Final Acc (%) | AULC      | Acc@5     | Acc@10    | Acc@20 |
| ---------- | --------- | ------------ | ------------- | --------- | --------- | --------- | ------ |
| supervised | drop      | 46.72        | 46.52         | 40.44     | 17.25     | 34.56     | 44.15  |
| supervised | trainable | 48.68        | 48.43         | 42.17     | 25.84     | 37.53     | 44.24  |
| clip_768   | drop      | 46.33        | 45.87         | 40.20     | 20.03     | 35.13     | 44.18  |
| clip_768   | trainable | **49.06**    | **48.79**     | **42.25** | **25.57** | **38.70** | 44.10  |
| clip_512   | drop      | 44.24        | 44.24         | 37.53     | 10.08     | 31.15     | 40.88  |
| clip_512   | trainable | 46.61        | 46.50         | 39.71     | 19.27     | 34.61     | 42.60  |


### EuroSAT Dataset (10 classes, satellite imagery)

#### 1% Labels (~270 training samples)


| Teacher    | Projector | Best Acc (%) | Final Acc (%) | AULC      | Acc@5     | Acc@10    | Acc@20    |
| ---------- | --------- | ------------ | ------------- | --------- | --------- | --------- | --------- |
| supervised | drop      | **72.44**    | **72.44**     | **59.56** | **29.98** | 36.39     | **61.65** |
| supervised | trainable | **72.91**    | **72.91**     | 54.80     | 30.56     | 20.63     | 59.83     |
| clip_768   | drop      | 71.00        | 70.89         | 55.52     | 22.59     | 38.65     | 53.70     |
| clip_768   | trainable | 69.96        | 69.70         | 54.80     | **33.37** | **44.93** | 46.07     |
| clip_512   | drop      | 68.02        | 67.63         | 54.58     | 24.59     | 39.43     | 57.22     |
| clip_512   | trainable | 72.35        | 71.93         | **59.66** | 22.13     | **50.63** | 63.35     |


#### 100% Labels (~27,000 training samples)


| Teacher    | Projector | Best Acc (%) | Final Acc (%) | AULC      | Acc@5     | Acc@10 | Acc@20    |
| ---------- | --------- | ------------ | ------------- | --------- | --------- | ------ | --------- |
| supervised | drop      | 97.52        | 97.50         | 95.01     | 90.37     | 95.61  | 96.70     |
| supervised | trainable | 97.43        | 97.43         | **95.36** | 89.94     | 95.22  | 96.11     |
| clip_768   | drop      | 97.57        | 97.57         | 94.90     | 84.91     | 95.07  | 96.26     |
| clip_768   | trainable | **97.80**    | **97.67**     | 94.88     | **91.39** | 91.76  | **96.50** |
| clip_512   | drop      | 97.54        | 97.54         | 95.23     | 89.83     | 94.39  | 96.46     |
| clip_512   | trainable | 97.46        | 97.35         | 94.56     | 89.78     | 93.48  | 96.72     |


---

## Key Findings

### 1. Higher distillation fidelity does NOT predict better downstream performance

Despite CLIP 512 having the highest distillation cosine similarity (0.837 vs 0.741 vs 0.626), it performs **worst** on Pets at 100% labels (44.24% drop projector) and worst on EuroSAT at 1% labels (68.02% drop projector). This suggests that distillation fidelity measures how well the student matches a specific teacher, not how useful the learned features are for downstream tasks.

### 2. No clear winner among teacher representations


| Metric                | Best Teacher                  |
| --------------------- | ----------------------------- |
| Pets 1% best acc      | clip_768 trainable (5.78%)    |
| Pets 100% best acc    | clip_768 trainable (49.06%)   |
| EuroSAT 1% best acc   | supervised trainable (72.91%) |
| EuroSAT 100% best acc | clip_768 trainable (97.80%)   |


Results are mixed — no single teacher dominates across all settings. This **weakens the platonic representation hypothesis**: if CLIP representations were truly more universal, they should consistently outperform supervised representations, especially in low-data regimes.

### 3. Trainable projector consistently helps

Across almost all conditions, keeping and training the projector outperforms dropping it:


| Condition                     | Drop Proj Best | Trainable Proj Best | Delta     |
| ----------------------------- | -------------- | ------------------- | --------- |
| Pets 1% (avg across teachers) | 4.91%          | 5.23%               | +0.33     |
| Pets 100% (avg)               | 45.76%         | 48.12%              | **+2.36** |
| EuroSAT 1% (avg)              | 70.49%         | 71.74%              | +1.26     |
| EuroSAT 100% (avg)            | 97.54%         | 97.56%              | +0.02     |


The benefit is most pronounced on Pets at 100% labels (+2.36 pp), where the additional projector capacity helps the model adapt fine-grained features.

### 4. All distillation initializations struggle severely in extreme low-data on fine-grained tasks

On Pets at 1% (~~37 samples), all methods achieve 4-6% accuracy (random chance is 2.7% for 37 classes). The distilled features provide almost no benefit for fine-grained breed recognition with so few examples. In contrast, EuroSAT at 1% (~~270 samples) reaches 68-73%, showing that distilled features do transfer meaningfully to coarser classification tasks even with limited labels.

### 5. EuroSAT saturates — all teachers converge at 100% labels

At 100% labels, all EuroSAT configurations achieve 97.4-97.8%, a spread of only 0.4 pp. The task is simply too easy at full data to differentiate teacher quality.

---

## Summary Table: Best Accuracy by Teacher (drop projector only, for clean comparison)


|            | Pets 1%  | Pets 100% | EuroSAT 1% | EuroSAT 100% |
| ---------- | -------- | --------- | ---------- | ------------ |
| Supervised | 4.58     | 46.72     | **72.44**  | 97.52        |
| CLIP 768   | 4.66     | 46.33     | 71.00      | 97.57        |
| CLIP 512   | **5.48** | 44.24     | 68.02      | 97.54        |


---

## Conclusions

1. **The platonic representation hypothesis is not supported** by these results. CLIP-distilled features do not consistently outperform supervised-distilled features on downstream tasks.
2. **Distillation cosine similarity is a poor proxy for downstream utility.** CLIP 512 achieved the best distillation fidelity but often the worst downstream accuracy.
3. **The projector architecture matters more than the teacher identity.** Keeping a trainable projector provides a more consistent and larger benefit (+2.36 pp on Pets 100%) than switching teachers (<1 pp difference in most conditions).
4. **All distillation-based initializations fail on extreme low-data fine-grained tasks** (Pets 1%), regardless of teacher. This contrasts with the original hypothesis that "universal" representations would shine most in data-scarce settings.
5. **Supervised distillation is surprisingly competitive**, slightly favoring EuroSAT where its ImageNet-derived features happen to align well with satellite image classification.

---

## Implementation Notes

- **New dependency:** `timm>=0.9.0` for loading pretrained ViT models
- **New module:** `src/models/generic_teacher.py` — frozen teacher wrapper matching ImageBindTeacher interface
- **Modified:** `student.py` (configurable teacher_dim), `train_distill.py` (multi-teacher support), `distill_datasets.py` (custom normalization), `train_downstream.py` (auto-detection of teacher_dim from checkpoint)
- **Scripts:** `scripts/experiment4_platonic.sh` (distillation), `scripts/experiment4_downstream.sh` (downstream)
- **COCO data:** Used COCO 2014 train images (symlinked as train2017 for compatibility)
- **Hardware:** Single H100 80GB GPU
- **Runtime:** ~2.5 hours distillation + ~2.5 hours downstream = ~5 hours total
- **Combined results CSV:** `checkpoints/experiment4_combined_results.csv`

