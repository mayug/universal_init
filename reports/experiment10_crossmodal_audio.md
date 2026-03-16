# Experiment 10: Cross-Modal Audio Distillation via Text Geometry

## Summary

We test whether text-based embedding geometry transfers cross-modally to audio — a modality the teacher has never seen. A MobileNetV3 audio student learns to match teacher text embeddings (from audio captions), then evaluates on ESC-50 environmental sound classification.

**Key finding:** Raw CLIP text distillation fails catastrophically for audio (avg -40pp vs SBERT), but **whitening CLIP embeddings completely fixes this** — whitened CLIP matches SBERT (3-3 win split, avg diff <1pp). The root cause is CLIP's "embedding cone" problem: AudioCaps captions cluster in a narrow cone (0.67 mean pairwise cosine sim) because CLIP treats them as similar visual scenes. Whitening (subtract mean, divide by std, re-normalize) spreads the embeddings and recovers the discriminative geometry underneath.

## Setup

### Models

| Role | Model | Embed Dim | Parameters |
|------|-------|-----------|------------|
| Teacher A | CLIP ViT-L/14 text encoder | 768 | frozen |
| Teacher B | Sentence-BERT (all-mpnet-base-v2) | 768 | frozen |
| Student | MobileNetV3-Large (audio) | 960 → 768 | 3.0M |

### Distillation

- **Data:** AudioCaps (~46K audio-caption pairs, 10s clips)
- **Protocol:** Student processes audio waveforms → mel spectrogram → backbone → projector → L2-normalized embedding. Teacher processes the corresponding text caption → L2-normalized embedding. Cosine embedding loss.
- **Training:** 20 epochs, BS=256, lr=1e-3 (cosine), AdamW

### Distillation Results

| Teacher | Val Cosine Similarity |
|---------|----------------------|
| CLIP text (raw) | **0.861** |
| CLIP text (whitened) | 0.423 |
| Sentence-BERT (raw) | 0.724 |
| Sentence-BERT (whitened) | 0.469 |

Note: Higher alignment to raw CLIP did NOT translate to better downstream features. Whitened CLIP has lower alignment but much better geometry.

### Downstream Evaluation

- **Dataset:** ESC-50 (50 environmental sound classes, 2000 clips, 5-fold CV)
- **Conditions:** Random init (A), CLIP text distilled (B), CLIP text whitened (D), AudioSet pretrained ceiling (C), Sentence-BERT distilled (E)
- **Label fractions:** 100%, 10%, 1%
- **Modes:** Fine-tune, Linear probe

## Results

### ESC-50 5-Fold Cross-Validation Accuracy (%)

| Condition | 100% FT | 100% LP | 10% FT | 10% LP | 1% FT | 1% LP |
|-----------|---------|---------|--------|--------|-------|-------|
| A: Random | 70.2 | 13.4 | 13.8 | 5.2 | 8.2 | 3.8 |
| B: CLIP text (raw) | 75.0 | 4.2 | 19.5 | 3.4 | 8.1 | 9.3 |
| D: CLIP text (whitened) | 83.9 | 75.6 | 57.9 | 56.6 | **43.6** | **43.6** |
| E: SBERT (raw) | **86.0** | 74.5 | **60.8** | **58.0** | 41.0 | 41.8 |
| F: SBERT (whitened) | 82.4 | 73.2 | 54.4 | 52.9 | 32.2 | 36.2 |
| C: AudioSet pretrained | **96.9** | **89.2** | **73.8** | 54.7 | 34.8 | 30.3 |

### Head-to-Head: Raw CLIP vs Sentence-BERT

| Setting | CLIP raw | SBERT | Winner | Delta |
|---------|----------|-------|--------|-------|
| 100% Fine-tune | 75.0 | 86.0 | SBERT | +11.0pp |
| 100% Linear Probe | 4.2 | 74.5 | SBERT | +70.2pp |
| 10% Fine-tune | 19.5 | 60.8 | SBERT | +41.3pp |
| 10% Linear Probe | 3.4 | 58.0 | SBERT | +54.7pp |
| 1% Fine-tune | 8.1 | 41.0 | SBERT | +33.0pp |
| 1% Linear Probe | 9.3 | 41.8 | SBERT | +32.5pp |

**SBERT wins 6/6 comparisons**, average margin: **+40.4 percentage points**.

### Head-to-Head: Whitened CLIP vs Sentence-BERT

| Setting | CLIP-W | SBERT | Winner | Delta |
|---------|--------|-------|--------|-------|
| 100% Fine-tune | 83.9 | 86.0 | SBERT | +2.2pp |
| 100% Linear Probe | 75.6 | 74.5 | CLIP-W | +1.1pp |
| 10% Fine-tune | 57.9 | 60.8 | SBERT | +2.9pp |
| 10% Linear Probe | 56.6 | 58.0 | SBERT | +1.5pp |
| 1% Fine-tune | 43.6 | 41.0 | CLIP-W | +2.6pp |
| 1% Linear Probe | 43.6 | 41.8 | CLIP-W | +1.8pp |

**CLIP-W wins 3/6, SBERT wins 3/6** — essentially tied (avg diff: -0.2pp). Whitening completely closes the 40pp gap.

### Impact of Whitening on CLIP

| Setting | Raw CLIP | Whitened CLIP | Delta |
|---------|----------|---------------|-------|
| 100% Fine-tune | 75.0 | 83.9 | +8.9pp |
| 100% Linear Probe | 4.2 | 75.6 | **+71.4pp** |
| 10% Fine-tune | 19.5 | 57.9 | +38.4pp |
| 10% Linear Probe | 3.4 | 56.6 | +53.2pp |
| 1% Fine-tune | 8.1 | 43.6 | +35.5pp |
| 1% Linear Probe | 9.3 | 43.6 | +34.3pp |

Average improvement: **+40.3 percentage points**. Whitening transforms CLIP from useless to competitive.

### Impact of Whitening on SBERT

| Setting | Raw SBERT | Whitened SBERT | Delta |
|---------|-----------|----------------|-------|
| 100% Fine-tune | 86.0 | 82.4 | -3.6pp |
| 100% Linear Probe | 74.5 | 73.2 | -1.3pp |
| 10% Fine-tune | 60.8 | 54.4 | -6.4pp |
| 10% Linear Probe | 58.0 | 52.9 | -5.2pp |
| 1% Fine-tune | 41.0 | 32.2 | -8.8pp |
| 1% Linear Probe | 41.8 | 36.2 | -5.7pp |

Average change: **-5.2 percentage points**. Whitening hurts SBERT across all settings.

### Whitening Asymmetry

| Teacher | Raw pairwise cosine | Whitening effect | Avg change |
|---------|--------------------:|:----------------:|:----------:|
| CLIP text | 0.663 (cone) | Fixes cone | **+40.3pp** |
| SBERT | 0.256 (spread) | Distorts geometry | **-5.2pp** |

Whitening is a targeted fix for the cone problem, not a universal improvement. SBERT's embeddings are already well-distributed; whitening disrupts their natural geometric structure. This confirms the diagnosis: CLIP's audio failure was specifically caused by the narrow embedding cone, not by a fundamental deficiency in CLIP's learned semantics.

### Gap Closure vs AudioSet Pretrained Ceiling

At 1% labels (most relevant for sample efficiency):

| Condition | Fine-tune | vs Ceiling (34.8%) | Linear Probe | vs Ceiling (30.3%) |
|-----------|-----------|-------------------|--------------|-------------------|
| Random | 8.2% | 0% closed | 3.8% | 0% closed |
| CLIP raw | 8.1% | 0% closed | 9.3% | 21% closed |
| CLIP whitened | **43.6%** | **133% (exceeds ceiling)** | **43.6%** | **150% (exceeds ceiling)** |
| SBERT distilled | **41.0%** | **123% (exceeds ceiling)** | **41.8%** | **143% (exceeds ceiling)** |

**Both whitened CLIP and SBERT distillation exceed AudioSet pretraining at 1% labels.** Text-based geometry — from models that have never heard audio — produces features that outperform 2M labeled audio clips when supervision is scarce.

## Analysis

### Why Does Raw CLIP Fail for Audio?

The root cause is the **embedding cone problem**: CLIP's text encoder maps all AudioCaps captions into a narrow cone because they describe similar everyday scenes.

| Metric | CLIP raw | CLIP whitened | SBERT |
|--------|----------|---------------|-------|
| Mean pairwise cosine sim | **0.663** | 0.001 | 0.256 |
| Std of pairwise sim | 0.095 | 0.119 | **0.148** |
| Per-dim embedding variance | 0.000426 | — | **0.000979** (2.3x) |

CLIP maps "a dog barking while wind blows," "someone playing piano," and "water flowing and birds chirping" into nearly the same embedding (0.66 mean pairwise cosine). The student achieves 0.86 alignment by learning to project all audio to roughly the same point — high alignment, zero discriminative power.

**The cone is domain-specific.** On COCO image captions, CLIP's mean pairwise cosine is only 0.36 — much more spread. The cone only appears for AudioCaps because CLIP's text encoder, trained to match images, treats all audio descriptions as "everyday scene descriptions" without strong differentiation.

### Why Does Whitening Fix It?

Whitening (subtract mean, divide by per-dimension std, re-normalize to unit sphere) transforms CLIP's narrow cone into a well-spread distribution:

| Metric | CLIP raw | CLIP whitened |
|--------|----------|---------------|
| Mean pairwise cosine | 0.663 | 0.001 |
| Rank correlation with raw | 1.000 | 0.484 |
| Rank correlation with SBERT | 0.399 | 0.506 |

Key observations:
1. **Whitening doesn't just rescale — it changes relative geometry** (rank correlation 0.48 with raw). The discriminative structure was partially obscured by the dominant shared direction.
2. **Whitened CLIP's geometry is closer to SBERT's** (0.51 rank correlation) than raw CLIP is (0.40). This explains why downstream performance converges.
3. **The underlying discriminative information was always there** — whitening extracts it by removing the uninformative shared component. This is analogous to PCA-whitening in classical feature engineering.

### Implications for the Platonic Representation Hypothesis

The vision experiments (Exp 7-9) showed CLIP's geometry consistently outperformed supervised geometry for vision tasks, supporting the "platonic" / universal representation hypothesis. The raw CLIP audio results initially appeared to be a **counterexample** — but whitening reveals a more nuanced picture:

- **CLIP's discriminative geometry IS useful for audio** — it was just compressed into a narrow subspace by the dominant vision-centric shared direction
- **The "platonic" structure is present but needs extraction** for out-of-distribution domains. Whitening is one simple extraction method.
- **After whitening, CLIP ≈ SBERT for audio** — the underlying semantic content is roughly equivalent, supporting the platonic hypothesis that different models converge on similar semantic structure
- **The practical implication**: when applying CLIP to a new domain, always check for cone collapse and whiten if needed

### Text Distillation Exceeding AudioSet Pretraining

At 1% labels, both whitened CLIP (43.6%) and SBERT (41.0%) exceed AudioSet-pretrained features (34.8% FT). This is remarkable because:
- AudioSet pretraining uses 2M labeled audio clips vs AudioCaps' 46K audio-caption pairs
- Neither text teacher has ever processed audio
- This advantage disappears at higher label fractions (AudioSet catches up by 10%, dominates at 100%)

The explanation: AudioSet pretraining learns audio-specific discriminative features optimized for its 527-class ontology. Text distillation learns semantically structured features that generalize better with minimal supervision. With 1% labels (1 sample per class), semantic structure matters more than audio-specific features.

## Technical Notes

- **Whitening procedure:** Compute per-dimension mean and std over all AudioCaps training caption embeddings (45K samples). At distillation time: `emb = (emb - mean) / (std + 1e-8)`, then L2-normalize. Stats stored in `checkpoints/clip_text_audiocaps_whitening_stats.pt`.
- **BatchNorm calibration:** Audio mel spectrograms have statistics far from BN defaults (mean=0, var=1). We added BN running stat recalibration before each evaluation pass, computing cumulative (not EMA) statistics from training data. Without this, eval-mode predictions collapsed to a single class.
- **AudioSet pretrained weights:** EfficientAT mn10_as weights required key remapping (adding `backbone.` prefix) and SE layer reshaping (Linear [out,in] → Conv2d [out,in,1,1]) to be compatible with torchvision's MobileNetV3.
- **1% label fraction:** With 50 classes and only ~16 training samples, we use 1 sample per class instead of stratified split (which requires ≥1 sample per class per split).
- **AMP disabled:** Mixed precision caused NaN overflow in log-mel computation. Not needed on B200 GPU (183GB).
