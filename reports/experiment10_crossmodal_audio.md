# Experiment 10: Cross-Modal Audio Distillation via Text Geometry

## Summary

We test whether text-based embedding geometry transfers cross-modally to audio — a modality the teacher has never seen. A MobileNetV3 audio student learns to match teacher text embeddings (from audio captions), then evaluates on ESC-50 environmental sound classification.

**Key finding:** Sentence-BERT distillation massively outperforms CLIP text distillation (6/6 comparisons, avg +40pp), reversing the pattern seen in vision experiments where CLIP consistently dominates. This suggests CLIP's "platonic" geometry is vision-centric rather than truly universal.

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
| CLIP text | **0.861** |
| Sentence-BERT | 0.724 |

Note: Higher alignment to CLIP did NOT translate to better downstream features.

### Downstream Evaluation

- **Dataset:** ESC-50 (50 environmental sound classes, 2000 clips, 5-fold CV)
- **Conditions:** Random init (A), CLIP text distilled (B), AudioSet pretrained ceiling (C), Sentence-BERT distilled (E)
- **Label fractions:** 100%, 10%, 1%
- **Modes:** Fine-tune, Linear probe

## Results

### ESC-50 5-Fold Cross-Validation Accuracy (%)

| Condition | 100% FT | 100% LP | 10% FT | 10% LP | 1% FT | 1% LP |
|-----------|---------|---------|--------|--------|-------|-------|
| A: Random | 70.2 | 13.4 | 13.8 | 5.2 | 8.2 | 3.8 |
| B: CLIP text distilled | 75.0 | 4.2 | 19.5 | 3.4 | 8.1 | 9.3 |
| E: SBERT distilled | **86.0** | **74.5** | **60.8** | **58.0** | **41.0** | **41.8** |
| C: AudioSet pretrained | **96.9** | **89.2** | **73.8** | 54.7 | 34.8 | 30.3 |

### Head-to-Head: CLIP vs Sentence-BERT

| Setting | CLIP | SBERT | Winner | Delta |
|---------|------|-------|--------|-------|
| 100% Fine-tune | 75.0 | 86.0 | SBERT | +11.0pp |
| 100% Linear Probe | 4.2 | 74.5 | SBERT | +70.2pp |
| 10% Fine-tune | 19.5 | 60.8 | SBERT | +41.3pp |
| 10% Linear Probe | 3.4 | 58.0 | SBERT | +54.7pp |
| 1% Fine-tune | 8.1 | 41.0 | SBERT | +33.0pp |
| 1% Linear Probe | 9.3 | 41.8 | SBERT | +32.5pp |

**SBERT wins 6/6 comparisons**, average margin: **+40.4 percentage points**.

### Gap Closure vs AudioSet Pretrained Ceiling

At 1% labels (most relevant for sample efficiency):

| Condition | Fine-tune | vs Ceiling (34.8%) | Linear Probe | vs Ceiling (30.3%) |
|-----------|-----------|-------------------|--------------|-------------------|
| Random | 8.2% | 0% closed | 3.8% | 0% closed |
| CLIP distilled | 8.1% | 0% closed | 9.3% | 21% closed |
| SBERT distilled | **41.0%** | **123% (exceeds ceiling)** | **41.8%** | **143% (exceeds ceiling)** |

**SBERT distillation exceeds AudioSet pretraining at 1% labels.** This is surprising: a text model that has never heard audio produces features that, at extreme low-data, outperform features from a model trained on 2M labeled audio clips.

## Analysis

### Why Does SBERT Beat CLIP for Audio?

1. **CLIP's geometry is vision-centric.** CLIP's text encoder is trained to match images. For a caption like "A dog barking while wind blows," CLIP's embedding reflects visual content (what a dog looks like, what wind looks like) rather than acoustic content. Sentence-BERT captures pure textual semantics — semantic similarity between "dog barking" and "animal sounds" — which maps more directly to audio category structure.

2. **Higher distillation alignment ≠ better features.** CLIP achieved 0.861 cosine similarity (vs 0.724 for SBERT) during distillation, meaning the student more closely matched CLIP's geometry. Yet this geometry was less useful for audio classification. The student faithfully learned CLIP's vision-biased embedding structure, which is poorly suited for audio.

3. **SBERT's semantic clustering helps audio.** ESC-50 categories are defined semantically ("dog bark", "rain", "clock tick"). SBERT naturally clusters semantically related text, and this clustering transfers well to audio: sounds that have similar text descriptions end up with similar embeddings. CLIP's clustering is instead optimized for visual discrimination.

### Root Cause: CLIP's Caption Embeddings Are Clustered in a Narrow Cone

To understand the 40pp gap, we measured the pairwise cosine similarity of 500 AudioCaps caption embeddings under each teacher:

| Metric | CLIP text | SBERT |
|--------|-----------|-------|
| Mean pairwise cosine sim | **0.673** | 0.248 |
| Std of pairwise sim | 0.089 | **0.145** |
| Per-dim embedding variance | 0.000426 | **0.000979** (2.3x) |
| Effective rank | 322 / 768 | 273 / 768 |

CLIP maps all AudioCaps captions — "a dog barking while wind blows," "someone playing piano," "water flowing and birds chirping" — into a narrow cone (0.67 mean pairwise similarity). These are all "everyday scene descriptions" and CLIP's text encoder, trained to match images, doesn't differentiate them strongly. The student achieves 0.86 cosine similarity by learning to project all audio to roughly the same point — high alignment, zero discriminative power.

SBERT spreads captions by semantic content (0.25 mean pairwise similarity, 1.6x higher spread), so "dog barking" is far from "piano playing." The student must learn genuinely different representations for different sounds, producing features that transfer to classification.

**The signal-to-noise problem:** With 0.000426 per-dimension variance in CLIP embeddings, the discriminative information lives in tiny perturbations around a shared direction. The student operates in a low-SNR regime and cannot reliably extract this signal. SBERT's 2.3x higher variance gives the student a much stronger learning signal.

### Implications for the Platonic Representation Hypothesis

The vision experiments (Exp 7-9) showed CLIP's geometry consistently outperformed supervised geometry for vision tasks, supporting the "platonic" / universal representation hypothesis. This audio experiment provides a **counterexample**: CLIP's text geometry is worse than SBERT's for audio, despite CLIP being "multimodal."

This suggests:
- **CLIP's representations are "platonic" within vision-adjacent spaces**, not universally
- **Semantic text representations** (SBERT) may be more universal for cross-modal transfer to non-visual modalities
- The "universality" of a representation depends on the target modality, not just the source model's training diversity

### SBERT Exceeding AudioSet Pretraining

At 1% labels, SBERT-distilled features (41%) exceed AudioSet-pretrained features (35%). This is remarkable because:
- AudioSet pretraining uses 2M labeled audio clips vs AudioCaps' 46K audio-caption pairs
- The SBERT teacher never processed audio
- This advantage disappears at higher label fractions (AudioSet catches up by 10%, dominates at 100%)

The explanation: AudioSet pretraining learns audio-specific discriminative features optimized for its 527-class ontology. SBERT distillation learns semantically structured features that generalize better with minimal supervision. With 1% labels (1 sample per class), semantic structure matters more than audio-specific features.

## Technical Notes

- **BatchNorm calibration:** Audio mel spectrograms have statistics far from BN defaults (mean=0, var=1). We added BN running stat recalibration before each evaluation pass, computing cumulative (not EMA) statistics from training data. Without this, eval-mode predictions collapsed to a single class.
- **AudioSet pretrained weights:** EfficientAT mn10_as weights required key remapping (adding `backbone.` prefix) and SE layer reshaping (Linear [out,in] → Conv2d [out,in,1,1]) to be compatible with torchvision's MobileNetV3.
- **1% label fraction:** With 50 classes and only ~16 training samples, we use 1 sample per class instead of stratified split (which requires ≥1 sample per class per split).
- **AMP disabled:** Mixed precision caused NaN overflow in log-mel computation. Not needed on B200 GPU (183GB).
