# Experiment 10b: Audio CKA-Only Distillation — Does Structural Loss Bypass the Cone?

## Summary

Experiment 10 showed that CLIP text distillation fails catastrophically for audio due to the "embedding cone" problem (mean pairwise cosine 0.663), while whitening fixes it. But that experiment used **embedding (cosine) loss** — a point-wise loss that directly inherits the cone's pathology by pressuring the student to map all audio to roughly the same point.

Here we test whether **CKA-only loss** — which transfers only relational structure (which samples are similar/dissimilar), not absolute positions — naturally bypasses the cone problem without any whitening. This is also the cleanest test of the project's central claim: that geometric structure from foundation models is the active ingredient for downstream transfer.

**Key findings:**

1. **CKA-only completely bypasses CLIP's cone problem** — CLIP CKA-only achieves 67.1% linear probe at 100% labels vs 4.2% for CLIP embedding loss (raw). A +63pp improvement with no preprocessing.
2. **CLIP CKA-only converges with SBERT CKA-only** — 80.9 vs 81.4 (FT) and 67.1 vs 69.9 (LP) at 100%. When only structure is transferred, the two teachers produce nearly identical features, supporting the platonic representation hypothesis.
3. **CKA-only is weaker than whitened embedding loss** — whitened CLIP beats CKA-only CLIP by ~8-11pp, and raw SBERT embedding loss beats SBERT CKA-only by ~5pp. Consistent with Exp 9: point-wise content adds value beyond structure.
4. **Both CKA-only approaches exceed AudioSet pretraining at 1% labels** (linear probe) — text geometry, transferred structurally through a model that has never heard audio, beats 2M labeled audio clips.

## Setup

### Distillation Phase

Same as Experiment 10 except `--loss cka_only` instead of `--loss embedding`. No whitening applied — the point is that CKA should not need it.

| Teacher | Loss | Epochs | BS | LR | Final CKA Loss | CKA Value | Cosine Sim |
|---------|------|--------|-----|-----|---------------|-----------|------------|
| CLIP text | cka_only | 20 | 256 | 1e-3 | 0.443 | **0.557** | 0.036 |
| SBERT | cka_only | 20 | 256 | 1e-3 | 0.322 | **0.678** | -0.012 |

**Observations:**
- SBERT achieves higher CKA (0.68 vs 0.56) — consistent with its better-spread embeddings being easier to structurally align to
- Cosine similarity stays near zero for both (~0.03 and -0.01), confirming CKA provides no pressure to match individual embeddings
- Both runs still improving at epoch 20 — longer training could close the remaining gap

### Downstream Phase

- **Dataset:** ESC-50 (50 environmental sound classes, 2000 clips, 5-fold CV)
- **Conditions:** CLIP text CKA-only (G), Sentence-BERT CKA-only (H)
- **Label fractions:** 100%, 10%, 1%
- **Modes:** Fine-tune, Linear probe
- **12 runs total**, all launched in parallel on B200 GPU

## Results

### Full Comparison Table

| Condition | 100% FT | 100% LP | 10% FT | 10% LP | 1% FT | 1% LP |
|-----------|---------|---------|--------|--------|-------|-------|
| A: Random | 70.2 | 13.4 | 13.8 | 5.2 | 8.2 | 3.8 |
| B: CLIP embed (raw) | 75.0 | 4.2 | 19.5 | 3.4 | 8.1 | 9.3 |
| D: CLIP whitened | 83.9 | 75.6 | 57.9 | 56.6 | 43.6 | 43.6 |
| **G: CLIP CKA-only** | **80.9** | **67.1** | **50.3** | **47.4** | **32.6** | **32.4** |
| E: SBERT embed (raw) | **86.0** | 74.5 | **60.8** | **58.0** | **41.0** | **41.8** |
| F: SBERT whitened | 82.4 | 73.2 | 54.4 | 52.9 | 32.2 | 36.2 |
| **H: SBERT CKA-only** | **81.4** | **69.9** | **52.9** | **49.6** | **30.2** | **30.4** |
| C: AudioSet pretrained | **96.9** | **89.2** | **73.8** | 54.7 | 34.8 | 30.3 |

### CKA-Only Bypasses the Cone Problem

The most dramatic comparison is CLIP embed (raw) vs CLIP CKA-only:

| Label Fraction | Mode | CLIP Embed (B) | CLIP CKA-only (G) | Delta |
|---------------|------|----------------|-------------------|-------|
| 100% | FT | 75.0 | **80.9** | +5.9 |
| 100% | LP | 4.2 | **67.1** | **+62.9** |
| 10% | FT | 19.5 | **50.3** | **+30.8** |
| 10% | LP | 3.4 | **47.4** | **+44.0** |
| 1% | FT | 8.1 | **32.6** | **+24.5** |
| 1% | LP | 9.3 | **32.4** | **+23.1** |

**CKA-only wins 6/6, average improvement +31.9pp.** The embedding (cosine) loss forces the student to match CLIP's cone — all audio maps to roughly the same point, achieving high alignment but zero discriminative power. CKA loss only requires the student to preserve *which* audio clips the teacher considers similar or dissimilar. The cone's absolute position is irrelevant.

### CKA-Only vs Whitening for CLIP

| Label Fraction | Mode | CLIP Whitened (D) | CLIP CKA-only (G) | Delta |
|---------------|------|-------------------|-------------------|-------|
| 100% | FT | **83.9** | 80.9 | -3.0 |
| 100% | LP | **75.6** | 67.1 | -8.5 |
| 10% | FT | **57.9** | 50.3 | -7.6 |
| 10% | LP | **56.6** | 47.4 | -9.2 |
| 1% | FT | **43.6** | 32.6 | -11.0 |
| 1% | LP | **43.6** | 32.4 | -11.2 |

**Whitened CLIP wins 6/6, average margin -8.4pp.** Both approaches solve the cone problem, but whitened embedding loss transfers more information — it preserves the relational structure (whitening fixes the geometry) *and* provides point-wise anchoring (cosine loss places each sample at a specific location). CKA-only transfers only the structure.

This is consistent with the Exp 9 vision result where CKA+cosine combined beat CKA-only on 52/63 comparisons.

### CLIP vs SBERT Under CKA-Only

This is the key platonic hypothesis test. When only geometric structure is transferred (no point-wise content), do CLIP and SBERT converge?

| Label Fraction | Mode | CLIP CKA-only (G) | SBERT CKA-only (H) | Delta |
|---------------|------|-------------------|---------------------|-------|
| 100% | FT | 80.9 | **81.4** | +0.5 |
| 100% | LP | 67.1 | **69.9** | +2.8 |
| 10% | FT | 50.3 | **52.9** | +2.6 |
| 10% | LP | 47.4 | **49.6** | +2.2 |
| 1% | FT | **32.6** | 30.2 | -2.4 |
| 1% | LP | **32.4** | 30.4 | -2.0 |

**Win rate: 4-2 in favor of SBERT, but margins are small (avg 2.1pp).** Compare with the raw embedding loss comparison where SBERT dominated CLIP by 40pp on average. Once we strip away point-wise content and compare only geometric structure, CLIP and SBERT produce near-identical downstream features.

This is strong evidence for representational convergence: two very different models (CLIP trained on image-text pairs, SBERT trained on text-only paraphrases) arrive at similar relational structure of semantic concepts.

### CKA-Only for SBERT: Structure vs Content

| Label Fraction | Mode | SBERT Embed (E) | SBERT CKA-only (H) | Delta |
|---------------|------|-----------------|---------------------|-------|
| 100% | FT | **86.0** | 81.4 | -4.6 |
| 100% | LP | **74.5** | 69.9 | -4.6 |
| 10% | FT | **60.8** | 52.9 | -7.9 |
| 10% | LP | **58.0** | 49.6 | -8.4 |
| 1% | FT | **41.0** | 30.2 | -10.8 |
| 1% | LP | **41.8** | 30.4 | -11.4 |

**Embedding loss wins 6/6, average margin -7.9pp.** Same pattern as vision Exp 9 and CLIP whitened vs CKA-only above: point-wise content consistently adds value beyond geometric structure. The gap widens at lower label fractions, where the anchoring information from embedding content is most valuable.

### Gap Closure vs AudioSet Pretrained Ceiling

| Condition | 1% FT | Gap closed | 1% LP | Gap closed |
|-----------|-------|-----------|-------|-----------|
| Random | 8.2% | 0% | 3.8% | 0% |
| CLIP embed (raw) | 8.1% | 0% | 9.3% | 21% |
| CLIP CKA-only | **32.6%** | **92%** | **32.4%** | **108% (exceeds)** |
| CLIP whitened | **43.6%** | **133% (exceeds)** | **43.6%** | **150% (exceeds)** |
| SBERT embed | **41.0%** | **123% (exceeds)** | **41.8%** | **143% (exceeds)** |
| SBERT CKA-only | **30.2%** | **83%** | **30.4%** | **100% (matches)** |
| AudioSet pretrained | 34.8% | 100% | 30.3% | 100% |

Both CKA-only approaches match or exceed AudioSet pretraining at 1% labels (linear probe). Structural transfer from text geometry — through a model that has never heard audio — matches 2M labeled audio clips.

## Analysis

### Two Solutions to the Cone Problem

The Exp 10 results presented whitening as the fix for CLIP's cone. CKA-only reveals a second, independent solution:

| Approach | How it handles the cone | Result |
|----------|------------------------|--------|
| **Whitened embedding loss** | Preprocesses embeddings to remove the cone, then transfers content + structure | Best overall (+40pp vs raw) |
| **CKA-only loss** | Ignores the cone entirely — only transfers relative similarities | Strong (+32pp vs raw) but weaker than whitening |
| **Raw embedding loss** | Inherits the cone directly — student collapses | Fails catastrophically |

The two approaches work for different reasons:
- **Whitening** fixes the input: spread the embeddings, then the point-wise loss works normally
- **CKA-only** fixes the objective: use a loss that is invariant to the cone's presence

### Why CKA-Only is Weaker Than Whitened Embedding Loss

CKA transfers only batch-level similarity structure. Embedding loss (after whitening) additionally provides:

1. **Per-sample anchoring** — each audio clip maps to a specific location in embedding space, not just a relative position
2. **Denser gradient signal** — one scalar per sample vs one scalar per batch
3. **Cross-batch consistency** — CKA is computed within batches; embedding loss anchors to a global coordinate system

The 8pp average gap between whitened CLIP and CKA-only CLIP is the "price" of losing this information.

### Convergence of CLIP and SBERT Under CKA-Only

The near-equivalence of CLIP and SBERT under CKA-only (avg 2.1pp difference) is the cleanest evidence yet for representational convergence:

| Comparison | Avg absolute difference |
|------------|------------------------|
| CLIP embed vs SBERT embed (raw) | **40.3pp** (SBERT dominates) |
| CLIP CKA-only vs SBERT CKA-only | **2.1pp** (near-identical) |
| CLIP whitened embed vs SBERT embed | **2.6pp** (near-identical, per Exp 10) |

Three different ways to "equalize" the comparison (remove the cone, remove content, or remove the cone) all converge on the same answer: the underlying relational structure of CLIP and SBERT is approximately the same. The 40pp gap from raw embedding loss was entirely an artifact of the cone, not a fundamental difference in semantic organization.

### Connection to Vision CKA-Only (Exp 9)

| Finding | Vision (Exp 9) | Audio (Exp 10b) |
|---------|---------------|-----------------|
| CKA-only vs CKA+cosine | CKA+cos wins 52/63 | Embed/whitened wins 6/6 |
| CLIP vs supervised/SBERT (CKA-only, LP) | CLIP wins 15/15, +10pp | SBERT wins 4/6, +2pp |
| Structure alone sufficient? | No, content adds ~5-40pp | No, content adds ~5-11pp |

The audio results reinforce the vision findings: geometric structure is genuinely transferable and useful, but it is not the whole story. Point-wise embedding content consistently provides additional value.

The key audio-specific insight is that CKA-only *solves a problem that doesn't exist in vision*. CLIP's vision embeddings on ImageNet have mean pairwise cosine of 0.34 (well-spread), so the cone is irrelevant for vision distillation. For audio, where the cone causes embedding loss to fail, CKA-only provides a robust alternative.

## Implications for the Platonic Representation Hypothesis

| Prediction | Result | Evidence |
|-----------|--------|----------|
| CKA-only bypasses cone for CLIP | **Strongly supported** | +63pp over raw embed (LP 100%) |
| CLIP ≈ SBERT under structural transfer | **Supported** | 2.1pp avg difference vs 40pp with raw embed |
| Structure alone sufficient | **Not supported** | Whitened embed beats CKA-only by 8pp avg |
| CKA-only exceeds AudioSet ceiling at 1% | **Supported (LP)** | 32.4% vs 30.3% |

The platonic hypothesis gains further support: CLIP and SBERT converge to the same relational structure despite radically different training. But as in vision (Exp 9), structure is a necessary but not sufficient component of good distillation — point-wise content matters.

**The practical ranking for audio distillation:**
1. SBERT embedding loss (best overall — good geometry + good content)
2. CLIP whitened embedding loss (close second — fixes cone, transfers content)
3. CKA-only, either teacher (solid — bypasses cone, but loses content)
4. CLIP raw embedding loss (fails — cone collapses the student)

## Technical Notes

- **No AMP** for audio distillation — mel spectrograms cause NaN under mixed precision (log of near-zero values)
- **CKA loss operates on projected embeddings** — Linear(960, 768) projector, same architecture as embedding loss runs
- **Best model selection for CKA-only** uses `1 - cka_loss` instead of val cosine similarity (cosine is not optimized)
- **Checkpoint naming** includes loss type: `audiocaps_{teacher}_cka_only_distilled_best.pth`
- **BatchNorm calibration** before each eval pass (cumulative mean, momentum=None) — same as Exp 10
- **Downstream runs parallelized**: all 12 runs launched simultaneously on B200 (5.8GB total, ~30 min wall-clock)

## Scripts and Checkpoints

- **Distillation code:** `src/train_audio_distill.py` (with loss-aware checkpoint naming)
- **Downstream script:** `scripts/experiment10b_audio_cka_only_downstream.sh`
- **Checkpoints:** `checkpoints/audiocaps_clip_text_cka_only_distilled_best.pth`, `checkpoints/audiocaps_sentence_bert_cka_only_distilled_best.pth`
- **Runtime:** ~20 min per distillation run + ~30 min downstream (parallel) = ~70 min total
