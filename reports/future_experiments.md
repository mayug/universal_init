# Future Experiments

Experiment ideas generated from Experiment 8 results and discussion. Ordered by priority.

---

## Experiment 9: CKA-Only Distillation (Isolating the Platonic Claim)

**Status:** Next up

**Core idea:** Remove point-wise embedding matching (cosine loss) and distill using CKA loss *only*. The student has no pressure to match individual teacher embeddings — it only needs to reproduce the relational geometry of the teacher's representation (which images are similar/dissimilar).

**Why this is the cleanest test of the platonic hypothesis:** The platonic representation hypothesis is about *structure* — large models converge toward the same geometric organization of concepts. If that structure drives downstream utility, CKA-only should work. If point-wise embedding content matters more than geometry, CKA-only will fail, and "platonic" is the wrong explanation.

**Design:**

| Condition | Loss | What it tests |
|-----------|------|---------------|
| CKA-only | CKA loss only (no cosine) | Pure structural alignment — the platonic claim |
| Cosine-only | Cosine embedding loss only | Pure point-wise matching — no structural pressure |
| CKA + cosine (existing) | Combined | Both (baseline from Exp 7/8) |

Run all 3 with clip_l teacher on ImageNet, then downstream evaluation. If CKA-only >= cosine-only, strong evidence for the platonic structure hypothesis. If cosine-only >> CKA-only, the platonic framing is wrong.

**Concern:** CKA-only may be harder to optimize since the loss operates on batch-level statistics, not per-sample. Need to monitor whether distillation converges at all.

---

## Experiment 10: Intermediate Checkpoint Evaluation (CKA Dose-Response)

**Status:** Planned

**Core idea:** Same teacher (clip_l), same training paradigm, but evaluate students at *different levels of alignment* by saving distillation checkpoints every 2 epochs and running downstream on each.

**Why this eliminates confounds:** The teacher is identical, the training paradigm is identical, the only variable is how closely the student has approached the teacher's representation (backbone CKA). If downstream accuracy monotonically improves with CKA, that's hard to explain by any mechanism other than "getting closer to the teacher's structure helps."

**Design:**
- Re-distill clip_l on ImageNet, saving checkpoints at epochs 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
- Measure backbone CKA at each checkpoint (already logged during training)
- Run downstream (linprobe + fine-tune) on each checkpoint for 2-3 key datasets (VOC, Pets, Imagenette) at 1%, 10%, 100%
- Plot: backbone CKA (x-axis) vs downstream accuracy (y-axis)

**Expected outcome if platonic hypothesis holds:** Monotonic increasing curve. Each step closer to the teacher's representation produces a measurably better initialization.

**Expected outcome if hypothesis is wrong:** Plateau or non-monotonic relationship — accuracy saturates early while CKA keeps improving, meaning the extra structural alignment doesn't help.

---

## Experiment 11: Larger Student Models (Testing the Capacity Bottleneck)

**Status:** Planned

**Core idea:** Exp 8 showed ViT-L teachers don't beat ViT-B downstream, suggesting the student (RegNetY-400MF, 4.3M params, 440-dim) is the bottleneck. Test whether larger students can absorb more from the teacher and whether the platonic advantage emerges at higher student capacity.

**Design:**
- Distill clip_l into RegNetY-400MF (4.3M), RegNetY-1.6GF (11.2M), and RegNetY-4GF (21.1M)
- Compare backbone CKA achieved and downstream accuracy
- Compare gap to ImageNet pretrained for each student size

**Predictions if platonic hypothesis holds:**
- Larger students achieve higher backbone CKA (can represent more structure)
- CKA → downstream correlation strengthens with student capacity
- Gap to ImageNet pretrained shrinks with student size

**Predictions if student is just the bottleneck:**
- Larger students improve regardless of teacher (even random → pretrained gap shrinks)
- Teacher choice matters less as student capacity increases

---

## Experiment 12: Per-Layer CKA Distillation (Intermediate Feature Alignment)

**Status:** Planned (requires architecture work)

**Core idea:** Instead of only aligning the final embedding (a single 1024-dim vector), align *intermediate features* between student CNN blocks and teacher ViT transformer blocks using layer-wise CKA.

**Why this matters:** Current distillation compresses all teacher knowledge into one final embedding vector — an extreme information bottleneck. ImageNet pretraining learns a hierarchy of features (edges → textures → parts → objects) across layers, and this hierarchy is what makes it transferable. Final-embedding CKA only captures the last layer's summary, losing all hierarchical information.

**Design:**
- Define correspondences between student layers and teacher layers:
  - Student stem → Teacher patch embed + early blocks
  - Student stage 1-2 → Teacher mid blocks
  - Student stage 3-4 → Teacher late blocks
  - Student projector → Teacher final embedding
- Compute CKA between corresponding feature maps (using spatial average pooling to match dimensions)
- Total loss = sum of per-layer CKA losses (possibly weighted)

**Analogy:** FitNets (Romero et al., 2015) but with CKA instead of MSE. CKA is better suited to the CNN↔ViT architectural mismatch because it measures representational similarity structure rather than requiring matched dimensions.

**Implementation complexity:** Moderate — requires hooks to extract intermediate features from both teacher and student, spatial pooling to flatten feature maps, and CKA computation at each layer. The main challenge is choosing good layer correspondences between a CNN (RegNetY) and a ViT.

**Expected impact:** If the platonic hypothesis is about hierarchical convergence (not just final-layer convergence), per-layer CKA could dramatically close the gap to ImageNet pretraining by forcing the student to learn similar low/mid-level features, not just match a final summary vector.

---

## Priority Order

1. **Experiment 9 (CKA-only)** — cheapest, most directly tests the platonic claim, ~12h
2. **Experiment 10 (intermediate checkpoints)** — cheap, dose-response curve, ~18h
3. **Experiment 11 (larger students)** — moderate cost, tests bottleneck hypothesis, ~48h
4. **Experiment 12 (per-layer CKA)** — highest implementation cost but potentially highest impact
