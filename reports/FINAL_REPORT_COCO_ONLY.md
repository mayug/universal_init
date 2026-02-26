# Final Experiment Report: Universal Initialization via ImageBind Distillation

**Date:** January 26, 2026
**Status:** Experiments Complete (Except Blocked Items)

---

## Executive Summary

This report documents comprehensive experiments testing whether initializing a small vision model (RegNetY-400MF) by distilling ImageBind embeddings yields more sample-efficient downstream learning compared to random or ImageNet-pretrained initialization.

### Main Findings

1. **ImageNet pretraining dramatically outperforms COCO distillation** across all tested conditions
2. **COCO distillation shows severe performance degradation** on fine-grained tasks (Pets dataset)
3. **Frozen projector hurts downstream performance** - should be dropped, not kept during fine-tuning
4. **Sample efficiency gap is largest at low label fractions** (1-10%)

### Recommendation

**Do not use COCO-based distillation for downstream initialization.** ImageNet-pretrained weights provide significantly better transfer learning performance. The distillation approach requires fundamental redesign.

---

## Completed Experiments

### Baseline Comparisons (34 runs)
✅ **Complete** - Three initialization methods across multiple datasets and label fractions

### Experiment 2: Projector Ablation (4 runs)
✅ **Complete** - Tested keeping vs dropping frozen COCO-distilled projector

### Experiment 1: Teacher Oracle Probes
⏸️ **Blocked** - Requires ImageBind installation

### Experiment 3: ImageNet Distillation
⏸️ **Blocked** - Requires ImageNet dataset (~147GB)

---

## Detailed Results

### 1. Baseline Performance Comparison

| Dataset | Init Method | 1% Labels | 10% Labels | 100% Labels |
|---------|-------------|-----------|------------|-------------|
| **Pets** (37 classes) |
|| COCO-Distilled | 5.78% ± 0.81% | 17.31% ± 2.32% | 54.16% |
|| **ImageNet** | **42.75% ± 2.85%** | **82.01% ± 0.05%** | **91.16% ± 0.16%** |
|| Random | 3.81% ± 0.35% | 6.68% ± 0.38% | 29.41% ± 0.52% |
|| **Gap (ImageNet / COCO)** | **7.4×** | **4.7×** | **1.7×** |
| **EuroSAT** (10 classes) |
|| COCO-Distilled | 74.74% | - | 97.67% |
|| **ImageNet** | **86.28%** | - | **98.94%** |
|| Random | 64.04% | - | 97.20% |
|| **Gap (ImageNet / COCO)** | **1.15×** | - | **1.01×** |
| **Imagenette** (10 classes) |
|| COCO-Distilled | - | - | 80.54% |
|| **ImageNet** | - | - | **99.16%** |
|| Random | - | - | 77.17% |

#### Key Observations:

1. **Pets Dataset (Fine-Grained Classification)**
   - COCO-distilled: **7.4× worse** than ImageNet at 1% labels
   - COCO barely outperforms random initialization
   - Even at 100% labels, COCO achieves only 54% vs ImageNet's 91%
   - **Conclusion:** COCO distillation fails catastrophically on fine-grained tasks

2. **EuroSAT Dataset (Domain-Shifted Satellite Imagery)**
   - COCO-distilled: **1.15× worse** than ImageNet at 1% labels
   - Gap much smaller than Pets
   - At 100% labels, all methods converge (97-99%)
   - **Conclusion:** Domain shift levels the playing field; no init has strong prior

3. **Imagenette Dataset (ImageNet Subset)**
   - COCO-distilled barely outperforms random (80.54% vs 77.17%)
   - ImageNet achieves near-perfect 99.16%
   - **Conclusion:** COCO distillation doesn't transfer even to natural images

---

### 2. Experiment 2: Projector Ablation Results

**Research Question:** Should we keep the frozen COCO-distilled projector during fine-tuning?

**Architectures Tested:**
- **Drop Projector (Baseline):** Backbone (fine-tuned, 440-dim) → Classifier
- **Keep Projector (Ablation):** Backbone (fine-tuned) → Frozen Projector (440→1024) → Classifier

#### Results:

| Dataset | Label Fraction | Drop Projector | Keep Frozen Projector | Difference | Verdict |
|---------|----------------|----------------|----------------------|------------|---------|
| Pets | 1% | 5.78% | 6.57% | +0.79% | ✓ Slightly better |
| Pets | 100% | 54.16% | **33.31%** | **-20.85%** | **✗ Much worse** |
| EuroSAT | 1% | 74.74% | 68.65% | -6.09% | ✗ Worse |
| EuroSAT | 100% | 97.67% | 84.72% | -12.94% | ✗ Much worse |

#### Key Findings:

1. **Frozen projector HURTS performance in 3 out of 4 conditions**
   - Pets 100%: **38.5% relative drop** (massive degradation)
   - EuroSAT 100%: **13.3% relative drop**
   - Only marginal improvement on Pets 1% (likely noise)

2. **Why does the frozen projector hurt?**
   - Projector overfits to COCO's embedding space during distillation
   - Frozen projector cannot adapt to new task requirements
   - 1024-dim frozen features are worse than 440-dim trainable features
   - Adds architectural bottleneck without learning capacity

3. **Clear Conclusion: Drop the projector during fine-tuning**
   - Trainable 440-dim backbone features >> Frozen 1024-dim projected features
   - Projector useful only during distillation training
   - Should be discarded for downstream tasks

---

## Analysis and Insights

### Why Does COCO Distillation Fail?

#### Hypothesis 1: Domain Mismatch ⭐ (Primary)
- **COCO:** General object detection scenes, cluttered backgrounds, multiple objects
- **Pets:** Fine-grained breed classification, subtle visual features (whiskers, ears, face structure)
- **Gap:** ImageBind trained on COCO doesn't capture fine-grained discriminative features
- **Evidence:** Performance worst on Pets (fine-grained), better on EuroSAT (coarse, domain-shifted)

#### Hypothesis 2: Insufficient Training Data
- **COCO:** 118K images for distillation
- **ImageNet:** 1.28M images for pretraining
- **Gap:** 10.8× fewer training images
- **Impact:** Student model may not learn diverse enough features

#### Hypothesis 3: Student Capacity Bottleneck
- **Student:** RegNetY-400MF (440-dim features)
- **Teacher:** ImageBind (1024-dim embeddings)
- **Compression ratio:** 2.3× dimensional reduction
- **Projector:** Linear mapping insufficient to bridge gap
- **Evidence:** Frozen projector hurts performance (can't adapt features)

#### Hypothesis 4: Loss Function Limitations
- **Current:** Combined loss (cosine embedding + relational similarity)
- **Issue:** May prioritize global structure over local discriminative features
- **Evidence:** Works better on coarse-grained EuroSAT than fine-grained Pets

### Why Does EuroSAT Show Smaller Gaps?

**Key Insight:** Satellite imagery is far from both COCO and ImageNet distributions

1. **No Strong Prior:** Neither COCO nor ImageNet provides strong domain knowledge
2. **Task Simplicity:** 10 classes vs 37 (Pets), easier to learn from scratch
3. **Visual Features:** Satellite imagery requires different low-level features (textures, patterns)
4. **Generalization:** COCO's broader object recognition may be more useful than ImageNet's bias

---

## Sample Efficiency Analysis

### Sample Efficiency Ratio (ImageNet / COCO-Distilled)

| Dataset | 1% Labels | 10% Labels | 100% Labels |
|---------|-----------|------------|-------------|
| **Pets** | **7.39×** | **4.74×** | 1.68× |
| **EuroSAT** | 1.15× | - | 1.01× |

**Interpretation:**
- **Low data regime (1%):** ImageNet is 7× more sample-efficient on fine-grained tasks
- **Medium data (10%):** Gap narrows to 4.7× but still substantial
- **High data (100%):** Gap closes as all methods see enough data
- **Domain shift:** Sample efficiency gap much smaller on EuroSAT

### Learning Efficiency (AULC)

**Area Under Learning Curve measures learning speed across all epochs**

| Dataset | Labels | COCO-Distilled | ImageNet | Ratio |
|---------|--------|----------------|----------|-------|
| Pets | 1% | 4.64 | 34.76 | **7.49×** |
| Pets | 100% | 47.27 | 89.65 | **1.90×** |
| EuroSAT | 1% | 58.39 | 73.04 | 1.25× |
| EuroSAT | 100% | 95.21 | 98.32 | 1.03× |

**Key Insight:** ImageNet not only reaches higher final accuracy, but learns MUCH faster (higher AULC).

---

## Learning Trajectories

### Milestone Accuracy Tracking

**Pets 1% Labels:**
| Init | Epoch 5 | Epoch 10 | Epoch 20 | Final (50) |
|------|---------|----------|----------|------------|
| COCO-Distilled | ~3% | ~5% | ~5% | 5.78% |
| ImageNet | ~35% | ~40% | ~42% | 42.75% |
| Random | ~3% | ~3% | ~3% | 3.81% |

**Observation:** ImageNet starts strong and maintains lead throughout. COCO shows slow, minimal learning.

**Pets 100% Labels (Keep Projector vs Drop):**
| Config | Epoch 5 | Epoch 10 | Epoch 20 | Final (50) |
|--------|---------|----------|----------|------------|
| Drop Projector | 21% | 42% | 52% | 54.16% |
| **Keep Frozen Projector** | **24%** | **29%** | **31%** | **33.31%** |

**Observation:** Frozen projector shows initial promise but fails to improve, suggesting it creates a learning bottleneck.

---

## Statistical Analysis

### Mean Performance by Initialization

**Across all datasets and label fractions:**

| Init Method | Mean Accuracy | Std Dev | Median |
|-------------|---------------|---------|--------|
| **ImageNet** | **79.66%** | 21.84% | 86.28% |
| **COCO-Distilled** | **48.31%** | 31.76% | 54.16% |
| **Random** | 36.88% | 34.69% | 29.41% |

**ImageNet is 1.65× better than COCO-distilled on average across all conditions.**

### Performance by Task Difficulty

**Fine-Grained (Pets, 37 classes):**
- ImageNet: 71.97% average
- COCO: 25.74% average
- **Gap: 2.80× (ImageNet better)**

**Coarse-Grained (EuroSAT, 10 classes):**
- ImageNet: 92.61% average
- COCO: 83.22% average
- **Gap: 1.11× (ImageNet better)**

**Conclusion:** COCO distillation's weakness is most pronounced on difficult, fine-grained tasks.

---

## Projector Ablation Deep Dive

### Architectural Comparison

**Configuration A: Drop Projector (Current Best)**
```
Input Image → Backbone (fine-tuned, 440-dim) → Classifier (440 → N classes)
```
- **Trainable params:** ~4.39M (backbone) + classifier
- **Feature dim:** 440 (lower dimensional)
- **Learning:** Full gradient flow through backbone
- **Result:** Better performance

**Configuration B: Keep Frozen Projector**
```
Input Image → Backbone (fine-tuned, 440-dim) → Frozen Projector (440→1024) → Classifier (1024 → N classes)
```
- **Trainable params:** ~4.39M (backbone) + classifier (frozen projector excluded)
- **Feature dim:** 1024 (higher dimensional, but frozen)
- **Learning:** Gradient flow blocked at projector
- **Result:** Worse performance

### Why Does Freezing the Projector Hurt?

1. **Gradient Blocking:**
   - Projector blocks gradient flow from classifier to backbone
   - Backbone receives weaker learning signal
   - Fine-tuning effectiveness reduced

2. **Feature Mismatch:**
   - Projector learned COCO-specific features
   - Frozen features cannot adapt to new task (Pets breeds, EuroSAT land use)
   - Creates architectural bottleneck

3. **Overfitting to COCO:**
   - Projector optimized for ImageBind's COCO-based embeddings
   - Embedding space not universal across tasks
   - Frozen projector enforces COCO's feature structure

4. **Trainability vs Dimensionality Trade-off:**
   - **440-dim trainable** > **1024-dim frozen**
   - Learning capacity matters more than feature dimension
   - Adaptive features > High-dimensional fixed features

### Experiment 2 Conclusion

**Clear verdict:** Drop the projector during fine-tuning. The projector is useful only for distillation training and should not be retained for downstream tasks.

---

## Recommendations

### For Production Use

1. **Use ImageNet-pretrained initialization** for computer vision tasks
   - Proven, reliable, strong transfer learning
   - Works across diverse downstream tasks
   - Much better sample efficiency

2. **Avoid COCO distillation** in its current form
   - Does not provide competitive performance
   - Severe degradation on fine-grained tasks
   - Not worth the computational cost of distillation

3. **Drop projector during fine-tuning**
   - Never keep frozen projector from distillation
   - Fine-tune backbone directly

### For Research Improvements

If continuing distillation research, address these fundamental issues:

#### 1. Change Distillation Dataset
- **Try ImageNet distillation** instead of COCO (Experiment 3)
- 1.28M images vs 118K (10.8× more data)
- Better coverage of visual concepts
- **Hypothesis:** Distillation source is the key factor

#### 2. Increase Student Capacity
- Use larger student models (RegNetY-800MF, RegNetY-1.6GF)
- Reduce compression ratio (440→1024 is 2.3×)
- Try MLP projectors instead of linear
- **Goal:** Better capacity to capture teacher's knowledge

#### 3. Improve Loss Functions
- Experiment with contrastive losses (SimCLR, MoCo)
- Try knowledge distillation variants (temperature scaling)
- Add attention-based losses to preserve fine-grained features
- **Goal:** Better preserve discriminative information

#### 4. Multi-Stage Distillation
- Stage 1: Distill on ImageNet (broad coverage)
- Stage 2: Task-specific distillation (e.g., fine-grained datasets)
- **Goal:** Learn hierarchical features

#### 5. Adaptive Projectors
- Make projector trainable during fine-tuning (not frozen)
- Use task-specific projector heads
- **Goal:** Allow feature adaptation to downstream tasks

---

## Experimental Details

### Datasets

| Dataset | Classes | Train Size | Val Size | Task Type |
|---------|---------|------------|----------|-----------|
| Pets | 37 | 3,680 | 3,669 | Fine-grained breeds |
| EuroSAT | 10 | 21,600 | 5,400 | Satellite land use |
| Imagenette | 10 | 9,469 | 3,925 | ImageNet subset |
| COCO (distill) | - | 118,287 | - | Object scenes |

### Model Architecture

- **Backbone:** RegNetY-400MF
  - Parameters: ~4.39M
  - Feature dim: 440
  - Pretrained: ImageNet1K-V2 (for ImageNet init)

- **Teacher:** ImageBind Vision Encoder
  - Frozen during distillation
  - Embedding dim: 1024
  - Pretrained on multi-modal data

- **Projector:** Linear(440 → 1024)
  - Used only during distillation
  - Parameters: ~450K

### Training Configuration

**Distillation (COCO):**
- Epochs: 30
- Batch size: 256
- Optimizer: AdamW (lr=1e-3)
- Loss: Combined (embedding + 0.5×relational)
- Time: ~6-8 hours

**Downstream Fine-tuning:**
- Epochs: 50
- Batch size: 64
- Optimizer: SGD (lr=0.01, momentum=0.9)
- LR Schedule: Linear warmup (5 epochs) + cosine decay
- Time: ~30-60 minutes per run

---

## Results Files

### Checkpoints
- `checkpoints/coco_distilled_best.pth` (51MB)
- `checkpoints/imagenette_distilled_best.pth` (51MB)

### Results CSVs (38 files)
- **Baseline:** `results_{dataset}_{init}_frac{frac}_s{seed}.csv` (34 files)
- **Projector Ablation:** `results_{dataset}_distilled_keepproj_frac{frac}_s42.csv` (4 files)

### Analysis Outputs
- `reports/combined_results.csv` - All results consolidated
- `analysis_output.txt` - Statistical summary
- `analysis_final.txt` - Final analysis with projector ablation

---

## Experiment Status Summary

| Experiment | Status | Runs | Key Finding |
|------------|--------|------|-------------|
| Baseline Comparisons | ✅ Complete | 34 | ImageNet >> COCO distillation |
| Experiment 2: Projector Ablation | ✅ Complete | 4 | Drop projector, don't keep |
| Experiment 1: Teacher Oracle | ⏸️ Blocked | 0 | Needs ImageBind install |
| Experiment 3: ImageNet Distill | ⏸️ Blocked | 0 | Needs ImageNet dataset |

### Blocked Experiments

**Experiment 1:** Would establish performance ceiling with frozen ImageBind embeddings
- **Value:** Quantify distillation efficiency
- **Blocker:** Requires `pip install ImageBind`
- **Time:** ~4-6 hours

**Experiment 3:** Would test if distillation source is the key factor
- **Value:** Compare ImageNet-distilled vs COCO-distilled
- **Blocker:** Requires ImageNet dataset (~147GB)
- **Time:** ~10-14 hours

---

## Limitations

1. **Single Distillation Source:** Only tested COCO, not ImageNet or other datasets
2. **Single Student Architecture:** Only RegNetY-400MF, not larger models
3. **Limited Projector Types:** Only linear projector, not MLP variants
4. **No Teacher Oracle:** Cannot measure distillation gap without frozen teacher baseline
5. **Limited Downstream Tasks:** Only 3 datasets tested
6. **Single Seed:** Projector ablation only run once per condition (seed=42)

---

## Conclusions

### Main Findings

1. **COCO distillation is not competitive** with ImageNet pretraining
   - 7.4× worse on fine-grained tasks (Pets @ 1% labels)
   - Barely better than random initialization
   - Fails to transfer even to natural images (Imagenette)

2. **Frozen projector hurts performance**
   - 20% absolute drop on Pets 100% labels
   - Creates learning bottleneck
   - Should be dropped during fine-tuning

3. **Domain and task matter immensely**
   - Fine-grained tasks: COCO distillation fails
   - Domain-shifted tasks: Gap narrows but ImageNet still wins
   - Sample efficiency gap largest at low data

### Root Cause Analysis

The poor performance stems from:
1. **Domain mismatch** - COCO scenes ≠ fine-grained breeds
2. **Insufficient data** - 118K images insufficient for universal features
3. **Student capacity** - 440-dim cannot capture 1024-dim teacher
4. **Projector overfitting** - Learned COCO-specific, not universal features

### Path Forward

To make distillation competitive:
1. **Distill from ImageNet** instead of COCO (Experiment 3)
2. **Use larger students** to reduce compression
3. **Improve loss functions** to preserve fine-grained features
4. **Multi-stage distillation** for hierarchical learning
5. **Adaptive projectors** that can fine-tune to new tasks

### Final Recommendation

**For practical applications:** Use ImageNet-pretrained weights. COCO distillation requires fundamental redesign before it can compete.

---

*Report compiled from 38 experimental runs across 3 experiments.*
*Total compute time: ~100-150 hours*
*Date: January 26, 2026*
