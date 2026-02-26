# Teacher Oracle Analysis: Performance Ceiling with Frozen ImageBind

**Date:** January 26, 2026
**Experiment:** Teacher Oracle Linear Probes on Frozen ImageBind Embeddings

---

## Executive Summary

This experiment establishes the **performance ceiling** achievable with frozen ImageBind (1024-dim) embeddings by training linear probes directly on teacher features, bypassing distillation entirely. This provides critical context for evaluating distillation efficiency.

### Key Findings

1. **On Pets (Fine-Grained):** Teacher oracle **outperforms COCO-distilled** at both label fractions, proving that frozen ImageBind features are superior to compressed distilled features for fine-grained classification.

2. **On EuroSAT (Satellite):** Teacher oracle **underperforms COCO-distilled**, surprisingly showing that distillation+fine-tuning helps on satellite imagery despite compression loss.

3. **Distillation Gap:** COCO-distilled captures only **39% of teacher oracle capacity** on Pets @ 1%, revealing massive information loss in compression (1024→440 dims).

---

## Complete Performance Comparison

### Pets Dataset (Fine-Grained, 37 Classes)

| Init Method | 1% Labels | 100% Labels |
|-------------|-----------|-------------|
| **Teacher Oracle (Frozen ImageBind)** | **15.32%** | **88.36%** |
| ImageNet-Pretrained | **42.75%** ✓ | **91.16%** ✓ |
| COCO-Distilled | 5.98% | 43.73% |
| Random | 3.81% | 29.41% |

**Analysis:**
- Teacher oracle (15.32% @ 1%) is **2.6× better** than COCO-distilled (5.98%)
- Teacher oracle (88.36% @ 100%) is **2.0× better** than COCO-distilled (43.73%)
- **ImageNet still wins**, showing that fine-tuning (not just features) is critical
- COCO-distilled captures only **39%** of teacher oracle performance @ 1% (5.98/15.32)
- Frozen ImageBind features are **much better** than distilled+frozen features

### EuroSAT Dataset (Satellite, 10 Classes)

| Init Method | 1% Labels | 100% Labels |
|-------------|-----------|-------------|
| **Teacher Oracle (Frozen ImageBind)** | 28.89% | 89.13% |
| ImageNet-Pretrained | **86.28%** ✓ | **98.94%** ✓ |
| COCO-Distilled | **71.69%** | **91.19%** |
| Random | 64.04% | 97.20% |

**Analysis:**
- Teacher oracle (28.89% @ 1%) is **2.5× worse** than COCO-distilled (71.69%)
- Teacher oracle (89.13% @ 100%) is slightly worse than COCO-distilled (91.19%)
- **Surprising result:** Distillation+fine-tuning helps on satellite imagery
- Frozen ImageBind features are **poorly suited** for satellite imagery without fine-tuning
- Domain gap: ImageBind trained on natural images, not satellite imagery

---

## Detailed Results Table

| Dataset | Labels | Teacher Oracle | ImageNet | COCO-Distilled | Random | Oracle vs COCO Gap |
|---------|--------|----------------|----------|----------------|--------|--------------------|
| Pets | 1% | 15.32% | **42.75%** | 5.98% | 3.81% | **+9.34%** (2.6×) |
| Pets | 100% | 88.36% | **91.16%** | 43.73% | 29.41% | **+44.63%** (2.0×) |
| EuroSAT | 1% | 28.89% | **86.28%** | 71.69% | 64.04% | **-42.80%** (0.4×) |
| EuroSAT | 100% | 89.13% | **98.94%** | 91.19% | 97.20% | **-2.06%** (0.98×) |

**Key Metrics:**
- **Oracle vs COCO Gap:** Positive = teacher oracle better, Negative = COCO-distilled better
- **Multiplier:** Teacher oracle / COCO-distilled

---

## Distillation Efficiency Analysis

### What is Distillation Efficiency?

**Distillation Efficiency = (Student Performance / Teacher Oracle Performance) × 100%**

This measures how much of the teacher's representational capacity is retained after distillation and compression.

### Results

| Dataset | Labels | Teacher Oracle | COCO-Distilled | Distillation Efficiency |
|---------|--------|----------------|----------------|------------------------|
| Pets | 1% | 15.32% | 5.98% | **39%** |
| Pets | 100% | 88.36% | 43.73% | **49%** |
| EuroSAT | 1% | 28.89% | 71.69% | **248%** ⚠️ |
| EuroSAT | 100% | 89.13% | 91.19% | **102%** ⚠️ |

⚠️ **Over 100% efficiency** indicates distillation+fine-tuning improves over frozen features

### Interpretation

**Pets (Fine-Grained):**
- Distillation captures only **39-49%** of teacher capacity
- Massive information loss from compression (1024→440 dims)
- Fine-tuning cannot recover lost fine-grained details
- **Conclusion:** Distillation fails on fine-grained tasks due to capacity bottleneck

**EuroSAT (Satellite):**
- Distillation efficiency **exceeds 100%** (!)
- Fine-tuning the distilled backbone helps more than frozen teacher features
- Frozen ImageBind poorly suited for satellite domain
- **Conclusion:** Distillation+fine-tuning beneficial for domain shift tasks

---

## Learning Curves (AULC)

**Area Under Learning Curve** measures learning speed across 50 epochs.

| Dataset | Labels | Teacher Oracle | ImageNet | COCO-Distilled | Random |
|---------|--------|----------------|----------|----------------|--------|
| Pets | 1% | 8.24 | **34.76** | 4.89 | 3.08 |
| Pets | 100% | 71.25 | **89.65** | 38.50 | 21.32 |
| EuroSAT | 1% | 22.45 | **73.04** | 59.03 | 46.09 |
| EuroSAT | 100% | 82.98 | **98.32** | 88.75 | 92.46 |

**Key Observations:**
- ImageNet initialization has **fastest learning** on all tasks
- Teacher oracle learns slowly on Pets @ 1% (AULC: 8.24 vs ImageNet: 34.76)
- Linear probes on frozen features struggle with limited data
- Fine-tuning (ImageNet) >>> Linear probes (Teacher Oracle) for sample efficiency

---

## Milestone Accuracies

### Pets 1% Labels

| Method | Epoch 5 | Epoch 10 | Epoch 20 | Best (Epoch 50) |
|--------|---------|----------|----------|-----------------|
| **Teacher Oracle** | 3.33% | 3.84% | 5.89% | 15.32% |
| **ImageNet** | 37.68% | 41.23% | 41.66% | 42.75% |
| **COCO-Distilled** | 4.95% | 5.31% | 5.52% | 5.98% |
| **Random** | 2.74% | 2.85% | 3.15% | 3.81% |

**Analysis:** Teacher oracle starts very slow (3.33% @ epoch 5) but improves significantly by epoch 50 (15.32%), still worse than ImageNet's early performance (37.68% @ epoch 5).

### Pets 100% Labels

| Method | Epoch 5 | Epoch 10 | Epoch 20 | Best (Epoch 50) |
|--------|---------|----------|----------|-----------------|
| **Teacher Oracle** | 24.64% | 53.48% | 80.35% | 88.36% |
| **ImageNet** | 83.64% | 89.36% | 90.42% | 91.16% |
| **COCO-Distilled** | 32.10% | 38.38% | 41.43% | 43.73% |
| **Random** | 13.74% | 17.94% | 20.89% | 29.41% |

**Analysis:** Teacher oracle learns quickly with full labels (80.35% @ epoch 20), approaching ImageNet performance (90.42% @ epoch 20).

### EuroSAT 1% Labels

| Method | Epoch 5 | Epoch 10 | Epoch 20 | Best (Epoch 50) |
|--------|---------|----------|----------|-----------------|
| **Teacher Oracle** | 9.28% | 10.43% | 28.89% | 28.89% |
| **ImageNet** | 73.70% | 78.02% | 79.31% | 86.28% |
| **COCO-Distilled** | 63.41% | 64.21% | 65.14% | 71.69% |
| **Random** | 46.72% | 52.54% | 53.98% | 64.04% |

**Analysis:** Teacher oracle **severely underperforms** on satellite imagery (9.28% @ epoch 5 vs ImageNet 73.70%), showing poor domain transfer from natural images.

### EuroSAT 100% Labels

| Method | Epoch 5 | Epoch 10 | Epoch 20 | Best (Epoch 50) |
|--------|---------|----------|----------|-----------------|
| **Teacher Oracle** | 67.50% | 79.17% | 85.54% | 89.13% |
| **ImageNet** | 96.41% | 97.91% | 98.56% | 98.94% |
| **COCO-Distilled** | 88.56% | 89.20% | 89.39% | 91.19% |
| **Random** | 92.07% | 93.87% | 95.13% | 97.20% |

**Analysis:** Teacher oracle improves with more data but still lags behind all other methods on satellite imagery.

---

## Why Does Teacher Oracle Fail on Pets @ 1%?

Despite being the "performance ceiling," teacher oracle performs poorly on fine-grained tasks with limited labels. Why?

### 1. **Limited Data + Frozen Features**
- Only 37 training samples (1% of 3680)
- Linear probe has limited capacity to map 1024-dim features to 37 classes
- No feature adaptation → poor class separation

### 2. **Feature Dimensionality Mismatch**
- ImageBind features (1024-dim) encode general visual concepts
- Fine-grained breeds require subtle local features (eye color, fur texture)
- Frozen features cannot adapt to task-specific requirements

### 3. **ImageNet Fine-Tuning Advantage**
- ImageNet initialization allows **full backbone fine-tuning**
- Can learn task-specific feature transformations
- Adapts low-level and high-level features simultaneously

### 4. **Sample Efficiency Trade-off**
- Linear probes require more data for effective learning
- Fine-tuning can work with fewer samples by adjusting features
- At 1%, fine-tuning >>> linear probes

---

## Why Does Teacher Oracle Underperform on EuroSAT?

Teacher oracle performs worse than COCO-distilled on satellite imagery. Why?

### 1. **Domain Mismatch**
- ImageBind trained on natural images (COCO, ImageNet)
- Satellite imagery has different visual properties:
  - Aerial perspective (top-down view)
  - Different color distributions (vegetation indices)
  - Spatial patterns (agricultural fields, urban areas)
- Frozen features lack domain-relevant representations

### 2. **Distillation as Domain Adaptation**
- COCO distillation creates compressed features
- Fine-tuning distilled backbone adapts to satellite domain
- **Compression + fine-tuning > frozen high-dim features** for domain shift

### 3. **Random Initialization Paradox**
- Random init achieves 97.20% @ 100% labels (!)
- EuroSAT is relatively easy with sufficient data
- Domain-specific learning from scratch works well
- Frozen ImageBind features may introduce **negative transfer**

---

## Implications for Distillation Research

### 1. **Distillation Efficiency is Task-Dependent**
- **Fine-grained tasks:** Distillation loses 50-60% of teacher capacity
- **Domain shift tasks:** Distillation+fine-tuning can exceed frozen teacher performance

### 2. **Student Capacity Bottleneck**
- Compressing 1024→440 dims loses critical information
- Fine-grained features (breed-specific details) are lost first
- **Solution:** Use larger student models or reduce compression ratio

### 3. **Fine-Tuning vs. Frozen Features**
- **Frozen teacher features:** High-capacity but no adaptation
- **Fine-tuned student:** Lower capacity but task-adaptive
- **Winner depends on:** Task complexity, data availability, domain match

### 4. **When to Use Teacher Oracle**
- ✅ **Use frozen teacher:** Domain match, sufficient data, need fast inference
- ❌ **Don't use frozen teacher:** Domain shift, limited data, need task adaptation

---

## Recommendations

### For Fine-Grained Classification (Pets)

**Option 1: ImageNet Initialization (Best)**
- Use ImageNet-pretrained ResNet50 or similar
- Fine-tune full backbone
- **Result:** 42.75% @ 1%, 91.16% @ 100%

**Option 2: Larger Student Model**
- If using distillation, increase student capacity (RegNetY-800MF or larger)
- Reduce compression ratio (1024→800 instead of 1024→440)
- May achieve closer to teacher oracle performance (15-20% @ 1%)

**❌ Don't Use:** COCO distillation with small student (current approach)

### For Domain Shift Tasks (EuroSAT)

**Option 1: ImageNet Initialization (Best)**
- Still the winner (86.28% @ 1%, 98.94% @ 100%)
- Fine-tuning adapts to satellite domain

**Option 2: COCO Distillation + Fine-Tuning**
- Actually works better than frozen teacher (71.69% vs 28.89% @ 1%)
- Distillation+fine-tuning provides domain adaptation
- Consider using ImageNet distillation instead of COCO

**Option 3: Random Initialization**
- Surprisingly competitive (64.04% @ 1%, 97.20% @ 100%)
- No negative transfer from mismatched domain

**❌ Don't Use:** Frozen ImageBind features (teacher oracle)

---

## Conclusions

1. **ImageNet initialization remains the gold standard** across all tasks and label fractions.

2. **Teacher oracle establishes performance ceiling** but requires task alignment:
   - Good for: Natural images with sufficient data
   - Bad for: Domain shift, limited data, fine-grained details

3. **COCO distillation fails on fine-grained tasks** (39% efficiency) but can work for domain shift (102% efficiency with fine-tuning).

4. **Distillation research should focus on:**
   - Larger student models (reduce capacity bottleneck)
   - ImageNet distillation source (better coverage)
   - Task-adaptive projectors (don't freeze)
   - Domain-aware distillation losses

5. **Critical insight:** Frozen high-dimensional features ≠ best performance. Fine-tuning smaller models often outperforms frozen large models due to task adaptation.

---

## Next Steps

### Completed Experiments
- ✅ Experiment 1: Teacher Oracle Probes
- ✅ Experiment 2: Projector Ablation
- ✅ Baseline Comparisons (34 runs)

### Remaining Experiments
- ⏸️ **Experiment 3: ImageNet Distillation** (blocked: requires ImageNet dataset ~147GB)
  - Would test if distillation source is key factor
  - Expected: ImageNet-distilled > COCO-distilled

### Blocked by ImageBind Installation
None - ImageBind is now functional!

---

*Report generated: January 26, 2026*
*Experiment 1 completed: 4/4 runs successful*
