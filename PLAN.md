# Implementation Plan: ImageBind-Distilled Initialization

## Project Structure

```
universal_init/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── student.py          # RegNetY-400MF with projector head
│   │   └── teacher.py          # ImageBind wrapper (frozen)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── distill_datasets.py # Imagenette, COCO loaders
│   │   └── downstream_datasets.py # Pets, Flowers102, DTD, EuroSAT
│   ├── losses/
│   │   ├── __init__.py
│   │   └── distillation.py     # Cosine embedding loss + relational loss
│   ├── train_distill.py        # Distillation training script
│   ├── train_downstream.py     # Downstream fine-tuning script
│   └── evaluate.py             # Evaluation utilities
├── scripts/
│   ├── phase0_sanity.sh        # Imagenette sanity check
│   ├── phase1_distill.sh       # COCO distillation
│   ├── phase1_downstream.sh    # All downstream experiments
│   └── ablations.sh            # Loss and projector ablations
├── configs/                    # Default config values (reference only)
├── checkpoints/                # Saved models
├── data/                       # Dataset storage (symlink or actual)
├── results/                    # CSV results, plots
├── requirements.txt
├── setup.py
└── CLAUDE.md
```

## Implementation Phases

### Phase 0: Setup & Sanity Check

**Step 1: Environment & Dependencies**
- PyTorch 2.x with CUDA
- torchvision (for RegNetY-400MF pretrained weights)
- ImageBind (from facebookresearch/ImageBind)
- wandb
- Standard: numpy, pandas, matplotlib, tqdm

**Step 2: Model Implementation**

*Teacher (teacher.py)*:
- Load ImageBind vision encoder
- Freeze all parameters
- Method: `encode(images) -> embeddings` (L2 normalized)
- ImageBind outputs 1024-dim embeddings

*Student (student.py)*:
- RegNetY-400MF backbone from torchvision
- Three init modes: 'random', 'imagenet', 'distilled'
- Projector head: `Linear(440 -> 1024)` or small MLP `440 -> 512 -> 1024`
- Methods:
  - `get_features(images) -> backbone_features` (440-dim for RegNetY-400MF)
  - `project(features) -> embeddings` (1024-dim, L2 normalized)
  - `forward(images) -> embeddings`

**Step 3: Distillation Loss (distillation.py)**

```python
def embedding_loss(student_emb, teacher_emb):
    # Both L2-normalized
    return 1 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()

def relational_loss(student_emb, teacher_emb):
    # Cosine similarity matrices
    S_student = student_emb @ student_emb.T  # [B, B]
    S_teacher = teacher_emb @ teacher_emb.T  # [B, B]
    return F.mse_loss(S_student, S_teacher)

def combined_loss(student_emb, teacher_emb, lambda_rel=0.5):
    return embedding_loss(student_emb, teacher_emb) + lambda_rel * relational_loss(student_emb, teacher_emb)
```

**Step 4: Data Loaders**

*Distillation datasets*:
- Imagenette: Use `torchvision.datasets.Imagenette` or manual download
- COCO: Use `torchvision.datasets.CocoDetection` (images only, ignore annotations)

*Downstream datasets*:
- Oxford-IIIT Pets: `torchvision.datasets.OxfordIIITPet`
- Flowers102: `torchvision.datasets.Flowers102`
- DTD: `torchvision.datasets.DTD`
- EuroSAT: `torchvision.datasets.EuroSAT`

All with standard ImageNet normalization and augmentation.

**Step 5: Distillation Training Script (train_distill.py)**

Key arguments:
```
--dataset {imagenette, coco}
--epochs 100
--batch_size 256
--lr 1e-3
--weight_decay 1e-4
--loss {embedding, combined}
--lambda_rel 0.5
--projector {linear, mlp}
--output_dir checkpoints/
--wandb_project universal_init
--seed 42
```

Training loop:
1. Load teacher (frozen ImageBind)
2. Initialize student (random init backbone)
3. For each epoch:
   - Compute teacher embeddings (no grad)
   - Compute student embeddings
   - Compute loss, backprop
   - Log: loss, cosine similarity mean, wandb
4. Save backbone weights (without projector) as `distilled_backbone.pth`
5. Optionally save projector weights separately

**Step 6: Phase 0 Sanity Check**
```bash
# scripts/phase0_sanity.sh
python src/train_distill.py \
    --dataset imagenette \
    --epochs 20 \
    --batch_size 128 \
    --lr 1e-3 \
    --loss combined \
    --seed 42

# Quick downstream check
python src/train_downstream.py \
    --dataset imagenette \
    --init distilled \
    --checkpoint checkpoints/imagenette_distilled.pth \
    --epochs 20 \
    --label_fraction 1.0 \
    --seed 42
```

Success criteria:
- Distillation loss decreases smoothly
- Mean cosine similarity between student/teacher > 0.7 by end
- No NaN/instability

---

### Phase 1: Main Experiments

**Step 7: COCO Distillation**
```bash
# scripts/phase1_distill.sh
python src/train_distill.py \
    --dataset coco \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --loss combined \
    --lambda_rel 0.5 \
    --seed 42
```

**Step 8: Downstream Fine-tuning Script (train_downstream.py)**

Key arguments:
```
--dataset {pets, flowers102, dtd, eurosat}
--init {random, imagenet, distilled}
--checkpoint <path>  # for distilled init
--epochs 50
--batch_size 64
--lr 1e-2
--weight_decay 1e-4
--label_fraction {0.01, 0.1, 1.0}
--seed 42
--wandb_project universal_init
```

Training:
1. Load backbone with specified init
2. Add classifier head: `Linear(440, num_classes)`
3. Fine-tune entire network
4. Log: loss, accuracy per epoch, learning curves

**Step 9: Downstream Experiments**
```bash
# scripts/phase1_downstream.sh
for dataset in pets flowers102 dtd eurosat; do
    for init in random imagenet distilled; do
        for frac in 0.01 0.1 1.0; do
            for seed in 1 2 3; do
                python src/train_downstream.py \
                    --dataset $dataset \
                    --init $init \
                    --checkpoint checkpoints/coco_distilled.pth \
                    --label_fraction $frac \
                    --epochs 50 \
                    --seed $seed
            done
        done
    done
done
```

**Step 10: Ablations**
```bash
# scripts/ablations.sh

# Loss ablation: embedding-only vs combined
python src/train_distill.py --dataset coco --loss embedding --seed 42
python src/train_distill.py --dataset coco --loss combined --seed 42

# Projector ablation: keep vs discard
# (implement --keep_projector flag in train_downstream.py)
```

---

## Metrics & Logging (W&B)

**Distillation metrics**:
- `train/loss`
- `train/embedding_loss`
- `train/relational_loss`
- `train/cosine_sim_mean` (student vs teacher)
- `train/cosine_sim_std`

**Downstream metrics**:
- `train/loss`, `train/acc`
- `val/loss`, `val/acc`
- `val/acc_at_epoch_5`, `val/acc_at_epoch_10`, `val/acc_at_epoch_20`
- `val/best_acc`
- `val/aulc` (Area Under Learning Curve - trapezoidal integration of acc vs epoch)

**W&B organization**:
- Project: `universal_init`
- Groups: `phase0`, `distillation`, `downstream`
- Tags: dataset, init type, label fraction

---

## Evaluation & Analysis (evaluate.py)

1. **Learning curves**: Plot accuracy vs epoch for each init, grouped by dataset and label fraction
2. **AULC comparison**: Bar plots of AULC across inits
3. **Statistical tests**: Mean ± std across seeds, significance testing
4. **Summary tables**: Best accuracy and AULC for all combinations

---

## Implementation Order

1. [ ] Setup: requirements.txt, project structure, wandb init
2. [ ] Models: teacher.py (ImageBind wrapper), student.py (RegNetY-400MF + projector)
3. [ ] Losses: distillation.py
4. [ ] Data: distill_datasets.py (Imagenette, COCO)
5. [ ] Training: train_distill.py
6. [ ] **Phase 0 checkpoint**: Run sanity check on Imagenette
7. [ ] Data: downstream_datasets.py (Pets, Flowers102, DTD, EuroSAT)
8. [ ] Training: train_downstream.py
9. [ ] Scripts: phase1_distill.sh, phase1_downstream.sh
10. [ ] **Phase 1 checkpoint**: Run COCO distillation + all downstream experiments
11. [ ] Ablations: loss variants, projector handling
12. [ ] Analysis: evaluate.py, generate plots and tables

---

## Key Implementation Details

**ImageBind loading**:
```python
# Requires cloning facebookresearch/ImageBind or pip install
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
# Use model.forward({ModalityType.VISION: images}) or extract vision encoder
```

**RegNetY-400MF**:
```python
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights

# ImageNet pretrained
model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)

# Random init
model = regnet_y_400mf(weights=None)

# Feature dim: 440 (from model.fc.in_features after removing classifier)
```

**Label fraction sampling**:
- Use stratified sampling to maintain class balance
- Same subset across seeds for fair comparison (set random state)

**Data augmentation**:
- Distillation: RandomResizedCrop(224), RandomHorizontalFlip, Normalize
- Downstream train: Same as distillation
- Downstream val: Resize(256), CenterCrop(224), Normalize

---

## Questions/Decisions Made

1. **Projector architecture**: Start with simple Linear(440 → 1024), can ablate MLP later
2. **Optimizer**: AdamW for distillation, SGD with momentum for downstream (standard for fine-tuning)
3. **LR schedule**: Cosine annealing for both
4. **Epochs**: 100 for distillation, 50 for downstream (can adjust based on convergence)
