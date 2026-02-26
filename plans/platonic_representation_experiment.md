# Plan: Platonic Representation Experiment — CLIP vs Supervised Distillation

## Context

The project's goal is to show that initializing from a more "platonic" representation (one trained on more modalities / richer supervision) produces more efficient downstream learning. Current experiments compared ImageBind distillation vs ImageNet-pretrained weights, but this conflates **training method** (distillation vs supervised pretraining) with **representation quality**.

**Fix:** Use the same distillation pipeline with different teachers to isolate representation quality:
- **CLIP ViT-B/16** (more platonic — trained on image-text pairs, contrastive)
- **Supervised ViT-B/16** (less platonic — trained on ImageNet classification only)

Additionally, test whether CLIP's projection head (768→512) contributes to platonic-ness:
- **Option A:** Both teachers at 768-dim (strip heads) — isolates ViT encoder quality
- **Option B:** CLIP at 512-dim (native projection) — tests if projection head matters

For each distilled checkpoint, evaluate with **drop projector** and **trainable projector**.

---

## Experiment Matrix

### Phase 1: Distillation on COCO (3 runs, ~6-8 hrs each)

| ID | Teacher | Output Dim | timm model | head config |
|----|---------|-----------|------------|-------------|
| D1 | Supervised ViT-B/16 | 768 | `vit_base_patch16_224.augreg_in1k` | `num_classes=0` |
| D2 | CLIP ViT-B/16 (pre-proj) | 768 | `vit_base_patch16_clip_224.openai` | `num_classes=0` |
| D3 | CLIP ViT-B/16 (with proj) | 512 | `vit_base_patch16_clip_224.openai` | default head |

### Phase 2: Downstream on COCO checkpoints (24 runs)

For each of D1, D2, D3:
- 2 projector modes: **drop** and **trainable**
- 2 datasets: **Pets** and **EuroSAT**
- 2 fractions: **1%** and **100%**
= 8 runs per checkpoint × 3 checkpoints = 24 runs

Plus existing random init baseline (already completed).

### Phase 3 (optional): Repeat distillation on ImageNet

Same 3 teacher configs but distilling on ImageNet (1.28M images). Run after COCO results are analyzed. Much slower (~20+ hrs/run).

### Comparisons

**Option A (768 vs 768):** D2 vs D1 — same dim, same projector, only teacher training differs
**Option B (512 vs 768):** D3 vs D1 — native CLIP space vs supervised space
**Projection head effect:** D3 vs D2 — does CLIP's 768→512 projection help?

---

## Implementation

### Step 1: Create generic teacher wrapper

**New file:** `src/models/generic_teacher.py`

```python
class GenericTeacher(nn.Module):
    """Frozen teacher model from timm for distillation."""

    def __init__(self, model_name: str, device: str = "cuda", use_head: bool = False):
        # model_name: timm model name (e.g., "vit_base_patch16_clip_224.openai")
        # use_head: False = strip head (768-dim), True = keep native head (e.g., CLIP 512-dim)
        num_classes = 0 if not use_head else None  # None = keep default head
        self.model = timm.create_model(model_name, pretrained=True,
                                        **({"num_classes": 0} if not use_head else {}))
        self.model.eval()
        self.model.to(device)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Detect embed_dim by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            self.embed_dim = self.model(dummy).shape[-1]

    def get_transform_config(self):
        """Return model-specific normalization config from timm."""
        return timm.data.resolve_model_data_config(self.model)
        # Returns dict with 'mean', 'std', 'input_size', 'interpolation', etc.

    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        emb = self.model(images)
        return F.normalize(emb, p=2, dim=-1)  # L2-normalize like ImageBind teacher
```

Interface matches existing `ImageBindTeacher`: frozen, returns L2-normalized embeddings, exposes `embed_dim`.

**Transform handling:** Each teacher has different normalization stats:
- CLIP: mean≈(0.48, 0.46, 0.41), std≈(0.27, 0.26, 0.28)
- Supervised: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

The `get_transform_config()` method returns model-specific stats. The data loader in `train_distill.py` will use these stats. The student (random init) adapts to whatever normalization is used.

### Step 2: Make TEACHER_DIM configurable in StudentModel

**Modify:** `src/models/student.py`

- Change `TEACHER_DIM = 1024` from class constant to `__init__` parameter
- `for_distillation()` accepts `teacher_dim` parameter
- `for_downstream()` accepts `teacher_dim` parameter (needed for trainable projector)
- ProjectorHead out_dim uses teacher_dim instead of hardcoded 1024

Key changes:
```python
def __init__(self, ..., teacher_dim: int = 1024):
    self.teacher_dim = teacher_dim
    # ... use self.teacher_dim instead of self.TEACHER_DIM for projector out_dim
```

### Step 3: Update train_distill.py to support generic teachers

**Modify:** `src/train_distill.py`

- Add `--teacher` argument: `choices=["imagebind", "clip_768", "clip_512", "supervised"]`
- When teacher is not "imagebind", instantiate `GenericTeacher` instead of `ImageBindTeacher`
- Pass `teacher.embed_dim` to `StudentModel.for_distillation(teacher_dim=...)`
- Save `teacher_dim` in checkpoint for downstream loading

Teacher configs:
```python
TEACHER_CONFIGS = {
    "imagebind": {"class": "ImageBindTeacher"},  # existing, 1024-dim
    "clip_768": {"model": "vit_base_patch16_clip_224.openai", "num_classes": 0},   # 768-dim
    "clip_512": {"model": "vit_base_patch16_clip_224.openai", "num_classes": -1},  # 512-dim (keep proj head)
    "supervised": {"model": "vit_base_patch16_224.augreg_in1k", "num_classes": 0}, # 768-dim
}
```

Note: For `clip_512`, need to check whether `num_classes=-1` or just using default (not passing num_classes) preserves the head. Likely: just don't pass `num_classes` to `timm.create_model()`, then the default `Linear(768, 512)` head is kept.

### Step 4: Update train_downstream.py to read teacher_dim from checkpoint

**Modify:** `src/train_downstream.py`

- Add `--teacher_dim` argument (default: auto-detect from checkpoint)
- When loading distilled checkpoint, read `teacher_dim` from saved args or infer from projector weights shape
- Pass to `StudentModel.for_downstream(teacher_dim=...)`
- Update result CSV filename to include teacher name (e.g., `results_pets_distilled_clip768_frac0.01_s42.csv`)

Auto-detection logic:
```python
if args.teacher_dim is None and args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'args' in ckpt and hasattr(ckpt['args'], 'teacher_dim'):
        teacher_dim = ckpt['args'].teacher_dim
    elif 'student_state_dict' in ckpt:
        # Infer from projector weight shape: head.projector.weight is [teacher_dim, 440]
        teacher_dim = ckpt['student_state_dict']['head.projector.weight'].shape[0]
    else:
        teacher_dim = 1024  # Legacy default
```

### Step 5: Create experiment scripts

**New file:** `scripts/experiment4_platonic.sh`

Three distillation runs on COCO:
```bash
# D1: Supervised ViT-B/16 teacher (768-dim)
python src/train_distill.py --teacher supervised --dataset coco --epochs 30 ...

# D2: CLIP ViT-B/16 pre-projection (768-dim)
python src/train_distill.py --teacher clip_768 --dataset coco --epochs 30 ...

# D3: CLIP ViT-B/16 with projection (512-dim)
python src/train_distill.py --teacher clip_512 --dataset coco --epochs 30 ...
```

**New file:** `scripts/experiment4_downstream.sh`

24 downstream runs (3 checkpoints × 2 projector modes × 2 datasets × 2 fractions):
```bash
for CKPT in supervised_distilled clip768_distilled clip512_distilled; do
  for PROJ in "" "--keep_projector --train_projector"; do
    for DATASET in pets eurosat; do
      for FRAC in 0.01 1.0; do
        python src/train_downstream.py --init distilled --checkpoint $CKPT ...
      done
    done
  done
done
```

---

## Critical Files

### New Files
1. `src/models/generic_teacher.py` — Generic frozen teacher wrapper using timm
2. `scripts/experiment4_platonic.sh` — Distillation runs
3. `scripts/experiment4_downstream.sh` — Downstream evaluation runs

### Modified Files
1. `src/models/student.py` — Make `TEACHER_DIM` configurable via `teacher_dim` param
2. `src/train_distill.py` — Add `--teacher` arg, support GenericTeacher, save teacher_dim in checkpoint
3. `src/train_downstream.py` — Add `--teacher_dim` arg, auto-detect from checkpoint

### Existing Files (reuse as-is)
- `src/losses/distillation.py` — Loss functions work with any embedding dim (already generic)
- `src/data/distill_datasets.py` — COCO dataloader (already exists)
- `src/data/downstream_datasets.py` — Pets/EuroSAT loaders (already exist)

---

## Verification

### 1. Distillation sanity check
```bash
# Quick test: 1 epoch each to verify all 3 teachers work
python src/train_distill.py --teacher supervised --dataset coco --epochs 1 --no_wandb
python src/train_distill.py --teacher clip_768 --dataset coco --epochs 1 --no_wandb
python src/train_distill.py --teacher clip_512 --dataset coco --epochs 1 --no_wandb
```
- All should complete without errors
- Check cosine similarity is increasing
- Check checkpoint contains teacher_dim in saved args

### 2. Downstream sanity check
```bash
# Verify both projector modes work with new checkpoints
python src/train_downstream.py --init distilled --checkpoint <ckpt> --dataset pets --label_fraction 1.0 --epochs 5 --no_wandb
python src/train_downstream.py --init distilled --checkpoint <ckpt> --dataset pets --label_fraction 1.0 --epochs 5 --keep_projector --train_projector --no_wandb
```

### 3. Full experiment run
```bash
./scripts/experiment4_platonic.sh      # ~3 × 6-8 hours distillation
./scripts/experiment4_downstream.sh    # ~24 × 30 min downstream
```

### 4. Expected results structure
```
checkpoints/
├── supervised_distilled_best.pth
├── clip768_distilled_best.pth
├── clip512_distilled_best.pth
├── results_pets_distilled_supervised_frac0.01_s42.csv
├── results_pets_distilled_supervised_trainproj_frac0.01_s42.csv
├── results_pets_distilled_clip768_frac0.01_s42.csv
├── results_pets_distilled_clip768_trainproj_frac0.01_s42.csv
├── results_pets_distilled_clip512_frac0.01_s42.csv
├── results_pets_distilled_clip512_trainproj_frac0.01_s42.csv
└── ... (same pattern for eurosat, frac1.0)
```
