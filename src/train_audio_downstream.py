#!/usr/bin/env python3
"""Audio downstream evaluation with fold-based cross-validation.

Evaluates audio representations on ESC-50 (5-fold CV) and UrbanSound8K
(10-fold CV) classification tasks. Supports four initialization conditions:
  - random: Random initialization (floor)
  - audioset_pretrained: EfficientAT mn10_as weights (ceiling)
  - distilled: Weights from cross-modal distillation checkpoint
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_student import AudioStudentModel
from src.data.audio_datasets import (
    ESC50Dataset,
    UrbanSound8KDataset,
    audio_label_collate_fn,
    create_label_fraction_subset,
)


DATASET_INFO = {
    "esc50": {
        "class": ESC50Dataset,
        "num_classes": 50,
        "num_folds": 5,
        "max_duration": 5.0,
    },
    "urbansound8k": {
        "class": UrbanSound8KDataset,
        "num_classes": 10,
        "num_folds": 10,
        "max_duration": 4.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audio downstream evaluation")

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_INFO.keys()),
                        help="Downstream dataset")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        choices=[0.01, 0.1, 1.0],
                        help="Fraction of training labels to use")

    # Cross-validation
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (1-indexed). If None, run all folds.")
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Audio sample rate")

    # Model initialization
    parser.add_argument("--init", type=str, default="random",
                        choices=["random", "audioset_pretrained", "distilled"],
                        help="Weight initialization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to distilled checkpoint (required if init=distilled)")
    parser.add_argument("--keep_projector", action="store_true",
                        help="Keep projector from distillation")
    parser.add_argument("--train_projector", action="store_true",
                        help="Make projector trainable")
    parser.add_argument("--teacher_dim", type=int, default=None,
                        help="Teacher embedding dim (auto-detected from checkpoint)")
    parser.add_argument("--teacher_name", type=str, default=None,
                        help="Teacher name for result filenames")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone for linear probing")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")

    # System
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for results")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="universal_init",
                        help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    args = parser.parse_args()

    if args.init == "distilled" and args.checkpoint is None:
        parser.error("--checkpoint is required when --init=distilled")

    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, num_epochs, warmup_epochs, steps_per_epoch):
    """Create learning rate scheduler with warmup and cosine decay."""
    import math

    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch
        if warmup_steps == 0:
            warmup_steps = 1
        if total_steps <= warmup_steps:
            total_steps = warmup_steps + 1

        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(progress * math.pi))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_fold_loaders(args, test_fold, info):
    """Create train and test dataloaders for a single fold.

    For fold-based CV, the test fold is held out and all other folds
    form the training set.
    """
    num_folds = info["num_folds"]
    all_folds = list(range(1, num_folds + 1))
    train_folds = [f for f in all_folds if f != test_fold]

    dataset_cls = info["class"]
    dataset_root = os.path.join(args.data_root, args.dataset)

    train_dataset = dataset_cls(
        root=dataset_root,
        folds=train_folds,
        sample_rate=args.sample_rate,
        max_duration=info["max_duration"],
    )
    test_dataset = dataset_cls(
        root=dataset_root,
        folds=[test_fold],
        sample_rate=args.sample_rate,
        max_duration=info["max_duration"],
    )

    # Apply label fraction subsampling to training set
    if args.label_fraction < 1.0:
        train_dataset = create_label_fraction_subset(
            train_dataset, args.label_fraction, args.seed
        )
        print(f"  Using {len(train_dataset)} training samples ({args.label_fraction*100:.0f}%)")

    effective_batch_size = min(args.batch_size, len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > effective_batch_size,
        collate_fn=audio_label_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=audio_label_collate_fn,
    )

    return train_loader, test_loader


def create_model(args, num_classes, teacher_dim):
    """Create and configure the audio student model."""
    model = AudioStudentModel.for_downstream(
        num_classes=num_classes,
        init_mode=args.init,
        checkpoint_path=args.checkpoint,
        keep_projector=args.keep_projector,
        train_projector=args.train_projector,
        teacher_dim=teacher_dim,
        freeze_backbone=args.freeze_backbone,
        sample_rate=args.sample_rate,
    ).to(args.device)
    return model


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, amp, freeze_backbone=False):
    """Train for one epoch."""
    model.train()
    if freeze_backbone:
        model.backbone.backbone.eval()  # Keep BN running stats frozen
    total_loss = 0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(dataloader, desc="Train", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if amp:
            with autocast():
                outputs = model.mel_forward(waveforms, normalize=False)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model.mel_forward(waveforms, normalize=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * waveforms.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def calibrate_bn(model, train_loader, device, num_batches=None):
    """Recalibrate BatchNorm running stats using training data.

    The mel spectrogram range differs significantly from BN's default
    initialization (mean=0, var=1). The exponential moving average used
    during training may not converge fast enough, causing eval-mode
    predictions to collapse. This function resets BN stats and recomputes
    them from scratch using the training data.
    """
    model.train()
    # Reset running stats and temporarily use cumulative moving average
    # (momentum=None) instead of EMA for accurate calibration
    bn_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers.append((m, m.momentum))
            m.reset_running_stats()
            m.momentum = None  # Use cumulative mean (1/N) instead of EMA

    # Accumulate stats
    count = 0
    for waveforms, _ in train_loader:
        waveforms = waveforms.to(device, non_blocking=True)
        model.mel_forward(waveforms, normalize=False)
        count += 1
        if num_batches is not None and count >= num_batches:
            break

    # Restore original momentum
    for m, orig_momentum in bn_layers:
        m.momentum = orig_momentum


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(dataloader, desc="Eval", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model.mel_forward(waveforms, normalize=False)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * waveforms.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def compute_aulc(accuracies: list) -> float:
    """Compute Area Under Learning Curve using trapezoidal rule."""
    if len(accuracies) < 2:
        return accuracies[0] if accuracies else 0.0
    return float(np.trapz(accuracies) / (len(accuracies) - 1))


def train_single_fold(args, test_fold, info, teacher_dim):
    """Train and evaluate on a single fold. Returns best test accuracy."""
    num_classes = info["num_classes"]

    print(f"\n--- Fold {test_fold} ---")
    set_seed(args.seed + test_fold)  # Different seed per fold for reproducibility

    # Data
    train_loader, test_loader = make_fold_loaders(args, test_fold, info)
    print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model (fresh init for each fold)
    model = create_model(args, num_classes, teacher_dim)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer, args.epochs, args.warmup_epochs, len(train_loader)
    )
    scaler = GradScaler() if args.amp else None

    # Training loop
    best_acc = 0
    test_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,
            args.device, args.amp,
            freeze_backbone=args.freeze_backbone,
        )
        # Recalibrate BN running stats before evaluation.
        # Mel spectrograms have a very different distribution than BN's
        # default (mean=0, var=1), and EMA-based updates during training
        # may not converge fast enough, causing eval predictions to collapse.
        calibrate_bn(model, train_loader, args.device)

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, args.device
        )

        test_accs.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch}/{args.epochs} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")

    aulc = compute_aulc(test_accs)
    print(f"  Fold {test_fold} result: Best={best_acc:.2f}%, AULC={aulc:.2f}")

    return {
        "fold": test_fold,
        "best_acc": best_acc,
        "final_acc": test_accs[-1] if test_accs else 0,
        "aulc": aulc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    info = DATASET_INFO[args.dataset]
    num_folds = info["num_folds"]

    # Auto-detect teacher_dim and teacher_name from checkpoint
    teacher_dim = args.teacher_dim or 768  # default for CLIP/SBERT
    teacher_name = args.teacher_name
    if args.init == "distilled" and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        if args.teacher_dim is None and "student_state_dict" in ckpt:
            student_state = ckpt["student_state_dict"]
            if "head.projector.weight" in student_state:
                teacher_dim = student_state["head.projector.weight"].shape[0]
            elif "head.weight" in student_state:
                teacher_dim = student_state["head.weight"].shape[0]
        if teacher_name is None:
            teacher_name = ckpt_args.get("teacher", "unknown")
        del ckpt
        print(f"Auto-detected teacher_dim={teacher_dim}, teacher_name={teacher_name}")

    # Initialize W&B
    if not args.no_wandb:
        init_label = args.init
        if args.init == "distilled" and teacher_name:
            init_label = f"distilled_{teacher_name}"
        suffix = "_linprobe" if args.freeze_backbone else ""
        run_name = f"audio_{args.dataset}_{init_label}{suffix}_frac{args.label_fraction}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["audio_downstream", args.dataset, args.init,
                  f"frac{args.label_fraction}", "experiment10"],
        )

    print("=" * 60)
    print("Audio Downstream Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({info['num_classes']} classes)")
    print(f"Init: {args.init}")
    print(f"Label fraction: {args.label_fraction}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Cross-validation: {num_folds}-fold")
    if args.freeze_backbone:
        print(f"Mode: LINEAR PROBE (backbone frozen)")
    if args.fold:
        print(f"Running single fold: {args.fold}")
    print("=" * 60)

    # Determine which folds to run
    if args.fold is not None:
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(1, num_folds + 1))

    # Run folds
    fold_results = []
    for test_fold in folds_to_run:
        result = train_single_fold(args, test_fold, info, teacher_dim)
        fold_results.append(result)

        # Log per-fold result to W&B
        if not args.no_wandb:
            wandb.log({
                f"fold_{test_fold}/best_acc": result["best_acc"],
                f"fold_{test_fold}/aulc": result["aulc"],
            })

    # Aggregate results
    best_accs = [r["best_acc"] for r in fold_results]
    aulcs = [r["aulc"] for r in fold_results]

    mean_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)
    mean_aulc = np.mean(aulcs)
    std_aulc = np.std(aulcs)

    print("\n" + "=" * 60)
    print(f"Cross-Validation Results ({len(folds_to_run)} folds)")
    print("=" * 60)
    print(f"Best Accuracy: {mean_acc:.2f} +/- {std_acc:.2f}%")
    print(f"AULC: {mean_aulc:.2f} +/- {std_aulc:.2f}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_acc']:.2f}%")
    print("=" * 60)

    # Log aggregated results to W&B
    if not args.no_wandb:
        wandb.log({
            "cv/mean_best_acc": mean_acc,
            "cv/std_best_acc": std_acc,
            "cv/mean_aulc": mean_aulc,
            "cv/std_aulc": std_aulc,
        })
        wandb.summary.update({
            "cv/mean_best_acc": mean_acc,
            "cv/std_best_acc": std_acc,
            "cv/mean_aulc": mean_aulc,
        })

    # Save results to CSV
    import pandas as pd

    init_label = args.init
    if args.init == "distilled" and teacher_name:
        init_label = f"distilled_{teacher_name}"
    suffix = "_linprobe" if args.freeze_backbone else ""
    fold_str = f"_fold{args.fold}" if args.fold else "_allcv"

    results_path = os.path.join(
        args.output_dir,
        f"results_audio_{args.dataset}_{init_label}{suffix}_frac{args.label_fraction}{fold_str}_s{args.seed}.csv"
    )

    rows = []
    for r in fold_results:
        rows.append({
            "dataset": args.dataset,
            "init": args.init,
            "teacher_name": teacher_name,
            "label_fraction": args.label_fraction,
            "seed": args.seed,
            "freeze_backbone": args.freeze_backbone,
            "fold": r["fold"],
            "best_acc": r["best_acc"],
            "final_acc": r["final_acc"],
            "aulc": r["aulc"],
        })
    # Summary row
    rows.append({
        "dataset": args.dataset,
        "init": args.init,
        "teacher_name": teacher_name,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "freeze_backbone": args.freeze_backbone,
        "fold": "mean",
        "best_acc": mean_acc,
        "final_acc": np.mean([r["final_acc"] for r in fold_results]),
        "aulc": mean_aulc,
    })

    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
