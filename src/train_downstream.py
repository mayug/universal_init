#!/usr/bin/env python3
"""Downstream fine-tuning on classification tasks."""

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

from src.models.student import StudentModel
from src.data.downstream_datasets import get_downstream_dataloaders, get_num_classes, is_multilabel_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream fine-tuning")

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pets", "flowers102", "dtd", "eurosat", "imagenette", "voc",
                                 "pathmnist", "dermamnist", "bloodmnist"],
                        help="Downstream dataset")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        choices=[0.01, 0.1, 1.0],
                        help="Fraction of training labels to use")

    # Model initialization
    parser.add_argument("--init", type=str, default="random",
                        choices=["random", "imagenet", "distilled"],
                        help="Weight initialization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to distilled checkpoint (required if init=distilled)")
    parser.add_argument("--keep_projector", action="store_true",
                        help="Keep projector from distillation (only with init=distilled)")
    parser.add_argument("--train_projector", action="store_true",
                        help="Make projector trainable (requires --keep_projector)")
    parser.add_argument("--teacher_dim", type=int, default=None,
                        help="Teacher embedding dim (auto-detected from checkpoint if not set)")
    parser.add_argument("--teacher_name", type=str, default=None,
                        help="Teacher name for result filenames (auto-detected from checkpoint)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone for linear probing (only train classifier head)")

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
                        help="Output directory for checkpoints")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="universal_init",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    args = parser.parse_args()

    # Validation
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

    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch

        # Guard against division by zero for very small datasets
        if warmup_steps == 0:
            warmup_steps = 1
        if total_steps <= warmup_steps:
            total_steps = warmup_steps + 1

        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(progress * np.pi))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, amp, freeze_backbone=False):
    """Train for one epoch."""
    model.train()
    if freeze_backbone:
        model.backbone.eval()  # Keep BN running stats frozen for linear probing
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
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
    return np.trapz(accuracies) / (len(accuracies) - 1)


def compute_map(all_targets, all_scores):
    """Compute mean Average Precision for multi-label classification."""
    from sklearn.metrics import average_precision_score
    per_class_ap = average_precision_score(all_targets, all_scores, average=None)
    return float(np.nanmean(per_class_ap))


def train_epoch_multilabel(model, dataloader, criterion, optimizer, scheduler, scaler, device, amp, freeze_backbone=False):
    """Train for one epoch (multi-label)."""
    model.train()
    if freeze_backbone:
        model.backbone.eval()
    total_loss = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_multilabel(model, dataloader, criterion, device):
    """Evaluate model on validation set (multi-label, returns mAP)."""
    model.eval()
    total_loss = 0
    total = 0
    all_targets = []
    all_scores = []

    for images, labels in tqdm(dataloader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_targets.append(labels.cpu().numpy())
        all_scores.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    mAP = compute_map(all_targets, all_scores)
    return total_loss / total if total > 0 else 0.0, mAP * 100.0


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"{args.dataset}_{args.init}_frac{args.label_fraction}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["downstream", args.dataset, args.init, f"frac{args.label_fraction}"],
        )

    print("=" * 60)
    print("Downstream Fine-tuning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Init: {args.init}")
    print(f"Label fraction: {args.label_fraction}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    if args.freeze_backbone:
        print(f"Mode: LINEAR PROBE (backbone frozen)")
    print("=" * 60)

    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    print(f"\nNumber of classes: {num_classes}")

    # Auto-detect teacher_dim and teacher_name from checkpoint
    teacher_dim = args.teacher_dim or 1024  # default for legacy ImageBind
    teacher_name = args.teacher_name
    if args.init == "distilled" and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        if args.teacher_dim is None and "student_state_dict" in ckpt:
            # Detect from projector weight shape
            student_state = ckpt["student_state_dict"]
            if "head.projector.weight" in student_state:
                teacher_dim = student_state["head.projector.weight"].shape[0]
            elif "head.weight" in student_state:
                teacher_dim = student_state["head.weight"].shape[0]
        if teacher_name is None:
            teacher_name = ckpt_args.get("teacher", "imagebind")
        del ckpt  # free memory
        print(f"Auto-detected teacher_dim={teacher_dim}, teacher_name={teacher_name}")

    # Create model
    print(f"\nCreating model with {args.init} initialization...")
    if args.keep_projector:
        if args.train_projector:
            print("Keeping TRAINABLE projector from distillation")
        else:
            print("Keeping FROZEN projector from distillation")
    model = StudentModel.for_downstream(
        num_classes=num_classes,
        init_mode=args.init,
        checkpoint_path=args.checkpoint,
        keep_projector=args.keep_projector,
        train_projector=args.train_projector,
        teacher_dim=teacher_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, _ = get_downstream_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=args.label_fraction,
        seed=args.seed,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Determine if multi-label
    multilabel = is_multilabel_dataset(args.dataset)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer,
        args.epochs,
        args.warmup_epochs,
        len(train_loader),
    )
    scaler = GradScaler() if args.amp else None

    # Metric name for multi-label vs single-label
    metric_name = "mAP" if multilabel else "Acc"

    # Training loop
    print("\nStarting training...")
    best_metric = 0
    val_metrics = []
    milestone_vals = {}  # Metric at epoch 5, 10, 20

    for epoch in range(1, args.epochs + 1):
        if multilabel:
            train_loss = train_epoch_multilabel(
                model, train_loader, criterion,
                optimizer, scheduler, scaler,
                args.device, args.amp,
                freeze_backbone=args.freeze_backbone,
            )
            val_loss, val_metric = evaluate_multilabel(
                model, val_loader, criterion, args.device,
            )
            train_metric = None  # mAP not computed per-batch during training
        else:
            train_loss, train_metric = train_epoch(
                model, train_loader, criterion,
                optimizer, scheduler, scaler,
                args.device, args.amp,
                freeze_backbone=args.freeze_backbone,
            )
            val_loss, val_metric = evaluate(model, val_loader, criterion, args.device)

        val_metrics.append(val_metric)

        # Track milestone values
        if epoch in [5, 10, 20]:
            milestone_key = f"val/{metric_name.lower()}_at_epoch_{epoch}"
            milestone_vals[milestone_key] = val_metric

        # Log metrics
        metrics = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            f"val/{metric_name.lower()}": val_metric,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
        if train_metric is not None:
            metrics[f"train/{metric_name.lower()}"] = train_metric

        if not args.no_wandb:
            wandb.log(metrics, step=epoch)

        # Print epoch summary
        train_str = f"Loss: {train_loss:.4f}" if train_metric is None else f"{metric_name}: {train_metric:.2f}%"
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train {train_str} | "
              f"Val {metric_name}: {val_metric:.2f}% | "
              f"Best: {max(best_metric, val_metric):.2f}%")

        # Track best
        if val_metric > best_metric:
            best_metric = val_metric

    # Compute AULC
    aulc = compute_aulc(val_metrics)

    # Log final metrics
    final_metrics = {
        f"val/best_{metric_name.lower()}": best_metric,
        f"val/final_{metric_name.lower()}": val_metrics[-1],
        f"val/aulc_{metric_name.lower()}": aulc,
        **milestone_vals,
    }

    if not args.no_wandb:
        wandb.log(final_metrics)
        wandb.summary.update(final_metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best {metric_name}: {best_metric:.2f}%")
    print(f"Final {metric_name}: {val_metrics[-1]:.2f}%")
    print(f"AULC ({metric_name}): {aulc:.2f}")
    for key, value in milestone_vals.items():
        print(f"{key}: {value:.2f}%")
    print("=" * 60)

    # Save results to CSV
    if args.freeze_backbone:
        suffix = "_linprobe"
    elif args.keep_projector:
        suffix = "_trainproj" if args.train_projector else "_keepproj"
    else:
        suffix = ""
    init_label = args.init
    if args.init == "distilled" and teacher_name and teacher_name != "imagebind":
        init_label = f"distilled_{teacher_name}"
    results_path = os.path.join(
        args.output_dir,
        f"results_{args.dataset}_{init_label}{suffix}_frac{args.label_fraction}_s{args.seed}.csv"
    )
    import pandas as pd
    results_row = {
        "dataset": args.dataset,
        "init": args.init,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "keep_projector": args.keep_projector,
        "train_projector": args.train_projector,
        "freeze_backbone": args.freeze_backbone,
    }
    if multilabel:
        results_row.update({
            "best_mAP": best_metric,
            "final_mAP": val_metrics[-1],
            "aulc_mAP": aulc,
            "best_acc": None,
            "final_acc": None,
            "aulc": None,
        })
    else:
        results_row.update({
            "best_acc": best_metric,
            "final_acc": val_metrics[-1],
            "aulc": aulc,
            "best_mAP": None,
            "final_mAP": None,
            "aulc_mAP": None,
        })
    results_row.update(milestone_vals)
    results_df = pd.DataFrame([results_row])
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
