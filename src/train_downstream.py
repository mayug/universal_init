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
from src.data.downstream_datasets import get_downstream_dataloaders, get_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream fine-tuning")

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pets", "flowers102", "dtd", "eurosat", "imagenette"],
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

        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(progress * np.pi))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, amp):
    """Train for one epoch."""
    model.train()
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

    return total_loss / total, 100.0 * correct / total


def compute_aulc(accuracies: list) -> float:
    """Compute Area Under Learning Curve using trapezoidal rule."""
    if len(accuracies) < 2:
        return accuracies[0] if accuracies else 0.0
    return np.trapz(accuracies) / (len(accuracies) - 1)


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
    print("=" * 60)

    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    print(f"\nNumber of classes: {num_classes}")

    # Create model
    print(f"\nCreating model with {args.init} initialization...")
    model = StudentModel.for_downstream(
        num_classes=num_classes,
        init_mode=args.init,
        checkpoint_path=args.checkpoint,
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

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
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

    # Training loop
    print("\nStarting training...")
    best_acc = 0
    val_accuracies = []
    milestone_accs = {}  # Accuracy at epoch 5, 10, 20

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,
            args.device, args.amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        val_accuracies.append(val_acc)

        # Track milestone accuracies
        if epoch in [5, 10, 20]:
            milestone_accs[f"val/acc_at_epoch_{epoch}"] = val_acc

        # Log metrics
        metrics = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }

        if not args.no_wandb:
            wandb.log(metrics, step=epoch)

        # Print epoch summary
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}% | "
              f"Best: {max(best_acc, val_acc):.2f}%")

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc

    # Compute AULC
    aulc = compute_aulc(val_accuracies)

    # Log final metrics
    final_metrics = {
        "val/best_acc": best_acc,
        "val/final_acc": val_accuracies[-1],
        "val/aulc": aulc,
        **milestone_accs,
    }

    if not args.no_wandb:
        wandb.log(final_metrics)
        wandb.summary.update(final_metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Final accuracy: {val_accuracies[-1]:.2f}%")
    print(f"AULC: {aulc:.2f}")
    for key, value in milestone_accs.items():
        print(f"{key}: {value:.2f}%")
    print("=" * 60)

    # Save results to CSV
    results_path = os.path.join(
        args.output_dir,
        f"results_{args.dataset}_{args.init}_frac{args.label_fraction}_s{args.seed}.csv"
    )
    import pandas as pd
    results_df = pd.DataFrame([{
        "dataset": args.dataset,
        "init": args.init,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "best_acc": best_acc,
        "final_acc": val_accuracies[-1],
        "aulc": aulc,
        **milestone_accs,
    }])
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
