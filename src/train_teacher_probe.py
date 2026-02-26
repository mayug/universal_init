#!/usr/bin/env python3
"""Train linear probes on frozen ImageBind embeddings (teacher oracle baseline)."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import ImageBindTeacher
from src.data.downstream_datasets import get_downstream_dataloaders, get_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher oracle linear probe")

    # Data
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pets", "flowers102", "dtd", "eurosat", "imagenette"],
                        help="Downstream dataset")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        choices=[0.01, 0.1, 1.0],
                        help="Fraction of training labels to use")

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

    # System
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

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

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings(teacher, dataloader, device):
    """Extract ImageBind embeddings for entire dataset."""
    embeddings = []
    labels = []

    print("Extracting embeddings...")
    for images, targets in tqdm(dataloader, desc="Extracting"):
        images = images.to(device, non_blocking=True)
        embs = teacher.encode(images)  # [B, 1024], L2-normalized
        embeddings.append(embs.cpu())
        labels.append(targets)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Extracted {len(embeddings)} embeddings of shape {embeddings.shape}")
    return embeddings, labels


def train_epoch(probe, train_embs, train_labels, criterion, optimizer, device):
    """Train linear probe for one epoch."""
    probe.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create mini-batches
    batch_size = 256
    num_batches = (len(train_embs) + batch_size - 1) // batch_size

    indices = torch.randperm(len(train_embs))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_embs))
        batch_indices = indices[start_idx:end_idx]

        embs = train_embs[batch_indices].to(device)
        labels = train_labels[batch_indices].to(device)

        optimizer.zero_grad()
        outputs = probe(embs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_probe(probe, val_embs, val_labels, criterion, device):
    """Evaluate linear probe."""
    probe.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Evaluate in batches to avoid OOM
    batch_size = 256
    num_batches = (len(val_embs) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(val_embs))

        embs = val_embs[start_idx:end_idx].to(device)
        labels = val_labels[start_idx:end_idx].to(device)

        outputs = probe(embs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * embs.size(0)
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


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"{args.dataset}_teacher_oracle_frac{args.label_fraction}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["teacher_oracle", args.dataset, f"frac{args.label_fraction}"],
        )

    print("=" * 60)
    print("Teacher Oracle Linear Probe")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Label fraction: {args.label_fraction}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    print(f"\nNumber of classes: {num_classes}")

    # Load frozen ImageBind teacher
    print("\nLoading ImageBind teacher model...")
    teacher = ImageBindTeacher(device=args.device).load()
    print("Teacher model loaded and frozen")

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

    # Extract embeddings once (teacher is frozen)
    train_embs, train_labels = extract_embeddings(teacher, train_loader, args.device)
    val_embs, val_labels = extract_embeddings(teacher, val_loader, args.device)

    # Create linear probe
    probe = nn.Linear(1024, num_classes).to(args.device)
    print(f"\nLinear probe: {1024} → {num_classes}")
    print(f"Trainable parameters: {sum(p.numel() for p in probe.parameters()):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        probe.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print("\nStarting training...")
    best_acc = 0
    val_accuracies = []
    milestone_accs = {}  # Accuracy at epoch 5, 10, 20

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            probe, train_embs, train_labels,
            criterion, optimizer, args.device
        )
        val_loss, val_acc = evaluate_probe(
            probe, val_embs, val_labels,
            criterion, args.device
        )

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
        f"results_{args.dataset}_teacher_oracle_frac{args.label_fraction}_s{args.seed}.csv"
    )
    import pandas as pd
    results_df = pd.DataFrame([{
        "dataset": args.dataset,
        "init": "teacher_oracle",
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
