#!/usr/bin/env python3
"""Distillation training: train student to match ImageBind embeddings."""

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import ImageBindTeacher
from src.models.student import StudentModel
from src.data.distill_datasets import get_distill_dataloader
from src.losses.distillation import (
    embedding_loss,
    relational_loss,
    combined_loss,
    compute_similarity_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation training")

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=["imagenette", "coco"],
                        help="Distillation dataset")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")

    # Model
    parser.add_argument("--projector", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Projector head type")
    parser.add_argument("--projector_hidden_dim", type=int, default=512,
                        help="Hidden dim for MLP projector")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")

    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["embedding", "combined"],
                        help="Loss function")
    parser.add_argument("--lambda_rel", type=float, default=0.5,
                        help="Weight for relational loss")

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
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

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
    import numpy as np
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
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    teacher,
    student,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch,
):
    """Train for one epoch."""
    student.train()

    total_loss = 0
    total_emb_loss = 0
    total_rel_loss = 0
    total_cos_sim = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, images in enumerate(pbar):
        images = images.to(args.device, non_blocking=True)

        # Get teacher embeddings (no grad, already frozen)
        with torch.no_grad():
            teacher_emb = teacher(images)

        # Forward pass through student
        optimizer.zero_grad()

        if args.amp:
            with autocast():
                student_emb = student(images, normalize=True)
                if args.loss == "embedding":
                    loss = embedding_loss(student_emb, teacher_emb)
                    loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
                else:
                    loss, loss_dict = combined_loss(
                        student_emb, teacher_emb, args.lambda_rel
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_emb = student(images, normalize=True)
            if args.loss == "embedding":
                loss = embedding_loss(student_emb, teacher_emb)
                loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
            else:
                loss, loss_dict = combined_loss(
                    student_emb, teacher_emb, args.lambda_rel
                )

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_similarity_metrics(student_emb, teacher_emb)

        # Accumulate
        total_loss += loss_dict["total_loss"]
        total_emb_loss += loss_dict.get("embedding_loss", 0)
        total_rel_loss += loss_dict.get("relational_loss", 0)
        total_cos_sim += metrics["cosine_sim_mean"]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "cos_sim": f"{metrics['cosine_sim_mean']:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
        })

    # Average metrics
    avg_metrics = {
        "train/loss": total_loss / num_batches,
        "train/embedding_loss": total_emb_loss / num_batches,
        "train/relational_loss": total_rel_loss / num_batches,
        "train/cosine_sim_mean": total_cos_sim / num_batches,
        "train/lr": scheduler.get_last_lr()[0],
    }

    return avg_metrics


def save_checkpoint(student, optimizer, scheduler, epoch, args, filename):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "backbone_state_dict": student.get_backbone_state_dict(),
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"distill_{args.dataset}_{args.loss}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["distillation", args.dataset, args.loss],
        )

    print("=" * 60)
    print("Distillation Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Loss: {args.loss}")
    print(f"Projector: {args.projector}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load teacher
    print("\nLoading ImageBind teacher...")
    teacher = ImageBindTeacher(device=args.device).load()
    print(f"Teacher loaded. Embedding dim: {teacher.embed_dim}")

    # Create student
    print("\nCreating student model...")
    student = StudentModel.for_distillation(
        projector_type=args.projector,
        projector_hidden_dim=args.projector_hidden_dim,
    ).to(args.device)
    print(f"Student backbone dim: {student.BACKBONE_DIM}")
    print(f"Student projector: {args.projector}")

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    dataloader = get_distill_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer,
        args.epochs,
        args.warmup_epochs,
        len(dataloader),
    )
    scaler = GradScaler() if args.amp else None

    # Training loop
    print("\nStarting training...")
    best_cos_sim = 0

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            teacher, student, dataloader,
            optimizer, scheduler, scaler,
            args, epoch,
        )

        # Log to W&B
        if not args.no_wandb:
            wandb.log(metrics, step=epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Loss: {metrics['train/loss']:.4f}")
        print(f"  Cosine sim: {metrics['train/cosine_sim_mean']:.4f}")

        # Save best model
        if metrics["train/cosine_sim_mean"] > best_cos_sim:
            best_cos_sim = metrics["train/cosine_sim_mean"]
            save_checkpoint(
                student, optimizer, scheduler, epoch, args,
                f"{args.dataset}_distilled_best.pth"
            )

        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            save_checkpoint(
                student, optimizer, scheduler, epoch, args,
                f"{args.dataset}_distilled_epoch{epoch}.pth"
            )

    # Save final checkpoint
    save_checkpoint(
        student, optimizer, scheduler, args.epochs, args,
        f"{args.dataset}_distilled_final.pth"
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best cosine similarity: {best_cos_sim:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
