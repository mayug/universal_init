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
from src.data.distill_datasets import get_distill_dataloader, get_distill_dataloaders_with_val
from src.losses.distillation import (
    embedding_loss,
    relational_loss,
    combined_loss,
    cka_combined_loss,
    compute_similarity_metrics,
)
from src.losses.validation_metrics import (
    validate_distillation,
    validate_with_linear_probe,
)


TEACHER_CONFIGS = {
    "imagebind":   {"class": "ImageBindTeacher"},
    "supervised":  {"model": "vit_base_patch16_224.augreg_in1k",   "use_head": False},  # 768-dim
    "clip_768":    {"model": "vit_base_patch16_clip_224.openai",    "use_head": False},  # 768-dim
    "clip_512":    {"model": "vit_base_patch16_clip_224.openai",    "use_head": True},   # 512-dim
}


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation training")

    # Teacher
    parser.add_argument("--teacher", type=str, default="imagebind",
                        choices=list(TEACHER_CONFIGS.keys()),
                        help="Teacher model for distillation")

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=["imagenette", "coco", "imagenet"],
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
                        choices=["embedding", "combined", "cka_combined"],
                        help="Loss function")
    parser.add_argument("--lambda_rel", type=float, default=0.5,
                        help="Weight for relational loss")
    parser.add_argument("--lambda_cka", type=float, default=0.5,
                        help="Weight for CKA loss (used with cka_combined)")

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

    # Validation
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--val_max_batches", type=int, default=50,
                        help="Max batches for validation (for speed)")
    parser.add_argument("--probe_every", type=int, default=10,
                        help="Run linear probe every N epochs (0 to disable)")
    parser.add_argument("--probe_dataset", type=str, default="imagenette",
                        help="Dataset for linear probe evaluation")

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
    total_cka_loss = 0
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

            # Compute loss in float32 outside autocast for numerical stability.
            # CKA's backward through sqrt/division can overflow under AMP scaling.
            student_emb_f32 = student_emb.float()
            teacher_emb_f32 = teacher_emb.float()
            if args.loss == "embedding":
                loss = embedding_loss(student_emb_f32, teacher_emb_f32)
                loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_combined":
                loss, loss_dict = cka_combined_loss(
                    student_emb_f32, teacher_emb_f32, args.lambda_cka
                )
            else:  # combined (relational)
                loss, loss_dict = combined_loss(
                    student_emb_f32, teacher_emb_f32, args.lambda_rel
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_emb = student(images, normalize=True)
            if args.loss == "embedding":
                loss = embedding_loss(student_emb, teacher_emb)
                loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_combined":
                loss, loss_dict = cka_combined_loss(
                    student_emb, teacher_emb, args.lambda_cka
                )
            else:  # combined (relational)
                loss, loss_dict = combined_loss(
                    student_emb, teacher_emb, args.lambda_rel
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_similarity_metrics(student_emb, teacher_emb)

        # Accumulate
        total_loss += loss_dict["total_loss"]
        total_emb_loss += loss_dict.get("embedding_loss", 0)
        total_rel_loss += loss_dict.get("relational_loss", 0)
        total_cka_loss += loss_dict.get("cka_loss", 0)
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
        "train/cka_loss": total_cka_loss / num_batches,
        "train/cka_value": 1.0 - (total_cka_loss / num_batches) if total_cka_loss > 0 else 0,
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
        run_name = args.wandb_run_name or f"distill_{args.teacher}_{args.dataset}_{args.loss}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["distillation", args.dataset, args.loss, args.teacher],
        )

    print("=" * 60)
    print("Distillation Training")
    print("=" * 60)
    print(f"Teacher: {args.teacher}")
    print(f"Dataset: {args.dataset}")
    print(f"Loss: {args.loss}")
    print(f"Projector: {args.projector}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Validation fraction: {args.val_fraction}")
    if args.loss == "cka_combined":
        print(f"Lambda CKA: {args.lambda_cka}")
    print("=" * 60)

    # Load teacher
    config = TEACHER_CONFIGS[args.teacher]
    if args.teacher == "imagebind":
        print("\nLoading ImageBind teacher...")
        teacher = ImageBindTeacher(device=args.device).load()
    else:
        print(f"\nLoading {args.teacher} teacher ({config['model']})...")
        from src.models.generic_teacher import GenericTeacher
        teacher = GenericTeacher(
            model_name=config["model"],
            device=args.device,
            use_head=config.get("use_head", False),
        )
    print(f"Teacher loaded. Embedding dim: {teacher.embed_dim}")

    # Create student
    print("\nCreating student model...")
    student = StudentModel.for_distillation(
        projector_type=args.projector,
        projector_hidden_dim=args.projector_hidden_dim,
        teacher_dim=teacher.embed_dim,
    ).to(args.device)
    print(f"Student backbone dim: {student.BACKBONE_DIM}")
    print(f"Student projector: {args.projector}")

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Get teacher-specific transform config
    transform_kwargs = {}
    if args.teacher != "imagebind":
        data_config = teacher.get_transform_config()
        transform_kwargs = {"mean": list(data_config["mean"]), "std": list(data_config["std"])}

    # Load data with train/val split
    print(f"\nLoading {args.dataset} dataset with validation split...")
    train_loader, val_loader = get_distill_dataloaders_with_val(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        **transform_kwargs,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")

    # For backward compatibility
    dataloader = train_loader

    # Load probe dataset if needed
    probe_train_loader = None
    probe_val_loader = None
    probe_num_classes = None
    if args.probe_every > 0:
        print(f"\nLoading {args.probe_dataset} for linear probe evaluation...")
        from src.data.downstream_datasets import get_downstream_dataloaders
        probe_train_loader, probe_val_loader, probe_num_classes = get_downstream_dataloaders(
            dataset_name=args.probe_dataset,
            data_root=args.data_root,
            batch_size=64,
            num_workers=args.num_workers,
            label_fraction=1.0,
            seed=args.seed,
        )
        print(f"Probe dataset: {args.probe_dataset}, {probe_num_classes} classes")

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
    best_val_cos_sim = 0
    best_train_cos_sim = 0

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            teacher, student, dataloader,
            optimizer, scheduler, scaler,
            args, epoch,
        )

        # Run validation
        if epoch % args.val_every == 0:
            print("Running validation...")
            val_metrics = validate_distillation(
                teacher=teacher,
                student=student,
                val_loader=val_loader,
                device=args.device,
                max_batches=args.val_max_batches,
            )
            metrics.update(val_metrics)

            # Check for overfitting
            train_cos = metrics["train/cosine_sim_mean"]
            val_cos = metrics["val/cosine_mean"]
            gap = train_cos - val_cos
            metrics["val/train_val_gap"] = gap

            print(f"  Val cosine: {val_cos:.4f} (train-val gap: {gap:.4f})")
            print(f"  Val R@1: {metrics['val/retrieval_R@1']:.4f}, R@5: {metrics['val/retrieval_R@5']:.4f}")
            print(f"  Val RSA corr: {metrics['val/rsa_correlation']:.4f}")
            print(f"  Val CKA (proj): {metrics['val/cka_projected']:.4f}, CKA (backbone): {metrics['val/cka_backbone']:.4f}")
            print(f"  Backbone eff. rank: {metrics['val/backbone_collapse_effective_rank']:.1f}")

        # Run linear probe periodically
        if args.probe_every > 0 and epoch % args.probe_every == 0:
            print("Running linear probe...")
            probe_metrics = validate_with_linear_probe(
                student=student,
                probe_train_loader=probe_train_loader,
                probe_val_loader=probe_val_loader,
                num_classes=probe_num_classes,
                device=args.device,
            )
            metrics.update(probe_metrics)
            print(f"  Probe accuracy: {metrics['val/probe_accuracy']:.4f}")

        # Log to W&B
        if not args.no_wandb:
            wandb.log(metrics, step=epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {metrics['train/loss']:.4f}")
        print(f"  Train Cosine sim: {metrics['train/cosine_sim_mean']:.4f}")

        # Save best model based on validation cosine similarity
        val_cos_sim = metrics.get("val/cosine_mean", metrics["train/cosine_sim_mean"])
        if val_cos_sim > best_val_cos_sim:
            best_val_cos_sim = val_cos_sim
            if args.loss == "cka_combined":
                ckpt_name = f"{args.dataset}_{args.teacher}_cka_l{args.lambda_cka}_distilled_best.pth"
            else:
                ckpt_name = f"{args.dataset}_{args.teacher}_distilled_best.pth"
            save_checkpoint(
                student, optimizer, scheduler, epoch, args,
                ckpt_name
            )

        # Track best training cosine sim
        if metrics["train/cosine_sim_mean"] > best_train_cos_sim:
            best_train_cos_sim = metrics["train/cosine_sim_mean"]

        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            if args.loss == "cka_combined":
                periodic_name = f"{args.dataset}_{args.teacher}_cka_l{args.lambda_cka}_distilled_epoch{epoch}.pth"
            else:
                periodic_name = f"{args.dataset}_{args.teacher}_distilled_epoch{epoch}.pth"
            save_checkpoint(
                student, optimizer, scheduler, epoch, args,
                periodic_name
            )

    # Save final checkpoint
    if args.loss == "cka_combined":
        final_name = f"{args.dataset}_{args.teacher}_cka_l{args.lambda_cka}_distilled_final.pth"
    else:
        final_name = f"{args.dataset}_{args.teacher}_distilled_final.pth"
    save_checkpoint(
        student, optimizer, scheduler, args.epochs, args,
        final_name
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best train cosine similarity: {best_train_cos_sim:.4f}")
    print(f"Best val cosine similarity: {best_val_cos_sim:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
