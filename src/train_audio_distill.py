#!/usr/bin/env python3
"""Cross-modal audio distillation: train audio student to match text teacher embeddings.

The teacher (CLIP text or Sentence-BERT) processes text captions while
the student processes the corresponding audio spectrograms. This tests
whether a text/vision model's geometric structure can transfer to a
modality it has never seen.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_student import AudioStudentModel
from src.models.text_teacher import CLIPTextTeacher, SentenceBERTTeacher
from src.data.audio_datasets import get_audiocaps_dataloaders_with_val
from src.losses.distillation import (
    embedding_loss,
    cka_loss,
    cka_combined_loss,
    compute_similarity_metrics,
)


TEACHER_CONFIGS = {
    "clip_text": {
        "class": CLIPTextTeacher,
        "kwargs": {"model_name": "openai/clip-vit-large-patch14"},
    },
    "sentence_bert": {
        "class": SentenceBERTTeacher,
        "kwargs": {"model_name": "all-mpnet-base-v2"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-modal audio distillation")

    # Teacher
    parser.add_argument("--teacher", type=str, default="clip_text",
                        choices=list(TEACHER_CONFIGS.keys()),
                        help="Text teacher model for distillation")

    # Data
    parser.add_argument("--dataset", type=str, default="audiocaps",
                        choices=["audiocaps"],
                        help="Distillation dataset")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--sample_rate", type=int, default=32000,
                        help="Audio sample rate")
    parser.add_argument("--max_duration", type=float, default=10.0,
                        help="Maximum audio clip duration in seconds")

    # Model
    parser.add_argument("--projector", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Projector head type")
    parser.add_argument("--projector_hidden_dim", type=int, default=512,
                        help="Hidden dim for MLP projector")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of warmup epochs")

    # Loss
    parser.add_argument("--loss", type=str, default="embedding",
                        choices=["embedding", "cka_combined", "cka_only"],
                        help="Loss function")
    parser.add_argument("--lambda_cka", type=float, default=0.5,
                        help="Weight for CKA loss (used with cka_combined)")

    # SpecAugment
    parser.add_argument("--freqm", type=int, default=48,
                        help="Frequency masking parameter for SpecAugment")
    parser.add_argument("--timem", type=int, default=192,
                        help="Time masking parameter for SpecAugment")

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
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="universal_init",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # Whitening
    parser.add_argument("--whiten", type=str, default=None,
                        help="Path to whitening stats (mean/std) to apply to teacher embeddings")

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
    import math

    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch

        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(progress * math.pi))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def encode_teacher_batch(teacher, captions, teacher_name, whiten_stats=None):
    """Encode a batch of captions through the teacher model.

    Handles the different interfaces of CLIP vs Sentence-BERT.
    Optionally whitens embeddings using precomputed mean/std.
    """
    with torch.no_grad():
        if teacher_name == "clip_text":
            tokens = teacher.tokenize(captions)
            teacher_emb = teacher(tokens)
        else:  # sentence_bert
            teacher_emb = teacher.encode(captions)

        if whiten_stats is not None:
            mean = whiten_stats["mean"].to(teacher_emb.device)
            std = whiten_stats["std"].to(teacher_emb.device)
            teacher_emb = (teacher_emb - mean) / (std + 1e-8)
            teacher_emb = torch.nn.functional.normalize(teacher_emb, p=2, dim=-1)

    return teacher_emb


def train_epoch(teacher, student, dataloader, optimizer, scheduler, scaler, args, epoch):
    """Train for one epoch."""
    student.train()
    teacher_name = args.teacher

    total_loss = 0
    total_emb_loss = 0
    total_cka_loss = 0
    total_cos_sim = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (waveforms, captions) in enumerate(pbar):
        waveforms = waveforms.to(args.device, non_blocking=True)

        # Get teacher embeddings from text captions
        teacher_emb = encode_teacher_batch(teacher, captions, teacher_name, args.whiten_stats)

        optimizer.zero_grad()

        if args.amp:
            with autocast():
                # Student processes audio: waveform -> mel -> embedding
                student_emb = student.mel_forward(waveforms, normalize=True)

            # Compute loss in float32 for numerical stability
            student_emb_f32 = student_emb.float()
            teacher_emb_f32 = teacher_emb.float()

            if args.loss == "embedding":
                loss = embedding_loss(student_emb_f32, teacher_emb_f32)
                loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_only":
                loss = cka_loss(student_emb_f32, teacher_emb_f32)
                loss_dict = {"cka_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_combined":
                loss, loss_dict = cka_combined_loss(
                    student_emb_f32, teacher_emb_f32, args.lambda_cka
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_emb = student.mel_forward(waveforms, normalize=True)

            if args.loss == "embedding":
                loss = embedding_loss(student_emb, teacher_emb)
                loss_dict = {"embedding_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_only":
                loss = cka_loss(student_emb, teacher_emb)
                loss_dict = {"cka_loss": loss.item(), "total_loss": loss.item()}
            elif args.loss == "cka_combined":
                loss, loss_dict = cka_combined_loss(
                    student_emb, teacher_emb, args.lambda_cka
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_similarity_metrics(student_emb, teacher_emb)

        # Skip NaN batches in accumulation (AMP can produce NaN on overflow)
        import math
        if math.isnan(loss_dict["total_loss"]):
            pbar.set_postfix({"loss": "NaN (skipped)", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            continue

        # Accumulate
        total_loss += loss_dict["total_loss"]
        total_emb_loss += loss_dict.get("embedding_loss", 0)
        total_cka_loss += loss_dict.get("cka_loss", 0)
        total_cos_sim += metrics["cosine_sim_mean"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "cos_sim": f"{metrics['cosine_sim_mean']:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
        })

    avg_metrics = {
        "train/loss": total_loss / num_batches,
        "train/embedding_loss": total_emb_loss / num_batches,
        "train/cka_loss": total_cka_loss / num_batches,
        "train/cosine_sim_mean": total_cos_sim / num_batches,
        "train/lr": scheduler.get_last_lr()[0],
    }

    return avg_metrics


@torch.no_grad()
def validate(teacher, student, val_loader, args):
    """Run validation pass."""
    student.eval()
    teacher_name = args.teacher

    total_loss = 0
    total_cos_sim = 0
    num_batches = 0

    for waveforms, captions in tqdm(val_loader, desc="Val", leave=False):
        waveforms = waveforms.to(args.device, non_blocking=True)
        teacher_emb = encode_teacher_batch(teacher, captions, teacher_name, args.whiten_stats)
        student_emb = student.mel_forward(waveforms, normalize=True)

        loss = embedding_loss(student_emb, teacher_emb)
        metrics = compute_similarity_metrics(student_emb, teacher_emb)

        total_loss += loss.item()
        total_cos_sim += metrics["cosine_sim_mean"]
        num_batches += 1

    if num_batches == 0:
        return {"val/loss": 0, "val/cosine_mean": 0}

    return {
        "val/loss": total_loss / num_batches,
        "val/cosine_mean": total_cos_sim / num_batches,
    }


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

    # Load whitening stats if specified
    if args.whiten:
        args.whiten_stats = torch.load(args.whiten, map_location="cpu", weights_only=True)
        print(f"Loaded whitening stats from {args.whiten}")
    else:
        args.whiten_stats = None

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"audio_distill_{args.teacher}_{args.loss}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["audio_distillation", args.teacher, args.loss, "experiment10"],
        )

    print("=" * 60)
    print("Cross-Modal Audio Distillation")
    print("=" * 60)
    print(f"Teacher: {args.teacher}")
    print(f"Dataset: {args.dataset}")
    print(f"Loss: {args.loss}")
    print(f"Projector: {args.projector}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Max duration: {args.max_duration}s")
    print(f"SpecAugment: freqm={args.freqm}, timem={args.timem}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load teacher
    config = TEACHER_CONFIGS[args.teacher]
    teacher = config["class"](device=args.device, **config["kwargs"])
    print(f"Teacher loaded. Embedding dim: {teacher.embed_dim}")

    # Create student
    print("\nCreating audio student model...")
    student = AudioStudentModel.for_distillation(
        teacher_dim=teacher.embed_dim,
        projector_type=args.projector,
        projector_hidden_dim=args.projector_hidden_dim,
        sample_rate=args.sample_rate,
        freqm=args.freqm,
        timem=args.timem,
    ).to(args.device)
    print(f"Student backbone dim: {student.BACKBONE_DIM}")
    print(f"Student teacher dim target: {teacher.embed_dim}")

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader = get_audiocaps_dataloaders_with_val(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

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
        len(train_loader),
    )
    scaler = GradScaler() if args.amp else None

    # Training loop
    print("\nStarting training...")
    best_val_cos_sim = 0

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            teacher, student, train_loader,
            optimizer, scheduler, scaler,
            args, epoch,
        )

        # Validation
        val_metrics = validate(teacher, student, val_loader, args)
        metrics.update(val_metrics)

        train_cos = metrics["train/cosine_sim_mean"]
        val_cos = metrics["val/cosine_mean"]
        gap = train_cos - val_cos
        metrics["val/train_val_gap"] = gap

        # Log to W&B
        if not args.no_wandb:
            wandb.log(metrics, step=epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {metrics['train/loss']:.4f}")
        print(f"  Train Cosine sim: {train_cos:.4f}")
        print(f"  Val Cosine sim: {val_cos:.4f} (gap: {gap:.4f})")

        # Save best model
        # For CKA-only: use CKA (structural alignment) instead of cosine similarity
        if args.loss == "cka_only":
            val_metric = 1.0 - metrics.get("train/cka_loss", 1.0)
        else:
            val_metric = val_cos
        if val_metric > best_val_cos_sim:
            best_val_cos_sim = val_metric
            if args.loss == "cka_only":
                ckpt_name = f"audiocaps_{args.teacher}_cka_only_distilled_best.pth"
            elif args.loss == "cka_combined":
                ckpt_name = f"audiocaps_{args.teacher}_cka_l{args.lambda_cka}_distilled_best.pth"
            else:
                ckpt_name = f"audiocaps_{args.teacher}_distilled_best.pth"
            save_checkpoint(student, optimizer, scheduler, epoch, args, ckpt_name)

        # Periodic checkpoints
        if epoch % args.save_every == 0:
            if args.loss == "cka_only":
                periodic_name = f"audiocaps_{args.teacher}_cka_only_distilled_epoch{epoch}.pth"
            elif args.loss == "cka_combined":
                periodic_name = f"audiocaps_{args.teacher}_cka_l{args.lambda_cka}_distilled_epoch{epoch}.pth"
            else:
                periodic_name = f"audiocaps_{args.teacher}_distilled_epoch{epoch}.pth"
            save_checkpoint(student, optimizer, scheduler, epoch, args, periodic_name)

    # Save final checkpoint
    if args.loss == "cka_only":
        final_name = f"audiocaps_{args.teacher}_cka_only_distilled_final.pth"
    elif args.loss == "cka_combined":
        final_name = f"audiocaps_{args.teacher}_cka_l{args.lambda_cka}_distilled_final.pth"
    else:
        final_name = f"audiocaps_{args.teacher}_distilled_final.pth"
    save_checkpoint(student, optimizer, scheduler, args.epochs, args, final_name)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val cosine similarity: {best_val_cos_sim:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
