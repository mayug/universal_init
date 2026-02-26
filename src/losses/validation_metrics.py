"""Validation metrics for monitoring distillation quality and detecting overfitting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


def compute_cosine_stats(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor
) -> Dict[str, float]:
    """Compute cosine similarity statistics."""
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    return {
        "cosine_mean": cos_sim.mean().item(),
        "cosine_std": cos_sim.std().item(),
        "cosine_min": cos_sim.min().item(),
        "cosine_max": cos_sim.max().item(),
    }


def compute_retrieval_metrics(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """
    Compute student→teacher retrieval metrics.

    For each student embedding, find nearest neighbors in teacher space
    and check if the correct teacher (same index) is retrieved.

    Args:
        student_emb: [N, D] normalized student embeddings
        teacher_emb: [N, D] normalized teacher embeddings
        k_values: tuple of k values for R@k computation

    Returns:
        Dictionary with R@k metrics
    """
    # Compute similarity matrix: student queries, teacher gallery
    # [N, N] - each row is similarities of student_i to all teachers
    sim_matrix = student_emb @ teacher_emb.T

    # Ground truth: diagonal should have highest similarity
    n = student_emb.shape[0]
    gt_indices = torch.arange(n, device=student_emb.device)

    # Get top-k indices for each student
    max_k = max(k_values)
    _, topk_indices = sim_matrix.topk(max_k, dim=1)

    metrics = {}
    for k in k_values:
        # Check if ground truth index is in top-k
        hits = (topk_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
        metrics[f"retrieval_R@{k}"] = hits.float().mean().item()

    return metrics


def compute_rsa_correlation(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor
) -> Dict[str, float]:
    """
    Compute Representational Similarity Analysis (RSA) correlation.

    Measures how well student preserves the relational structure of teacher.
    Uses Pearson correlation between flattened similarity matrices.

    Args:
        student_emb: [N, D] normalized student embeddings
        teacher_emb: [N, D] normalized teacher embeddings

    Returns:
        Dictionary with RSA metrics
    """
    # Compute similarity matrices
    S_student = student_emb @ student_emb.T
    S_teacher = teacher_emb @ teacher_emb.T

    # Extract upper triangular (excluding diagonal) for correlation
    n = S_student.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1, device=S_student.device)

    s_vec = S_student[triu_indices[0], triu_indices[1]]
    t_vec = S_teacher[triu_indices[0], triu_indices[1]]

    # Pearson correlation
    s_centered = s_vec - s_vec.mean()
    t_centered = t_vec - t_vec.mean()

    numerator = (s_centered * t_centered).sum()
    denominator = torch.sqrt((s_centered ** 2).sum() * (t_centered ** 2).sum())

    correlation = (numerator / (denominator + 1e-8)).item()

    # Also compute MSE between matrices
    mse = F.mse_loss(S_student, S_teacher).item()

    return {
        "rsa_correlation": correlation,
        "rsa_mse": mse,
    }


def compute_collapse_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics to detect representation collapse.

    Collapse indicators:
    - Low variance across embedding dimensions
    - High average pairwise similarity (embeddings too similar)
    - Low effective rank

    Args:
        embeddings: [N, D] embeddings (normalized or unnormalized)

    Returns:
        Dictionary with collapse detection metrics
    """
    # Normalize for similarity computations
    emb_norm = F.normalize(embeddings, p=2, dim=-1)

    # 1. Variance per dimension, then average
    dim_variance = embeddings.var(dim=0)
    avg_dim_variance = dim_variance.mean().item()
    min_dim_variance = dim_variance.min().item()

    # 2. Average pairwise cosine similarity (excluding self)
    sim_matrix = emb_norm @ emb_norm.T
    n = sim_matrix.shape[0]
    # Zero out diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    avg_pairwise_sim = sim_matrix[mask].mean().item()

    # 3. Effective rank (using singular values)
    # Effective rank = exp(entropy of normalized singular values)
    try:
        _, S, _ = torch.svd(embeddings - embeddings.mean(dim=0, keepdim=True))
        S_norm = S / S.sum()
        # Add small epsilon for numerical stability
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()
    except:
        effective_rank = -1.0  # SVD failed

    # 4. Uniformity loss (from contrastive learning literature)
    # Lower is better, measures how spread out embeddings are on hypersphere
    uniformity = torch.pdist(emb_norm, p=2).pow(2).mul(-2).exp().mean().log().item()

    return {
        "collapse_avg_dim_variance": avg_dim_variance,
        "collapse_min_dim_variance": min_dim_variance,
        "collapse_avg_pairwise_sim": avg_pairwise_sim,
        "collapse_effective_rank": effective_rank,
        "collapse_uniformity": uniformity,
    }


class LinearProbe(nn.Module):
    """Simple linear probe for evaluating backbone features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def run_linear_probe(
    student_backbone: nn.Module,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 0.01,
) -> Dict[str, float]:
    """
    Run a quick linear probe on backbone features (without projector).

    This tests if the backbone itself has learned useful features,
    independent of the projector that will be discarded.

    Args:
        student_backbone: Student backbone (not full model)
        probe_train_loader: Training data with labels
        probe_val_loader: Validation data with labels
        num_classes: Number of classes
        device: Device to use
        epochs: Training epochs
        lr: Learning rate

    Returns:
        Dictionary with probe accuracy metrics
    """
    student_backbone.eval()

    # Get feature dimension from a sample
    with torch.no_grad():
        sample_batch = next(iter(probe_train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_img = sample_batch[0]
        else:
            sample_img = sample_batch
        sample_feat = student_backbone(sample_img[:1].to(device))
        feat_dim = sample_feat.shape[-1]

    # Create and train linear probe
    probe = LinearProbe(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Extract features once (frozen backbone)
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    imgs, lbls = batch[0], batch[1]
                else:
                    continue  # Skip if no labels
                imgs = imgs.to(device)
                feats = student_backbone(imgs)
                features.append(feats.cpu())
                labels.append(lbls)
        return torch.cat(features), torch.cat(labels)

    train_feats, train_labels = extract_features(probe_train_loader)
    val_feats, val_labels = extract_features(probe_val_loader)

    # Train probe
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = probe(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # Evaluate probe
    probe.eval()
    with torch.no_grad():
        val_feats = val_feats.to(device)
        val_labels = val_labels.to(device)
        logits = probe(val_feats)
        preds = logits.argmax(dim=1)
        accuracy = (preds == val_labels).float().mean().item()

    return {
        "probe_accuracy": accuracy,
    }


@torch.no_grad()
def validate_distillation(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run comprehensive validation on distillation.

    Args:
        teacher: Teacher model (frozen)
        student: Student model (in eval mode)
        val_loader: Validation data loader
        device: Device to use
        max_batches: Limit number of batches for speed

    Returns:
        Dictionary with all validation metrics
    """
    teacher.eval()
    student.eval()

    all_student_emb = []
    all_teacher_emb = []
    all_student_backbone_emb = []

    num_batches = len(val_loader) if max_batches is None else min(max_batches, len(val_loader))

    for i, images in enumerate(tqdm(val_loader, desc="Validation", total=num_batches)):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)

        # Get embeddings
        teacher_emb = teacher(images)
        student_emb = student(images, normalize=True)

        # Also get backbone features (without projector)
        backbone_emb = student.get_features(images)

        all_student_emb.append(student_emb.cpu())
        all_teacher_emb.append(teacher_emb.cpu())
        all_student_backbone_emb.append(backbone_emb.cpu())

    # Concatenate all embeddings
    student_emb = torch.cat(all_student_emb, dim=0)
    teacher_emb = torch.cat(all_teacher_emb, dim=0)
    backbone_emb = torch.cat(all_student_backbone_emb, dim=0)

    # Move to device for computations
    student_emb = student_emb.to(device)
    teacher_emb = teacher_emb.to(device)
    backbone_emb = backbone_emb.to(device)

    # Compute all metrics
    metrics = {}

    # 1. Cosine similarity stats
    metrics.update({f"val/{k}": v for k, v in compute_cosine_stats(student_emb, teacher_emb).items()})

    # 2. Retrieval metrics
    metrics.update({f"val/{k}": v for k, v in compute_retrieval_metrics(student_emb, teacher_emb).items()})

    # 3. RSA correlation
    metrics.update({f"val/{k}": v for k, v in compute_rsa_correlation(student_emb, teacher_emb).items()})

    # 4. Collapse stats (on projected embeddings)
    metrics.update({f"val/proj_{k}": v for k, v in compute_collapse_stats(student_emb).items()})

    # 5. Collapse stats (on backbone embeddings - what we actually keep)
    metrics.update({f"val/backbone_{k}": v for k, v in compute_collapse_stats(backbone_emb).items()})

    return metrics


def validate_with_linear_probe(
    student: nn.Module,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run linear probe validation on backbone features.

    Args:
        student: Full student model
        probe_train_loader: Labeled data for probe training
        probe_val_loader: Labeled data for probe evaluation
        num_classes: Number of classes
        device: Device to use

    Returns:
        Dictionary with probe metrics
    """
    # Extract just the backbone
    backbone = student.backbone
    backbone.eval()

    metrics = run_linear_probe(
        student_backbone=backbone,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        num_classes=num_classes,
        device=device,
        epochs=10,
        lr=0.01,
    )

    return {f"val/{k}": v for k, v in metrics.items()}
