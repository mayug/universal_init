"""Distillation losses for matching student embeddings to teacher."""

import torch
import torch.nn.functional as F


def embedding_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """
    Cosine embedding loss between student and teacher embeddings.

    Args:
        student_emb: L2-normalized student embeddings [B, D]
        teacher_emb: L2-normalized teacher embeddings [B, D]

    Returns:
        loss: Scalar loss value (1 - mean cosine similarity)
    """
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    return 1.0 - cos_sim.mean()


def relational_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """
    Relational loss matching within-batch similarity structure.

    Encourages student to preserve the geometric relationships between
    samples as captured by the teacher, not just individual point matches.

    Args:
        student_emb: L2-normalized student embeddings [B, D]
        teacher_emb: L2-normalized teacher embeddings [B, D]

    Returns:
        loss: MSE between student and teacher similarity matrices
    """
    # Compute cosine similarity matrices [B, B]
    # Since embeddings are L2-normalized, this is just matrix multiplication
    S_student = student_emb @ student_emb.T
    S_teacher = teacher_emb @ teacher_emb.T

    return F.mse_loss(S_student, S_teacher)


def combined_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    lambda_rel: float = 0.5
) -> tuple[torch.Tensor, dict]:
    """
    Combined embedding and relational loss.

    Args:
        student_emb: L2-normalized student embeddings [B, D]
        teacher_emb: L2-normalized teacher embeddings [B, D]
        lambda_rel: Weight for relational loss component

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual loss components for logging
    """
    emb_loss = embedding_loss(student_emb, teacher_emb)
    rel_loss = relational_loss(student_emb, teacher_emb)

    total_loss = emb_loss + lambda_rel * rel_loss

    loss_dict = {
        "embedding_loss": emb_loss.item(),
        "relational_loss": rel_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


def compute_similarity_metrics(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor
) -> dict:
    """
    Compute similarity metrics between student and teacher embeddings.

    Args:
        student_emb: L2-normalized student embeddings [B, D]
        teacher_emb: L2-normalized teacher embeddings [B, D]

    Returns:
        metrics: Dictionary with similarity statistics
    """
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)

    return {
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_std": cos_sim.std().item(),
        "cosine_sim_min": cos_sim.min().item(),
        "cosine_sim_max": cos_sim.max().item(),
    }
