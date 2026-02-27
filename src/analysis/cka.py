"""Linear CKA (Centered Kernel Alignment) implementation.

Based on Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
"""

import torch
import numpy as np


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: Feature matrix of shape [N, D1]
        Y: Feature matrix of shape [N, D2]

    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"

    # Center features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute cross-covariance and self-covariances
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = (YtX * YtX).sum()
    denominator = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    if denominator < 1e-10:
        return 0.0

    return (numerator / denominator).item()


def cka_matrix(feature_dict: dict) -> tuple:
    """Compute pairwise CKA matrix for all models.

    Args:
        feature_dict: Dict mapping model name -> feature tensor [N, D]

    Returns:
        (matrix, names): numpy array of shape [K, K] and list of model names
    """
    names = list(feature_dict.keys())
    K = len(names)
    matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(i, K):
            score = linear_cka(feature_dict[names[i]], feature_dict[names[j]])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix, names
