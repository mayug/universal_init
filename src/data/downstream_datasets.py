"""Data loaders for downstream classification datasets."""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import (
    OxfordIIITPet,
    Flowers102,
    DTD,
    EuroSAT,
    ImageFolder,
)
from sklearn.model_selection import train_test_split
import numpy as np


def get_train_transform():
    """Training augmentation for downstream fine-tuning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transform():
    """Validation transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# Dataset configurations
DATASET_INFO = {
    "pets": {
        "class": OxfordIIITPet,
        "num_classes": 37,
        "train_split": "trainval",
        "test_split": "test",
    },
    "flowers102": {
        "class": Flowers102,
        "num_classes": 102,
        "train_split": "train",
        "test_split": "test",
    },
    "dtd": {
        "class": DTD,
        "num_classes": 47,
        "train_split": "train",
        "test_split": "test",
    },
    "eurosat": {
        "class": EuroSAT,
        "num_classes": 10,
        "train_split": None,  # Single split, need to create our own
        "test_split": None,
    },
    "imagenette": {
        "class": None,  # Uses ImageFolder
        "num_classes": 10,
        "train_split": "train",
        "test_split": "val",
    },
}


def get_labels_from_dataset(dataset) -> np.ndarray:
    """Extract labels from a dataset for stratified sampling."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    elif hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    else:
        # Fallback: iterate through dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return np.array(labels)


def create_label_fraction_subset(
    dataset,
    label_fraction: float,
    seed: int = 42
) -> Subset:
    """
    Create a stratified subset of the dataset.

    Args:
        dataset: Full dataset
        label_fraction: Fraction of data to use (0.01, 0.1, or 1.0)
        seed: Random seed for reproducibility

    Returns:
        Subset of the dataset
    """
    if label_fraction >= 1.0:
        return dataset

    labels = get_labels_from_dataset(dataset)
    indices = np.arange(len(dataset))

    # Stratified split
    _, subset_indices = train_test_split(
        indices,
        test_size=label_fraction,
        stratify=labels,
        random_state=seed
    )

    return Subset(dataset, subset_indices)


def load_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    transform,
):
    """Load a specific dataset split."""
    info = DATASET_INFO[dataset_name]

    if dataset_name == "imagenette":
        # Use ImageFolder for Imagenette
        split_dir = "train" if split == "train" else "val"
        path = os.path.join(data_root, "imagenette2", split_dir)
        return ImageFolder(path, transform=transform)

    elif dataset_name == "eurosat":
        # EuroSAT doesn't have predefined splits
        dataset = EuroSAT(
            root=data_root,
            download=True,
            transform=transform
        )
        # Create train/test split (80/20)
        labels = get_labels_from_dataset(dataset)
        indices = np.arange(len(dataset))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=42  # Fixed seed for consistent splits
        )
        if split == "train":
            return Subset(dataset, train_idx)
        else:
            return Subset(dataset, test_idx)

    elif dataset_name == "pets":
        return OxfordIIITPet(
            root=data_root,
            split=info["train_split"] if split == "train" else info["test_split"],
            target_types="category",
            download=True,
            transform=transform
        )

    elif dataset_name == "flowers102":
        return Flowers102(
            root=data_root,
            split=info["train_split"] if split == "train" else info["test_split"],
            download=True,
            transform=transform
        )

    elif dataset_name == "dtd":
        return DTD(
            root=data_root,
            split=info["train_split"] if split == "train" else info["test_split"],
            download=True,
            transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_downstream_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 8,
    label_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Get train and validation data loaders for downstream classification.

    Args:
        dataset_name: One of 'pets', 'flowers102', 'dtd', 'eurosat', 'imagenette'
        data_root: Root directory for datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        label_fraction: Fraction of training data to use (for few-shot experiments)
        seed: Random seed for label fraction sampling

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes in the dataset
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from: {list(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[dataset_name]
    num_classes = info["num_classes"]

    # Load datasets
    train_dataset = load_dataset(
        dataset_name, data_root, "train", get_train_transform()
    )
    val_dataset = load_dataset(
        dataset_name, data_root, "test", get_val_transform()
    )

    # Apply label fraction subsampling to training data
    if label_fraction < 1.0:
        train_dataset = create_label_fraction_subset(
            train_dataset, label_fraction, seed
        )
        print(f"Using {len(train_dataset)} training samples ({label_fraction*100:.0f}%)")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a dataset."""
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_INFO[dataset_name]["num_classes"]
