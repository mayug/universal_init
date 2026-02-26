"""Data loaders for distillation pretraining (Imagenette, COCO)."""

import os
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_train_transform():
    """Standard training augmentation for distillation."""
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


class ImagenetteDataset(Dataset):
    """Imagenette dataset for sanity checking distillation."""

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root: Path to imagenette2 directory
            split: 'train' or 'val'
            transform: Image transforms
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform or get_train_transform()

        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(
                f"Imagenette not found at {self.root}. "
                "Download with: wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
            )

        self.dataset = ImageFolder(str(split_dir), transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image  # For distillation, we only need images


class COCOImagesDataset(Dataset):
    """COCO dataset (images only) for distillation pretraining."""

    def __init__(self, root: str, split: str = "train2017", transform=None):
        """
        Args:
            root: Path to COCO directory (should contain train2017/, val2017/, etc.)
            split: 'train2017' or 'val2017'
            transform: Image transforms
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform or get_train_transform()

        self.image_dir = self.root / split
        if not self.image_dir.exists():
            raise ValueError(
                f"COCO images not found at {self.image_dir}. "
                "Download from https://cocodataset.org/#download"
            )

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class ImageNetDataset(Dataset):
    """ImageNet dataset (images only) for distillation pretraining."""

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root: Path to ImageNet directory (should contain train/, val/ subdirectories)
            split: 'train' or 'val'
            transform: Image transforms
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform or get_train_transform()

        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(
                f"ImageNet {split} directory not found at {split_dir}. "
                f"Expected structure: {root}/{split}/n01440764/..."
            )

        # Use ImageFolder to handle class subdirectories
        self.dataset = ImageFolder(str(split_dir), transform=self.transform)

        print(f"Found {len(self.dataset)} images in ImageNet {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Discard label for distillation
        return image


def get_distill_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    split: str = "train",
) -> DataLoader:
    """
    Get data loader for distillation pretraining.

    Args:
        dataset_name: 'imagenette', 'coco', or 'imagenet'
        data_root: Root directory containing datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        split: Data split to use

    Returns:
        DataLoader for the specified dataset
    """
    transform = get_train_transform()

    if dataset_name == "imagenette":
        dataset = ImagenetteDataset(
            root=os.path.join(data_root, "imagenette2"),
            split=split if split != "train" else "train",
            transform=transform
        )
    elif dataset_name == "coco":
        coco_split = "train2017" if split == "train" else "val2017"
        dataset = COCOImagesDataset(
            root=os.path.join(data_root, "coco"),
            split=coco_split,
            transform=transform
        )
    elif dataset_name == "imagenet":
        dataset = ImageNetDataset(
            root=os.path.join(data_root, "imagenet"),
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: imagenette, coco, imagenet")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for relational loss
    )

    return dataloader


def get_distill_dataloaders_with_val(
    dataset_name: str,
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders for distillation with holdout split.

    Creates a validation split from the training data to monitor for overfitting.

    Args:
        dataset_name: 'imagenette', 'coco', or 'imagenet'
        data_root: Root directory containing datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_fraction: Fraction of data to use for validation
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    if dataset_name == "imagenette":
        # For Imagenette, use built-in train/val split
        train_dataset = ImagenetteDataset(
            root=os.path.join(data_root, "imagenette2"),
            split="train",
            transform=train_transform
        )
        val_dataset = ImagenetteDataset(
            root=os.path.join(data_root, "imagenette2"),
            split="val",
            transform=val_transform
        )
    elif dataset_name == "coco":
        # For COCO, split train2017 into train/val
        full_dataset = COCOImagesDataset(
            root=os.path.join(data_root, "coco"),
            split="train2017",
            transform=None  # Will set per-subset
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(total_size * val_fraction)
        train_size = total_size - val_size

        # Create reproducible split
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(
            range(total_size),
            [train_size, val_size],
            generator=generator
        )
        train_indices = list(train_indices)
        val_indices = list(val_indices)

        # Create subset datasets with different transforms
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

        print(f"COCO split: {len(train_dataset)} train, {len(val_dataset)} val")
    elif dataset_name == "imagenet":
        # For ImageNet, use built-in train/val split
        train_dataset = ImageNetDataset(
            root=os.path.join(data_root, "imagenet"),
            split="train",
            transform=train_transform
        )
        val_dataset = ImageNetDataset(
            root=os.path.join(data_root, "imagenet"),
            split="val",
            transform=val_transform
        )
        print(f"ImageNet: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: imagenette, coco, imagenet")

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
        drop_last=False,
    )

    return train_loader, val_loader


class TransformSubset(Dataset):
    """Subset of a dataset with a custom transform."""

    def __init__(self, dataset: Dataset, indices: list, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get raw image from parent dataset
        real_idx = self.indices[idx]
        img_path = self.dataset.image_files[real_idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
