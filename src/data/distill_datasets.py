"""Data loaders for distillation pretraining (Imagenette, COCO)."""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
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
        dataset_name: 'imagenette' or 'coco'
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: imagenette, coco")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for relational loss
    )

    return dataloader
