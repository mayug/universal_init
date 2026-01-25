"""ImageBind teacher model wrapper for distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageBindTeacher(nn.Module):
    """Frozen ImageBind vision encoder for generating target embeddings."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = None
        self.embed_dim = 1024  # ImageBind embedding dimension

    def load(self):
        """Load ImageBind model. Call this after initialization."""
        # ImageBind requires special import handling
        try:
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
            self.ModalityType = ModalityType
        except ImportError:
            raise ImportError(
                "ImageBind not found. Install it with:\n"
                "  git clone https://github.com/facebookresearch/ImageBind.git\n"
                "  cd ImageBind && pip install -e ."
            )

        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to ImageBind embeddings.

        Args:
            images: Tensor of shape [B, C, H, W], normalized with ImageNet stats

        Returns:
            embeddings: L2-normalized tensor of shape [B, 1024]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # ImageBind expects dict input with modality keys
        inputs = {self.ModalityType.VISION: images}
        embeddings = self.model(inputs)
        vision_emb = embeddings[self.ModalityType.VISION]

        # L2 normalize
        vision_emb = F.normalize(vision_emb, p=2, dim=-1)

        return vision_emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for encode()."""
        return self.encode(images)


def get_imagebind_transform():
    """Get the preprocessing transform for ImageBind.

    ImageBind uses standard ImageNet normalization with 224x224 input.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_imagebind_train_transform():
    """Get training augmentation transform compatible with ImageBind."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
