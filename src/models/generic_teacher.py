"""Generic frozen teacher wrapper using timm models (CLIP, supervised ViT, etc.)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config


class GenericTeacher(nn.Module):
    """Frozen teacher model using any timm-compatible pretrained model.

    Interface matches ImageBindTeacher: frozen, returns L2-normalized embeddings,
    exposes embed_dim.

    Args:
        model_name: timm model identifier (e.g. 'vit_base_patch16_clip_224.openai')
        device: Device to place model on
        use_head: If False, strip classification head to get raw features.
                  If True, keep native head (e.g. CLIP projection layer).
    """

    def __init__(self, model_name: str, device: str = "cuda", use_head: bool = False):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_head = use_head

        # Create model: num_classes=0 strips the head
        if use_head:
            self.model = timm.create_model(model_name, pretrained=True)
        else:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        self.model = self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Detect embed_dim via dummy forward pass
        self.embed_dim = self._detect_embed_dim()

        # Cache data config for transforms
        self._data_config = resolve_data_config(self.model.pretrained_cfg)

        print(f"GenericTeacher: {model_name}")
        print(f"  use_head={use_head}, embed_dim={self.embed_dim}")
        print(f"  mean={self._data_config['mean']}, std={self._data_config['std']}")

    @torch.no_grad()
    def _detect_embed_dim(self) -> int:
        """Detect output embedding dimension via dummy forward pass."""
        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        out = self.model(dummy)
        if out.dim() == 1:
            return out.shape[0]
        return out.shape[-1]

    def get_transform_config(self) -> dict:
        """Returns timm data config with 'mean', 'std', 'input_size', etc."""
        return self._data_config

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalized embeddings.

        Args:
            images: Tensor of shape [B, C, H, W]

        Returns:
            embeddings: L2-normalized tensor of shape [B, embed_dim]
        """
        emb = self.model(images)
        return F.normalize(emb, p=2, dim=-1)
