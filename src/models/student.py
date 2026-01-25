"""Student model: RegNetY-400MF with projector head for distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights


class ProjectorHead(nn.Module):
    """Projector head to map backbone features to teacher embedding space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = None,
        use_mlp: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if use_mlp and hidden_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.projector = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class StudentModel(nn.Module):
    """
    Student model for distillation.

    Uses RegNetY-400MF backbone with optional projector head.
    Supports three initialization modes:
    - 'random': Random initialization
    - 'imagenet': ImageNet pretrained weights
    - 'distilled': Load weights from distillation checkpoint
    """

    BACKBONE_DIM = 440  # RegNetY-400MF feature dimension
    TEACHER_DIM = 1024  # ImageBind embedding dimension

    def __init__(
        self,
        init_mode: str = "random",
        checkpoint_path: str = None,
        projector_type: str = "linear",  # 'linear' or 'mlp'
        projector_hidden_dim: int = 512,
        num_classes: int = None,  # If set, adds classifier head instead of projector
    ):
        super().__init__()
        self.init_mode = init_mode
        self.projector_type = projector_type

        # Load backbone
        if init_mode == "imagenet":
            self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        else:
            self.backbone = regnet_y_400mf(weights=None)

        # Remove classifier head from backbone
        self.backbone.fc = nn.Identity()

        # Add projector or classifier
        if num_classes is not None:
            # Downstream mode: classifier head
            self.head = nn.Linear(self.BACKBONE_DIM, num_classes)
            self.mode = "classifier"
        else:
            # Distillation mode: projector head
            use_mlp = projector_type == "mlp"
            self.head = ProjectorHead(
                in_dim=self.BACKBONE_DIM,
                out_dim=self.TEACHER_DIM,
                hidden_dim=projector_hidden_dim if use_mlp else None,
                use_mlp=use_mlp
            )
            self.mode = "projector"

        # Load distilled weights if specified
        if init_mode == "distilled" and checkpoint_path is not None:
            self.load_distilled_weights(checkpoint_path)

    def load_distilled_weights(self, checkpoint_path: str):
        """Load backbone weights from distillation checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "backbone_state_dict" in checkpoint:
            state_dict = checkpoint["backbone_state_dict"]
        elif "state_dict" in checkpoint:
            # Extract backbone weights from full model state dict
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("backbone."):
                    state_dict[k.replace("backbone.", "")] = v
        else:
            state_dict = checkpoint

        # Load backbone weights
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading backbone: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading backbone: {unexpected}")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass through backbone and head.

        Args:
            x: Input images [B, C, H, W]
            normalize: If True and in projector mode, L2-normalize output

        Returns:
            embeddings: [B, TEACHER_DIM] if projector mode, [B, num_classes] if classifier
        """
        features = self.get_features(x)
        out = self.head(features)

        if normalize and self.mode == "projector":
            out = F.normalize(out, p=2, dim=-1)

        return out

    def get_backbone_state_dict(self) -> dict:
        """Get only backbone weights for saving after distillation."""
        return self.backbone.state_dict()

    @classmethod
    def for_distillation(
        cls,
        projector_type: str = "linear",
        projector_hidden_dim: int = 512
    ) -> "StudentModel":
        """Create student model configured for distillation."""
        return cls(
            init_mode="random",
            projector_type=projector_type,
            projector_hidden_dim=projector_hidden_dim,
            num_classes=None
        )

    @classmethod
    def for_downstream(
        cls,
        num_classes: int,
        init_mode: str = "random",
        checkpoint_path: str = None
    ) -> "StudentModel":
        """Create student model configured for downstream classification."""
        return cls(
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes
        )
