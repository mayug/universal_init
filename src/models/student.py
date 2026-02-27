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
        keep_projector: bool = False,  # If True, keep projector in downstream mode
        train_projector: bool = False,  # If True, projector is trainable (not frozen)
        teacher_dim: int = 1024,  # Teacher embedding dimension
    ):
        super().__init__()
        self.init_mode = init_mode
        self.projector_type = projector_type
        self.keep_projector = keep_projector
        self.train_projector = train_projector
        self.teacher_dim = teacher_dim

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
            if keep_projector and init_mode == "distilled":
                # Keep projector from distillation + classifier on top
                use_mlp = projector_type == "mlp"
                self.projector = ProjectorHead(
                    in_dim=self.BACKBONE_DIM,
                    out_dim=self.teacher_dim,
                    hidden_dim=projector_hidden_dim if use_mlp else None,
                    use_mlp=use_mlp
                )
                self.head = nn.Linear(self.teacher_dim, num_classes)
                # Mode depends on whether projector is trainable
                if train_projector:
                    self.mode = "classifier_with_trainable_projector"
                else:
                    self.mode = "classifier_with_projector"  # Frozen projector
            else:
                # Drop projector (default): classifier directly on backbone
                self.projector = None
                self.head = nn.Linear(self.BACKBONE_DIM, num_classes)
                self.mode = "classifier"
        else:
            # Distillation mode: projector head
            use_mlp = projector_type == "mlp"
            self.projector = None
            self.head = ProjectorHead(
                in_dim=self.BACKBONE_DIM,
                out_dim=self.teacher_dim,
                hidden_dim=projector_hidden_dim if use_mlp else None,
                use_mlp=use_mlp
            )
            self.mode = "projector"

        # Load distilled weights if specified
        if init_mode == "distilled" and checkpoint_path is not None:
            self.load_distilled_weights(
                checkpoint_path,
                load_projector=keep_projector,
                freeze_projector=not train_projector
            )

    def load_distilled_weights(self, checkpoint_path: str, load_projector: bool = False, freeze_projector: bool = True):
        """
        Load backbone weights from distillation checkpoint.

        Args:
            checkpoint_path: Path to distillation checkpoint
            load_projector: If True, also load projector weights
            freeze_projector: If True, freeze projector weights (default). If False, projector is trainable.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats for backbone
        if "backbone_state_dict" in checkpoint:
            backbone_state = checkpoint["backbone_state_dict"]
        elif "state_dict" in checkpoint:
            # Extract backbone weights from full model state dict
            backbone_state = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("backbone."):
                    backbone_state[k.replace("backbone.", "")] = v
        else:
            backbone_state = checkpoint

        # Load backbone weights
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"Missing keys when loading backbone: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading backbone: {unexpected}")

        # Optionally load and freeze projector
        if load_projector and self.projector is not None:
            # Extract projector weights from checkpoint
            if "student_state_dict" in checkpoint:
                # Full student model saved
                student_state = checkpoint["student_state_dict"]
                projector_state = {}
                for k, v in student_state.items():
                    if k.startswith("head."):
                        # In distillation mode, head is the projector
                        projector_state[k.replace("head.", "projector.")] = v

                if projector_state:
                    # Remap keys to match projector structure
                    final_projector_state = {}
                    for k, v in projector_state.items():
                        final_projector_state[k.replace("projector.", "")] = v

                    self.projector.load_state_dict(final_projector_state, strict=False)

                    # Optionally freeze projector weights
                    if freeze_projector:
                        for param in self.projector.parameters():
                            param.requires_grad = False
                        print("Loaded and froze projector from distillation checkpoint")
                    else:
                        print("Loaded projector from distillation checkpoint (trainable)")
            else:
                print("Warning: Could not find projector weights in checkpoint")

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

        if self.mode == "classifier_with_projector":
            # Pass through frozen projector, then classifier
            with torch.no_grad():
                projected = self.projector(features)  # [B, 1024]
            out = self.head(projected)  # [B, num_classes]
        elif self.mode == "classifier_with_trainable_projector":
            # Pass through trainable projector, then classifier
            projected = self.projector(features)  # [B, 1024]
            out = self.head(projected)  # [B, num_classes]
        else:
            # Standard path
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
        projector_hidden_dim: int = 512,
        teacher_dim: int = 1024,
    ) -> "StudentModel":
        """Create student model configured for distillation."""
        return cls(
            init_mode="random",
            projector_type=projector_type,
            projector_hidden_dim=projector_hidden_dim,
            num_classes=None,
            teacher_dim=teacher_dim,
        )

    @classmethod
    def for_downstream(
        cls,
        num_classes: int,
        init_mode: str = "random",
        checkpoint_path: str = None,
        keep_projector: bool = False,
        train_projector: bool = False,
        teacher_dim: int = 1024,
        freeze_backbone: bool = False,
    ) -> "StudentModel":
        """
        Create student model configured for downstream classification.

        Args:
            num_classes: Number of output classes
            init_mode: 'random', 'imagenet', or 'distilled'
            checkpoint_path: Path to distillation checkpoint (for distilled mode)
            keep_projector: If True, keep projector from distillation
            train_projector: If True, projector is trainable (requires keep_projector=True)
            teacher_dim: Teacher embedding dimension (must match checkpoint)
            freeze_backbone: If True, freeze backbone for linear probing
        """
        model = cls(
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            keep_projector=keep_projector,
            train_projector=train_projector,
            teacher_dim=teacher_dim,
        )
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
        return model
