"""Audio student model: MobileNetV3 backbone with projector head for cross-modal distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mn.model import MobileNetV3Audio, AugmentMelSTFT
from src.models.student import ProjectorHead


class AudioStudentModel(nn.Module):
    """Audio student model for cross-modal distillation.

    Uses MobileNetV3-Large backbone (adapted for 1-channel mel spectrograms)
    with optional projector head for distillation or classifier head for
    downstream tasks.

    Supports three initialization modes:
    - 'random': Random initialization
    - 'audioset_pretrained': EfficientAT mn10 AudioSet pretrained weights
    - 'distilled': Load weights from cross-modal distillation checkpoint

    The mel spectrogram transform is included in the model so raw waveforms
    can be passed directly (via the `mel_forward` method), or pre-computed
    spectrograms can be passed to `forward`.
    """

    BACKBONE_DIM = 960  # MobileNetV3-Large feature dimension

    def __init__(
        self,
        init_mode: str = "random",
        checkpoint_path: str = None,
        projector_type: str = "linear",
        projector_hidden_dim: int = 512,
        num_classes: int = None,
        keep_projector: bool = False,
        train_projector: bool = False,
        teacher_dim: int = 768,
        freeze_backbone: bool = False,
        sample_rate: int = 32000,
        n_mels: int = 128,
        hop_size: int = 320,
        n_fft: int = 1024,
        win_length: int = 800,
        freqm: int = 48,
        timem: int = 192,
    ):
        super().__init__()
        self.init_mode = init_mode
        self.projector_type = projector_type
        self.keep_projector = keep_projector
        self.train_projector = train_projector
        self.teacher_dim = teacher_dim

        # Mel spectrogram transform (included in model for convenience)
        self.mel = AugmentMelSTFT(
            n_mels=n_mels,
            sr=sample_rate,
            win_length=win_length,
            hopsize=hop_size,
            n_fft=n_fft,
            freqm=freqm,
            timem=timem,
        )

        # Load backbone
        if init_mode == "audioset_pretrained":
            self.backbone = MobileNetV3Audio(pretrained_name="mn10_as")
        else:
            self.backbone = MobileNetV3Audio()

        # Add projector or classifier (same pattern as vision StudentModel)
        if num_classes is not None:
            # Downstream mode: classifier head
            if keep_projector and init_mode == "distilled":
                use_mlp = projector_type == "mlp"
                self.projector = ProjectorHead(
                    in_dim=self.BACKBONE_DIM,
                    out_dim=self.teacher_dim,
                    hidden_dim=projector_hidden_dim if use_mlp else None,
                    use_mlp=use_mlp,
                )
                self.head = nn.Linear(self.teacher_dim, num_classes)
                if train_projector:
                    self.mode = "classifier_with_trainable_projector"
                else:
                    self.mode = "classifier_with_projector"
            else:
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
                use_mlp=use_mlp,
            )
            self.mode = "projector"

        # Load distilled weights if specified
        if init_mode == "distilled" and checkpoint_path is not None:
            self.load_distilled_weights(
                checkpoint_path,
                load_projector=keep_projector,
                freeze_projector=not train_projector,
            )

        # Freeze backbone if requested (for linear probing)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def load_distilled_weights(
        self, checkpoint_path: str, load_projector: bool = False, freeze_projector: bool = True
    ):
        """Load backbone weights from distillation checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "backbone_state_dict" in checkpoint:
            backbone_state = checkpoint["backbone_state_dict"]
        elif "state_dict" in checkpoint:
            backbone_state = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("backbone."):
                    backbone_state[k.replace("backbone.", "")] = v
        else:
            backbone_state = checkpoint

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"Missing keys when loading backbone: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading backbone: {unexpected}")

        # Optionally load and freeze projector
        if load_projector and self.projector is not None:
            if "student_state_dict" in checkpoint:
                student_state = checkpoint["student_state_dict"]
                projector_state = {}
                for k, v in student_state.items():
                    if k.startswith("head."):
                        projector_state[k.replace("head.", "")] = v
                if projector_state:
                    self.projector.load_state_dict(projector_state, strict=False)
                    if freeze_projector:
                        for param in self.projector.parameters():
                            param.requires_grad = False
                        print("Loaded and froze projector from distillation checkpoint")
                    else:
                        print("Loaded projector from distillation checkpoint (trainable)")
            else:
                print("Warning: Could not find projector weights in checkpoint")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features from mel spectrogram.

        Args:
            x: Mel spectrogram [B, 1, n_mels, T].

        Returns:
            features: [B, BACKBONE_DIM] feature vector.
        """
        return self.backbone.get_features(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Forward pass through backbone and head.

        Args:
            x: Mel spectrogram [B, 1, n_mels, T].
            normalize: If True and in projector mode, L2-normalize output.

        Returns:
            embeddings: [B, teacher_dim] if projector mode, [B, num_classes] if classifier.
        """
        features = self.get_features(x)

        if self.mode == "classifier_with_projector":
            with torch.no_grad():
                projected = self.projector(features)
            out = self.head(projected)
        elif self.mode == "classifier_with_trainable_projector":
            projected = self.projector(features)
            out = self.head(projected)
        else:
            out = self.head(features)

        if normalize and self.mode == "projector":
            out = F.normalize(out, p=2, dim=-1)

        return out

    def mel_forward(self, waveform: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """End-to-end forward from raw waveform.

        Args:
            waveform: [B, num_samples] raw audio.
            normalize: If True and in projector mode, L2-normalize output.

        Returns:
            Output embeddings or logits.
        """
        mel = self.mel(waveform)  # [B, 1, n_mels, T]
        return self.forward(mel, normalize=normalize)

    def get_backbone_state_dict(self) -> dict:
        """Get only backbone weights for saving after distillation."""
        return self.backbone.state_dict()

    @classmethod
    def for_distillation(
        cls,
        teacher_dim: int = 768,
        projector_type: str = "linear",
        projector_hidden_dim: int = 512,
        sample_rate: int = 32000,
        freqm: int = 48,
        timem: int = 192,
    ) -> "AudioStudentModel":
        """Create audio student configured for distillation."""
        return cls(
            init_mode="random",
            projector_type=projector_type,
            projector_hidden_dim=projector_hidden_dim,
            num_classes=None,
            teacher_dim=teacher_dim,
            sample_rate=sample_rate,
            freqm=freqm,
            timem=timem,
        )

    @classmethod
    def for_downstream(
        cls,
        num_classes: int,
        init_mode: str = "random",
        checkpoint_path: str = None,
        keep_projector: bool = False,
        train_projector: bool = False,
        teacher_dim: int = 768,
        freeze_backbone: bool = False,
        sample_rate: int = 32000,
    ) -> "AudioStudentModel":
        """Create audio student configured for downstream classification.

        Args:
            num_classes: Number of output classes.
            init_mode: 'random', 'audioset_pretrained', or 'distilled'.
            checkpoint_path: Path to distillation checkpoint (for distilled mode).
            keep_projector: If True, keep projector from distillation.
            train_projector: If True, projector is trainable.
            teacher_dim: Teacher embedding dimension (must match checkpoint).
            freeze_backbone: If True, freeze backbone for linear probing.
            sample_rate: Audio sample rate.
        """
        return cls(
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            keep_projector=keep_projector,
            train_projector=train_projector,
            teacher_dim=teacher_dim,
            freeze_backbone=freeze_backbone,
            sample_rate=sample_rate,
            freqm=0,  # No augment masking for downstream eval
            timem=0,
        )
