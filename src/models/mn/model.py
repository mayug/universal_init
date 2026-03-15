"""MobileNetV3 audio backbone compatible with EfficientAT mn10 weights.

This is a minimal implementation of the MobileNetV3-based audio model from
EfficientAT (https://github.com/fschmid56/EfficientAT). It wraps torchvision's
MobileNetV3-Large, adapting it for single-channel mel spectrogram input.

Key properties:
  - Input: log-mel spectrogram [B, 1, 128, T]
  - Backbone output: 960-dim feature vector
  - ~4.9M parameters (backbone only)
  - Compatible with EfficientAT mn10_as pretrained weights when loaded via
    the provided weight-conversion utility.
"""

import torch
import torch.nn as nn
import torchaudio


class AugmentMelSTFT(nn.Module):
    """Convert raw waveforms to log-mel spectrograms.

    Replaces EfficientAT's custom AugmentMelSTFT with a torchaudio-based
    implementation that produces compatible output.

    Args:
        n_mels: Number of mel filter banks.
        sr: Sample rate.
        win_length: Window length in samples.
        hopsize: Hop size in samples.
        n_fft: FFT size.
        freqm: Frequency masking parameter (SpecAugment). 0 to disable.
        timem: Time masking parameter (SpecAugment). 0 to disable.
        training_only_augment: If True, masking only applied during training.
    """

    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        freqm: int = 0,
        timem: int = 0,
        training_only_augment: bool = True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.hopsize = hopsize
        self.training_only_augment = training_only_augment

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hopsize,
            n_mels=n_mels,
            power=2.0,
        )

        self.freqm = torchaudio.transforms.FrequencyMasking(freqm) if freqm > 0 else None
        self.timem = torchaudio.transforms.TimeMasking(timem) if timem > 0 else None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram.

        Args:
            waveform: [B, num_samples] raw audio waveform.

        Returns:
            log_mel: [B, 1, n_mels, T] log-mel spectrogram.
        """
        # Compute mel spectrogram
        mel = self.mel_spec(waveform)  # [B, n_mels, T]

        # Log scale (add small epsilon for numerical stability)
        log_mel = torch.log(mel + 1e-7)

        # SpecAugment masking (training only)
        if self.training or not self.training_only_augment:
            if self.freqm is not None:
                log_mel = self.freqm(log_mel)
            if self.timem is not None:
                log_mel = self.timem(log_mel)

        # Add channel dimension: [B, n_mels, T] -> [B, 1, n_mels, T]
        log_mel = log_mel.unsqueeze(1)

        return log_mel


class MobileNetV3Audio(nn.Module):
    """MobileNetV3-Large backbone adapted for audio classification.

    Modifies torchvision's MobileNetV3-Large to accept single-channel
    mel spectrogram input and outputs 960-dim features.

    Args:
        pretrained_name: If set, load pretrained weights from a checkpoint file.
            Supported: 'mn10_as' for AudioSet-pretrained EfficientAT weights.
        num_classes: Number of output classes. If None, returns 960-dim features.
        in_channels: Number of input channels (1 for mel spectrograms).
        width_mult: Width multiplier for MobileNetV3 (1.0 for mn10).
    """

    BACKBONE_DIM = 960  # MobileNetV3-Large feature dimension

    def __init__(
        self,
        pretrained_name: str = None,
        num_classes: int = None,
        in_channels: int = 1,
        width_mult: float = 1.0,
    ):
        super().__init__()
        from torchvision.models.mobilenetv3 import (
            mobilenet_v3_large,
            MobileNet_V3_Large_Weights,
        )

        # Build base model without pretrained weights (we'll load audio-specific ones)
        self.backbone = mobilenet_v3_large(weights=None, width_mult=width_mult)

        # Replace first conv to accept 1-channel input instead of 3-channel
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Remove the classifier (we only want features)
        self.backbone.classifier = nn.Identity()

        # Optional classifier head
        if num_classes is not None:
            self.classifier = nn.Linear(self.BACKBONE_DIM, num_classes)
        else:
            self.classifier = None

        # Load pretrained weights if specified
        if pretrained_name is not None:
            self._load_pretrained(pretrained_name)

    def _load_pretrained(self, name: str):
        """Load pretrained weights from checkpoint file.

        For AudioSet-pretrained weights (mn10_as), expects the checkpoint at:
            checkpoints/mn10_as.pt
        Download from: https://github.com/fschmid56/EfficientAT/releases
        """
        import os

        if name == "mn10_as":
            # Try common locations
            search_paths = [
                "checkpoints/mn10_as.pt",
                "data/mn10_as.pt",
                os.path.expanduser("~/.cache/efficientat/mn10_as.pt"),
            ]
            ckpt_path = None
            for p in search_paths:
                if os.path.exists(p):
                    ckpt_path = p
                    break

            if ckpt_path is None:
                print(
                    f"WARNING: AudioSet pretrained weights not found. "
                    f"Searched: {search_paths}. "
                    f"Download from https://github.com/fschmid56/EfficientAT/releases "
                    f"and place at checkpoints/mn10_as.pt"
                )
                return

            print(f"Loading AudioSet pretrained weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # EfficientAT saves full model state dict — need to filter/remap keys
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # EfficientAT checkpoint keys are like "features.0.0.weight",
            # but our model wraps MobileNetV3 as self.backbone, so our keys
            # are "backbone.features.0.0.weight". Also remap SE layer keys:
            # EfficientAT uses "conc_se_layers.0.fc1/fc2" (Linear, 2D) while
            # torchvision uses "fc1/fc2" (Conv2d, 4D). Remap names and reshape.
            remapped = {}
            for k, v in state_dict.items():
                new_key = f"backbone.{k}"
                # Remap EfficientAT's SE layer naming to torchvision's
                is_se = False
                if ".conc_se_layers.0.fc1" in new_key or ".conc_se_layers.0.fc2" in new_key:
                    new_key = new_key.replace(".conc_se_layers.0.fc1", ".fc1")
                    new_key = new_key.replace(".conc_se_layers.0.fc2", ".fc2")
                    is_se = True
                # Reshape SE weights: EfficientAT Linear [out,in] -> torchvision Conv2d [out,in,1,1]
                if is_se and "weight" in k and v.dim() == 2:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                remapped[new_key] = v

            missing, unexpected = self.load_state_dict(remapped, strict=False)
            # Filter out expected missing/unexpected keys
            missing = [k for k in missing if "classifier" not in k]
            unexpected = [k for k in unexpected if "classifier" not in k]
            if missing:
                print(f"  Missing keys: {len(missing)}: {missing[:5]}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}: {unexpected[:5]}")
            loaded = len(remapped) - len(unexpected)
            print(f"  Loaded {loaded}/{len(remapped)} keys from AudioSet pretrained weights")
        else:
            raise ValueError(f"Unknown pretrained model: {name}")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features.

        Args:
            x: Mel spectrogram [B, 1, n_mels, T].

        Returns:
            features: [B, 960] feature vector.
        """
        # MobileNetV3 features expect [B, C, H, W]
        features = self.backbone.features(x)
        # Global average pooling
        features = self.backbone.avgpool(features)
        features = features.flatten(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mel spectrogram [B, 1, n_mels, T].

        Returns:
            If classifier: [B, num_classes] logits.
            If no classifier: [B, 960] features.
        """
        features = self.get_features(x)
        if self.classifier is not None:
            return self.classifier(features)
        return features
