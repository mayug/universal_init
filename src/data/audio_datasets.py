"""Audio dataset loaders for cross-modal distillation and downstream evaluation.

Datasets:
  - AudioCaps: Audio-caption pairs for cross-modal distillation.
  - ESC-50: Environmental sound classification (5-fold CV).
  - UrbanSound8K: Urban sound classification (10-fold CV).

All datasets return raw waveforms. Mel spectrogram conversion happens in
the model's preprocessing module (AugmentMelSTFT), keeping the data pipeline
clean and modality-agnostic.
"""

import os
import csv
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile is required for audio datasets: pip install soundfile")

try:
    import soxr
    _HAS_SOXR = True
except ImportError:
    _HAS_SOXR = False


def load_audio(
    path: str,
    target_sr: int = 32000,
    max_duration: float = None,
    mono: bool = True,
) -> torch.Tensor:
    """Load and preprocess an audio file.

    Uses soundfile for loading (avoids torchcodec/FFmpeg dependency).
    Uses soxr for high-quality resampling if available, otherwise linear interpolation.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.
        max_duration: Maximum duration in seconds (truncate/pad).
        mono: If True, convert to mono.

    Returns:
        waveform: [num_samples] tensor (mono) or [channels, num_samples].
    """
    # Load with soundfile (returns numpy array)
    data, sr = sf.read(path, dtype="float32")

    # Convert to tensor
    waveform = torch.from_numpy(data)

    # Handle mono/stereo: soundfile returns [samples] for mono, [samples, channels] for stereo
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, samples]
    else:
        waveform = waveform.T  # [channels, samples]

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        if _HAS_SOXR:
            # High-quality resampling with soxr
            audio_np = waveform.numpy()
            # soxr expects [samples, channels]
            resampled = soxr.resample(audio_np.T, sr, target_sr)
            waveform = torch.from_numpy(resampled.T)
        else:
            # Fallback: linear interpolation
            target_len = int(waveform.shape[-1] * target_sr / sr)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False
            ).squeeze(0)

    # Squeeze to 1D for mono
    if mono:
        waveform = waveform.squeeze(0)

    # Pad or truncate to fixed length
    if max_duration is not None:
        target_len = int(target_sr * max_duration)
        if waveform.shape[-1] < target_len:
            pad_len = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]

    return waveform


# ============================================================
# AudioCaps: Audio-caption pairs for cross-modal distillation
# ============================================================

class AudioCapsDataset(Dataset):
    """AudioCaps dataset for cross-modal distillation.

    Returns (waveform, caption) pairs. The teacher encodes the caption
    and the student encodes the audio spectrogram.

    Expected directory structure:
        root/
        ├── train/          # Audio files: {youtube_id}_{start_time}.wav
        ├── val/
        ├── test/
        └── dataset/        # Metadata CSVs from HuggingFace
            ├── train.csv   # columns: audiocap_id,youtube_id,start_time,caption
            ├── val.csv
            └── test.csv

    AudioCaps can be obtained from:
    - HuggingFace: https://huggingface.co/datasets/OpenSound/AudioCaps
    - Or download AudioSet clips + AudioCaps captions separately.

    Args:
        root: Path to AudioCaps root directory.
        split: One of 'train', 'val', 'test'.
        sample_rate: Target sample rate.
        max_duration: Maximum clip duration in seconds.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 32000,
        max_duration: float = 10.0,
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load metadata
        self.entries = self._load_metadata()
        print(f"AudioCaps {split}: {len(self.entries)} entries")

    def _load_metadata(self) -> List[dict]:
        """Load and filter metadata CSV, keeping only entries with existing audio."""
        # Try multiple CSV locations
        csv_candidates = [
            self.root / "dataset" / f"{self.split}.csv",
            self.root / f"{self.split}.csv",
            self.root / "metadata" / f"{self.split}.csv",
        ]

        csv_path = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_path = candidate
                break

        if csv_path is None:
            raise FileNotFoundError(
                f"AudioCaps metadata CSV not found. Tried: {csv_candidates}"
            )

        # Audio directory
        audio_dir_candidates = [
            self.root / self.split,
            self.root / "audio" / self.split,
            self.root / "data" / self.split,
        ]
        audio_dir = None
        for candidate in audio_dir_candidates:
            if candidate.exists():
                audio_dir = candidate
                break

        if audio_dir is None:
            raise FileNotFoundError(
                f"AudioCaps audio directory not found. Tried: {audio_dir_candidates}"
            )

        entries = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                youtube_id = row.get("youtube_id", row.get("ytid", ""))
                start_time = row.get("start_time", row.get("start", ""))
                caption = row.get("caption", "")

                # Try common audio filename patterns
                audio_candidates = [
                    audio_dir / f"{youtube_id}_{start_time}.wav",
                    audio_dir / f"{youtube_id}.wav",
                    audio_dir / f"Y{youtube_id}.wav",
                    audio_dir / f"{youtube_id}_{start_time}.flac",
                ]

                audio_path = None
                for ac in audio_candidates:
                    if ac.exists():
                        audio_path = ac
                        break

                if audio_path is not None and caption:
                    entries.append({
                        "audio_path": str(audio_path),
                        "caption": caption,
                    })

        if len(entries) == 0:
            # Fallback: scan audio directory directly and match with CSV
            print(f"WARNING: No entries matched by filename. Scanning {audio_dir}...")
            all_audio = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))
            print(f"  Found {len(all_audio)} audio files in {audio_dir}")

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform = load_audio(
            entry["audio_path"],
            target_sr=self.sample_rate,
            max_duration=self.max_duration,
        )
        return waveform, entry["caption"]


# ============================================================
# ESC-50: Environmental Sound Classification (50 classes, 5-fold CV)
# ============================================================

class ESC50Dataset(Dataset):
    """ESC-50 dataset for downstream evaluation.

    Environmental Sound Classification dataset with 50 classes, 2000 clips,
    and 5 predefined folds for cross-validation.

    Expected directory structure:
        root/
        ├── audio/          # 2000 .wav files (5s each, 44.1kHz)
        └── meta/
            └── esc50.csv   # columns: filename,fold,target,category,esc10,...

    Download from: https://github.com/karolpiczak/ESC-50

    Args:
        root: Path to ESC-50 root directory.
        folds: List of fold numbers to include (1-5). Use for CV splits.
        sample_rate: Target sample rate.
        max_duration: Maximum clip duration in seconds.
    """

    NUM_CLASSES = 50

    def __init__(
        self,
        root: str,
        folds: Optional[List[int]] = None,
        sample_rate: int = 32000,
        max_duration: float = 5.0,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load metadata
        meta_path = self.root / "meta" / "esc50.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")

        self.entries = []
        self.labels = []
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold = int(row["fold"])
                if folds is not None and fold not in folds:
                    continue

                audio_path = self.root / "audio" / row["filename"]
                if not audio_path.exists():
                    continue

                self.entries.append({
                    "audio_path": str(audio_path),
                    "label": int(row["target"]),
                    "fold": fold,
                    "category": row["category"],
                })
                self.labels.append(int(row["target"]))

        self.labels = np.array(self.labels)
        fold_str = str(folds) if folds else "all"
        print(f"ESC-50 (folds={fold_str}): {len(self.entries)} clips, {self.NUM_CLASSES} classes")

    @property
    def targets(self):
        """For compatibility with label fraction subsampling."""
        return self.labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform = load_audio(
            entry["audio_path"],
            target_sr=self.sample_rate,
            max_duration=self.max_duration,
        )
        return waveform, entry["label"]


# ============================================================
# UrbanSound8K: Urban Sound Classification (10 classes, 10-fold CV)
# ============================================================

class UrbanSound8KDataset(Dataset):
    """UrbanSound8K dataset for downstream evaluation.

    Urban sound classification with 10 classes, 8732 clips, and 10
    predefined folds for cross-validation. Folds must not be reshuffled
    per the dataset authors' instructions.

    Expected directory structure:
        root/
        ├── audio/
        │   ├── fold1/      # Audio files per fold
        │   ├── fold2/
        │   └── ...
        └── metadata/
            └── UrbanSound8K.csv  # columns: slice_file_name,fsID,start,end,
                                  #          salience,fold,classID,class

    Download from: https://urbansounddataset.weebly.com/urbansound8k.html

    Args:
        root: Path to UrbanSound8K root directory.
        folds: List of fold numbers to include (1-10). Use for CV splits.
        sample_rate: Target sample rate.
        max_duration: Maximum clip duration in seconds.
    """

    NUM_CLASSES = 10

    def __init__(
        self,
        root: str,
        folds: Optional[List[int]] = None,
        sample_rate: int = 32000,
        max_duration: float = 4.0,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load metadata
        meta_path = self.root / "metadata" / "UrbanSound8K.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"UrbanSound8K metadata not found at {meta_path}")

        self.entries = []
        self.labels = []
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold = int(row["fold"])
                if folds is not None and fold not in folds:
                    continue

                audio_path = (
                    self.root / "audio" / f"fold{fold}" / row["slice_file_name"]
                )
                if not audio_path.exists():
                    continue

                self.entries.append({
                    "audio_path": str(audio_path),
                    "label": int(row["classID"]),
                    "fold": fold,
                    "class_name": row["class"],
                })
                self.labels.append(int(row["classID"]))

        self.labels = np.array(self.labels)
        fold_str = str(folds) if folds else "all"
        print(f"UrbanSound8K (folds={fold_str}): {len(self.entries)} clips, {self.NUM_CLASSES} classes")

    @property
    def targets(self):
        """For compatibility with label fraction subsampling."""
        return self.labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform = load_audio(
            entry["audio_path"],
            target_sr=self.sample_rate,
            max_duration=self.max_duration,
        )
        return waveform, entry["label"]


# ============================================================
# Collate functions
# ============================================================

def audio_caption_collate_fn(batch):
    """Collate function for AudioCaps (waveform, caption) pairs.

    Pads waveforms to the longest in the batch.

    Returns:
        waveforms: [B, max_samples] tensor.
        captions: List of caption strings.
    """
    waveforms, captions = zip(*batch)

    # Pad to longest waveform in batch
    max_len = max(w.shape[-1] for w in waveforms)
    padded = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded.append(w)

    waveforms = torch.stack(padded)
    return waveforms, list(captions)


def audio_label_collate_fn(batch):
    """Collate function for (waveform, label) pairs.

    Pads waveforms to the longest in the batch.

    Returns:
        waveforms: [B, max_samples] tensor.
        labels: [B] int tensor.
    """
    waveforms, labels = zip(*batch)

    max_len = max(w.shape[-1] for w in waveforms)
    padded = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded.append(w)

    waveforms = torch.stack(padded)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels


# ============================================================
# Helper functions
# ============================================================

def create_label_fraction_subset(
    dataset: Dataset,
    label_fraction: float,
    seed: int = 42,
) -> Dataset:
    """Create a stratified subset of an audio dataset.

    Args:
        dataset: Dataset with .targets attribute.
        label_fraction: Fraction of data to keep.
        seed: Random seed.

    Returns:
        Subset of the dataset.
    """
    if label_fraction >= 1.0:
        return dataset

    labels = dataset.targets
    indices = np.arange(len(dataset))
    n_desired = max(1, int(len(dataset) * label_fraction))
    n_classes = len(np.unique(labels))

    if n_desired < n_classes:
        # Too few samples for stratified split — select at least 1 per class
        rng = np.random.RandomState(seed)
        subset_indices = []
        for c in range(n_classes):
            class_idx = indices[labels == c]
            subset_indices.append(rng.choice(class_idx, size=1)[0])
        subset_indices = np.array(subset_indices)
    else:
        from sklearn.model_selection import train_test_split
        _, subset_indices = train_test_split(
            indices,
            test_size=label_fraction,
            stratify=labels,
            random_state=seed,
        )

    return Subset(dataset, subset_indices)


def get_audiocaps_dataloader(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    split: str = "train",
    sample_rate: int = 32000,
    max_duration: float = 10.0,
) -> DataLoader:
    """Get AudioCaps dataloader for distillation."""
    dataset = AudioCapsDataset(
        root=os.path.join(data_root, "audiocaps"),
        split=split,
        sample_rate=sample_rate,
        max_duration=max_duration,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=audio_caption_collate_fn,
    )


def get_audiocaps_dataloaders_with_val(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    sample_rate: int = 32000,
    max_duration: float = 10.0,
) -> Tuple[DataLoader, DataLoader]:
    """Get AudioCaps train and val dataloaders."""
    train_loader = get_audiocaps_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="train",
        sample_rate=sample_rate,
        max_duration=max_duration,
    )
    val_loader = get_audiocaps_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        split="val",
        sample_rate=sample_rate,
        max_duration=max_duration,
    )
    return train_loader, val_loader
