#!/usr/bin/env python3
"""Download AudioCaps from HuggingFace and save in our expected format.

Uses Audio(decode=False) to bypass torchcodec, then decodes raw bytes
with soundfile. This avoids FFmpeg dependency issues.

Creates:
    data/audiocaps/
    ├── train/          # .wav files
    ├── val/
    ├── test/
    └── dataset/
        ├── train.csv   # audiocap_id,youtube_id,start_time,caption
        ├── val.csv
        └── test.csv

Usage:
    python scripts/download_audiocaps.py --data_root ./data
"""

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import soundfile as sf
import numpy as np
from tqdm import tqdm


def download_split_mapped(hf_split, our_name, out_root):
    """Download a HF split and save with our naming convention.

    Uses Audio(decode=False) to get raw bytes, then decodes with soundfile.
    """
    from datasets import load_dataset, Audio

    print(f"\nDownloading AudioCaps '{hf_split}' -> '{our_name}'...")

    # Load with decode=False to bypass torchcodec requirement
    ds = load_dataset("OpenSound/AudioCaps", split=hf_split)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds)} samples")

    audio_dir = out_root / our_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = out_root / "dataset"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{our_name}.csv"
    written = 0
    skipped = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audiocap_id", "youtube_id", "start_time", "caption"])

        for i, sample in enumerate(tqdm(ds, desc=f"Processing {our_name}")):
            try:
                audio_data = sample["audio"]
                caption = sample.get("caption", "")
                audiocap_id = sample.get("audiocap_id", i)
                youtube_id = sample.get("youtube_id", f"unknown_{i}")
                start_time = sample.get("start_time", 0)

                # audio_data is dict with 'bytes' and 'path' when decode=False
                raw_bytes = audio_data["bytes"]
                if raw_bytes is None:
                    # Try loading from path
                    audio_path = audio_data.get("path")
                    if audio_path and os.path.exists(audio_path):
                        array, sr = sf.read(audio_path, dtype="float32")
                    else:
                        skipped += 1
                        continue
                else:
                    # Decode raw bytes with soundfile
                    array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

                # Save as wav
                filename = f"{youtube_id}_{start_time}.wav"
                filepath = audio_dir / filename
                sf.write(str(filepath), array, sr)

                writer.writerow([audiocap_id, youtube_id, start_time, caption])
                written += 1

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Warning: Skipped sample {i}: {e}")
                elif skipped == 6:
                    print("  (suppressing further warnings)")

    print(f"  Saved {written} samples to {audio_dir}")
    print(f"  Metadata: {csv_path}")
    if skipped:
        print(f"  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "validation", "test"])
    args = parser.parse_args()

    out_root = Path(args.data_root) / "audiocaps"
    out_root.mkdir(parents=True, exist_ok=True)

    # Map HF split names to our directory names
    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for hf_split in args.splits:
        our_name = split_map.get(hf_split, hf_split)
        download_split_mapped(hf_split, our_name, out_root)

    print("\nDone! AudioCaps saved to", out_root)


if __name__ == "__main__":
    main()
