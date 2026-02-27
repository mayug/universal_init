#!/usr/bin/env python3
"""CKA similarity analysis between distilled student representations.

Computes pairwise linear CKA between backbone features from students
distilled with different teachers, plus optional baselines and teachers.

Usage:
    # Students only
    python src/analyze_cka.py --dataset pets --output_dir ./results

    # Full comparison including teachers and baselines
    python src/analyze_cka.py --dataset pets --include_teachers --include_baselines --output_dir ./results
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentModel
from src.data.downstream_datasets import get_downstream_dataloaders, get_num_classes
from src.analysis.cka import linear_cka, cka_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="CKA similarity analysis")

    parser.add_argument("--dataset", type=str, default="pets",
                        choices=["pets", "flowers102", "dtd", "eurosat", "imagenette"],
                        help="Dataset for feature extraction")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing distilled checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for CKA results")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples to use for CKA computation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    # Optional model groups
    parser.add_argument("--include_baselines", action="store_true",
                        help="Include random and ImageNet-pretrained baselines")
    parser.add_argument("--include_teachers", action="store_true",
                        help="Include teacher models (requires timm)")

    # Teacher configs
    parser.add_argument("--teachers", type=str, nargs="+",
                        default=["supervised", "clip_768", "clip_512"],
                        help="Teacher names to analyze")

    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, max_samples=2000):
    """Extract backbone features from a model.

    Args:
        model: Model with get_features() method
        dataloader: Data loader
        device: Device
        max_samples: Maximum number of samples to extract

    Returns:
        Feature tensor of shape [N, D]
    """
    model.eval()
    features = []
    n_collected = 0

    for images, _ in tqdm(dataloader, desc="Extracting", leave=False):
        if n_collected >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        feats = model.get_features(images)
        features.append(feats.cpu())
        n_collected += feats.shape[0]

    features = torch.cat(features, dim=0)[:max_samples]
    return features


@torch.no_grad()
def extract_teacher_features(teacher_model, dataloader, device, max_samples=2000):
    """Extract features from a teacher model.

    Args:
        teacher_model: Teacher model (GenericTeacher or similar)
        dataloader: Data loader
        device: Device
        max_samples: Maximum number of samples to extract

    Returns:
        Feature tensor of shape [N, D]
    """
    teacher_model.eval()
    features = []
    n_collected = 0

    for images, _ in tqdm(dataloader, desc="Extracting teacher", leave=False):
        if n_collected >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        feats = teacher_model(images)
        features.append(feats.cpu())
        n_collected += feats.shape[0]

    features = torch.cat(features, dim=0)[:max_samples]
    return features


def load_distilled_student(checkpoint_path, device):
    """Load a distilled student backbone for feature extraction."""
    num_classes = 10  # dummy, we only use get_features()
    model = StudentModel.for_downstream(
        num_classes=num_classes,
        init_mode="distilled",
        checkpoint_path=checkpoint_path,
    ).to(device)
    model.eval()
    return model


def print_cka_matrix(matrix, names):
    """Pretty-print the CKA matrix."""
    # Header
    max_name_len = max(len(n) for n in names)
    header = " " * (max_name_len + 2) + "  ".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        row += "  ".join(f"{matrix[i, j]:10.4f}" for j in range(len(names)))
        print(row)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("CKA Similarity Analysis")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Device: {device}")
    print(f"Include baselines: {args.include_baselines}")
    print(f"Include teachers: {args.include_teachers}")
    print("=" * 60)

    # Load data
    num_classes = get_num_classes(args.dataset)
    _, val_loader, _ = get_downstream_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_fraction=1.0,
        seed=42,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    feature_dict = {}

    # 1. Load distilled students
    for teacher_name in args.teachers:
        ckpt_path = os.path.join(args.checkpoint_dir, f"coco_{teacher_name}_distilled_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}, skipping")
            continue

        print(f"\nExtracting features: student_{teacher_name}")
        model = load_distilled_student(ckpt_path, device)
        features = extract_features(model, val_loader, device, args.max_samples)
        feature_dict[f"student_{teacher_name}"] = features
        del model
        torch.cuda.empty_cache()

    # 2. Optional baselines
    if args.include_baselines:
        for init_mode in ["random", "imagenet"]:
            print(f"\nExtracting features: {init_mode}")
            model = StudentModel.for_downstream(
                num_classes=num_classes,
                init_mode=init_mode,
            ).to(device)
            model.eval()
            features = extract_features(model, val_loader, device, args.max_samples)
            feature_dict[init_mode] = features
            del model
            torch.cuda.empty_cache()

    # 3. Optional teachers
    if args.include_teachers:
        try:
            from src.models.generic_teacher import GenericTeacher

            teacher_configs = {
                "teacher_supervised": ("vit_base_patch16_224.augreg_in1k", False),
                "teacher_clip_768": ("vit_base_patch16_clip_224.openai", False),
                "teacher_clip_512": ("vit_base_patch16_clip_224.openai", True),
            }

            for name, (model_name, use_head) in teacher_configs.items():
                print(f"\nExtracting features: {name}")
                teacher = GenericTeacher(model_name, device=device, use_head=use_head)
                features = extract_teacher_features(teacher, val_loader, device, args.max_samples)
                feature_dict[name] = features
                del teacher
                torch.cuda.empty_cache()

        except ImportError:
            print("WARNING: Could not import GenericTeacher, skipping teacher features")

    if len(feature_dict) < 2:
        print("ERROR: Need at least 2 models for CKA comparison")
        sys.exit(1)

    # Compute CKA matrix
    print(f"\nComputing CKA matrix for {len(feature_dict)} models...")
    matrix, names = cka_matrix(feature_dict)

    # Print results
    print("\n" + "=" * 60)
    print("CKA Similarity Matrix")
    print("=" * 60)
    print_cka_matrix(matrix, names)

    # Save to CSV
    csv_path = os.path.join(args.output_dir, f"cka_matrix_{args.dataset}.csv")
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(csv_path)
    print(f"\nCKA matrix saved to {csv_path}")

    # Print key observations
    print("\n" + "=" * 60)
    print("Key Observations")
    print("=" * 60)

    # Find most/least similar pairs (excluding diagonal)
    n = len(names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost similar pairs:")
    for name1, name2, score in pairs[:3]:
        print(f"  {name1} <-> {name2}: {score:.4f}")

    print("\nLeast similar pairs:")
    for name1, name2, score in pairs[-3:]:
        print(f"  {name1} <-> {name2}: {score:.4f}")

    # Student-student average (if multiple students)
    student_names = [n for n in names if n.startswith("student_")]
    if len(student_names) >= 2:
        student_scores = []
        for i, n1 in enumerate(student_names):
            for j, n2 in enumerate(student_names):
                if i < j:
                    idx1 = names.index(n1)
                    idx2 = names.index(n2)
                    student_scores.append(matrix[idx1, idx2])
        avg_student_cka = np.mean(student_scores)
        print(f"\nAvg student-student CKA: {avg_student_cka:.4f}")
        if avg_student_cka > 0.8:
            print("  → HIGH convergence: students may be bottlenecked by capacity")
        elif avg_student_cka < 0.5:
            print("  → LOW convergence: teacher signal is preserved despite capacity constraints")
        else:
            print("  → MODERATE convergence: partial teacher signal preservation")


if __name__ == "__main__":
    main()
