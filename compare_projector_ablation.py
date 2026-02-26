#!/usr/bin/env python3
"""Compare projector ablation results in detail."""

import pandas as pd
import glob

# Load all results
csv_files = glob.glob("checkpoints/results_*.csv")
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    if 'keep_projector' not in df.columns:
        df['keep_projector'] = False
    dfs.append(df)

results = pd.concat(dfs, ignore_index=True)

# Filter for distilled init only
distilled = results[results['init'] == 'distilled'].copy()

print("="*80)
print("PROJECTOR ABLATION: Detailed Comparison")
print("="*80)
print()

# Compare for each dataset and label fraction
for dataset in ['pets', 'eurosat']:
    print(f"\n{dataset.upper()}:")
    print("-"*80)

    for frac in [0.01, 1.0]:
        print(f"\n  Label Fraction: {frac:.0%}")

        # Get drop projector baseline
        drop = distilled[(distilled['dataset'] == dataset) &
                        (distilled['label_fraction'] == frac) &
                        (distilled['keep_projector'] == False)]

        # Get keep projector results
        keep = distilled[(distilled['dataset'] == dataset) &
                        (distilled['label_fraction'] == frac) &
                        (distilled['keep_projector'] == True)]

        if len(drop) > 0:
            drop_acc = drop['best_acc'].mean()
            drop_std = drop['best_acc'].std()
            drop_aulc = drop['aulc'].mean()
            print(f"    Drop Projector (baseline): {drop_acc:.2f}% ± {drop_std:.2f}% (AULC: {drop_aulc:.2f}) [n={len(drop)}]")

        if len(keep) > 0:
            keep_acc = keep['best_acc'].mean()
            keep_std = keep['best_acc'].std()
            keep_aulc = keep['aulc'].mean()
            print(f"    Keep Frozen Projector:     {keep_acc:.2f}% ± {keep_std:.2f}% (AULC: {keep_aulc:.2f}) [n={len(keep)}]")

        if len(drop) > 0 and len(keep) > 0:
            diff = keep_acc - drop_acc
            diff_pct = (diff / drop_acc) * 100
            print(f"    Difference: {diff:+.2f}% ({diff_pct:+.1f}% relative)")

            if diff > 0:
                print(f"    → Keeping frozen projector IMPROVES performance")
            else:
                print(f"    → Keeping frozen projector HURTS performance")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("Keeping the frozen COCO-distilled projector during fine-tuning:")
print()

# Overall summary
for dataset in ['pets', 'eurosat']:
    for frac in [0.01, 1.0]:
        drop = distilled[(distilled['dataset'] == dataset) &
                        (distilled['label_fraction'] == frac) &
                        (distilled['keep_projector'] == False)]
        keep = distilled[(distilled['dataset'] == dataset) &
                        (distilled['label_fraction'] == frac) &
                        (distilled['keep_projector'] == True)]

        if len(drop) > 0 and len(keep) > 0:
            diff = keep['best_acc'].mean() - drop['best_acc'].mean()
            status = "✓ Better" if diff > 0 else "✗ Worse"
            print(f"  {dataset:8s} {frac:>4.0%}: {status:10s} ({diff:+.2f}%)")

print()
