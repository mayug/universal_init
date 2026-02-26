#!/usr/bin/env python3
"""Generate a clean summary table comparing all initialization methods."""

import pandas as pd
import glob

def load_results():
    """Load all result CSV files."""
    csv_files = glob.glob("./checkpoints/results_*.csv")
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    results = pd.concat(dfs, ignore_index=True)
    return results

def generate_summary_table(results):
    """Generate clean summary table."""

    # Group by dataset, init, and label_fraction
    grouped = results.groupby(['dataset', 'init', 'label_fraction']).agg({
        'best_acc': ['mean', 'std', 'count'],
        'aulc': 'mean',
        'val/acc_at_epoch_5': 'mean',
        'val/acc_at_epoch_10': 'mean',
        'val/acc_at_epoch_20': 'mean'
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in grouped.columns.values]

    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: All Initialization Methods")
    print("="*100)

    for dataset in sorted(results['dataset'].unique()):
        dataset_data = grouped[grouped['dataset'] == dataset]

        print(f"\n### {dataset.upper()} Dataset")
        print("-" * 100)

        # Create pivot table
        pivot = dataset_data.pivot_table(
            index='init',
            columns='label_fraction',
            values='best_acc_mean',
            aggfunc='first'
        )

        # Format as percentages
        pivot = pivot.applymap(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

        print(pivot.to_string())
        print()

        # Show detailed metrics for each condition
        for _, row in dataset_data.iterrows():
            init = row['init']
            frac = row['label_fraction']
            acc = row['best_acc_mean']
            aulc = row['aulc_mean']
            n = int(row['best_acc_count'])

            epoch5 = row.get('val/acc_at_epoch_5_mean', None)
            epoch10 = row.get('val/acc_at_epoch_10_mean', None)
            epoch20 = row.get('val/acc_at_epoch_20_mean', None)

            print(f"  {init:20s} @ {int(frac*100):3d}% labels: "
                  f"Best={acc:5.2f}%  AULC={aulc:5.2f}  "
                  f"E5={epoch5:5.2f}%  E10={epoch10:5.2f}%  E20={epoch20:5.2f}%  "
                  f"[n={n}]")

        print()

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)

    # Calculate distillation efficiency for Pets
    pets_data = grouped[grouped['dataset'] == 'pets']

    print("\nDistillation Efficiency (Pets):")
    for frac in [0.01, 1.0]:
        oracle_row = pets_data[(pets_data['init'] == 'teacher_oracle') &
                               (pets_data['label_fraction'] == frac)]
        distilled_row = pets_data[(pets_data['init'] == 'distilled') &
                                  (pets_data['label_fraction'] == frac)]

        if not oracle_row.empty and not distilled_row.empty:
            oracle_acc = oracle_row['best_acc_mean'].values[0]
            distilled_acc = distilled_row['best_acc_mean'].values[0]
            efficiency = (distilled_acc / oracle_acc) * 100

            print(f"  @ {int(frac*100):3d}% labels: "
                  f"Teacher Oracle={oracle_acc:5.2f}%  "
                  f"COCO-Distilled={distilled_acc:5.2f}%  "
                  f"Efficiency={efficiency:5.1f}%")

    print("\nPerformance Gaps (vs ImageNet):")
    for dataset in ['pets', 'eurosat']:
        dataset_data = grouped[grouped['dataset'] == dataset]

        print(f"\n  {dataset.upper()}:")
        for frac in sorted(dataset_data['label_fraction'].unique()):
            imagenet_row = dataset_data[(dataset_data['init'] == 'imagenet') &
                                        (dataset_data['label_fraction'] == frac)]

            if imagenet_row.empty:
                continue

            imagenet_acc = imagenet_row['best_acc_mean'].values[0]

            for init in ['teacher_oracle', 'distilled', 'random']:
                init_row = dataset_data[(dataset_data['init'] == init) &
                                       (dataset_data['label_fraction'] == frac)]

                if not init_row.empty:
                    init_acc = init_row['best_acc_mean'].values[0]
                    gap = imagenet_acc - init_acc
                    multiplier = imagenet_acc / init_acc if init_acc > 0 else float('inf')

                    print(f"    @ {int(frac*100):3d}% labels, {init:20s}: "
                          f"{init_acc:5.2f}% (gap: {gap:+6.2f}%, {multiplier:.2f}× worse)")

    print("\n" + "="*100)

if __name__ == "__main__":
    results = load_results()
    generate_summary_table(results)
