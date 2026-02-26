#!/usr/bin/env python3
"""Analyze experiment results and generate reports."""

import pandas as pd
import glob
from pathlib import Path
import numpy as np

def load_all_results(results_dir="./checkpoints"):
    """Load all result CSV files."""
    csv_files = glob.glob(f"{results_dir}/results_*.csv")

    if not csv_files:
        print(f"No result files found in {results_dir}")
        return None

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not dfs:
        return None

    results = pd.concat(dfs, ignore_index=True)

    # Add keep_projector column if missing (for backward compatibility)
    if 'keep_projector' not in results.columns:
        results['keep_projector'] = False

    print(f"Loaded {len(results)} results from {len(csv_files)} files")
    return results

def analyze_by_init(results):
    """Analyze results grouped by initialization method."""
    print("\n" + "="*80)
    print("ANALYSIS: Performance by Initialization Method")
    print("="*80 + "\n")

    summary = results.groupby(['dataset', 'init', 'label_fraction']).agg({
        'best_acc': ['mean', 'std', 'count'],
        'aulc': ['mean', 'std'],
        'val/acc_at_epoch_5': ['mean'],
        'val/acc_at_epoch_10': ['mean'],
        'val/acc_at_epoch_20': ['mean']
    }).round(2)

    print(summary)
    print()

    return summary

def analyze_projector_ablation(results):
    """Analyze projector keep vs drop comparison."""
    print("\n" + "="*80)
    print("ANALYSIS: Projector Ablation (Keep vs Drop)")
    print("="*80 + "\n")

    # Filter for distilled init only
    distilled = results[results['init'] == 'distilled'].copy()

    if len(distilled) == 0:
        print("No distilled results found")
        return None

    if 'keep_projector' not in distilled.columns:
        print("No projector ablation data found (keep_projector column missing)")
        return None

    # Check if we have both keep and drop variants
    keep_counts = distilled['keep_projector'].value_counts()
    print(f"Results with keep_projector=True: {keep_counts.get(True, 0)}")
    print(f"Results with keep_projector=False: {keep_counts.get(False, 0)}")
    print()

    if True not in keep_counts:
        print("No results with keep_projector=True found yet")
        return None

    summary = distilled.groupby(['dataset', 'label_fraction', 'keep_projector']).agg({
        'best_acc': ['mean', 'std', 'count'],
        'aulc': ['mean', 'std']
    }).round(2)

    print(summary)
    print()

    return summary

def compare_inits(results):
    """Compare different initialization methods."""
    print("\n" + "="*80)
    print("COMPARISON: Distilled vs ImageNet vs Random")
    print("="*80 + "\n")

    for dataset in results['dataset'].unique():
        print(f"\nDataset: {dataset.upper()}")
        print("-" * 60)

        dataset_results = results[results['dataset'] == dataset]

        for frac in sorted(dataset_results['label_fraction'].unique()):
            print(f"\n  Label Fraction: {frac:.0%}")

            frac_results = dataset_results[dataset_results['label_fraction'] == frac]

            for init_method in ['distilled', 'imagenet', 'random']:
                init_results = frac_results[frac_results['init'] == init_method]

                if len(init_results) > 0:
                    mean_acc = init_results['best_acc'].mean()
                    std_acc = init_results['best_acc'].std()
                    mean_aulc = init_results['aulc'].mean()
                    count = len(init_results)

                    print(f"    {init_method:12s}: {mean_acc:5.2f}% ± {std_acc:.2f}% "
                          f"(AULC: {mean_aulc:5.2f}) [n={count}]")

def generate_summary_table(results):
    """Generate a summary table for the report."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Best Accuracy by Init and Label Fraction")
    print("="*80 + "\n")

    # Pivot table: dataset/init as rows, label fractions as columns
    for dataset in sorted(results['dataset'].unique()):
        print(f"\n{dataset.upper()}:")
        print("-" * 60)

        dataset_results = results[results['dataset'] == dataset].copy()

        # Group by init and label_fraction
        pivot = dataset_results.groupby(['init', 'label_fraction'])['best_acc'].mean().reset_index()
        pivot_table = pivot.pivot(index='init', columns='label_fraction', values='best_acc')

        # Format nicely
        print(pivot_table.to_string(float_format=lambda x: f"{x:.2f}%"))
        print()

def main():
    """Main analysis function."""
    print("="*80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*80)

    # Load all results
    results = load_all_results()

    if results is None or len(results) == 0:
        print("No results to analyze")
        return

    # Print summary
    print(f"\nTotal experiments: {len(results)}")
    print(f"Datasets: {', '.join(results['dataset'].unique())}")
    print(f"Init methods: {', '.join(results['init'].unique())}")
    print(f"Label fractions: {', '.join([f'{x:.0%}' for x in sorted(results['label_fraction'].unique())])}")

    # Run analyses
    analyze_by_init(results)
    compare_inits(results)
    generate_summary_table(results)
    analyze_projector_ablation(results)

    # Save combined results
    output_path = "reports/combined_results.csv"
    Path("reports").mkdir(exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nCombined results saved to: {output_path}")

if __name__ == "__main__":
    main()
