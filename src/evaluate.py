#!/usr/bin/env python3
"""Evaluation and analysis utilities for generating summary tables and plots."""

import argparse
import os
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all CSV result files from directory."""
    csv_files = glob(os.path.join(results_dir, "results_*.csv"))
    if not csv_files:
        raise ValueError(f"No result files found in {results_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds for each configuration."""
    group_cols = ["dataset", "init", "label_fraction"]
    metric_cols = ["best_acc", "final_acc", "aulc"]

    # Add milestone columns if present
    for col in df.columns:
        if col.startswith("val/acc_at_epoch"):
            metric_cols.append(col)

    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()

    # Flatten column names
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary.columns
    ]

    return summary


def generate_summary_table(df: pd.DataFrame, metric: str = "best_acc") -> pd.DataFrame:
    """Generate summary table comparing inits across datasets and label fractions."""
    summary = compute_summary_stats(df)

    # Pivot to get init methods as columns
    pivot = summary.pivot_table(
        index=["dataset", "label_fraction"],
        columns="init",
        values=[f"{metric}_mean", f"{metric}_std"],
        aggfunc="first"
    )

    # Format as "mean ± std"
    result = pd.DataFrame(index=pivot.index)
    for init in ["random", "imagenet", "distilled"]:
        mean_col = (f"{metric}_mean", init)
        std_col = (f"{metric}_std", init)
        if mean_col in pivot.columns:
            result[init] = pivot[mean_col].apply(lambda x: f"{x:.1f}") + " ± " + \
                          pivot[std_col].apply(lambda x: f"{x:.1f}")

    return result


def plot_learning_curves(df: pd.DataFrame, output_dir: str):
    """Plot learning curves comparing inits (requires per-epoch data from W&B)."""
    # This requires loading from W&B or having per-epoch CSVs
    # For now, just plot AULC comparison
    pass


def plot_aulc_comparison(df: pd.DataFrame, output_dir: str):
    """Plot AULC comparison across datasets and inits."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    datasets = df["dataset"].unique()

    for ax, dataset in zip(axes.flat, datasets):
        data = df[df["dataset"] == dataset]

        # Group by init and label_fraction
        grouped = data.groupby(["init", "label_fraction"])["aulc"].agg(["mean", "std"])
        grouped = grouped.reset_index()

        # Plot bars
        x = np.arange(3)  # 3 label fractions
        width = 0.25

        for i, init in enumerate(["random", "imagenet", "distilled"]):
            init_data = grouped[grouped["init"] == init]
            ax.bar(
                x + i * width,
                init_data["mean"],
                width,
                yerr=init_data["std"],
                label=init,
                capsize=3
            )

        ax.set_title(dataset)
        ax.set_xlabel("Label Fraction")
        ax.set_ylabel("AULC")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aulc_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved AULC comparison plot to {output_dir}/aulc_comparison.png")


def plot_accuracy_heatmap(df: pd.DataFrame, output_dir: str, metric: str = "best_acc"):
    """Plot heatmap of accuracy differences (distilled - baseline)."""
    summary = compute_summary_stats(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, baseline in zip(axes, ["random", "imagenet"]):
        # Compute difference: distilled - baseline
        distilled = summary[summary["init"] == "distilled"].set_index(["dataset", "label_fraction"])
        base = summary[summary["init"] == baseline].set_index(["dataset", "label_fraction"])

        diff = distilled[f"{metric}_mean"] - base[f"{metric}_mean"]
        diff = diff.reset_index()
        diff_pivot = diff.pivot(index="dataset", columns="label_fraction", values=0)

        # Plot heatmap
        sns.heatmap(
            diff_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": f"Δ {metric} (%)"}
        )
        ax.set_title(f"Distilled vs {baseline.capitalize()}")
        ax.set_xlabel("Label Fraction")
        ax.set_ylabel("Dataset")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_difference_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved accuracy difference heatmap to {output_dir}/{metric}_difference_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("--results_dir", type=str, default="./checkpoints",
                        help="Directory containing result CSV files")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} experiment results")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Init methods: {df['init'].unique()}")
    print(f"Label fractions: {df['label_fraction'].unique()}")
    print(f"Seeds: {df['seed'].unique()}")

    # Generate summary table
    print("\n" + "=" * 60)
    print("Best Accuracy Summary (mean ± std)")
    print("=" * 60)
    summary_table = generate_summary_table(df, "best_acc")
    print(summary_table.to_string())

    # Save summary table
    summary_table.to_csv(os.path.join(args.output_dir, "summary_best_acc.csv"))

    print("\n" + "=" * 60)
    print("AULC Summary (mean ± std)")
    print("=" * 60)
    aulc_table = generate_summary_table(df, "aulc")
    print(aulc_table.to_string())
    aulc_table.to_csv(os.path.join(args.output_dir, "summary_aulc.csv"))

    # Generate plots
    print("\nGenerating plots...")
    plot_aulc_comparison(df, args.output_dir)
    plot_accuracy_heatmap(df, args.output_dir, "best_acc")
    plot_accuracy_heatmap(df, args.output_dir, "aulc")

    # Compute win rates
    print("\n" + "=" * 60)
    print("Win Rate Analysis")
    print("=" * 60)

    summary = compute_summary_stats(df)

    # Distilled vs Random
    distilled = summary[summary["init"] == "distilled"].set_index(["dataset", "label_fraction"])
    random_init = summary[summary["init"] == "random"].set_index(["dataset", "label_fraction"])
    imagenet = summary[summary["init"] == "imagenet"].set_index(["dataset", "label_fraction"])

    wins_vs_random = (distilled["best_acc_mean"] > random_init["best_acc_mean"]).sum()
    wins_vs_imagenet = (distilled["best_acc_mean"] > imagenet["best_acc_mean"]).sum()
    total = len(distilled)

    print(f"Distilled beats Random: {wins_vs_random}/{total} ({100*wins_vs_random/total:.0f}%)")
    print(f"Distilled beats ImageNet: {wins_vs_imagenet}/{total} ({100*wins_vs_imagenet/total:.0f}%)")

    # Low-label regime analysis (1% and 10%)
    low_label = summary[summary["label_fraction"] <= 0.1]
    distilled_low = low_label[low_label["init"] == "distilled"].set_index(["dataset", "label_fraction"])
    random_low = low_label[low_label["init"] == "random"].set_index(["dataset", "label_fraction"])
    imagenet_low = low_label[low_label["init"] == "imagenet"].set_index(["dataset", "label_fraction"])

    wins_vs_random_low = (distilled_low["best_acc_mean"] > random_low["best_acc_mean"]).sum()
    wins_vs_imagenet_low = (distilled_low["best_acc_mean"] > imagenet_low["best_acc_mean"]).sum()
    total_low = len(distilled_low)

    print(f"\nLow-label regime (1%, 10%):")
    print(f"Distilled beats Random: {wins_vs_random_low}/{total_low} ({100*wins_vs_random_low/total_low:.0f}%)")
    print(f"Distilled beats ImageNet: {wins_vs_imagenet_low}/{total_low} ({100*wins_vs_imagenet_low/total_low:.0f}%)")

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
