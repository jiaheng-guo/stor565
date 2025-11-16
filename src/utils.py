from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _plot_metric_boxplots(df: pd.DataFrame, metric: str, output_path: Optional[Path], title_prefix: str) -> None:
    metric_df = df[["dataset", "model", metric]].dropna()
    if metric_df.empty:
        raise ValueError(f"No {metric} scores available to plot.")

    datasets = sorted(metric_df["dataset"].unique())
    n_cols = 3
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for ax, dataset in zip(axes.flat, datasets):
        subset = metric_df[metric_df["dataset"] == dataset]
        sns.boxplot(
            data=subset,
            x="model",
            y=metric,
            ax=ax,
            palette="Set2",
        )
        ax.set_title(f"{title_prefix} | {dataset}")
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_ylim(
            max(0.0, subset[metric].min() - 0.05),
            min(1.05, subset[metric].max() + 0.05),
        )
        ax.tick_params(axis="x", rotation=20)

    total_axes = n_rows * n_cols
    for ax in axes.flat[len(datasets) : total_axes]:
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_auc(fold_df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    _plot_metric_boxplots(fold_df, "roc_auc", output_path, "AUC")


def plot_accuracy(fold_df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    _plot_metric_boxplots(fold_df, "accuracy", output_path, "Accuracy")


def plot_f1_score(fold_df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    _plot_metric_boxplots(fold_df, "f1_weighted", output_path, "F1-score")


def plot_training_time(results_df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    if "training_time" not in results_df:
        raise ValueError("No training_time column found in results.")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="dataset", y="training_time", hue="model", palette="Set2")
    plt.ylabel("Training Time (s)")
    plt.xlabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    plt.title("Training Time per Dataset and Model")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()
