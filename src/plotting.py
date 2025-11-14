from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_roc_auc(results: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    """Render (and optionally save) a ROC-AUC comparison chart."""
    auc_df = results[["dataset", "model", "test_roc_auc"]].dropna()
    if auc_df.empty:
        raise ValueError("No ROC-AUC scores available to plot.")

    auc_df = auc_df.sort_values(["dataset", "model"])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=auc_df, x="dataset", y="test_roc_auc", hue="model", palette="viridis")
    plt.ylabel("ROC-AUC (Test)")
    plt.xlabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    plt.title("Boosting Models ROC-AUC Across Classification Datasets")
    plt.ylim(0.5, 1.0)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()
