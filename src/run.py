from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .data_configs import CLASSIFICATION_DATASETS, DatasetConfig
from .evaluation import BOOSTING_ALGOS, evaluate_dataset
from .modeling import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from .plotting import plot_roc_auc


def run_all_datasets(
    datasets: Tuple[DatasetConfig, ...] = tuple(CLASSIFICATION_DATASETS),
    make_plot: bool = True,
    plot_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    results = []
    reports: Dict[str, str] = {}

    for algorithm in BOOSTING_ALGOS:
        if algorithm == "xgboost" and not XGBOOST_AVAILABLE:
            print("Skipping XGBoost — package not installed.")
            continue
        if algorithm == "lightgbm" and not LIGHTGBM_AVAILABLE:
            print("Skipping LightGBM — package not installed.")
            continue
        for cfg in datasets:
            print(f"===== Evaluating {cfg.pretty_name} | {algorithm.upper()} =====")
            result, report = evaluate_dataset(cfg, algorithm)
            results.append(result)
            reports[f"{cfg.key}_{algorithm}"] = report

    results_df = pd.DataFrame(results)
    if make_plot and not results_df.empty:
        plot_roc_auc(results_df, plot_path)
    return results_df, reports


if __name__ == "__main__":
    df, reports = run_all_datasets()
    df.to_csv("outputs/results.csv", index=False)
    with open("outputs/reports.txt", "w") as fh:
        for key, rep in reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")
