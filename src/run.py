from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from .evaluation import BOOSTING_ALGOS, evaluate_dataset
from .preprocess import (
    CLASSIFICATION_DATASETS,
    REGRESSION_DATASETS,
    DatasetConfig,
)
from .utils import (
    plot_accuracy,
    plot_f1_score,
    plot_mae,
    plot_r2,
    plot_rmse,
    plot_roc_auc,
    plot_training_time,
)


def run_clf_datasets(
    datasets: Tuple[DatasetConfig, ...] = tuple(CLASSIFICATION_DATASETS),
    make_plot: bool = True,
    auc_path: Path | None = None,
    acc_path: Path | None = None,
    f1_path: Path | None = None,
    time_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    results = []
    reports: Dict[str, str] = {}
    fold_rows = []

    for algorithm in BOOSTING_ALGOS:
        for cfg in tqdm(datasets, desc=f"Evaluating classification datasets with {algorithm}"):
            result, report, folds = evaluate_dataset(cfg, algorithm)
            results.append(result)
            reports[f"{cfg.key}_{algorithm}"] = report
            fold_rows.extend(folds)

    results_df = pd.DataFrame(results)
    fold_df = pd.DataFrame(fold_rows)
    if make_plot:
        if not fold_df.empty:
            plot_roc_auc(fold_df, auc_path)
            plot_accuracy(fold_df, acc_path)
            plot_f1_score(fold_df, f1_path)
        if not results_df.empty:
            plot_training_time(results_df, time_path)
    return results_df, reports, fold_df


def run_reg_datasets(
    datasets: Tuple[DatasetConfig, ...] = tuple(REGRESSION_DATASETS),
    make_plot: bool = True,
    time_path: Path | None = None,
    rmse_path: Path | None = None,
    mae_path: Path | None = None,
    r2_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    results = []
    reports: Dict[str, str] = {}
    fold_rows: list[dict] = []

    for algorithm in BOOSTING_ALGOS:
        for cfg in tqdm(datasets, desc=f"Evaluating regression datasets with {algorithm}"):
            result, report, folds = evaluate_dataset(cfg, algorithm)
            results.append(result)
            reports[f"{cfg.key}_{algorithm}"] = report
            fold_rows.extend(folds)

    results_df = pd.DataFrame(results)
    fold_df = pd.DataFrame(fold_rows)
    if make_plot:
        plot_training_time(results_df, time_path)
        if not fold_df.empty:
            plot_rmse(fold_df, rmse_path)
            plot_mae(fold_df, mae_path)
            plot_r2(fold_df, r2_path)
    return results_df, reports, fold_df


if __name__ == "__main__":
    static_dir = Path("static")
    static_dir.mkdir(parents=True, exist_ok=True)

    # clf_df, clf_reports, fold_df = run_clf_datasets(
    #     auc_path=static_dir / "auroc.png",
    #     acc_path=static_dir / "accuracy.png",
    #     f1_path=static_dir / "f1_score.png",
    #     time_path=static_dir / "clf_training_time.png",
    # )
    # clf_df.to_csv(static_dir / "clf_results.csv", index=False)
    # fold_df.to_csv(static_dir / "clf_fold_metrics.csv", index=False)

    # with open(static_dir / "clf_reports.txt", "w") as fh:
        # for key, rep in clf_reports.items():
            # fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")

    reg_df, reg_reports, reg_fold_df = run_reg_datasets(
        time_path=static_dir / "reg_training_time.png",
        rmse_path=static_dir / "rmse.png",
        mae_path=static_dir / "mae.png",
        r2_path=static_dir / "r2.png",
    )
    reg_df.to_csv(static_dir / "reg_results.csv", index=False)
    reg_fold_df.to_csv(static_dir / "reg_fold_metrics.csv", index=False)

    with open(static_dir / "reg_reports.txt", "w") as fh:
        for key, rep in reg_reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")
