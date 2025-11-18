import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from .evaluation import BOOSTING_ALGOS, evaluate_dataset, tune_hyperparameters
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
    tuned: bool = False,
    n_trials: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    results = []
    reports: Dict[str, str] = {}
    fold_rows = []

    for algorithm in BOOSTING_ALGOS:
        for cfg in tqdm(datasets, desc=f"{'Tuned' if tuned else 'Baseline'} classification | {algorithm}"):
            model_params = None
            best_score = None
            if tuned:
                model_params, best_score = tune_hyperparameters(cfg, algorithm, n_trials=n_trials)
            result, report, folds = evaluate_dataset(cfg, algorithm, model_params=model_params)
            result["tuned"] = tuned
            if tuned and best_score is not None:
                result["optuna_metric"] = best_score
                if model_params:
                    result["best_params"] = json.dumps(model_params)
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
            plot_training_time(results_df, time_path, n_cols=3)
    return results_df, reports, fold_df


def run_reg_datasets(
    datasets: Tuple[DatasetConfig, ...] = tuple(REGRESSION_DATASETS),
    make_plot: bool = True,
    time_path: Path | None = None,
    rmse_path: Path | None = None,
    mae_path: Path | None = None,
    r2_path: Path | None = None,
    tuned: bool = False,
    n_trials: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    results = []
    reports: Dict[str, str] = {}
    fold_rows: list[dict] = []

    for algorithm in BOOSTING_ALGOS:
        for cfg in tqdm(datasets, desc=f"{'Tuned' if tuned else 'Baseline'} regression | {algorithm}"):
            model_params = None
            best_score = None
            if tuned:
                model_params, best_score = tune_hyperparameters(cfg, algorithm, n_trials=n_trials)
            result, report, folds = evaluate_dataset(cfg, algorithm, model_params=model_params)
            result["tuned"] = tuned
            if tuned and best_score is not None:
                result["optuna_metric"] = best_score
                if model_params:
                    result["best_params"] = json.dumps(model_params)
            results.append(result)
            reports[f"{cfg.key}_{algorithm}"] = report
            fold_rows.extend(folds)

    results_df = pd.DataFrame(results)
    fold_df = pd.DataFrame(fold_rows)
    if make_plot:
        plot_training_time(results_df, time_path, n_cols=2)
        if not fold_df.empty:
            plot_rmse(fold_df, rmse_path)
            plot_mae(fold_df, mae_path)
            plot_r2(fold_df, r2_path)
    return results_df, reports, fold_df


def build_comparison(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    metrics: Tuple[str, ...],
) -> pd.DataFrame:
    merged = baseline_df.merge(
        tuned_df,
        on=["dataset", "model"],
        suffixes=("_baseline", "_tuned"),
    )
    for metric in metrics:
        merged[f"{metric}_delta"] = merged[f"{metric}_tuned"] - merged[f"{metric}_baseline"]
    return merged


if __name__ == "__main__":
    static_dir = Path("static")
    static_dir.mkdir(parents=True, exist_ok=True)

    clf_df, clf_reports, clf_fold_df = run_clf_datasets(
        auc_path=static_dir / "clf_baseline_auroc.png",
        acc_path=static_dir / "clf_baseline_accuracy.png",
        f1_path=static_dir / "clf_baseline_f1.png",
        time_path=static_dir / "clf_baseline_training_time.png",
    )
    clf_df.to_csv(static_dir / "clf_baseline_results.csv", index=False)
    clf_fold_df.to_csv(static_dir / "clf_baseline_fold_metrics.csv", index=False)
    with open(static_dir / "clf_baseline_reports.txt", "w") as fh:
        for key, rep in clf_reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")

    clf_tuned_df, clf_tuned_reports, clf_tuned_fold_df = run_clf_datasets(
        auc_path=static_dir / "clf_tuned_auroc.png",
        acc_path=static_dir / "clf_tuned_accuracy.png",
        f1_path=static_dir / "clf_tuned_f1.png",
        time_path=static_dir / "clf_tuned_training_time.png",
        tuned=True,
        n_trials=20,
    )
    clf_tuned_df.to_csv(static_dir / "clf_tuned_results.csv", index=False)
    clf_tuned_fold_df.to_csv(static_dir / "clf_tuned_fold_metrics.csv", index=False)
    with open(static_dir / "clf_tuned_reports.txt", "w") as fh:
        for key, rep in clf_tuned_reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")

    clf_comparison = build_comparison(
        clf_df,
        clf_tuned_df,
        metrics=("test_accuracy", "test_f1_weighted", "test_roc_auc"),
    )
    clf_comparison.to_csv(static_dir / "clf_tuning_comparison.csv", index=False)
    print("Classification baseline vs tuned:")
    print(
        clf_comparison[
            [
                "dataset",
                "model",
                "test_accuracy_baseline",
                "test_accuracy_tuned",
                "test_accuracy_delta",
                "test_f1_weighted_delta",
                "test_roc_auc_delta",
            ]
        ]
    )

    reg_df, reg_reports, reg_fold_df = run_reg_datasets(
        time_path=static_dir / "reg_baseline_training_time.png",
        rmse_path=static_dir / "reg_baseline_rmse.png",
        mae_path=static_dir / "reg_baseline_mae.png",
        r2_path=static_dir / "reg_baseline_r2.png",
    )
    reg_df.to_csv(static_dir / "reg_baseline_results.csv", index=False)
    reg_fold_df.to_csv(static_dir / "reg_baseline_fold_metrics.csv", index=False)
    with open(static_dir / "reg_baseline_reports.txt", "w") as fh:
        for key, rep in reg_reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")

    reg_tuned_df, reg_tuned_reports, reg_tuned_fold_df = run_reg_datasets(
        time_path=static_dir / "reg_tuned_training_time.png",
        rmse_path=static_dir / "reg_tuned_rmse.png",
        mae_path=static_dir / "reg_tuned_mae.png",
        r2_path=static_dir / "reg_tuned_r2.png",
        tuned=True,
        n_trials=20,
    )
    reg_tuned_df.to_csv(static_dir / "reg_tuned_results.csv", index=False)
    reg_tuned_fold_df.to_csv(static_dir / "reg_tuned_fold_metrics.csv", index=False)
    with open(static_dir / "reg_tuned_reports.txt", "w") as fh:
        for key, rep in reg_tuned_reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")

    reg_comparison = build_comparison(
        reg_df,
        reg_tuned_df,
        metrics=("test_rmse", "test_mae", "test_r2"),
    )
    reg_comparison.to_csv(static_dir / "reg_tuning_comparison.csv", index=False)
    print("Regression baseline vs tuned:")
    print(
        reg_comparison[
            [
                "dataset",
                "model",
                "test_rmse_baseline",
                "test_rmse_tuned",
                "test_rmse_delta",
                "test_mae_delta",
                "test_r2_delta",
            ]
        ]
    )
