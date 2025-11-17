from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from .preprocess import CLASSIFICATION_DATASETS, DatasetConfig
from .evaluation import BOOSTING_ALGOS, evaluate_dataset
from .utils import plot_accuracy, plot_f1_score, plot_roc_auc, plot_training_time


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
        for cfg in tqdm(datasets, desc=f"Evaluating datasets with {algorithm}"):
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

def run_reg_datasets():
    pass 

if __name__ == "__main__":
    static_dir = Path("static")
    static_dir.mkdir(parents=True, exist_ok=True)

    df, reports, fold_df = run_clf_datasets(
        auc_path=static_dir / "auroc.png",
        acc_path=static_dir / "accuracy.png",
        f1_path=static_dir / "f1_score.png",
        time_path=static_dir / "training_time.png",
    )
    df.to_csv(static_dir / "results.csv", index=False)
    fold_df.to_csv(static_dir / "fold_roc_auc.csv", index=False)
    with open(static_dir / "reports.txt", "w") as fh:
        for key, rep in reports.items():
            fh.write(f"{key}\n{'=' * 40}\n{rep}\n\n")
