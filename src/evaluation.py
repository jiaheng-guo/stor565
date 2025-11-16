import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from .data_configs import DatasetConfig
from .modeling import BOOSTING_ALGOS, build_model_pipeline, scoring


def _predict_proba(pipeline, X_test):
    model = pipeline
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        decision = np.asarray(decision)
        if decision.ndim == 1:
            decision = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
            return decision
    return None


def _compute_metrics(
    task: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    encoder: LabelEncoder | None = None,
) -> Tuple[Dict[str, float], str]:
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr") if y_proba is not None else np.nan
        labels_true = encoder.inverse_transform(y_true) if encoder is not None else y_true
        labels_pred = encoder.inverse_transform(y_pred) if encoder is not None else y_pred
        report = classification_report(labels_true, labels_pred)
        return metrics, report

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    report = f"RMSE: {rmse:.4f} MAE: {mae:.4f} R²: {r2:.4f}"
    return metrics, report


def evaluate_dataset(
    config: DatasetConfig,
    algorithm: str,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, float], str, list[dict]]:
    X, y = config.loader()

    if config.task == "classification":
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        encoder = LabelEncoder()
        y_encoded = pd.Series(
            encoder.fit_transform(y),
            index=y.index,
            name=y.name or "target",
        )
    else:
        encoder = None
        y_encoded = y

    stratify = y_encoded if config.task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    base_pipeline = build_model_pipeline(X, config.task, algorithm, random_state=random_state)
    pipeline = clone(base_pipeline)
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start
    y_pred = pipeline.predict(X_test)
    y_proba = _predict_proba(pipeline, X_test)
    test_metrics, detailed_report = _compute_metrics(config.task, y_test, y_pred, y_proba, encoder)

    cv_results = cross_validate(
        base_pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring=scoring(config.task),
        n_jobs=-1,
    )
    cv_summary = {}
    fold_rows: list[dict] = []
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        scores = np.asarray(values)
        if config.task == "regression" and metric in {"rmse", "mae"}:
            scores = -scores
        cv_summary[f"{metric}_mean"] = scores.mean()
        cv_summary[f"{metric}_std"] = scores.std()
    if config.task == "classification":
        tracked_metrics = {}
        for metric in ("roc_auc", "accuracy", "f1_weighted"):
            key = f"test_{metric}"
            if key in cv_results:
                tracked_metrics[metric] = np.asarray(cv_results[key])
        if tracked_metrics:
            n_folds = len(next(iter(tracked_metrics.values())))
            for fold_idx in range(n_folds):
                row = {
                    "dataset": config.pretty_name,
                    "model": algorithm,
                    "fold": fold_idx,
                }
                for metric, scores in tracked_metrics.items():
                    row[metric] = scores[fold_idx]
                fold_rows.append(row)

    result = {
        "dataset": config.pretty_name,
        "task": config.task,
        "model": algorithm,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "training_time": training_time,
    }
    result.update({f"test_{k}": v for k, v in test_metrics.items()})
    result.update({f"cv_{k}": v for k, v in cv_summary.items()})
    return result, detailed_report, fold_rows


__all__ = [
    "evaluate_dataset",
    "BOOSTING_ALGOS",
]
