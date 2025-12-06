import json
import time
from typing import Dict, Tuple

import optuna

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocess import DatasetConfig
from .modeling import BOOSTING_ALGOS, build_model_pipeline, scoring


def _predict_proba(pipeline, X_test):
    model = pipeline
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2:
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba
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
        if y_proba is not None:
            if y_proba.ndim == 1 and len(np.unique(y_true)) > 2:
                metrics["roc_auc"] = np.nan
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        else:
            metrics["roc_auc"] = np.nan
        labels_true = encoder.inverse_transform(y_true) if encoder is not None else y_true
        labels_pred = encoder.inverse_transform(y_pred) if encoder is not None else y_pred
        report = classification_report(labels_true, labels_pred)
        return metrics, report

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    report = f"RMSE: {rmse:.4f} MAE: {mae:.4f} R²: {r2:.4f}"
    return metrics, report


def _suggest_params(trial: optuna.Trial, algorithm: str, task: str) -> Dict:
    alg = algorithm.lower()
    if alg == "adaboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 5e-1, log=True),
        }
        if task == "regression":
            params["loss"] = trial.suggest_categorical("loss", ["square", "linear", "exponential"])
        return params
    if alg == "gbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if alg == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 2e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if alg == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        return params
    raise ValueError(f"Unsupported algorithm for tuning: {algorithm}")


def tune_hyperparameters(
    config: DatasetConfig,
    algorithm: str,
    n_trials: int = 20,
    random_state: int = 42,
) -> Tuple[Dict, float]:
    X, y = config.loader()

    if config.task == "classification":
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        scoring_metric = "accuracy"
        direction = "maximize"
    else:
        encoder = None
        y_encoded = y
        scoring_metric = "neg_root_mean_squared_error"
        direction = "maximize"

    num_classes = len(np.unique(y_encoded)) if config.task == "classification" else None

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, algorithm, config.task)
        pipeline = build_model_pipeline(
            X,
            config.task,
            algorithm,
            random_state=random_state,
            num_classes=num_classes,
            model_params=params,
        )
        cv_results = cross_validate(
            pipeline,
            X,
            y_encoded,
            cv=5,
            scoring=scoring_metric,
            n_jobs=-1,
        )
        score = cv_results["test_score"].mean()
        return score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value
    if config.task == "regression":
        best_score = -best_score
    return best_params, best_score

def evaluate_dataset(
    config: DatasetConfig,
    algorithm: str,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    model_params: Dict | None = None,
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

    num_classes = len(np.unique(y_encoded)) if config.task == "classification" else None
    base_pipeline = build_model_pipeline(
        X,
        config.task,
        algorithm,
        random_state=random_state,
        num_classes=num_classes,
        model_params=model_params,
    )
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
    else:
        tracked_metrics = {}
        for metric in ("rmse", "mae", "r2"):
            key = f"test_{metric}"
            if key in cv_results:
                scores = np.asarray(cv_results[key])
                if metric in {"rmse", "mae"}:
                    scores = -scores
                tracked_metrics[metric] = scores
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
    "tune_hyperparameters",
]
