from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    XGBClassifier = XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    LGBMClassifier = LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


BOOSTING_ALGOS: List[str] = ["adaboost", "gbm", "xgboost", "lightgbm"]


def _build_model(task: str, algorithm: str, random_state: int):
    algorithm = algorithm.lower()
    if algorithm == "adaboost":
        if task == "classification":
            return AdaBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=random_state)
        return AdaBoostRegressor(n_estimators=500, learning_rate=0.3, loss="square", random_state=random_state)
    if algorithm == "gbm":
        if task == "classification":
            return GradientBoostingClassifier(random_state=random_state)
        return GradientBoostingRegressor(random_state=random_state)
    if algorithm == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("Install the `xgboost` package to run XGBoost models.")
        if task == "classification":
            return XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if algorithm == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("Install the `lightgbm` package to run LightGBM models.")
        if task == "classification":
            return LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
            )
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="regression_l2",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_model_pipeline(
    X: pd.DataFrame,
    task: str,
    algorithm: str,
    random_state: int = 42,
) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = _build_model(task, algorithm, random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def scoring(task: str) -> Dict[str, str]:
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
            "roc_auc": "roc_auc",
        }
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
