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
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


BOOSTING_ALGOS: List[str] = ["adaboost", "gbm", "xgboost", "lightgbm"]


def _build_model(
    task: str,
    algorithm: str,
    random_state: int,
    num_classes: int | None = None,
    model_params: Dict | None = None,
):
    algorithm = algorithm.lower()
    if algorithm == "adaboost":
        if task == "classification":
            params = dict(n_estimators=400, learning_rate=0.5, random_state=random_state)
            if model_params:
                params.update(model_params)
            return AdaBoostClassifier(**params)
        params = dict(n_estimators=500, learning_rate=0.3, loss="square", random_state=random_state)
        if model_params:
            params.update(model_params)
        return AdaBoostRegressor(**params)
    if algorithm == "gbm":
        if task == "classification":
            params = dict(random_state=random_state)
            if model_params:
                params.update(model_params)
            return GradientBoostingClassifier(**params)
        params = dict(random_state=random_state)
        if model_params:
            params.update(model_params)
        return GradientBoostingRegressor(**params)
    if algorithm == "xgboost":
        if task == "classification":
            params = dict(
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
            if model_params:
                params.update(model_params)
            return XGBClassifier(**params)
        params = dict(
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
        if model_params:
            params.update(model_params)
        return XGBRegressor(**params)
    if algorithm == "lightgbm":
        if task == "classification":
            params = dict(
                n_estimators=600,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
            )
            if num_classes is None or num_classes <= 2:
                params["objective"] = "binary"
            else:
                params["objective"] = "multiclass"
                params["num_class"] = num_classes
            if model_params:
                params.update(model_params)
            return LGBMClassifier(**params)
        params = dict(
            n_estimators=800,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="regression_l2",
            random_state=random_state,
            n_jobs=-1,
        )
        if model_params:
            params.update(model_params)
        return LGBMRegressor(**params)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_model_pipeline(
    X: pd.DataFrame,
    task: str,
    algorithm: str,
    random_state: int = 42,
    num_classes: int | None = None,
    model_params: Dict | None = None,
) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
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
    model = _build_model(task, algorithm, random_state, num_classes=num_classes, model_params=model_params)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def scoring(task: str) -> Dict[str, str]:
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
            "roc_auc": "roc_auc_ovr",
        }
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
