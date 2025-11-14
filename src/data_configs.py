from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_openml, load_breast_cancer


@dataclass(frozen=True)
class DatasetConfig:
    """Container describing a benchmark dataset."""

    key: str
    pretty_name: str
    loader: Callable[[], Tuple[pd.DataFrame, pd.Series]]
    notes: str = ""


def _replace_question_marks(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert '?' placeholders to NaNs so imputers can handle them."""
    return frame.replace("?", np.nan)


def load_adult_income() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    X = _replace_question_marks(dataset.data.copy())
    y = dataset.target.str.strip()
    return X, y


def load_heart_disease() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="heart-disease", version=1, as_frame=True)
    frame = dataset.frame.copy()
    target_col = dataset.details.get("default_target_attribute") if dataset.details else None
    if not target_col:
        target_candidates = [col for col in frame.columns if col.lower() in {"target", "class", "label"}]
        target_col = target_candidates[0] if target_candidates else frame.columns[-1]
    y = frame.pop(target_col)
    X = _replace_question_marks(frame)
    y = y.astype(str).str.lower()
    return X, y


def load_mushrooms() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="mushroom", version=1, as_frame=True)
    X = _replace_question_marks(dataset.data.copy())
    y = dataset.target.astype(str).str.lower()
    return X, y


def load_telco_churn() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(data_id=42178, as_frame=True)
    frame = dataset.frame.copy()
    frame.columns = [col.strip() for col in frame.columns]
    target = dataset.target
    if target is not None and not target.empty:
        y = target.astype(str).str.strip().str.lower()
        target_name = target.name
        if target_name in frame.columns:
            frame = frame.drop(columns=[target_name])
    else:
        target_candidates = [col for col in frame.columns if col.lower() in {"churn", "target", "class", "label"}]
        target_name = target_candidates[0] if target_candidates else frame.columns[-1]
        y = frame.pop(target_name).astype(str).str.strip().str.lower()
    if "customerID" in frame.columns:
        frame = frame.drop(columns=["customerID"])
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].replace(" ", np.nan), errors="coerce")
    X = _replace_question_marks(frame)
    return X, y


def load_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    y = frame.pop("target").map({0: "malignant", 1: "benign"})
    return frame, y


def load_credit_card_fraud(sample_size: int = 80_000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="creditcard", version=1, as_frame=True)
    frame = dataset.frame.copy()
    frame["Class"] = frame["Class"].astype(int)
    if sample_size and sample_size < len(frame):
        frac = sample_size / len(frame)
        samples = []
        for label, group in frame.groupby("Class"):
            n = max(1, int(round(len(group) * frac)))
            samples.append(group.sample(n=min(len(group), n), random_state=random_state))
        frame = pd.concat(samples).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y = frame.pop("Class").map({0: "legitimate", 1: "fraud"})
    return frame, y


def load_imdb_reviews(sample_size: int = 20_000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_dataset("imdb")
    frame = pd.concat(
        [
            dataset["train"].to_pandas()[["text", "label"]],
            dataset["test"].to_pandas()[["text", "label"]],
        ],
        ignore_index=True,
    )
    if sample_size and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)

    text = frame["text"].fillna("")
    letters_only = text.str.replace(r"[^A-Za-z]+", " ", regex=True).str.strip()
    word_counts = letters_only.str.split().str.len().fillna(0)
    char_lengths = text.str.len()
    avg_word_length = (letters_only.str.len() / word_counts.clip(lower=1)).fillna(0)
    upper_ratio = (
        text.str.replace(r"[^A-Z]", "", regex=True).str.len() / char_lengths.replace(0, np.nan)
    ).fillna(0)

    features = pd.DataFrame(
        {
            "char_length": char_lengths,
            "word_count": word_counts,
            "avg_word_length": avg_word_length,
            "exclamation_count": text.str.count("!"),
            "question_count": text.str.count(r"\?"),
            "upper_ratio": upper_ratio,
        }
    )
    y = frame["label"].map({0: "negative", 1: "positive"})
    return features.reset_index(drop=True), y.reset_index(drop=True)


CLASSIFICATION_DATASETS: List[DatasetConfig] = [
    DatasetConfig("adult", "Adult Income", load_adult_income, "Predict >50K annual income."),
    DatasetConfig("heart", "Heart Disease", load_heart_disease, "Binary heart disease diagnosis."),
    DatasetConfig("mushroom", "Mushroom Classification", load_mushrooms, "Edible vs poisonous mushrooms."),
    DatasetConfig("telco", "Telco Customer Churn", load_telco_churn, "Telecommunications churn prediction."),
    DatasetConfig("breast_cancer", "Breast Cancer", load_breast_cancer_dataset, "Wisconsin diagnostic dataset."),
    DatasetConfig("credit", "Credit Card Fraud", load_credit_card_fraud, "Highly imbalanced fraud detection."),
    DatasetConfig("imdb", "IMDB Movie Reviews", load_imdb_reviews, "Sentiment features engineered from reviews."),
]
