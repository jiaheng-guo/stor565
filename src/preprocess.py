from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass(frozen=True)
class DatasetConfig:
    key: str
    pretty_name: str
    loader: Callable[[], Tuple[pd.DataFrame, pd.Series]]
    task: str


def _replace_question_marks(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert '?' placeholders to NaNs so imputers can handle them."""
    return frame.replace("?", np.nan)

###############################################################################
# --------------------------------------------------------------------------- #
# Classification Datasets
# --------------------------------------------------------------------------- #
###############################################################################

MAX_CLF_SAMPLE_SIZE = 10_000
MAX_REG_SAMPLE_SIZE = 10_000

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


def load_credit_card_fraud(sample_size: int = MAX_CLF_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
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


def load_imdb_reviews(
    sample_size: int = MAX_CLF_SAMPLE_SIZE,
    random_state: int = 42,
    max_features: int = 50_000
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the IMDB dataset and preprocess it using TF–IDF features.
    Returns:
        X_df : pd.DataFrame   (TF–IDF features)
        y    : pd.Series      (labels)
    """

    # -----------------------------
    # Load IMDB (train + test)
    # -----------------------------
    ds = load_dataset("stanfordnlp/imdb")
    df = pd.concat(
        [ds["train"].to_pandas()[["text", "label"]],
         ds["test"].to_pandas()[["text", "label"]]],
        ignore_index=True
    )

    # Optional subsampling
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    # Extract text + labels
    texts = df["text"].fillna("").tolist()
    y = pd.Series(df["label"].values, name="label")

    # -----------------------------
    # TF–IDF Vectorizer
    # -----------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        max_df=0.8,
        max_features=max_features,
        ngram_range=(1, 2),
        strip_accents="unicode",
    )

    X_sparse = vectorizer.fit_transform(texts)

    # -----------------------------
    # Convert sparse matrix → DataFrame
    # -----------------------------
    feature_names = vectorizer.get_feature_names_out()
    X_df = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=feature_names)

    return X_df, y

def load_mnist(sample_size: int = MAX_CLF_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="mnist_784", version=1, as_frame=False)
    X = pd.DataFrame(dataset.data)
    y = pd.Series(dataset.target.astype(str))
    if sample_size and sample_size < len(X):
        rs = np.random.RandomState(random_state)
        idx = rs.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    X_array = X.to_numpy(dtype=np.float32)
    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)
    if X_array.shape[0] < 2:
        raise ValueError("MNIST PCA requires at least two samples.")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=random_state)
    comps = pca.fit_transform(X_array)
    comp_cols = [f"pc_{i+1}" for i in range(comps.shape[1])]
    X_pca = pd.DataFrame(comps, columns=comp_cols)
    return X_pca.reset_index(drop=True), y.reset_index(drop=True)


def load_higgs(sample_size: int = MAX_CLF_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="higgs", version=1, as_frame=True)
    frame = dataset.frame.copy()
    target_col = dataset.details.get("default_target_attribute") if dataset.details else None
    if not target_col:
        target_candidates = [col for col in frame.columns if col.lower() in {"class", "target", "label"}]
        target_col = target_candidates[0] if target_candidates else frame.columns[-1]
    if sample_size and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)
    y = frame.pop(target_col).astype(str).str.strip()
    return frame.reset_index(drop=True), y.reset_index(drop=True)


CLASSIFICATION_DATASETS: List[DatasetConfig] = [
    DatasetConfig("adult", "Adult Income", load_adult_income, "classification"),
    DatasetConfig("heart", "Heart Disease", load_heart_disease, "classification"),
    DatasetConfig("mushroom", "Mushroom", load_mushrooms, "classification"),
    DatasetConfig("telco", "Telco Customer Churn", load_telco_churn, "classification"),
    DatasetConfig("breast_cancer", "Breast Cancer", load_breast_cancer_dataset, "classification"),
    DatasetConfig("credit", "Credit Card Fraud", load_credit_card_fraud, "classification"),
    DatasetConfig("imdb", "IMDB Movie Reviews", load_imdb_reviews, "classification"),
    DatasetConfig("mnist", "MNIST", load_mnist, "classification"),
    DatasetConfig("higgs", "HIGGS", load_higgs, "classification"),
]

###############################################################################
# --------------------------------------------------------------------------- #
# Regression Datasets
# --------------------------------------------------------------------------- #
###############################################################################

def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    target = frame.pop("MedHouseVal")
    return frame, target


def load_ames_house_prices(sample_size: int = MAX_REG_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="ames_housing", version=1, as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.astype(float)
    if sample_size and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state)
        y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_wine_quality(sample_size: int = MAX_REG_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="wine-quality-white", version=1, as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.astype(float)
    if sample_size and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state)
        y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_superconductivity(sample_size: int = MAX_REG_SAMPLE_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_openml(name="superconduct", version=1, as_frame=True)
    frame = dataset.frame.copy()
    target_col = frame.columns[-1]
    if sample_size and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_state)
    y = frame.pop(target_col).astype(float)
    return frame.reset_index(drop=True), y.reset_index(drop=True)


REGRESSION_DATASETS: List[DatasetConfig] = [
    DatasetConfig("california", "California Housing", load_california_housing, "regression"),
    DatasetConfig("ames", "Ames House Prices", load_ames_house_prices, "regression"),
    DatasetConfig("wine_quality", "Wine Quality", load_wine_quality, "regression"),
    DatasetConfig("superconduct", "Superconductivity", load_superconductivity, "regression"),
]
