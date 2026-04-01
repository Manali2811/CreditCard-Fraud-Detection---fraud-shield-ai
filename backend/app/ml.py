from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
META_PATH = MODEL_DIR / "metadata.joblib"


def _detect_label_column(df: pd.DataFrame) -> str:
    candidates = ["Class", "class", "label", "is_fraud", "fraud", "target"]
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError("Could not detect label column. Expected one of: Class, label, is_fraud, fraud, target.")


def _prepare_dataframe(dataset_id: str) -> Tuple[pd.DataFrame, str]:
    ds = load_dataset(dataset_id)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    label_col = _detect_label_column(df)
    return df, label_col


def train_model(
    dataset_id: str = "David-Egea/Creditcard-fraud-detection",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    df, label_col = _prepare_dataframe(dataset_id)

    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].astype(int).copy()

    # Keep numeric columns only for a robust MVP pipeline.
    X = X.select_dtypes(include=["number", "bool"])
    if X.empty:
        raise ValueError("No numeric features found in dataset.")

    feature_names: List[str] = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump({"feature_names": feature_names, "dataset_id": dataset_id}, META_PATH)

    return {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "features": len(feature_names),
        "label_column": label_col,
        "roc_auc": float(auc),
        "precision_fraud": float(report.get("1", {}).get("precision", 0.0)),
        "recall_fraud": float(report.get("1", {}).get("recall", 0.0)),
        "f1_fraud": float(report.get("1", {}).get("f1-score", 0.0)),
    }


def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)
    return model, metadata


def predict_one(model, metadata: Dict[str, object], features: Dict[str, float], threshold: float = 0.5) -> Dict[str, object]:
    feature_names = metadata["feature_names"]
    row = {name: features.get(name, np.nan) for name in feature_names}
    frame = pd.DataFrame([row], columns=feature_names)
    proba = float(model.predict_proba(frame)[0][1])
    return {
        "is_fraud": bool(proba >= threshold),
        "fraud_probability": proba,
        "threshold": threshold,
    }

