"""Train a classifier on the Breast Cancer Wisconsin dataset."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the breast cancer dataset as a pandas DataFrame."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    return X, y


def explore(X: pd.DataFrame, y: pd.Series) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples:    {X.shape[0]}")
    print(f"Features:   {X.shape[1]}")
    print(f"Classes:    {dict(y.value_counts())}")
    print(f"Missing:    {X.isnull().sum().sum()}")
    print(f"Feature names (first 5): {list(X.columns[:5])}")


def train_logistic(X_train, X_test, y_train, y_test) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    return {
        "name": "Logistic Regression",
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
    }


def train_forest(X_train, X_test, y_train, y_test) -> dict:
    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "name": "Random Forest",
        "model": model,
        "scaler": None,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
    }


def evaluate(result: dict, y_test) -> None:
    print()
    print("=" * 60)
    print(f"MODEL: {result['name']}")
    print("=" * 60)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ROC AUC:  {result['roc_auc']:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, result["y_pred"], target_names=["malignant", "benign"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, result["y_pred"]))


def main() -> None:
    X, y = load_data()
    explore(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    results = [
        train_logistic(X_train, X_test, y_train, y_test),
        train_forest(X_train, X_test, y_train, y_test),
    ]

    for r in results:
        evaluate(r, y_test)

    best = max(results, key=lambda r: r["roc_auc"])
    print()
    print("=" * 60)
    print(f"BEST MODEL: {best['name']} (ROC AUC {best['roc_auc']:.4f})")
    print("=" * 60)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best["model"], model_dir / "best_model.pkl")
    if best["scaler"] is not None:
        joblib.dump(best["scaler"], model_dir / "scaler.pkl")
    joblib.dump(list(X.columns), model_dir / "feature_names.pkl")
    print(f"Saved to {model_dir}/")


if __name__ == "__main__":
    main()
