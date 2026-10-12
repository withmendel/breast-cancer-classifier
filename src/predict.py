"""Load the saved model and predict on new samples."""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer


def load_artifacts():
    model_dir = Path("models")
    model = joblib.load(model_dir / "best_model.pkl")
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    feature_names = joblib.load(model_dir / "feature_names.pkl")
    return model, scaler, feature_names


def predict(sample: np.ndarray, model, scaler, feature_names) -> dict:
    if scaler is not None:
        sample = scaler.transform(sample.reshape(1, -1))
    else:
        sample = sample.reshape(1, -1)

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]

    label = "benign" if pred == 1 else "malignant"
    confidence = float(prob[pred])
    return {"label": label, "confidence": confidence}


def main() -> None:
    model, scaler, feature_names = load_artifacts()

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Demo: predict on the first 10 test samples
    print("Predicting on first 10 samples of the dataset:")
    print("-" * 60)
    for i in range(10):
        result = predict(X[i], model, scaler, feature_names)
        truth = "benign" if y[i] == 1 else "malignant"
        match = "OK" if result["label"] == truth else "WRONG"
        print(
            f"  sample {i:2d}  predicted={result['label']:9s}  "
            f"confidence={result['confidence']:.3f}  truth={truth:9s}  {match}"
        )


if __name__ == "__main__":
    main()
