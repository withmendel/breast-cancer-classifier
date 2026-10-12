"""Sanity tests for the ML pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_data_loads():
    data = load_breast_cancer()
    assert data.data.shape == (569, 30)
    assert set(data.target) == {0, 1}


def test_split_is_stratified():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    ratio_train = y_train.mean()
    ratio_test = y_test.mean()
    assert abs(ratio_train - ratio_test) < 0.05


def test_model_can_overfit_tiny_set():
    """A model should reach 100% on a tiny dataset it's allowed to memorize."""
    data = load_breast_cancer()
    X, y = data.data[:20], data.target[:20]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.transform(X), y)
    assert (model.predict(scaler.transform(X)) == y).mean() == 1.0


if __name__ == "__main__":
    test_data_loads()
    test_split_is_stratified()
    test_model_can_overfit_tiny_set()
    print("All tests passed.")
