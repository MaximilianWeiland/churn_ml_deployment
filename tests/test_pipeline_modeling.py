import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.model.train_evaluate import train_evaluate_model


@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "age": rng.integers(20, 70, n).astype(float),
        "watch_hours": rng.uniform(0, 20, n),
        "last_login_days": rng.integers(1, 60, n).astype(float),
        "number_of_profiles": rng.integers(1, 5, n).astype(float),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)


MINIMAL_PARAMS = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1}


def test_train_evaluate_returns_xgboost_model(dataset, monkeypatch):
    import mlflow, mlflow.sklearn
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **kw: None)

    X_train, X_test, y_train, y_test = dataset
    model = train_evaluate_model(X_train, y_train, X_test, y_test, MINIMAL_PARAMS, threshold=0.4)
    assert isinstance(model, XGBClassifier)


def test_train_evaluate_model_predicts_correct_shape(dataset, monkeypatch):
    import mlflow, mlflow.sklearn
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **kw: None)

    X_train, X_test, y_train, y_test = dataset
    model = train_evaluate_model(X_train, y_train, X_test, y_test, MINIMAL_PARAMS, threshold=0.4)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_train_evaluate_model_predicts_binary(dataset, monkeypatch):
    import mlflow, mlflow.sklearn
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **kw: None)

    X_train, X_test, y_train, y_test = dataset
    model = train_evaluate_model(X_train, y_train, X_test, y_test, MINIMAL_PARAMS, threshold=0.4)
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1})
