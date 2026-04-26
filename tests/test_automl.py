import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from pipeline.automl import (
    train_all_models, get_feature_importances,
    get_classification_models, get_regression_models,
    build_stacking_ensemble,
)


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y)
    return X_df, y_series


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y)
    return X_df, y_series


def test_classification_model_count():
    models = get_classification_models()
    expected = {"logistic_regression", "random_forest", "extra_trees",
                "gradient_boosting", "xgboost", "lightgbm", "catboost"}
    assert set(models.keys()) == expected


def test_regression_model_count():
    models = get_regression_models()
    expected = {"linear_regression", "random_forest", "extra_trees",
                "gradient_boosting", "xgboost", "lightgbm", "catboost"}
    assert set(models.keys()) == expected


def test_train_classification_returns_all_models(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=0)
    assert len(trained) == 7
    expected = {"logistic_regression", "random_forest", "extra_trees",
                "gradient_boosting", "xgboost", "lightgbm", "catboost"}
    assert set(trained.keys()) == expected


def test_train_regression_returns_all_models(regression_data):
    X, y = regression_data
    trained = train_all_models(X, y, "regression", list(X.columns), cv_folds=0)
    assert len(trained) == 7
    expected = {"linear_regression", "random_forest", "extra_trees",
                "gradient_boosting", "xgboost", "lightgbm", "catboost"}
    assert set(trained.keys()) == expected


def test_trained_models_have_importances(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=0)
    for name, entry in trained.items():
        assert "feature_importances" in entry
        assert isinstance(entry["feature_importances"], dict)


def test_feature_importances_sorted_descending(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=0)
    rf_importances = trained["random_forest"]["feature_importances"]
    scores = list(rf_importances.values())
    assert scores == sorted(scores, reverse=True)


def test_trained_models_can_predict(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=0)
    for name, entry in trained.items():
        preds = entry["model"].predict(X)
        assert len(preds) == len(y)


def test_cv_scores_populated(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=3)
    for name, entry in trained.items():
        assert "cv_mean" in entry
        assert "cv_std" in entry
        if entry["cv_mean"] is not None:
            assert 0.0 <= entry["cv_mean"] <= 1.0


def test_class_weight_balanced(classification_data):
    X, y = classification_data
    # should not crash; class_weight is forwarded to supporting models
    trained = train_all_models(X, y, "binary_classification", list(X.columns),
                               class_weight="balanced", cv_folds=0)
    assert len(trained) == 7


def test_progress_callback_called(classification_data):
    X, y = classification_data
    calls = []
    def cb(name, done, total):
        calls.append((name, done, total))

    trained = train_all_models(X, y, "binary_classification", list(X.columns),
                               progress_callback=cb, cv_folds=0)
    assert len(calls) == len(trained)
    assert all(total == 7 for _, _, total in calls)


def test_stacking_ensemble_classif(classification_data):
    X, y = classification_data
    trained = train_all_models(X, y, "binary_classification", list(X.columns), cv_folds=0)
    top3 = {name: {"model": entry["model"]} for name, entry in list(trained.items())[:3]}
    stack = build_stacking_ensemble(top3, "binary_classification", X, y)
    assert stack is not None
    preds = stack.predict(X)
    assert len(preds) == len(y)
