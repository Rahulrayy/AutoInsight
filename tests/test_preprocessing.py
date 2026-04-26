import pandas as pd
import numpy as np
import pytest
from pipeline.preprocessing import (
    detect_skew,
    impute_missing,
    encode_categoricals,
    scale_numericals,
    detect_task_type,
    engineer_datetime_features,
    cap_outliers,
    target_encode,
    apply_feature_selection,
    run_preprocessing,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "salary": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
    })


@pytest.fixture
def sample_profile():
    return {
        "column_types": {
            "age": "numerical",
            "salary": "numerical",
            "gender": "categorical",
            "target": "categorical",
        },
        "near_constant_features": [],
        "missing_values": {},
    }


# ── existing tests ─────────────────────────────────────────────────────────────

def test_detect_skew_symmetric():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert detect_skew(series) == False


def test_detect_skew_skewed():
    series = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1000])
    assert detect_skew(series) == True


def test_impute_missing_numerical_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0]})
    result = impute_missing(df, {"a": "numerical"})
    assert result["a"].isnull().sum() == 0


def test_impute_missing_categorical_mode():
    df = pd.DataFrame({"b": ["cat", "cat", None, "dog", "cat"]})
    result = impute_missing(df, {"b": "categorical"})
    assert result["b"].isnull().sum() == 0
    assert result["b"].iloc[2] == "cat"


def test_encode_categoricals_low_cardinality():
    df = pd.DataFrame({"gender": ["M", "F", "M", "F", "M"]})
    result, encoders = encode_categoricals(df, {"gender": "categorical"})
    assert "gender" not in result.columns
    assert result.shape[1] >= 2


def test_encode_categoricals_high_cardinality():
    df = pd.DataFrame({"city": [f"city_{i}" for i in range(15)]})
    result, encoders = encode_categoricals(df, {"city": "categorical"})
    assert "city" in result.columns
    assert "city" in encoders


def test_scale_numericals():
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0, 60.0]})
    result, scaler = scale_numericals(df, {"age": "numerical"})
    assert abs(result["age"].mean()) < 1e-9


def test_detect_task_type_binary():
    df = pd.DataFrame({"target": ["yes", "no", "yes", "no"]})
    assert detect_task_type(df, "target") == "binary_classification"


def test_detect_task_type_regression():
    df = pd.DataFrame({"target": list(range(100))})
    assert detect_task_type(df, "target") == "regression"


def test_run_preprocessing_output_shapes(sample_df, sample_profile):
    X_train, X_test, y_train, y_test, artifacts = run_preprocessing(sample_df, "target", sample_profile)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_run_preprocessing_no_leakage(sample_df, sample_profile):
    _, _, _, _, artifacts = run_preprocessing(sample_df, "target", sample_profile)
    assert artifacts["scaler"] is not None
    assert "task_type" in artifacts


# ── new feature tests ──────────────────────────────────────────────────────────

def test_engineer_datetime_features_extracts_components():
    df = pd.DataFrame({"ts": ["2023-01-15 08:30:00", "2023-06-20 14:45:00"]})
    result, processed = engineer_datetime_features(df, {"ts": "datetime"})
    assert "ts" not in result.columns
    assert "ts_year" in result.columns
    assert "ts_month" in result.columns
    assert "ts_dayofweek" in result.columns
    assert "ts" in processed


def test_engineer_datetime_ignores_non_datetime():
    df = pd.DataFrame({"age": [25, 30], "name": ["a", "b"]})
    result, processed = engineer_datetime_features(df, {"age": "numerical", "name": "categorical"})
    assert list(result.columns) == ["age", "name"]
    assert processed == []


def test_cap_outliers_clips_extremes():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    X_test = pd.DataFrame({"a": [2.0, 999.0]})
    train_out, test_out, bounds = cap_outliers(X_train, X_test, iqr_factor=1.5)
    assert train_out["a"].max() < 1000.0
    assert test_out["a"].max() < 999.0
    assert "a" in bounds


def test_cap_outliers_preserves_normal_values():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"a": [2.0, 3.0]})
    train_out, test_out, bounds = cap_outliers(X_train, X_test)
    # no extreme outliers, values unchanged
    assert train_out["a"].tolist() == X_train["a"].tolist()


def test_target_encode_replaces_with_mean():
    X_train = pd.DataFrame({"cat": [0, 0, 1, 1, 2]})
    X_test = pd.DataFrame({"cat": [0, 1]})
    y_train = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])
    train_out, test_out, encoders = target_encode(X_train, X_test, y_train, ["cat"])
    # cat=0 → mean target = 0.5, cat=1 → 1.0, cat=2 → 0.0
    assert abs(train_out["cat"].iloc[0] - 0.5) < 1e-9
    assert abs(test_out["cat"].iloc[1] - 1.0) < 1e-9
    assert "cat" in encoders


def test_target_encode_unseen_gets_global_mean():
    X_train = pd.DataFrame({"cat": [0, 1]})
    X_test = pd.DataFrame({"cat": [99]})  # unseen category
    y_train = pd.Series([1.0, 0.0])
    _, test_out, _ = target_encode(X_train, X_test, y_train, ["cat"])
    assert abs(test_out["cat"].iloc[0] - 0.5) < 1e-9  # global mean = 0.5


def test_feature_selection_reduces_columns():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 50), columns=[f"f{i}" for i in range(50)])
    y = pd.Series(np.random.randint(0, 2, 100))
    X_train, X_test_dummy = X[:80], X[80:]
    train_sel, test_sel, selector, selected = apply_feature_selection(
        X_train, X_test_dummy, y[:80], "binary_classification", k=10
    )
    assert train_sel.shape[1] == 10
    assert test_sel.shape[1] == 10
    assert len(selected) == 10


def test_feature_selection_noop_when_few_features(sample_df, sample_profile):
    X_train, X_test, y_train, _, artifacts = run_preprocessing(sample_df, "target", sample_profile)
    # sample has fewer features than default k=30, so selector should be None
    assert artifacts["feature_selector"] is None


def test_scale_numericals_fit_then_transform():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [6.0, 7.0]})
    train_scaled, scaler = scale_numericals(train, {"a": "numerical"})
    test_scaled, _ = scale_numericals(test, {"a": "numerical"}, fit_scaler=scaler)
    # test values should be consistently scaled relative to train distribution
    assert test_scaled["a"].iloc[0] > train_scaled["a"].iloc[-1]
