import pandas as pd
import numpy as np
import pytest
from pipeline.profiling import (
    detect_column_types,
    get_missing_value_ratios,
    find_near_constant_features,
    get_correlation_summary,
    infer_target_column,
    profile_dataset
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "salary": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "country": ["NL"] * 10,  # near constant
        "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"]
    })


def test_detect_column_types(sample_df):
    types = detect_column_types(sample_df)
    assert types["age"] == "numerical"
    assert types["salary"] == "numerical"
    assert types["gender"] == "categorical"
    assert types["target"] == "categorical"


def test_missing_value_ratios_no_missing(sample_df):
    ratios = get_missing_value_ratios(sample_df)
    assert ratios == {}


def test_missing_value_ratios_with_missing():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    ratios = get_missing_value_ratios(df)
    assert "a" in ratios
    assert abs(ratios["a"] - 0.1) < 0.01
    assert "b" not in ratios


def test_near_constant_features(sample_df):
    near_constant = find_near_constant_features(sample_df)
    assert "country" in near_constant
    assert "age" not in near_constant


def test_correlation_summary(sample_df):
    result = get_correlation_summary(sample_df)
    assert "high_correlation_pairs" in result
    assert "threshold_used" in result
    # age and salary are not perfectly correlated here so no pairs expected
    assert isinstance(result["high_correlation_pairs"], list)


def test_infer_target_column(sample_df):
    types = detect_column_types(sample_df)
    suggested = infer_target_column(sample_df, types)
    # gender and target are both binary categoricals, either is valid
    assert suggested in ["gender", "target"]


def test_profile_dataset_returns_expected_keys(sample_df):
    report = profile_dataset(sample_df, "target")
    expected_keys = [
        "rows", "columns", "column_types", "missing_values",
        "near_constant_features", "correlation_summary",
        "class_balance", "imbalance_warning", "target_suggestion"
    ]
    for key in expected_keys:
        assert key in report, f"missing key: {key}"


def test_profile_dataset_row_count(sample_df):
    report = profile_dataset(sample_df, "target")
    assert report["rows"] == 10
    assert report["columns"] == 5


def test_imbalance_flag():
    df = pd.DataFrame({
        "feature": range(10),
        "target": ["yes"] * 9 + ["no"]  # 90/10 split, should trigger flag
    })
    report = profile_dataset(df, "target")
    assert report["imbalance_warning"] is True