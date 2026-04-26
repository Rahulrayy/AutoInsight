"""
End-to-end integration test: runs the full pipeline on a small synthetic dataset
and verifies that each stage produces the expected outputs.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from pipeline.profiling import profile_dataset
from pipeline.preprocessing import run_preprocessing
from pipeline.automl import train_all_models
from pipeline.evaluation import run_evaluation
from pipeline.shap_analysis import run_shap_for_top3


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    X, y = make_classification(n_samples=150, n_features=8, n_informative=5,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


@pytest.fixture
def reg_df():
    X, y = make_regression(n_samples=150, n_features=8, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


# ── classification pipeline ────────────────────────────────────────────────────

class TestClassificationPipeline:
    def test_profile_runs(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        assert profile["rows"] == 150
        assert "column_types" in profile
        assert "missing_values" in profile

    def test_preprocessing_runs(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        assert len(X_train) + len(X_test) == 150
        assert artifacts["task_type"] in ("binary_classification", "multiclass_classification")
        assert "scaler" in artifacts
        assert "feature_names" in artifacts

    def test_training_runs(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        assert len(trained) == 7
        for name, entry in trained.items():
            assert "model" in entry
            assert "feature_importances" in entry

    def test_evaluation_runs(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])
        assert "leaderboard" in results
        assert "best_model_name" in results
        lb = results["leaderboard"]
        assert len(lb) == 7
        # leaderboard should be sorted best first
        aucs = [row.get("auc") for row in lb if row.get("auc") is not None]
        if aucs:
            assert aucs == sorted(aucs, reverse=True)

    def test_shap_runs(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])
        shap_results = run_shap_for_top3(results["leaderboard"], trained, None,
                                         X_train, X_test, artifacts["task_type"])
        assert len(shap_results) >= 1
        for name, data in shap_results.items():
            assert "shap_values" in data
            assert "mean_abs_shap" in data
            assert len(data["mean_abs_shap"]) > 0

    def test_leaderboard_has_cv_scores(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=3)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])
        # at least some models should have CV scores
        has_cv = any(row.get("cv_mean") is not None for row in results["leaderboard"])
        assert has_cv

    def test_confusion_matrix_in_leaderboard(self, clf_df):
        profile = profile_dataset(clf_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(clf_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])
        for row in results["leaderboard"]:
            assert "confusion_matrix" in row
            assert "confusion_matrix_labels" in row


# ── regression pipeline ────────────────────────────────────────────────────────

class TestRegressionPipeline:
    def test_full_regression_pipeline(self, reg_df):
        profile = profile_dataset(reg_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(reg_df, "target", profile)
        assert artifacts["task_type"] == "regression"

        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])

        lb = results["leaderboard"]
        # regression leaderboard sorted by rmse ascending
        rmses = [row["rmse"] for row in lb]
        assert rmses == sorted(rmses)

        # predictions stored for scatter plot
        best_row = lb[0]
        assert "y_pred" in best_row
        assert "y_true" in best_row

    def test_regression_shap_runs(self, reg_df):
        profile = profile_dataset(reg_df, "target")
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(reg_df, "target", profile)
        trained = train_all_models(X_train, y_train, artifacts["task_type"],
                                   artifacts["feature_names"], cv_folds=0)
        results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])
        shap_results = run_shap_for_top3(results["leaderboard"], trained, None,
                                         X_train, X_test, artifacts["task_type"])
        assert len(shap_results) >= 1


# ── preprocessing robustness ───────────────────────────────────────────────────

class TestPreprocessingRobustness:
    def test_datetime_columns_are_engineered(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50, freq="D").astype(str),
            "value": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        })
        # explicitly mark the date column as datetime so the test is independent
        # of profiling detection (which can vary by pandas version)
        profile = {
            "column_types": {"date": "datetime", "value": "numerical", "target": "numerical"},
            "near_constant_features": [],
            "missing_values": {},
        }
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(df, "target", profile)
        feat_names = artifacts["feature_names"]
        has_dt_feat = any("_year" in f or "_month" in f for f in feat_names)
        assert has_dt_feat

    def test_high_cardinality_target_encoded(self):
        n = 100
        cats = [f"cat_{i % 20}" for i in range(n)]  # 20 unique > threshold
        df = pd.DataFrame({
            "high_card": cats,
            "num": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        })
        profile = {
            "column_types": {"high_card": "categorical", "num": "numerical", "target": "categorical"},
            "near_constant_features": [],
            "missing_values": {},
        }
        X_train, X_test, _, _, artifacts = run_preprocessing(df, "target", profile)
        # high_card should survive (as target-encoded float)
        assert "high_card" in artifacts["feature_names"] or any(
            "high_card" in f for f in artifacts["feature_names"]
        )
