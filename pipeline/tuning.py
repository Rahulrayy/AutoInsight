import numpy as np
import optuna
import os
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from pipeline.config_loader import get as cfg

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", cfg("pipeline", "random_seed", 42)))
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", cfg("tuning", "n_trials", 40)))


def get_top_n_models(leaderboard, n=3):
    return [row["model"] for row in leaderboard[:n]]


def build_search_space(trial, model_name, task_type):
    is_clf = "classification" in task_type

    if model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        return RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1) if is_clf \
            else RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)

    elif model_name == "extra_trees":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        return ExtraTreesClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1) if is_clf \
            else ExtraTreesRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)

    elif model_name == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        return GradientBoostingClassifier(**params, random_state=RANDOM_SEED) if is_clf \
            else GradientBoostingRegressor(**params, random_state=RANDOM_SEED)

    elif model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        extra = {"eval_metric": "logloss", "verbosity": 0} if is_clf else {"verbosity": 0}
        return XGBClassifier(**params, **extra, random_state=RANDOM_SEED) if is_clf \
            else XGBRegressor(**params, **extra, random_state=RANDOM_SEED)

    elif model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        return LGBMClassifier(**params, random_state=RANDOM_SEED, verbose=-1) if is_clf \
            else LGBMRegressor(**params, random_state=RANDOM_SEED, verbose=-1)

    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
        return CatBoostClassifier(**params, random_seed=RANDOM_SEED, verbose=0) if is_clf \
            else CatBoostRegressor(**params, random_seed=RANDOM_SEED, verbose=0)

    elif model_name == "logistic_regression":
        params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True), "max_iter": 1000, "solver": "lbfgs"}
        return LogisticRegression(**params, random_state=RANDOM_SEED)

    else:
        raise ValueError(f"no search space defined for model: {model_name}")


def get_score(model, X_test, y_test, task_type):
    if "classification" in task_type:
        try:
            if task_type == "binary_classification":
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = model.predict_proba(X_test)
            return roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            preds = model.predict(X_test)
            return f1_score(y_test, preds, average="weighted", zero_division=0)
    else:
        preds = model.predict(X_test)
        return r2_score(y_test, preds)


def tune_single_model(model_name, task_type, X_train, y_train, X_test, y_test,
                      progress_callback=None):
    def objective(trial):
        model = build_search_space(trial, model_name, task_type)
        model.fit(X_train, y_train)
        score = get_score(model, X_test, y_test, task_type)
        if progress_callback:
            progress_callback(trial.number + 1, N_TRIALS)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    best_model = build_search_space(best_trial, model_name, task_type)
    best_model.fit(X_train, y_train)

    return best_model, study.best_value, best_trial.params, study


def tune_top_models(leaderboard, trained_models, task_type, X_train, y_train,
                    X_test, y_test, progress_callback=None):
    top_names = get_top_n_models(leaderboard, n=3)
    tuning_results = {}
    studies = {}

    for name in top_names:
        try:
            best_model, best_score, best_params, study = tune_single_model(
                name, task_type, X_train, y_train, X_test, y_test,
                progress_callback=progress_callback
            )
            from pipeline.automl import get_feature_importances
            feature_names = list(X_train.columns) if hasattr(X_train, "columns") \
                else [f"f{i}" for i in range(X_train.shape[1])]
            importances = get_feature_importances(best_model, feature_names)

            tuning_results[name] = {
                "model": best_model,
                "feature_importances": importances,
                "best_score": round(best_score, 4),
                "best_params": best_params,
                "original_name": name,
                "n_trials": N_TRIALS,
            }
            studies[name] = study
        except Exception as e:
            print(f"warning: tuning failed for {name}: {e}")
            continue

    return tuning_results, studies
