import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from pipeline.config_loader import get as cfg

RANDOM_SEED = int(os.getenv("RANDOM_SEED", cfg("pipeline", "random_seed", 42)))


def get_classification_models(class_weight=None):
    cw = class_weight  # None or "balanced"
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight=cw, random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=None,
            class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_SEED
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            class_weight=cw, random_state=RANDOM_SEED, verbose=-1
        ),
        "catboost": CatBoostClassifier(
            iterations=100, learning_rate=0.1, depth=6,
            random_seed=RANDOM_SEED, verbose=0,
            auto_class_weights="Balanced" if cw == "balanced" else None
        ),
    }


def get_regression_models():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_SEED
        ),
        "xgboost": XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_SEED, verbosity=0
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            random_state=RANDOM_SEED, verbose=-1
        ),
        "catboost": CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=6,
            random_seed=RANDOM_SEED, verbose=0
        ),
    }


def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        if model.coef_.ndim == 2:
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_).flatten()
    else:
        return {}
    importance_dict = dict(zip(feature_names, importances.tolist()))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def _get_cv_scoring(task_type):
    if task_type == "binary_classification":
        return "roc_auc"
    elif "multiclass" in task_type:
        return "roc_auc_ovr_weighted"
    else:
        return "r2"


def _train_one(name, model, X_train, y_train, feature_names, task_type,
               cat_feature_indices, cv_folds):
    """Train one model and optionally compute CV scores. Returns (name, entry)."""
    # pass categorical feature indices to CatBoost natively
    if "catboost" in name and cat_feature_indices:
        try:
            model.set_params(cat_features=cat_feature_indices)
        except Exception:
            pass

    # cross-validation before final fit (uses fresh internal copies)
    cv_mean, cv_std = None, None
    if cv_folds and cv_folds > 1:
        try:
            scoring = _get_cv_scoring(task_type)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                     scoring=scoring, n_jobs=1)
            cv_mean = round(float(scores.mean()), 4)
            cv_std = round(float(scores.std()), 4)
        except Exception:
            pass

    model.fit(X_train, y_train)
    importances = get_feature_importances(model, feature_names)
    return name, {
        "model": model,
        "feature_importances": importances,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }


def train_all_models(X_train, y_train, task_type, feature_names,
                     progress_callback=None, class_weight=None,
                     cat_feature_indices=None, cv_folds=3,
                     max_workers=4):
    """Train all models in parallel, with optional CV scores and class-weight support."""
    if "classification" in task_type:
        models = get_classification_models(class_weight=class_weight)
    else:
        models = get_regression_models()

    trained = {}
    total = len(models)
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        futures = {
            executor.submit(
                _train_one, name, model, X_train, y_train,
                feature_names, task_type, cat_feature_indices or [], cv_folds
            ): name
            for name, model in models.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, entry = future.result()
                with lock:
                    trained[name] = entry
            except Exception as e:
                print(f"warning: {model_name} failed to train: {e}")

            with lock:
                completed += 1
                done = completed

            if progress_callback:
                progress_callback(model_name, done, total)

    if not trained:
        raise RuntimeError("all models failed to train, check your data")

    return trained


def build_stacking_ensemble(tuned_results, task_type, X_train, y_train):
    """Build and fit a stacking ensemble from the tuned top models."""
    estimators = [
        (name.replace("_tuned", ""), entry["model"])
        for name, entry in tuned_results.items()
    ]
    if len(estimators) < 2:
        return None

    if "classification" in task_type:
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        stack = StackingClassifier(estimators=estimators, final_estimator=meta,
                                   passthrough=False, cv=3, n_jobs=1)
    else:
        meta = LinearRegression()
        stack = StackingRegressor(estimators=estimators, final_estimator=meta,
                                  passthrough=False, cv=3, n_jobs=1)

    try:
        stack.fit(X_train, y_train)
        return stack
    except Exception as e:
        print(f"warning: stacking ensemble failed: {e}")
        return None
