import shap
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence


def get_explainer(model, X_train):
    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["forest", "tree", "boost", "catboost", "xgb", "lgbm", "gradient", "stacking"]):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass
    if "logistic" in model_type or "linear" in model_type:
        return shap.LinearExplainer(model, X_train)
    sample = shap.sample(X_train, 100)
    return shap.KernelExplainer(model.predict_proba, sample)


def compute_shap_values(model, X_train, X_test, task_type, max_samples=500):
    X_explain = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    explainer = get_explainer(model, X_train)

    try:
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        shap_explanation = explainer(X_explain)
        shap_values = shap_explanation.values

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

    return shap_values, X_explain


def get_mean_abs_shap(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True))


def compute_shap_for_instance(model, X_train, instance_df, task_type):
    """Compute SHAP values for a single instance. Returns (shap_vals, base_value, feature_names)."""
    explainer = get_explainer(model, X_train)
    try:
        shap_vals = explainer.shap_values(instance_df)
    except Exception:
        exp = explainer(instance_df)
        shap_vals = exp.values

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    # flatten to 1D for a single instance
    vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

    try:
        base = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) \
            else float(explainer.expected_value)
    except Exception:
        base = 0.0

    return vals, base, list(instance_df.columns)


def compute_dependence_data(shap_values, X_explain, feature_names, feature):
    """Return arrays for a SHAP dependence plot of the given feature."""
    if feature not in feature_names:
        return None, None, None
    idx = feature_names.index(feature)
    shap_col = shap_values[:, idx]
    feat_vals = X_explain.iloc[:, idx].values
    # use top interaction feature (highest absolute SHAP correlation)
    interactions = []
    for i, f in enumerate(feature_names):
        if i == idx:
            continue
        other_shap = shap_values[:, i]
        corr = np.corrcoef(shap_col, other_shap)[0, 1]
        interactions.append((f, abs(corr)))
    interactions.sort(key=lambda x: x[1], reverse=True)
    color_feature = interactions[0][0] if interactions else None
    color_vals = X_explain[color_feature].values if color_feature else None
    return feat_vals, shap_col, color_vals


def compute_pdp(model, X_train, feature_names, feature, task_type, n_points=50):
    """Compute Partial Dependence for a single feature."""
    if feature not in feature_names:
        return None, None
    feat_idx = feature_names.index(feature)
    try:
        kind = "average"
        pdp_result = partial_dependence(model, X_train, [feat_idx],
                                        kind=kind, grid_resolution=n_points)
        grid = pdp_result["grid_values"][0]
        avg = pdp_result["average"][0]
        return grid, avg
    except Exception:
        return None, None


def run_shap_for_top3(leaderboard, trained_models, tuned_results, X_train, X_test,
                      task_type, top_n=3):
    candidates = []
    for row in leaderboard[:top_n]:
        name = row["model"]
        tuned_key = name  # tuned_results uses original model name as key
        if tuned_results and tuned_key in tuned_results:
            model = tuned_results[tuned_key]["model"]
            label = f"{name} (tuned)"
        elif name in trained_models:
            model = trained_models[name]["model"]
            label = name
        else:
            continue
        candidates.append((name, label, model))

    feature_names = list(X_train.columns)
    results = {}

    for name, label, model in candidates:
        try:
            shap_values, X_explain = compute_shap_values(model, X_train, X_test, task_type)
            mean_abs = get_mean_abs_shap(shap_values, feature_names)
            results[name] = {
                "label": label,
                "shap_values": shap_values,
                "X_explain": X_explain,
                "feature_names": feature_names,
                "mean_abs_shap": mean_abs,
            }
        except Exception as e:
            print(f"warning: shap failed for {name}: {e}")
            continue

    return results
