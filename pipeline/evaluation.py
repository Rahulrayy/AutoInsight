import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix,
)


def evaluate_classification(model, X_test, y_test, model_name, task_type):
    y_pred = model.predict(X_test)

    try:
        if task_type == "binary_classification":
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = round(roc_auc_score(y_test, y_proba), 4)
        else:
            y_proba = model.predict_proba(X_test)
            auc = round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4)
    except Exception:
        auc = None

    avg = "binary" if task_type == "binary_classification" else "weighted"
    labels = sorted(y_test.unique().tolist()) if hasattr(y_test, "unique") else sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    return {
        "model": model_name,
        "auc": auc,
        "weighted_f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": [str(l) for l in labels],
        "y_pred": y_pred.tolist(),
    }


def evaluate_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred.tolist(),
        "y_true": list(y_test),
    }


def build_leaderboard(trained_models, X_test, y_test, task_type):
    rows = []

    for name, entry in trained_models.items():
        model = entry["model"]
        cv_mean = entry.get("cv_mean")
        cv_std = entry.get("cv_std")

        try:
            if "classification" in task_type:
                result = evaluate_classification(model, X_test, y_test, name, task_type)
            else:
                result = evaluate_regression(model, X_test, y_test, name)

            if cv_mean is not None:
                result["cv_mean"] = cv_mean
                result["cv_std"] = cv_std

            rows.append(result)
        except Exception as e:
            print(f"warning: could not evaluate {name}: {e}")
            continue

    if not rows:
        raise RuntimeError("no models could be evaluated")

    if "classification" in task_type:
        rows.sort(key=lambda x: (x.get("auc") if x.get("auc") is not None else x.get("weighted_f1") or 0), reverse=True)
    else:
        rows.sort(key=lambda x: x.get("rmse") if x.get("rmse") is not None else float("inf"))

    return rows


def get_best_model(leaderboard, trained_models):
    best_name = leaderboard[0]["model"]
    best_model = trained_models[best_name]["model"]
    best_importances = trained_models[best_name]["feature_importances"]
    return best_name, best_model, best_importances


def run_evaluation(trained_models, X_test, y_test, task_type):
    leaderboard = build_leaderboard(trained_models, X_test, y_test, task_type)
    best_name, best_model, best_importances = get_best_model(leaderboard, trained_models)

    return {
        "leaderboard": leaderboard,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_feature_importances": best_importances,
        "task_type": task_type,
    }
