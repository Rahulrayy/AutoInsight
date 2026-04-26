import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
import os

from pipeline.config_loader import get as cfg

RANDOM_SEED = int(os.getenv("RANDOM_SEED", cfg("pipeline", "random_seed", 42)))
TEST_SIZE = float(os.getenv("DEFAULT_TEST_SIZE", cfg("pipeline", "test_size", 0.2)))
HIGH_CARDINALITY_THRESHOLD = cfg("preprocessing", "high_cardinality_threshold", 10)
SKEW_THRESHOLD = cfg("preprocessing", "skew_threshold", 1.0)
OUTLIER_IQR_FACTOR = cfg("preprocessing", "outlier_iqr_factor", 3.0)
FEATURE_SELECTION_K = cfg("preprocessing", "feature_selection_k", 30)


def drop_useless_columns(df, near_constant_cols):
    cols_to_drop = [c for c in near_constant_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def engineer_datetime_features(X, column_types):
    """Extract year/month/day/hour/dayofweek from datetime columns instead of dropping them."""
    processed = []
    X = X.copy()
    for col, ctype in list(column_types.items()):
        if ctype != "datetime" or col not in X.columns:
            continue
        try:
            dt = pd.to_datetime(X[col], errors="coerce")
            X = X.drop(columns=[col])
            X[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
            X[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
            X[f"{col}_day"] = dt.dt.day.fillna(0).astype(int)
            X[f"{col}_hour"] = dt.dt.hour.fillna(0).astype(int)
            X[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
            processed.append(col)
        except Exception:
            X = X.drop(columns=[col], errors="ignore")
    return X, processed


def detect_skew(series):
    return abs(series.skew()) > SKEW_THRESHOLD


def impute_missing(df, column_types):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        col_type = column_types.get(col, "categorical")
        if col_type == "numerical":
            if detect_skew(df[col].dropna()):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "unknown")
    return df


def cap_outliers(X_train, X_test, iqr_factor=None):
    """IQR-based outlier capping fitted on training data, applied to both sets."""
    factor = iqr_factor or OUTLIER_IQR_FACTOR
    X_train = X_train.copy()
    X_test = X_test.copy()
    bounds = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lo = Q1 - factor * IQR
        hi = Q3 + factor * IQR
        bounds[col] = (lo, hi)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)
    return X_train, X_test, bounds


def encode_categoricals(df, column_types, fit_encoders=None):
    """Backward-compatible encoding: one-hot for low cardinality, label for high."""
    df = df.copy()
    encoders = fit_encoders or {}
    new_encoders = {}

    for col in list(df.columns):
        if column_types.get(col) != "categorical":
            continue
        n_unique = df[col].nunique()
        if n_unique <= HIGH_CARDINALITY_THRESHOLD:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                new_encoders[col] = le
    return df, {**encoders, **new_encoders}


def scale_numericals(df, column_types, fit_scaler=None):
    """Scale numerical columns. Backward-compatible signature."""
    df = df.copy()
    numeric_cols = [c for c in df.columns if column_types.get(c) == "numerical" and c in df.columns]
    if not numeric_cols:
        return df, fit_scaler

    if fit_scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        scaler = fit_scaler
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


def apply_feature_selection(X_train, X_test, y_train, task_type, k=None):
    """Select top-k features by mutual information. No-op when n_features <= k."""
    k = k or FEATURE_SELECTION_K
    if X_train.shape[1] <= k:
        return X_train, X_test, None, list(X_train.columns)

    score_fn = mutual_info_classif if "classification" in task_type else mutual_info_regression
    selector = SelectKBest(score_fn, k=k)
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    selected = [col for col, sel in zip(X_train.columns, mask) if sel]
    return X_train[selected], X_test[selected], selector, selected


def target_encode(X_train, X_test, y_train, high_card_cols):
    """Replace label-encoded high-cardinality columns with target-mean encoding.

    Fitted on training data only to avoid leakage.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    encoders = {}
    global_mean = float(np.mean(y_train))

    for col in high_card_cols:
        if col not in X_train.columns:
            continue
        temp = pd.DataFrame({"cat": X_train[col].values, "target": y_train.values})
        mapping = temp.groupby("cat")["target"].mean().to_dict()
        encoders[col] = {"mapping": mapping, "global_mean": global_mean}
        X_train[col] = X_train[col].map(mapping).fillna(global_mean)
        X_test[col] = X_test[col].map(mapping).fillna(global_mean)

    return X_train, X_test, encoders


def detect_task_type(df, target_col):
    series = df[target_col]
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary_classification"
    elif n_unique <= 20 and series.dtype == object:
        return "multiclass_classification"
    elif pd.api.types.is_numeric_dtype(series) and n_unique > 20:
        return "regression"
    else:
        return "multiclass_classification"


def run_preprocessing(df, target_col, profile_report):
    near_constant = profile_report.get("near_constant_features", [])
    column_types = profile_report.get("column_types", {})

    df = drop_useless_columns(df, near_constant)
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # engineer datetime features instead of dropping them
    X, datetime_cols_processed = engineer_datetime_features(X, column_types)

    task_type = detect_task_type(df, target_col)

    target_encoder = None
    if "classification" in task_type:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        target_encoder = le

    X = impute_missing(X, column_types)

    # identify high-cardinality categorical columns before encoding
    high_card_cols = [
        col for col in X.columns
        if column_types.get(col) == "categorical"
        and X[col].nunique() > HIGH_CARDINALITY_THRESHOLD
    ]

    # one-hot for low-card, label for high-card (label encoding has no target leakage)
    X, label_encoders = encode_categoricals(X, column_types)

    # all columns are now numerical after encoding
    updated_types = {col: "numerical" for col in X.columns}

    # split BEFORE any target-informed transformations
    stratify = y if "classification" in task_type else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify
    )

    # target encode high-card cols using training labels only (no leakage)
    surviving_high_card = [col for col in high_card_cols if col in X_train.columns]
    X_train, X_test, target_encoders = target_encode(X_train, X_test, y_train, surviving_high_card)

    # cap outliers (bounds fitted on train, applied to both)
    X_train, X_test, outlier_bounds = cap_outliers(X_train, X_test)

    # scale (fitted on train, applied to both)
    X_train, scaler = scale_numericals(X_train, updated_types)
    X_test, _ = scale_numericals(X_test, updated_types, fit_scaler=scaler)

    # feature selection (fitted on train)
    X_train, X_test, selector, selected_features = apply_feature_selection(
        X_train, X_test, y_train, task_type
    )

    # identify surviving categorical-origin feature indices for CatBoost
    final_cols = list(X_train.columns)
    cat_feature_indices = [
        final_cols.index(col)
        for col in surviving_high_card
        if col in final_cols
    ]

    artifacts = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "target_encoders": target_encoders,
        "target_encoder": target_encoder,
        "task_type": task_type,
        "feature_names": final_cols,
        "datetime_cols_processed": datetime_cols_processed,
        "outlier_bounds": outlier_bounds,
        "feature_selector": selector,
        "selected_features": selected_features,
        "cat_feature_indices": cat_feature_indices,
        "high_card_cols": surviving_high_card,
    }

    return X_train, X_test, y_train, y_test, artifacts
