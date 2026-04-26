import pandas as pd
import numpy as np
NEAR_CONSTANT_THRESHOLD = 0.95
HIGH_CORRELATION_THRESHOLD = 0.85


def detect_column_types(df):
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            # try parsing as datetime before labelling categorical
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                # only classify as datetime if most values parsed successfully
                if parsed.notna().mean() > 0.8:
                    types[col] = "datetime"
                else:
                    types[col] = "categorical"
            except Exception:
                types[col] = "categorical"
    return types


def get_missing_value_ratios(df):
    ratios = (df.isnull().sum() / len(df)).to_dict()
    # only return columns that actually have missing values
    return {col: round(ratio, 4) for col, ratio in ratios.items() if ratio > 0}


def find_near_constant_features(df):
    near_constant = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        if top_freq >= NEAR_CONSTANT_THRESHOLD:
            near_constant.append(col)
    return near_constant


def get_correlation_summary(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"high_correlation_pairs": [], "threshold_used": HIGH_CORRELATION_THRESHOLD}

    corr_matrix = numeric_df.corr().abs()

    # only look at upper triangle to avoid duplicate pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_pairs = [
        [col, row]
        for col in upper.columns
        for row in upper.index
        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > HIGH_CORRELATION_THRESHOLD
    ]

    return {
        "high_correlation_pairs": high_pairs,
        "threshold_used": HIGH_CORRELATION_THRESHOLD
    }


def get_class_balance(df, target_col):
    if target_col not in df.columns:
        return {}
    counts = df[target_col].value_counts(normalize=True).round(4).to_dict()
    return {str(k): v for k, v in counts.items()}


def infer_target_column(df, column_types):
    # best guess at target: low cardinality categorical or binary numeric
    candidates = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if column_types.get(col) == "categorical" and 2 <= n_unique <= 20:
            entropy = compute_entropy(df[col])
            candidates.append((col, entropy))
        elif column_types.get(col) == "numerical" and n_unique == 2:
            candidates.append((col, 1.0))

    if not candidates:
        return None

    # pick the column with highest entropy among candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def compute_entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))


def check_class_imbalance(class_balance):
    if not class_balance:
        return False
    max_ratio = max(class_balance.values())
    return max_ratio > 0.80  # flag if dominant class is more than 80 percent


def profile_dataset(df, target_col=None):
    column_types = detect_column_types(df)

    suggested_target = target_col if target_col else infer_target_column(df, column_types)

    class_balance = {}
    is_imbalanced = False
    if suggested_target and column_types.get(suggested_target) in ("categorical", "numerical"):
        class_balance = get_class_balance(df, suggested_target)
        is_imbalanced = check_class_imbalance(class_balance)

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_types": column_types,
        "missing_values": get_missing_value_ratios(df),
        "near_constant_features": find_near_constant_features(df),
        "correlation_summary": get_correlation_summary(df),
        "class_balance": {suggested_target: class_balance} if class_balance else {},
        "imbalance_warning": is_imbalanced,
        "target_suggestion": suggested_target
    }

    return report