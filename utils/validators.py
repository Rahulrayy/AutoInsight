import os

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))


def validate_target_column(target_col, column_names):
    if not target_col:
        raise ValueError("no target column selected")
    if target_col not in column_names:
        raise ValueError(f"column '{target_col}' not found in dataset")


def validate_file_extension(filename):
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"only csv files are supported, got: {filename}")


def validate_column_count(df):
    if df.shape[1] < 2:
        raise ValueError("dataset needs at least 2 columns")


def validate_row_count(df):
    if df.shape[0] < 10:
        raise ValueError("dataset needs at least 10 rows")


def validate_target_has_variance(df, target_col):
    if df[target_col].nunique() < 2:
        raise ValueError(f"target column '{target_col}' has only one unique value, nothing to predict")


def validate_api_key(api_key):
    if not api_key or not api_key.strip():
        raise ValueError("groq api key is missing, add it to your .env file")


def run_all_validations(df, target_col, api_key=None):
    # runs the full set of checks before the pipeline starts
    validate_column_count(df)
    validate_row_count(df)
    validate_target_column(target_col, list(df.columns))
    validate_target_has_variance(df, target_col)

    if api_key is not None:
        validate_api_key(api_key)