import pandas as pd
import chardet
import os


MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))


def detect_encoding(file_bytes):
    result = chardet.detect(file_bytes)
    return result.get("encoding", "utf-8") or "utf-8"


def check_file_size(file_obj):
    # seek to end to get size then reset
    file_obj.seek(0, 2)
    size_mb = file_obj.tell() / (1024 * 1024)
    file_obj.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"file is {size_mb:.1f}mb, limit is {MAX_FILE_SIZE_MB}mb")


def load_csv(file_obj):
    check_file_size(file_obj)

    raw_bytes = file_obj.read()
    encoding = detect_encoding(raw_bytes)

    try:
        df = pd.read_csv(
            pd.io.common.BytesIO(raw_bytes),
            encoding=encoding,
            on_bad_lines="skip"  # skip rows that cant be parsed instead of crashing
        )
    except Exception as e:
        raise ValueError(f"could not parse csv: {e}")

    validate_dataframe(df)
    return df


def validate_dataframe(df):
    if df.empty:
        raise ValueError("uploaded file is empty")

    if df.shape[1] < 2:
        raise ValueError("need at least 2 columns to do anything useful")

    if df.shape[0] < 10:
        raise ValueError("need at least 10 rows, this dataset is too small")

    # warn if more than 100k rows, caller can decide to sample
    if df.shape[0] > 100_000:
        print(f"warning: {df.shape[0]} rows detected, consider sampling for speed")

    # drop columns that are entirely empty
    all_null_cols = [c for c in df.columns if df[c].isnull().all()]
    if all_null_cols:
        df.drop(columns=all_null_cols, inplace=True)

    # strip whitespace from column names
    df.columns = df.columns.str.strip()

    return df


def get_basic_info(df):
    # just a quick dict summarising what id loaded used by the ui
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }