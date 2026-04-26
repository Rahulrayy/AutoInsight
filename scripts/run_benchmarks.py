import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # add project root to path

import argparse
import json
import time
import pandas as pd
from sklearn.datasets import fetch_california_housing

from pipeline.profiling import profile_dataset
from pipeline.preprocessing import run_preprocessing
from pipeline.automl import train_all_models
from pipeline.evaluation import run_evaluation


def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df, "Survived"


def load_heart_disease():
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd.read_csv(url)
    return df, "target"


def load_adult_income():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
    cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"]
    df = pd.read_csv(url, header=None, names=cols, skipinitialspace=True)
    return df, "income"


def load_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df, "MedHouseVal"


def load_diamonds():
    url = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv"
    df = pd.read_csv(url)
    return df, "price"


DATASET_LOADERS = {
    "titanic": load_titanic,
    "heart_disease": load_heart_disease,
    "adult_income": load_adult_income,
    "california_housing": load_california_housing,
    "diamonds": load_diamonds
}


def run_single_benchmark(name, loader_fn):
    print(f"\nrunning benchmark: {name}")
    start = time.time()

    df, target_col = loader_fn()
    print(f"  loaded {df.shape[0]} rows, {df.shape[1]} columns")

    profile = profile_dataset(df, target_col)
    X_train, X_test, y_train, y_test, artifacts = run_preprocessing(df, target_col, profile)
    trained = train_all_models(X_train, y_train, artifacts["task_type"], artifacts["feature_names"])
    results = run_evaluation(trained, X_test, y_test, artifacts["task_type"])

    elapsed = round(time.time() - start, 1)

    top = results["leaderboard"][0]
    primary_metric = "auc" if "classification" in artifacts["task_type"] else "r2"
    primary_score = top.get(primary_metric)

    # regression guard: fail loudly if primary score degrades below floor
    FLOORS = {
        "binary_classification": {"auc": 0.60},
        "multiclass_classification": {"auc": 0.55},
        "regression": {"r2": -1.0},  # any finite r2 is acceptable
    }
    floor_val = FLOORS.get(artifacts["task_type"], {}).get(primary_metric)
    if primary_score is not None and floor_val is not None and primary_score < floor_val:
        print(f"  WARN: {name} {primary_metric}={primary_score} below floor {floor_val}")

    summary = {
        "dataset": name,
        "rows": df.shape[0],
        "features": df.shape[1] - 1,
        "task_type": artifacts["task_type"],
        "best_model": results["best_model_name"],
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "leaderboard": results["leaderboard"],
        "runtime_seconds": elapsed,
        "passed": primary_score is not None and (floor_val is None or primary_score >= floor_val),
    }

    print(f"  best model: {results['best_model_name']}")
    print(f"  {primary_metric}: {primary_score}")
    print(f"  done in {elapsed}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="run autoinsight benchmarks")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=list(DATASET_LOADERS.keys()),
        help="which datasets to benchmark"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="path to save results json"
    )
    args = parser.parse_args()

    all_results = []

    for name in args.datasets:
        try:
            result = run_single_benchmark(name, DATASET_LOADERS[name])
            all_results.append(result)
        except Exception as e:
            print(f"  failed: {e}")
            all_results.append({"dataset": name, "error": str(e)})

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nall results saved to {args.output}")

    failed = [r["dataset"] for r in all_results if not r.get("passed", True)]
    if failed:
        print(f"\nFAILED benchmarks: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(all_results)} benchmarks passed.")


if __name__ == "__main__":
    main()