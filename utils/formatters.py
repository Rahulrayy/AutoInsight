import json


def leaderboard_to_display(leaderboard, task_type):
    # converts raw leaderboard list into something readable for the ui
    rows = []
    for i, entry in enumerate(leaderboard):
        row = {"rank": i + 1}
        row.update({k: v for k, v in entry.items() if k != "model"})
        row["model"] = entry["model"].replace("_", " ").title()
        rows.append(row)
    return rows


def importances_to_display(importances, top_n=10):
    # trims and formats feature importances for a bar chart or table
    top = dict(list(importances.items())[:top_n])
    return {
        "features": list(top.keys()),
        "scores": [round(v, 4) for v in top.values()]
    }


def profile_to_display(profile_report):
    # pulls out the parts of the profile that are worth showing in the ui
    return {
        "rows": profile_report["rows"],
        "columns": profile_report["columns"],
        "missing_value_count": len(profile_report.get("missing_values", {})),
        "near_constant_count": len(profile_report.get("near_constant_features", [])),
        "high_correlation_pairs": profile_report.get("correlation_summary", {}).get("high_correlation_pairs", []),
        "imbalance_warning": profile_report.get("imbalance_warning", False),
        "suggested_target": profile_report.get("target_suggestion")
    }


def build_full_report(profile_report, eval_results, llm_explanation):
    # assembles everything into one exportable json dict
    report = {
        "dataset_profile": profile_report,
        "leaderboard": eval_results["leaderboard"],
        "best_model": eval_results["best_model_name"],
        "task_type": eval_results["task_type"],
        "top_feature_importances": eval_results["best_feature_importances"],
        "llm_explanation": llm_explanation
    }
    return report


def report_to_json_string(report):
    return json.dumps(report, indent=2, default=str)  # default=str handles any non serializable types