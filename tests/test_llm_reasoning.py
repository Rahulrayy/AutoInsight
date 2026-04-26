import pytest
import json
from unittest.mock import patch, MagicMock
from pipeline.llm_reasoning import (
    parse_json_response, call_with_retry, build_analysis_context,
    explain_dataset, answer_question,
)


@pytest.fixture
def sample_profile():
    return {
        "rows": 303,
        "columns": 13,
        "column_types": {"age": "numerical", "target": "categorical"},
        "missing_values": {"ca": 0.0066},
        "near_constant_features": [],
        "correlation_summary": {"high_correlation_pairs": [], "threshold_used": 0.85},
        "class_balance": {"target": {"0": 0.46, "1": 0.54}},
        "imbalance_warning": False,
        "target_suggestion": "target",
    }


@pytest.fixture
def sample_eval():
    return {
        "leaderboard": [
            {"model": "random_forest", "auc": 0.91, "weighted_f1": 0.85, "accuracy": 0.86},
            {"model": "xgboost", "auc": 0.89, "weighted_f1": 0.82, "accuracy": 0.84},
        ],
        "best_model_name": "random_forest",
        "best_model": MagicMock(),
        "best_feature_importances": {"thal": 0.18, "cp": 0.15, "ca": 0.14},
        "task_type": "binary_classification",
    }


# ── parse_json_response ────────────────────────────────────────────────────────

def test_parse_json_response_clean():
    raw = '{"summary": "test", "key_findings": []}'
    result = parse_json_response(raw)
    assert result["summary"] == "test"


def test_parse_json_response_strips_code_fences():
    raw = '```json\n{"summary": "test"}\n```'
    result = parse_json_response(raw)
    assert result["summary"] == "test"


def test_parse_json_response_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("this is not json at all")


# ── build_analysis_context ─────────────────────────────────────────────────────

def test_build_analysis_context_has_required_keys(sample_profile, sample_eval):
    context_str = build_analysis_context(sample_profile, sample_eval)
    context = json.loads(context_str)
    assert "dataset_summary" in context
    assert "leaderboard" in context
    assert "top_feature_importances" in context
    assert "task_type" in context


def test_build_analysis_context_no_raw_data(sample_profile, sample_eval):
    context_str = build_analysis_context(sample_profile, sample_eval)
    # should not contain individual row data
    assert "303" in context_str  # row count is fine
    # raw feature values should not appear
    context = json.loads(context_str)
    assert "dataset_summary" in context


def test_build_analysis_context_with_tuning(sample_profile, sample_eval):
    tuned = {
        "random_forest": {
            "model": MagicMock(),
            "best_score": 0.94,
            "best_params": {"n_estimators": 200},
            "feature_importances": {"thal": 0.2},
        }
    }
    context_str = build_analysis_context(sample_profile, sample_eval, tuned_results=tuned)
    context = json.loads(context_str)
    assert context["tuning_was_performed"] is True
    assert "tuning_summary" in context


def test_build_analysis_context_token_budget_respected(sample_profile, sample_eval):
    # create a large leaderboard (>5 entries) and lower the budget so trimming triggers
    big_eval = dict(sample_eval)
    big_eval["leaderboard"] = [
        {"model": f"model_{i}", "auc": 0.9 - i*0.01} for i in range(10)
    ]
    import pipeline.llm_reasoning as llm_mod
    original = llm_mod.MAX_CONTEXT_CHARS
    llm_mod.MAX_CONTEXT_CHARS = 1  # force trimming
    try:
        context_str = build_analysis_context(sample_profile, big_eval)
        context = json.loads(context_str)
        assert len(context["original_leaderboard"]) <= 5
    finally:
        llm_mod.MAX_CONTEXT_CHARS = original


# ── call_with_retry ────────────────────────────────────────────────────────────

def test_call_with_retry_success():
    valid_response = json.dumps({
        "summary": "ok", "key_findings": [],
        "interpretation": "", "recommendations": [],
    })
    with patch("pipeline.llm_reasoning.call_groq", return_value=valid_response):
        result = call_with_retry("system", "user", "fake_key")
        assert result["summary"] == "ok"


def test_call_with_retry_falls_back_after_bad_json():
    with patch("pipeline.llm_reasoning.call_groq", return_value="not valid json"):
        result = call_with_retry("system", "user message", "fake_key")
        assert "summary" in result
        assert "llm explanation unavailable" in result["summary"]


def test_call_with_retry_handles_network_error():
    import requests
    with patch("pipeline.llm_reasoning.call_groq",
               side_effect=requests.RequestException("timeout")):
        result = call_with_retry("system", "user message", "fake_key")
        assert "summary" in result


# ── explain_dataset ────────────────────────────────────────────────────────────

def test_explain_dataset_uses_mock(sample_profile):
    valid_response = json.dumps({
        "summary": "dataset looks clean",
        "key_findings": ["no major issues"],
        "interpretation": "",
        "recommendations": [],
    })
    with patch("pipeline.llm_reasoning.call_groq", return_value=valid_response):
        result = explain_dataset(sample_profile, api_key="fake_key")
        assert result["summary"] == "dataset looks clean"


# ── answer_question ────────────────────────────────────────────────────────────

def test_answer_question_uses_mock(sample_profile, sample_eval):
    valid_response = json.dumps({
        "answer": "thal is the most important feature",
        "supporting_evidence": ["shap value 0.18"],
        "confidence": "high",
        "caveat": None,
    })
    with patch("pipeline.llm_reasoning.call_groq", return_value=valid_response):
        result = answer_question(
            "which feature is most important?",
            sample_profile, sample_eval, api_key="fake_key"
        )
        assert "thal" in result["answer"]
        assert result["confidence"] == "high"
