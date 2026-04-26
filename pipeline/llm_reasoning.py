import json
import os
import re
import time
import requests

from pipeline.config_loader import get as cfg

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = int(os.getenv("MAX_RETRIES_LLM", cfg("llm", "max_retries", 3)))
MAX_CONTEXT_CHARS = cfg("llm", "max_context_chars", 10000)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def get_model():
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def load_prompt(filename):
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _build_payload(system_prompt, user_message, stream=False):
    return {
        "model": get_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": cfg("llm", "temperature", 0.2),
        "max_tokens": cfg("llm", "max_tokens", 1500),
        "stream": stream,
    }


def call_groq(system_prompt, user_message, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_payload(system_prompt, user_message, stream=False)
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    if not response.ok:
        raise requests.HTTPError(f"{response.status_code}: {response.text}", response=response)
    return response.json()["choices"][0]["message"]["content"]


def stream_groq(system_prompt, user_message, api_key):
    """Generator that yields text chunks from the Groq streaming API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_payload(system_prompt, user_message, stream=True)
    with requests.post(GROQ_API_URL, headers=headers, json=payload,
                       stream=True, timeout=60) as resp:
        if not resp.ok:
            raise requests.HTTPError(f"{resp.status_code}: {resp.text}", response=resp)
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(text)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError):
                continue


def parse_json_response(raw_text):
    cleaned = raw_text.strip()
    # strip chain-of-thought thinking blocks emitted by qwen3 and similar models
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        raise json.JSONDecodeError("empty response (thinking tokens used up all budget)", "", 0)
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def _retry_after(response):
    """Extract how many seconds to wait from a 429 response."""
    if response is not None and "Retry-After" in response.headers:
        try:
            return float(response.headers["Retry-After"])
        except ValueError:
            pass
    try:
        body = response.json()
        msg = body.get("error", {}).get("message", "")
        m = re.search(r"try again in ([\d.]+)s", msg)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 15.0


def call_with_retry(system_prompt, user_message, api_key):
    last_error = None
    msg = user_message

    for attempt in range(MAX_RETRIES):
        try:
            raw = call_groq(system_prompt, msg, api_key)
            return parse_json_response(raw)
        except json.JSONDecodeError as e:
            last_error = e
            msg = msg + "\n\nyour previous response was not valid json. return only the json object, no preamble, no markdown."
        except requests.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code == 429:
                wait = _retry_after(e.response)
                time.sleep(min(wait + 1, 60))
                continue  # retry after wait
            break  # other HTTP errors are not retriable
        except requests.RequestException as e:
            last_error = e
            break

    return {
        "summary": "llm explanation unavailable",
        "key_findings": [],
        "interpretation": f"could not generate explanation after {MAX_RETRIES} attempts: {last_error}",
        "recommendations": [],
    }


_HEAVY_FIELDS = {"y_pred", "y_true", "confusion_matrix", "confusion_matrix_labels"}


def _clean_row(row):
    """Strip heavy per-prediction arrays from a leaderboard row before sending to LLM."""
    return {k: v for k, v in row.items() if k not in _HEAVY_FIELDS}


def _trim_context(context_dict):
    """Reduce context size when it exceeds the token budget."""
    for key in ("original_leaderboard", "leaderboard"):
        if key in context_dict:
            context_dict[key] = context_dict[key][:5]
    if "shap_summary" in context_dict:
        for label in context_dict["shap_summary"]:
            top = context_dict["shap_summary"][label].get("mean_abs_shap_top10", {})
            context_dict["shap_summary"][label]["mean_abs_shap_top10"] = dict(list(top.items())[:5])
    if "top_feature_importances" in context_dict:
        context_dict["top_feature_importances"] = dict(
            list(context_dict["top_feature_importances"].items())[:5]
        )
    return context_dict


def build_analysis_context(profile_report, eval_results, tuned_results=None, shap_results=None):
    top_features = dict(list(eval_results["best_feature_importances"].items())[:10])
    test_size = float(os.getenv("DEFAULT_TEST_SIZE", cfg("pipeline", "test_size", 0.2)))
    train_rows = round(profile_report["rows"] * (1 - test_size))
    test_rows = profile_report["rows"] - train_rows

    # always strip heavy arrays — y_pred/y_true/confusion_matrix must never reach the LLM
    clean_leaderboard = [_clean_row(r) for r in eval_results["leaderboard"]]

    context = {
        "dataset_summary": {
            "rows": profile_report["rows"],
            "columns": profile_report["columns"],
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_test_split": f"{round((1-test_size)*100)}/{round(test_size*100)}",
            "missing_value_columns": len(profile_report.get("missing_values", {})),
            "near_constant_features": profile_report.get("near_constant_features", []),
            "high_correlation_pairs": profile_report.get("correlation_summary", {}).get("high_correlation_pairs", []),
            "class_balance": profile_report.get("class_balance", {}),
            "imbalance_warning": profile_report.get("imbalance_warning", False),
        },
        "task_type": eval_results["task_type"],
        "original_leaderboard": clean_leaderboard,
        "best_model_before_tuning": eval_results["best_model_name"],
        "top_feature_importances": top_features,
        "leaderboard": clean_leaderboard,
    }

    if tuned_results:
        tuning_summary = []
        for name, result in tuned_results.items():
            original_row = next((r for r in eval_results["leaderboard"] if r["model"] == name), {})
            primary = "auc" if "classification" in eval_results["task_type"] else "r2"
            original_score = original_row.get(primary)
            tuning_summary.append({
                "model": name,
                "original_score": original_score,
                "tuned_score": result["best_score"],
                "improvement": round(result["best_score"] - original_score, 4) if original_score else None,
                "best_hyperparameters": result["best_params"],
            })
        best_tuned = max(tuning_summary, key=lambda x: x["tuned_score"])
        context["tuning_was_performed"] = True
        context["tuning_summary"] = tuning_summary
        context["best_model_after_tuning"] = best_tuned["model"]
        context["best_tuned_score"] = best_tuned["tuned_score"]
    else:
        context["tuning_was_performed"] = False

    if shap_results:
        shap_summary = {}
        for model_name, data in shap_results.items():
            label = data["label"]
            top_shap = dict(list(data["mean_abs_shap"].items())[:10])
            shap_summary[label] = {
                "mean_abs_shap_top10": top_shap,
                "note": "mean absolute shap — higher means more impact on predictions on average",
            }
        context["shap_analysis_available"] = True
        context["shap_summary"] = shap_summary
        best = eval_results["best_model_name"]
        if best in shap_results:
            context["top_feature_importances_source"] = "shap"
            context["top_feature_importances"] = dict(list(shap_results[best]["mean_abs_shap"].items())[:10])
        else:
            context["top_feature_importances_source"] = "model_native"
    else:
        context["shap_analysis_available"] = False

    # guard against oversized context
    raw = json.dumps(context)
    if len(raw) > MAX_CONTEXT_CHARS:
        context = _trim_context(context)

    return json.dumps(context, indent=2)


def explain_results(profile_report, eval_results, api_key, tuned_results=None, shap_results=None):
    system_prompt = load_prompt("system_prompt.txt")
    template = load_prompt("model_explanation.txt")
    context_json = build_analysis_context(profile_report, eval_results,
                                          tuned_results=tuned_results, shap_results=shap_results)
    user_message = template.replace("{{context}}", context_json)
    return call_with_retry(system_prompt, user_message, api_key)


def explain_dataset(profile_report, api_key):
    system_prompt = load_prompt("system_prompt.txt")
    template = load_prompt("dataset_analysis.txt")
    context_json = json.dumps({
        "rows": profile_report["rows"],
        "columns": profile_report["columns"],
        "column_types": profile_report["column_types"],
        "missing_values": profile_report["missing_values"],
        "near_constant_features": profile_report["near_constant_features"],
        "correlation_summary": profile_report["correlation_summary"],
        "class_balance": profile_report.get("class_balance", {}),
        "imbalance_warning": profile_report.get("imbalance_warning", False),
    }, indent=2)
    user_message = template.replace("{{context}}", context_json)
    return call_with_retry(system_prompt, user_message, api_key)


def answer_question(question, profile_report, eval_results, api_key,
                    tuned_results=None, shap_results=None):
    system_prompt = load_prompt("system_prompt.txt")
    template = load_prompt("qa_prompt.txt")
    context_json = build_analysis_context(profile_report, eval_results,
                                          tuned_results=tuned_results, shap_results=shap_results)
    user_message = (
        template
        .replace("{{context}}", context_json)
        .replace("{{question}}", question)
    )
    return call_with_retry(system_prompt, user_message, api_key)


def stream_answer_question(question, profile_report, eval_results, api_key,
                           tuned_results=None, shap_results=None):
    """Stream a plain-text answer to the user's question (no JSON wrapper)."""
    context_json = build_analysis_context(profile_report, eval_results,
                                          tuned_results=tuned_results, shap_results=shap_results)
    system_prompt = load_prompt("system_prompt.txt")
    user_message = (
        f"Results context:\n{context_json}\n\n"
        f"Question: {question}\n\n"
        "Answer in clear plain text (not JSON). Be concise and quantify with numbers from the context. "
        "End with a one-sentence caveat about limitations if relevant."
    )
    return stream_groq(system_prompt, user_message, api_key)
