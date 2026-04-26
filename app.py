import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import tempfile
from dotenv import load_dotenv

from pipeline.ingestion import load_csv, get_basic_info
from pipeline.profiling import profile_dataset
from pipeline.preprocessing import run_preprocessing
from pipeline.automl import train_all_models, build_stacking_ensemble
from pipeline.evaluation import run_evaluation
from pipeline.llm_reasoning import (
    explain_results, explain_dataset, answer_question, stream_answer_question
)
from pipeline.tuning import tune_single_model, tune_top_models
from pipeline.shap_analysis import (
    run_shap_for_top3, compute_shap_for_instance,
    compute_dependence_data, compute_pdp,
)
from pipeline.automl import get_feature_importances
from utils.validators import run_all_validations
from utils.formatters import (
    leaderboard_to_display, importances_to_display,
    profile_to_display, build_full_report, report_to_json_string,
)
from utils.logger import get_logger

load_dotenv()
logger = get_logger("app")

st.set_page_config(page_title="AutoInsight", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .main > div { padding: 0 !important; }
    .navbar {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 0 3rem;
        height: 60px;
        display: flex;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .navbar-brand { font-size: 1.2rem; font-weight: 700; color: #111827; letter-spacing: -0.3px; }
    .navbar-brand span { color: #2563eb; }
    .navbar-sub { font-size: 0.78rem; color: #9ca3af; font-weight: 400; }
    .section-heading { font-size: 1rem; font-weight: 600; color: #111827; margin: 0 0 0.4rem 0; }
    .section-sub { font-size: 0.82rem; color: #6b7280; margin-bottom: 1.2rem; }
    .badge { display: inline-block; background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.25rem 0.75rem; font-size: 0.78rem; font-weight: 600; }
    .badge-green { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
    .step-box { background: #f8f9fb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
    .step-done { border-left: 4px solid #16a34a; }
    .step-active { border-left: 4px solid #2563eb; }
    .finding { background: #f8faff; border-left: 3px solid #2563eb; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-size: 0.875rem; color: #374151; line-height: 1.55; }
    .rec { background: #f0fdf4; border-left: 3px solid #16a34a; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-size: 0.875rem; color: #374151; line-height: 1.55; }
    .body-text { font-size: 0.9rem; color: #374151; line-height: 1.65; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #ffffff; border-bottom: 1px solid #e5e7eb; padding: 0 3rem; margin-bottom: 0; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 1.25rem; font-size: 0.875rem; font-weight: 500; color: #6b7280; border-radius: 0; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #2563eb; border-bottom: 2px solid #2563eb; background: transparent; }
    .stTabs [data-baseweb="tab-panel"] { padding: 2.5rem 3rem; max-width: 1200px; margin: 0 auto; }
    [data-testid="stFileUploader"] { background: #fafafa; border: 2px dashed #d1d5db; border-radius: 12px; }
    .stButton > button { background: #2563eb; color: #ffffff; border: none; border-radius: 8px; padding: 0.6rem 1.75rem; font-weight: 600; font-size: 0.875rem; }
    .stButton > button:hover { background: #1d4ed8; color: #ffffff; }
    .stDownloadButton > button { background: #ffffff; color: #2563eb; border: 1.5px solid #2563eb; border-radius: 8px; font-weight: 600; font-size: 0.875rem; padding: 0.6rem 1.75rem; }
    .stDownloadButton > button:hover { background: #eff6ff; }
    [data-testid="stDataFrame"] > div { border-radius: 10px; border: 1px solid #e5e7eb; overflow: hidden; }
    .stTextInput input { border-radius: 8px; border: 1px solid #d1d5db; font-size: 0.875rem; padding: 0.55rem 0.9rem; }
    [data-testid="metric-container"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.1rem 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    [data-testid="metric-container"] label { font-size: 0.72rem; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; color: #111827; }
    .chat-user { background: #eff6ff; border-radius: 12px 12px 4px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.875rem; }
    .chat-bot { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px 12px 12px 4px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.875rem; }
</style>
""", unsafe_allow_html=True)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "mixtral-8x7b-32768",
    "qwen/qwen3-32b",
]


def get_api_key():
    return os.getenv("GROQ_API_KEY", "")


def render_sidebar():
    with st.sidebar:
        st.markdown("### Settings")
        model_choice = st.selectbox(
            "LLM Model", GROQ_MODELS,
            index=0,
            help="Groq model used for explanations and Q&A"
        )
        os.environ["GROQ_MODEL"] = model_choice

        st.markdown("---")
        st.markdown("**Session**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", use_container_width=True, help="Save session to file"):
                _save_session()
        with col2:
            restore_file = st.file_uploader("Restore", type=["pkl"],
                                            label_visibility="collapsed",
                                            help="Restore a previously saved session")
            if restore_file:
                _restore_session(restore_file)

        st.markdown("---")
        st.caption("Light/dark theme: use the ⋮ menu (top right)")


def _save_session():
    keys = ["df", "target_col", "profile", "eval_results", "artifacts",
            "trained_models", "X_train", "X_test", "y_train", "y_test",
            "tuned_results", "tuning_studies", "shap_results", "explanation",
            "pipeline_stage", "qa_history"]
    snapshot = {k: st.session_state[k] for k in keys if k in st.session_state}
    try:
        data = pickle.dumps(snapshot)
        st.sidebar.download_button("Download session", data=data,
                                   file_name="autoinsight_session.pkl",
                                   mime="application/octet-stream")
    except Exception as e:
        st.sidebar.error(f"Save failed: {e}")


def _restore_session(uploaded):
    try:
        snapshot = pickle.loads(uploaded.read())
        for k, v in snapshot.items():
            st.session_state[k] = v
        st.success("Session restored.")
        st.rerun()
    except Exception as e:
        st.error(f"Restore failed: {e}")


def navbar():
    st.markdown("""
    <div class="navbar">
        <div>
            <div class="navbar-brand">Auto<span>Insight</span></div>
            <div class="navbar-sub">automated tabular ml with llm explanations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def step_indicator(label, status):
    css = "step-done" if status == "done" else ("step-active" if status == "active" else "")
    icon = "✓" if status == "done" else ("→" if status == "active" else "·")
    st.markdown(f'<div class="step-box {css}"><strong>{icon} {label}</strong></div>',
                unsafe_allow_html=True)


# ── caching wrappers ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_profile(df_hash, df, target_col):
    return profile_dataset(df, target_col)


@st.cache_data(show_spinner=False)
def cached_preprocess(df_hash, df, target_col, profile_json):
    import json
    profile = json.loads(profile_json)
    return run_preprocessing(df, target_col, profile)


# ── upload tab ────────────────────────────────────────────────────────────────

def render_upload_tab():
    st.markdown('<p class="section-heading">Upload your dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a CSV file and select the column you want to predict.</p>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded_file is None:
        return None, None

    try:
        df = load_csv(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return None, None

    info = get_basic_info(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", info["columns"])
    c3.metric("File size", f"{info['memory_mb']} MB")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Preview first 20 rows", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Select target column</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">The column the models will learn to predict.</p>',
                unsafe_allow_html=True)
    target_col = st.selectbox("Target column", options=df.columns, label_visibility="collapsed")

    return df, target_col


def render_profile_section(df, target_col, api_key):
    with st.spinner("Profiling dataset..."):
        df_hash = str(pd.util.hash_pandas_object(df).sum()) + target_col
        profile = cached_profile(df_hash, df, target_col)

    display = profile_to_display(profile)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Dataset Profile</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{display['rows']:,}")
    c2.metric("Total columns", display["columns"])
    c3.metric("Missing value columns", display["missing_value_count"])
    c4.metric("Near-constant features", display["near_constant_count"])

    st.markdown("<br>", unsafe_allow_html=True)
    if display["imbalance_warning"]:
        st.warning("Class imbalance detected. The dominant class exceeds 80% of target values.")
    if display["high_correlation_pairs"]:
        st.warning(f"Highly correlated feature pairs: {display['high_correlation_pairs']}")
    if profile.get("missing_values"):
        with st.expander("Missing value breakdown"):
            missing_df = pd.DataFrame(list(profile["missing_values"].items()),
                                      columns=["Column", "Missing Ratio"])
            st.dataframe(missing_df, use_container_width=True)

    # column distribution drill-down
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Column Distribution</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Select a column to inspect its distribution.</p>',
                unsafe_allow_html=True)
    col_choice = st.selectbox("Column", options=list(df.columns), key="profile_col_select")
    if col_choice:
        col_data = df[col_choice].dropna()
        col_types = profile.get("column_types", {})
        if col_types.get(col_choice) == "numerical":
            fig = px.histogram(df, x=col_choice, nbins=30, title=f"Distribution of {col_choice}",
                               color_discrete_sequence=["#2563eb"])
        else:
            vc = col_data.value_counts().head(20).reset_index()
            vc.columns = ["value", "count"]
            fig = px.bar(vc, x="value", y="count", title=f"Top values in {col_choice}",
                         color_discrete_sequence=["#2563eb"])
        fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          font=dict(size=11, color="#374151"),
                          margin=dict(l=10, r=10, t=40, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # dataset analysis via LLM (if key available)
    if api_key:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">AI Dataset Analysis</p>', unsafe_allow_html=True)
        if st.button("Analyse dataset with AI", key="ai_dataset_btn"):
            with st.spinner("Generating dataset analysis..."):
                try:
                    da = explain_dataset(profile, api_key)
                    st.session_state["dataset_analysis"] = da
                except Exception as e:
                    st.error(f"Dataset analysis failed: {e}")

        da = st.session_state.get("dataset_analysis")
        if da:
            if da.get("summary") == "llm explanation unavailable":
                st.error(f"Dataset analysis failed: {da.get('interpretation', 'unknown error')}")
            else:
                if da.get("summary"):
                    st.markdown(f'<p class="body-text">{da["summary"]}</p>', unsafe_allow_html=True)
                flags = da.get("data_quality_flags") or []
                if flags:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="section-heading">Data Quality Flags</p>', unsafe_allow_html=True)
                    for f in flags:
                        st.markdown(f'<div class="finding">{f}</div>', unsafe_allow_html=True)
                note = da.get("class_balance_note")
                if note:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<p class="section-sub">Class balance: {note}</p>', unsafe_allow_html=True)
                prep = da.get("preprocessing_notes") or []
                if prep:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="section-heading">Preprocessing Notes</p>', unsafe_allow_html=True)
                    for p in prep:
                        st.markdown(f'<div class="rec">{p}</div>', unsafe_allow_html=True)

    return profile


# ── pipeline tab helpers ───────────────────────────────────────────────────────

def render_leaderboard(leaderboard, task_type):
    display_rows = leaderboard_to_display(leaderboard, task_type)
    best = display_rows[0]["model"]
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<p class="section-heading">Model Leaderboard</p>', unsafe_allow_html=True)
    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:right"><span class="badge badge-green">Best: {best}</span></div>',
                    unsafe_allow_html=True)

    clean_rows = [
        {k: v for k, v in row.items() if k not in ("confusion_matrix", "confusion_matrix_labels", "y_pred", "y_true")}
        for row in display_rows
    ]
    st.dataframe(pd.DataFrame(clean_rows).set_index("rank"), use_container_width=True)


def render_model_comparison(leaderboard, task_type):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Select two models to compare side by side.</p>',
                unsafe_allow_html=True)
    model_names = [r["model"] for r in leaderboard]
    if len(model_names) < 2:
        return
    col1, col2 = st.columns(2)
    with col1:
        m1 = st.selectbox("Model A", model_names, index=0, key="cmp_m1")
    with col2:
        m2 = st.selectbox("Model B", model_names, index=min(1, len(model_names)-1), key="cmp_m2")

    if m1 == m2:
        st.info("Select two different models.")
        return

    row1 = next((r for r in leaderboard if r["model"] == m1), {})
    row2 = next((r for r in leaderboard if r["model"] == m2), {})
    skip = {"model", "confusion_matrix", "confusion_matrix_labels", "y_pred", "y_true"}
    metrics = [k for k in row1 if k not in skip]

    cmp_data = {"Metric": metrics,
                m1.replace("_", " ").title(): [row1.get(m) for m in metrics],
                m2.replace("_", " ").title(): [row2.get(m) for m in metrics]}
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True)


def render_confusion_matrix(leaderboard, task_type):
    if "classification" not in task_type:
        return
    cm_entries = [r for r in leaderboard if r.get("confusion_matrix") and r.get("confusion_matrix_labels")]
    if not cm_entries:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Confusion Matrices</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Predicted vs. actual class counts on the held-out test set.</p>',
                unsafe_allow_html=True)

    cols = st.columns(min(len(cm_entries), 3))
    for i, row in enumerate(cm_entries):
        cm = np.array(row["confusion_matrix"])
        labels = row["confusion_matrix_labels"]
        model_label = row["model"].replace("_", " ").title()
        fig = go.Figure(go.Heatmap(
            z=cm, x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels],
            text=cm, texttemplate="%{text}",
            colorscale=[[0, "#eff6ff"], [1, "#1d4ed8"]], showscale=False,
        ))
        fig.update_layout(
            title=dict(text=model_label, font_size=12),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(size=11, color="#374151"),
            margin=dict(l=10, r=10, t=40, b=10), height=300,
        )
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True)


def render_regression_scatter(leaderboard, task_type):
    if "classification" in task_type:
        return
    best_row = leaderboard[0]
    if "y_pred" not in best_row or "y_true" not in best_row:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Predicted vs Actual</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Best model on the held-out test set.</p>', unsafe_allow_html=True)

    y_true = best_row["y_true"]
    y_pred = best_row["y_pred"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color="#2563eb", size=6, opacity=0.6),
                             name="Predictions"))
    lo, hi = min(y_true + y_pred), max(y_true + y_pred)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#d1d5db", dash="dash"), name="Perfect fit"))
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis_title="Actual", yaxis_title="Predicted",
        font=dict(size=11, color="#374151"),
        margin=dict(l=10, r=10, t=20, b=10), height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tuning_comparison(original_leaderboard, tuned_results, task_type):
    primary = "auc" if "classification" in task_type else "r2"
    original_scores = {row["model"]: row.get(primary) for row in original_leaderboard}
    rows = []
    for name, result in tuned_results.items():
        original = original_scores.get(name)
        tuned = result["best_score"]
        delta = round(tuned - original, 4) if original is not None else None
        rows.append({
            "model": name.replace("_", " ").title(),
            f"original {primary}": original,
            f"tuned {primary}": tuned,
            "improvement": f"+{delta}" if delta and delta > 0 else str(delta),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    for name, result in tuned_results.items():
        with st.expander(f"Best hyperparameters: {name.replace('_', ' ').title()}"):
            for param, val in result["best_params"].items():
                st.write(f"**{param}:** {val}")


def render_convergence_plots(studies):
    if not studies:
        return
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Tuning Convergence</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Best score found per Optuna trial.</p>', unsafe_allow_html=True)

    cols = st.columns(min(len(studies), 3))
    for i, (model_name, study) in enumerate(studies.items()):
        trials = study.trials
        trial_nums = [t.number + 1 for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]
        best_so_far = [max(values[:j+1]) for j in range(len(values))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trial_nums, y=values, mode="markers",
                                 marker=dict(color="#93c5fd", size=5), name="Trial score"))
        fig.add_trace(go.Scatter(x=trial_nums, y=best_so_far, mode="lines",
                                 line=dict(color="#2563eb", width=2), name="Best so far"))
        fig.update_layout(
            title=dict(text=model_name.replace("_", " ").title(), font_size=12),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(size=11, color="#374151"),
            xaxis_title="Trial", yaxis_title="Score",
            margin=dict(l=10, r=10, t=40, b=10), height=300,
            showlegend=False,
        )
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True)


def render_shap_plots(shap_results, X_train, trained_models, tuned_results, task_type):
    for model_name, data in shap_results.items():
        label = data["label"]
        shap_values = data["shap_values"]
        X_explain = data["X_explain"]
        feature_names = data["feature_names"]
        mean_abs = data["mean_abs_shap"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-heading">{label.replace("_", " ").title()}</p>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        top_features = list(mean_abs.keys())[:15]
        top_scores = [mean_abs[f] for f in top_features]

        with col1:
            fig_bar = px.bar(
                x=top_scores, y=top_features, orientation="h",
                labels={"x": "Mean |SHAP value|", "y": ""},
                color=top_scores, color_continuous_scale=[[0, "#93c5fd"], [1, "#1d4ed8"]],
                title="Mean Absolute SHAP",
            )
            fig_bar.update_layout(
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(size=11, color="#374151"), yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10),
                height=420, title_font_size=13,
            )
            fig_bar.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            top_n = 15
            top_feat_idx = [feature_names.index(f) for f in top_features[:top_n] if f in feature_names]
            shap_subset = shap_values[:, top_feat_idx]
            feat_subset = [feature_names[i] for i in top_feat_idx]
            X_vals = X_explain.iloc[:, top_feat_idx].values.astype(float)
            X_norm = (X_vals - X_vals.min(axis=0)) / ((X_vals.max(axis=0) - X_vals.min(axis=0)) + 1e-9)

            fig_bee = go.Figure()
            for i, feat in enumerate(feat_subset):
                y_jitter = np.full(len(shap_subset), i) + np.random.uniform(-0.3, 0.3, len(shap_subset))
                fig_bee.add_trace(go.Scatter(
                    x=shap_subset[:, i], y=y_jitter, mode="markers",
                    marker=dict(size=4, color=X_norm[:, i],
                                colorscale=[[0, "#3b82f6"], [1, "#ef4444"]], opacity=0.6),
                    name=feat, showlegend=False,
                    hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.3f}}<extra></extra>",
                ))
            fig_bee.update_layout(
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(size=11, color="#374151"),
                xaxis=dict(title="SHAP value", gridcolor="#f3f4f6", zeroline=True, zerolinecolor="#d1d5db"),
                yaxis=dict(tickvals=list(range(len(feat_subset))),
                           ticktext=[f.replace("_", " ") for f in feat_subset], gridcolor="#f3f4f6"),
                margin=dict(l=10, r=10, t=40, b=10), height=420,
                title="Beeswarm (blue=low, red=high)", title_font_size=13,
            )
            st.plotly_chart(fig_bee, use_container_width=True)

        # dependence plot for selected feature
        st.markdown('<p class="section-sub">SHAP Dependence Plot</p>', unsafe_allow_html=True)
        dep_feat = st.selectbox("Feature", top_features[:20], key=f"dep_{model_name}")
        feat_vals, shap_col, color_vals = compute_dependence_data(
            shap_values, X_explain, feature_names, dep_feat
        )
        if feat_vals is not None:
            fig_dep = go.Figure(go.Scatter(
                x=feat_vals, y=shap_col, mode="markers",
                marker=dict(size=5, color=color_vals if color_vals is not None else "#2563eb",
                            colorscale="RdBu", showscale=color_vals is not None, opacity=0.7),
                hovertemplate=f"{dep_feat}: %{{x:.3f}}<br>SHAP: %{{y:.3f}}<extra></extra>",
            ))
            fig_dep.update_layout(
                xaxis_title=dep_feat, yaxis_title=f"SHAP value for {dep_feat}",
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(size=11, color="#374151"),
                margin=dict(l=10, r=10, t=20, b=10), height=320,
            )
            st.plotly_chart(fig_dep, use_container_width=True)

        # PDP for selected feature
        st.markdown('<p class="section-sub">Partial Dependence Plot</p>', unsafe_allow_html=True)
        pdp_feat = st.selectbox("PDP Feature", top_features[:20], key=f"pdp_{model_name}")
        model_obj = (tuned_results or {}).get(model_name, {}).get("model") or \
                    trained_models.get(model_name, {}).get("model")
        if model_obj and pdp_feat:
            grid, avg = compute_pdp(model_obj, X_train, feature_names, pdp_feat, task_type)
            if grid is not None:
                fig_pdp = go.Figure(go.Scatter(
                    x=grid, y=avg, mode="lines+markers",
                    line=dict(color="#2563eb", width=2),
                    marker=dict(size=5),
                ))
                fig_pdp.update_layout(
                    xaxis_title=pdp_feat, yaxis_title="Partial dependence",
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(size=11, color="#374151"),
                    margin=dict(l=10, r=10, t=20, b=10), height=300,
                )
                st.plotly_chart(fig_pdp, use_container_width=True)

        st.divider()


def render_per_prediction(shap_results, trained_models, tuned_results, X_train, task_type):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Per-Prediction Explanation</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter feature values to get a SHAP waterfall for that prediction.</p>',
                unsafe_allow_html=True)

    if not shap_results:
        st.info("Run SHAP analysis first.")
        return

    best_model_name = list(shap_results.keys())[0]
    feature_names = shap_results[best_model_name]["feature_names"]
    model_obj = (tuned_results or {}).get(best_model_name, {}).get("model") or \
                trained_models.get(best_model_name, {}).get("model")
    if model_obj is None:
        return

    with st.expander("Enter feature values", expanded=False):
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            col_idx = X_train.columns.get_loc(feat) if feat in X_train.columns else 0
            default_val = float(X_train.iloc[:, col_idx].mean()) if feat in X_train.columns else 0.0
            input_vals[feat] = cols[i % 3].number_input(feat, value=default_val,
                                                          key=f"pred_input_{feat}")

        if st.button("Explain this prediction", key="explain_pred_btn"):
            instance = pd.DataFrame([input_vals])
            try:
                vals, base, fnames = compute_shap_for_instance(model_obj, X_train, instance, task_type)
                sorted_idx = np.argsort(np.abs(vals))[::-1][:15]
                top_feats = [fnames[i] for i in sorted_idx]
                top_vals = [vals[i] for i in sorted_idx]
                colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top_vals]

                fig = go.Figure(go.Bar(
                    x=top_vals, y=top_feats, orientation="h",
                    marker_color=colors,
                ))
                fig.update_layout(
                    title=f"SHAP Waterfall — base value: {base:.3f}",
                    xaxis_title="SHAP value (red=pushes up, blue=pushes down)",
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(size=11, color="#374151"),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=10), height=420,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not explain: {e}")


# ── llm tab helpers ────────────────────────────────────────────────────────────

def render_llm_explanation(explanation):
    if not explanation:
        st.info("No explanation generated yet.")
        return
    if explanation.get("summary") == "llm explanation unavailable":
        st.error(f"LLM call failed: {explanation.get('interpretation', 'unknown error')}")
        return

    st.markdown('<p class="section-heading">Summary</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="body-text">{explanation.get("summary", "")}</p>', unsafe_allow_html=True)

    findings = explanation.get("key_findings", [])
    if findings:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">Key Findings</p>', unsafe_allow_html=True)
        for f in findings:
            st.markdown(f'<div class="finding">{f}</div>', unsafe_allow_html=True)

    interp = explanation.get("interpretation", "")
    if interp:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">Interpretation</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="body-text">{interp}</p>', unsafe_allow_html=True)

    recs = explanation.get("recommendations", [])
    if recs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">Recommendations</p>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f'<div class="rec">{r}</div>', unsafe_allow_html=True)


def render_qa_section(profile, eval_results, api_key, tuned_results=None, shap_results=None):
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="section-heading">Ask a question</p>', unsafe_allow_html=True)

    context_note = "The model answers using only the computed results, not raw data."
    if shap_results and tuned_results:
        context_note = "The model has access to original results, tuning results, and SHAP values."
    elif shap_results:
        context_note = "The model has access to original results and SHAP values."
    st.markdown(f'<p class="section-sub">{context_note}</p>', unsafe_allow_html=True)

    # show conversation history
    history = st.session_state.get("qa_history", [])
    for entry in history:
        st.markdown(f'<div class="chat-user">{entry["question"]}</div>', unsafe_allow_html=True)
        answer_text = entry["answer"] if isinstance(entry["answer"], str) else entry["answer"].get("answer", "")
        confidence = entry["answer"].get("confidence", "") if isinstance(entry["answer"], dict) else ""
        badge = f' <span style="font-size:0.72rem;color:#9ca3af;">[{confidence}]</span>' if confidence else ""
        st.markdown(f'<div class="chat-bot">{answer_text}{badge}</div>', unsafe_allow_html=True)
        if isinstance(entry["answer"], dict):
            evidence = entry["answer"].get("supporting_evidence", [])
            caveat = entry["answer"].get("caveat")
            if evidence:
                with st.expander("Evidence", expanded=False):
                    for e in evidence:
                        st.markdown(f"- {e}")
            if caveat:
                st.caption(f"Note: {caveat}")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input("Question",
                                  placeholder="e.g. which features matter most and why?",
                                  label_visibility="collapsed", key="qa_input")
    with col_btn:
        ask = st.button("Ask", use_container_width=True, key="qa_ask_btn")

    if ask and question.strip():
        with st.spinner("Thinking..."):
            try:
                # streaming mode for a better UX feel
                stream_gen = stream_answer_question(
                    question, profile, eval_results, api_key,
                    tuned_results=tuned_results, shap_results=shap_results
                )
                answer_text = st.write_stream(stream_gen)

                # also fetch structured JSON for evidence + confidence
                structured = answer_question(
                    question, profile, eval_results, api_key,
                    tuned_results=tuned_results, shap_results=shap_results
                )
                structured["answer"] = answer_text  # replace with streamed text

                history.append({"question": question, "answer": structured})
                st.session_state["qa_history"] = history
                st.rerun()
            except Exception as e:
                st.error(f"Could not get answer: {e}")

    if history and st.button("Clear conversation", key="qa_clear"):
        st.session_state["qa_history"] = []
        st.rerun()


# ── pipeline state machine ─────────────────────────────────────────────────────

def run_automl_step(df, target_col, profile):
    with st.spinner("Preprocessing data..."):
        X_train, X_test, y_train, y_test, artifacts = run_preprocessing(df, target_col, profile)

    model_names_display = {
        "logistic_regression": "Logistic Regression",
        "linear_regression":   "Linear Regression",
        "random_forest":       "Random Forest",
        "extra_trees":         "Extra Trees",
        "gradient_boosting":   "Gradient Boosting",
        "xgboost":             "XGBoost",
        "lightgbm":            "LightGBM",
        "catboost":            "CatBoost",
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    def training_progress(model_name, done, total):
        label = model_names_display.get(model_name, model_name.replace("_", " ").title())
        status_text.markdown(f'<p class="section-sub">Training {label} ({done}/{total})…</p>',
                             unsafe_allow_html=True)
        progress_bar.progress(done / total)

    # use class_weight when imbalance detected
    class_weight = "balanced" if profile.get("imbalance_warning") else None

    trained_models = train_all_models(
        X_train, y_train,
        artifacts["task_type"],
        artifacts["feature_names"],
        progress_callback=training_progress,
        class_weight=class_weight,
        cat_feature_indices=artifacts.get("cat_feature_indices", []),
        cv_folds=3,
    )

    progress_bar.progress(1.0)
    status_text.empty()

    with st.spinner("Evaluating on test set..."):
        eval_results = run_evaluation(trained_models, X_test, y_test, artifacts["task_type"])

    st.session_state["eval_results"]   = eval_results
    st.session_state["artifacts"]      = artifacts
    st.session_state["trained_models"] = trained_models
    st.session_state["X_train"]        = X_train
    st.session_state["X_test"]         = X_test
    st.session_state["y_train"]        = y_train
    st.session_state["y_test"]         = y_test
    st.session_state["tuned_results"]  = None
    st.session_state["tuning_studies"] = None
    st.session_state["shap_results"]   = None
    st.session_state["explanation"]    = None
    st.session_state["pipeline_stage"] = "automl_done"


def run_tuning_step(eval_results, artifacts):
    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]
    task_type = artifacts["task_type"]
    top_names = [row["model"] for row in eval_results["leaderboard"][:3]]

    progress_bar = st.progress(0)
    status = st.empty()
    total_trials = len(top_names) * 40

    trial_counter = {"done": 0}

    def make_cb(idx):
        def cb(trial_num, total):
            trial_counter["done"] = idx * 40 + trial_num
            progress_bar.progress(min(trial_counter["done"] / total_trials, 1.0))
        return cb

    tuned_results = {}
    studies = {}

    for i, name in enumerate(top_names):
        status.markdown(f'<p class="section-sub">Tuning {name} ({i+1}/{len(top_names)})...</p>',
                        unsafe_allow_html=True)
        try:
            best_model, best_score, best_params, study = tune_single_model(
                name, task_type, X_train, y_train, X_test, y_test, progress_callback=make_cb(i)
            )
            importances = get_feature_importances(best_model, list(X_train.columns))
            tuned_results[name] = {
                "model": best_model,
                "best_score": round(best_score, 4),
                "best_params": best_params,
                "feature_importances": importances,
            }
            studies[name] = study
        except Exception as e:
            st.warning(f"Tuning failed for {name}: {e}")

    # build stacking ensemble from tuned models
    if len(tuned_results) >= 2:
        status.markdown('<p class="section-sub">Building stacking ensemble...</p>',
                        unsafe_allow_html=True)
        stack = build_stacking_ensemble(tuned_results, task_type, X_train, y_train)
        if stack is not None:
            importances = get_feature_importances(stack, list(X_train.columns))
            from pipeline.evaluation import evaluate_classification, evaluate_regression
            try:
                if "classification" in task_type:
                    stack_metrics = evaluate_classification(stack, X_test, y_test, "stacking_ensemble", task_type)
                else:
                    stack_metrics = evaluate_regression(stack, X_test, y_test, "stacking_ensemble")
                tuned_results["stacking_ensemble"] = {
                    "model": stack,
                    "best_score": stack_metrics.get("auc") or stack_metrics.get("r2") or 0,
                    "best_params": {"type": "stacking", "n_estimators": len(tuned_results)},
                    "feature_importances": importances,
                }
            except Exception as e:
                print(f"warning: stacking eval failed: {e}")

    progress_bar.progress(1.0)
    status.empty()
    st.session_state["tuned_results"]  = tuned_results
    st.session_state["tuning_studies"] = studies
    st.session_state["pipeline_stage"] = "tuning_done"


def run_shap_step(eval_results, artifacts):
    X_train        = st.session_state["X_train"]
    X_test         = st.session_state["X_test"]
    trained_models = st.session_state["trained_models"]
    tuned_results  = st.session_state.get("tuned_results")

    with st.spinner("Computing SHAP values for top 3 models..."):
        try:
            shap_results = run_shap_for_top3(
                eval_results["leaderboard"], trained_models, tuned_results,
                X_train, X_test, artifacts["task_type"]
            )
            st.session_state["shap_results"] = shap_results
        except Exception as e:
            st.warning(f"SHAP analysis failed: {e}")
            st.session_state["shap_results"] = {}

    st.session_state["pipeline_stage"] = "shap_done"


def run_llm_step(profile, eval_results, api_key):
    tuned_results = st.session_state.get("tuned_results")
    shap_results  = st.session_state.get("shap_results")

    with st.spinner("Generating LLM explanation..."):
        try:
            explanation = explain_results(
                profile, eval_results, api_key,
                tuned_results=tuned_results, shap_results=shap_results
            )
            st.session_state["explanation"] = explanation
        except Exception as e:
            logger.warning(f"llm failed: {e}")
            st.session_state["explanation"] = None
            st.error(f"LLM explanation failed: {e}")

    st.session_state["pipeline_stage"] = "complete"


def render_pipeline_tab(api_key):
    if "profile" not in st.session_state:
        st.info("Complete the upload step first.")
        return

    df         = st.session_state["df"]
    target_col = st.session_state["target_col"]
    profile    = st.session_state["profile"]
    stage      = st.session_state.get("pipeline_stage")

    steps_col, _ = st.columns([2, 3])
    with steps_col:
        step_indicator("AutoML", "done" if stage else "active")
        step_indicator("Tune Top 3 (optional)",
                       "done" if stage in ("tuning_done", "shap_done", "complete")
                       else ("active" if stage == "automl_done" else "pending"))
        step_indicator("SHAP Analysis",
                       "done" if stage in ("shap_done", "complete")
                       else ("active" if stage == "tuning_done" else "pending"))
        step_indicator("LLM Analysis",
                       "done" if stage == "complete"
                       else ("active" if stage == "shap_done" else "pending"))

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run AutoML", type="primary", key="automl_btn"):
        try:
            run_automl_step(df, target_col, profile)
            st.rerun()
        except Exception as e:
            logger.error(f"pipeline failed: {e}")
            st.error(f"Pipeline error: {e}")

    if not stage:
        return

    eval_results = st.session_state["eval_results"]
    artifacts    = st.session_state["artifacts"]

    render_leaderboard(eval_results["leaderboard"], artifacts["task_type"])
    render_model_comparison(eval_results["leaderboard"], artifacts["task_type"])
    render_confusion_matrix(eval_results["leaderboard"], artifacts["task_type"])
    render_regression_scatter(eval_results["leaderboard"], artifacts["task_type"])

    if st.session_state.get("tuned_results"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">Tuning Results</p>', unsafe_allow_html=True)
        render_tuning_comparison(eval_results["leaderboard"],
                                 st.session_state["tuned_results"], artifacts["task_type"])

    if st.session_state.get("tuning_studies"):
        render_convergence_plots(st.session_state["tuning_studies"])

    if stage == "automl_done":
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown('<p class="section-heading">Would you like to tune the top 3 models?</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Tuning runs 40 Optuna trials per model and builds a stacking ensemble. Skip to go straight to SHAP.</p>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tune Top 3 Models", use_container_width=True, key="tune_yes"):
                run_tuning_step(eval_results, artifacts)
                st.rerun()
        with col2:
            if st.button("Skip Tuning", use_container_width=True, key="tune_no"):
                st.session_state["pipeline_stage"] = "tuning_done"
                st.rerun()
        return

    if stage == "tuning_done":
        run_shap_step(eval_results, artifacts)
        st.rerun()
        return

    if st.session_state.get("shap_results"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">SHAP Analysis</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Top 3 models. Tuned versions used where available.</p>',
                    unsafe_allow_html=True)
        render_shap_plots(
            st.session_state["shap_results"],
            st.session_state["X_train"],
            st.session_state["trained_models"],
            st.session_state.get("tuned_results"),
            artifacts["task_type"],
        )
        render_per_prediction(
            st.session_state["shap_results"],
            st.session_state["trained_models"],
            st.session_state.get("tuned_results"),
            st.session_state["X_train"],
            artifacts["task_type"],
        )

    if stage == "shap_done":
        if api_key:
            run_llm_step(profile, eval_results, api_key)
            st.rerun()
        else:
            st.warning("No GROQ_API_KEY in .env. Skipping LLM analysis.")
            st.session_state["pipeline_stage"] = "complete"
            st.rerun()
        return


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = get_api_key()
    navbar()
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Profile", "Pipeline", "Explanation", "Download"])

    with tab1:
        df, target_col = render_upload_tab()
        if df is not None and target_col:
            try:
                run_all_validations(df, target_col)
                profile = render_profile_section(df, target_col, api_key)
                st.session_state["df"]         = df
                st.session_state["target_col"] = target_col
                st.session_state["profile"]    = profile
            except ValueError as e:
                st.error(str(e))

    with tab2:
        render_pipeline_tab(api_key)

    with tab3:
        if "eval_results" not in st.session_state:
            st.info("Run the pipeline first.")
        else:
            render_llm_explanation(st.session_state.get("explanation"))
            if "profile" in st.session_state and api_key:
                render_qa_section(
                    st.session_state["profile"],
                    st.session_state["eval_results"],
                    api_key,
                    tuned_results=st.session_state.get("tuned_results"),
                    shap_results=st.session_state.get("shap_results"),
                )

    with tab4:
        if "eval_results" not in st.session_state:
            st.info("Run the pipeline first.")
        else:
            report = build_full_report(
                st.session_state["profile"],
                st.session_state["eval_results"],
                st.session_state.get("explanation"),
            )
            json_str = report_to_json_string(report)
            st.markdown('<p class="section-heading">Download Report</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Full analysis exported as structured JSON.</p>',
                        unsafe_allow_html=True)
            st.download_button(
                label="Download full report as JSON",
                data=json_str,
                file_name="autoinsight_report.json",
                mime="application/json",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-heading">Report Preview</p>', unsafe_allow_html=True)
            st.json(report)


if __name__ == "__main__":
    main()
