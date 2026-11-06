"""
SMS Shield 2.0 — Intelligent SMS Spam Detection & NLP Analytics Platform
Main Application Entrypoint & Welcome Hub
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="SMS Shield 2.0 | NLP Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    inject_custom_css,
    render_header,
    render_sidebar_status,
    get_dataset,
    get_predictor,
    get_evaluation_metrics,
)

inject_custom_css()
render_header("Welcome Hub & Platform Overview", "Production-grade NLP pipeline, three-model consensus, character n-grams, and adversarial robustness defense.")
render_sidebar_status()

# Load cached system resources
try:
    raw_df, clean_df, dup_count = get_dataset()
    predictor = get_predictor()
    eval_data = get_evaluation_metrics()
    metrics_df = eval_data["metrics_df"]
except Exception as ex:
    st.error(f"Error initializing system resources: {ex}")
    st.info("Please verify the dataset exists in data/SMSSpamCollection and run `python src/train.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# Quick Telemetry Cards
# -----------------------------------------------------------------------------
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100
best_f1_model = metrics_df.loc[metrics_df["F1-Score"].idxmax(), "Model"]
best_f1_val = metrics_df["F1-Score"].max()

q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    st.metric("Unique Messages", f"{total_clean:,}", f"{dup_count:,} duplicates dropped")
with q2:
    st.metric("Spam Prevalence", f"{spam_pct:.1f}%", f"{spam_count:,} spam messages")
with q3:
    st.metric("Legitimate (Ham)", f"{100-spam_pct:.1f}%", f"{ham_count:,} ham messages")
with q4:
    st.metric("Active ML Models", "3 Models", "NB, LR, Linear SVM")
with q5:
    st.metric("Peak F1-Score", f"{best_f1_val:.2%}", best_f1_model)

st.divider()

# -----------------------------------------------------------------------------
# Quick Try-It-Now Section
# -----------------------------------------------------------------------------
st.markdown("### ⚡ Quick Live SMS Spam Detection")
st.write("Test the three-model consensus engine directly from this welcome hub, or navigate using the sidebar for in-depth analysis.")

quick_input = st.text_input(
    "Enter an SMS message to test:",
    value="WINNER!! You have won a £1,000 cash prize! Text CLAIM to 88088 now."
)

if quick_input.strip():
    pred_res = predictor.predict_single(quick_input)
    if pred_res.get("is_valid", False):
        models_data = pred_res["models"]
        is_spam_consensus = (pred_res["consensus"] == "SPAM")

        st.markdown(
            f"""
            <div style="background-color: {'#fef2f2' if is_spam_consensus else '#f0fdf4'}; border: 2px solid {'#ef4444' if is_spam_consensus else '#22c55e'}; border-radius: 10px; padding: 14px 20px; margin: 12px 0;">
                <span style="font-size: 0.85rem; font-weight: 700; color: {'#991b1b' if is_spam_consensus else '#166534'};">CONSENSUS VERDICT:</span>
                <span style="font-size: 1.3rem; font-weight: 800; color: {'#7f1d1d' if is_spam_consensus else '#14532d'}; margin-left: 10px;">
                    {'🚨 SPAM' if is_spam_consensus else '🟢 HAM'} ({pred_res['consensus_detail']})
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            nb = models_data["naive_bayes"]
            st.markdown(f"**Naive Bayes:** {'🚨 SPAM' if nb['pred']==1 else '🟢 HAM'} (Prob: `{nb['prob_spam']:.1%}`)")
        with qc2:
            lr = models_data["logistic_regression"]
            st.markdown(f"**Logistic Regression:** {'🚨 SPAM' if lr['pred']==1 else '🟢 HAM'} (Prob: `{lr['prob_spam']:.1%}`)")
        with qc3:
            svc = models_data["linear_svm"]
            st.markdown(f"**Linear SVM:** {'🚨 SPAM' if svc['pred']==1 else '🟢 HAM'} (Prob: `{svc['prob_spam']:.1%}`)")

st.divider()

# -----------------------------------------------------------------------------
# Application Module Directory Cards
# -----------------------------------------------------------------------------
st.markdown("### 🧭 Platform Navigation & Architecture")
st.write("Explore the five specialized modules accessible via the sidebar navigation:")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="cyber-card">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">📊 1. Global Dashboard</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                Visualizes dataset distributions, class imbalance, message character length curves,
                and high-level metric summaries across all three models.
            </p>
        </div>
        <div class="cyber-card">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">🔍 2. Message Analyzer</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                Interactive single-SMS classification interface with preset sample scenarios,
                multi-model consensus, probability bars, heuristic signal detection, and token contributions.
            </p>
        </div>
        <div class="cyber-card">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">🔬 3. NLP Analysis & N-Grams</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                Step-by-step 7-stage NLP pipeline demonstration, vocabulary frequency distributions,
                and comparative unigram, bigram, and trigram analyses.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    st.markdown(
        """
        <div class="cyber-card">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">📈 4. Model Evaluation & ROC/PR Curves</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                In-depth comparative analysis with side-by-side Confusion Matrices, ROC Curves,
                Precision-Recall Curves, 5-Fold Cross-Validation, and TF-IDF n-gram experiments.
            </p>
        </div>
        <div class="cyber-card">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">📁 5. Bulk SMS Analysis & Reports</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                Batch CSV processing engine with column mapping, agreement breakdown,
                interactive filtering, and one-click downloadable classification reports.
            </p>
        </div>
        <div class="cyber-card" style="border-left: 4px solid #f43f5e;">
            <h4 style="margin: 0 0 6px 0; color: #9f1239;">🛡️ 6. Robustness & Adversarial Defense</h4>
            <p style="margin: 0; color: #475569; font-size: 0.92rem;">
                Adversarial attack playground (leetspeak, spacing, delimiters), character-level
                TF-IDF evaluation, SMS text normalization inspector, and empirical robustness benchmarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
