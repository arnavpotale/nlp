"""
SMS Shield 4.0 — Overview & Quick Detection Hub
Author: SMS Shield NLP Project

Clean, minimalist landing page providing high-level system telemetry,
an interactive single-message inference scanner, and navigation to all six modules.
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="SMS Shield | Overview",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    inject_custom_css,
    render_header,
    render_sidebar_status,
    render_kpi_card,
    get_dataset,
    get_predictor,
    get_evaluation_metrics,
)
from src.ui_theme import PRIMARY_ACCENT, SECONDARY_ACCENT, SUCCESS_GREEN, DANGER_RED

inject_custom_css()
render_header(
    "SMS Shield",
    "NLP-based SMS spam classification and adversarial robustness intelligence platform."
)
render_sidebar_status()

# Load cached dataset and models
try:
    raw_df, clean_df, dup_count = get_dataset()
    predictor = get_predictor()
    eval_data = get_evaluation_metrics()
    metrics_df = eval_data["metrics_df"]
except Exception as ex:
    st.error(f"Error loading system resources: {ex}")
    st.info("Please verify data/SMSSpamCollection exists and run `python src/train.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. System Telemetry KPI Cards
# -----------------------------------------------------------------------------
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100
best_f1_idx = metrics_df["F1-Score"].idxmax()
best_f1_model = metrics_df.loc[best_f1_idx, "Model"]
best_f1_val = metrics_df["F1-Score"].max()

c1, c2, c3, c4 = st.columns(4)
render_kpi_card(c1, "Total Messages", f"{total_clean:,}", f"{dup_count:,} duplicates dropped")
render_kpi_card(c2, "Spam Ratio", f"{spam_pct:.1f}%", f"{spam_count:,} spam messages")
render_kpi_card(c3, "Ham Ratio", f"{100-spam_pct:.1f}%", f"{ham_count:,} legitimate messages")
render_kpi_card(c4, "Peak F1-Score", f"{best_f1_val:.2%}", best_f1_model)

st.write("")

# -----------------------------------------------------------------------------
# 2. Quick Detection Scanner
# -----------------------------------------------------------------------------
st.markdown("#### Quick Message Analysis")
st.caption("Select a sample message or enter your own to evaluate predictions across all three classifiers.")

demo_presets = [
    ("Lottery Scam", "WINNER!! As a valued customer you have won a £1,000 cash prize reward! Call 09061701461 to claim. Valid 12 hours."),
    ("Bank Phish", "URGENT! Your debit card has been suspended due to suspicious activity. Visit https://secure-bank-verify.com to re-activate."),
    ("Team Lunch", "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too."),
    ("Project Sync", "Ok sounds good, I will review the pull request and update the documentation before our sync at 3 PM."),
]

preset_cols = st.columns(4)
for idx, (btn_label, msg_text) in enumerate(demo_presets):
    if preset_cols[idx].button(btn_label, use_container_width=True):
        st.session_state["overview_scanner_input"] = msg_text

default_input_val = st.session_state.get(
    "overview_scanner_input",
    "WINNER!! You have won a £1,000 cash prize! Text CLAIM to 88088 now to collect your reward."
)

quick_input = st.text_input(
    "Message text:",
    value=default_input_val,
    label_visibility="collapsed"
)

if quick_input.strip():
    pred_res = predictor.predict_single(quick_input)
    if pred_res.get("is_valid", False):
        models_data = pred_res["models"]
        is_spam_consensus = (pred_res["consensus"] == "SPAM")

        box_class = "verdict-box-spam" if is_spam_consensus else "verdict-box-ham"
        pill_class = "saas-pill-spam" if is_spam_consensus else "saas-pill-ham"
        verdict_title = "Spam Detected" if is_spam_consensus else "Legitimate Message (Ham)"
        detail_text = (
            f"{pred_res['consensus_detail']} • {pred_res['spam_votes']} of 3 models flagged as spam"
            if is_spam_consensus
            else f"{pred_res['consensus_detail']} • {pred_res['ham_votes']} of 3 models verified as legitimate"
        )

        st.markdown(
            f"""
            <div class="{box_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div style="font-size: 0.76rem; font-weight: 600; text-transform: uppercase; color: {'#F87171' if is_spam_consensus else '#34D399'}; letter-spacing: 0.04em;">
                            Consensus Result
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">
                            {verdict_title}
                        </div>
                        <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px;">
                            {detail_text}
                        </div>
                    </div>
                    <span class="saas-pill {pill_class}">
                        {pred_res['consensus']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3 Model Breakdown Cards
        qc1, qc2, qc3 = st.columns(3)

        def render_compact_model_card(col, m_data):
            with col:
                is_m_spam = (m_data["pred"] == 1)
                st.markdown(
                    f"""
                    <div class="saas-card" style="padding: 14px 16px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.9rem; font-weight: 600; color: #F1F5F9;">{m_data['name']}</span>
                            <span class="saas-pill {'saas-pill-spam' if is_m_spam else 'saas-pill-ham'}" style="font-size: 0.7rem; padding: 2px 6px;">
                                {'Spam' if is_m_spam else 'Ham'}
                            </span>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.8rem; color: #9CA3AF;">
                            Spam probability: <strong style="color: #F1F5F9;">{m_data['prob_spam']:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(float(m_data["prob_spam"]))

        render_compact_model_card(qc1, models_data["naive_bayes"])
        render_compact_model_card(qc2, models_data["logistic_regression"])
        render_compact_model_card(qc3, models_data["linear_svm"])

st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Application Modules Navigation
# -----------------------------------------------------------------------------
st.markdown("#### Application Modules")
st.caption("Explore detailed analytics, linguistic explainability, model diagnostics, and robustness testing.")

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown(
        """
        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                📊 1. Global Dashboard
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Dataset distribution, class imbalance, message character length curves, and comparative model metrics.
            </div>
        </div>

        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                🔍 2. Message Analyzer
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Real-time classification with consensus voting, individual model probabilities, heuristic signals, and token influence.
            </div>
        </div>

        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                🔬 3. NLP Analysis
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Step-by-step 7-stage NLP pipeline trace, suspicious keyword spotter, and dataset-wide n-gram distributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m_col2:
    st.markdown(
        """
        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                📈 4. Model Evaluation
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Held-out test split evaluation (n=1,035), confusion matrices, ROC/PR curves, and 5-fold cross-validation.
            </div>
        </div>

        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                📁 5. Bulk Analysis
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Batch CSV processing, column mapping, consensus statistics, multi-criteria filtering, and report downloads.
            </div>
        </div>

        <div class="saas-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">
                🛡️ 6. Robustness Lab
            </div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                Adversarial attack playground, sub-word character TF-IDF evaluation, and automated SMS normalization recovery.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
