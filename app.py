"""
SMS Shield 4.0 — AI-Powered SMS Threat Detection & Intelligence Platform
Main Application Entrypoint & Executive Cyber Command Center
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="SMS Shield 4.0 | Threat Intelligence Platform",
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
    render_kpi_card,
    get_dataset,
    get_predictor,
    get_evaluation_metrics,
)
from src.ui_theme import PRIMARY_VIOLET, CYAN_ACCENT, SUCCESS_GREEN, DANGER_RED

# Instate cyber styling and persistent status
inject_custom_css()
render_header(
    "Executive Cyber Portal",
    "Enterprise-grade NLP SMS threat detection ensemble with character n-grams and adversarial evasion defense."
)
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
# 1. Executive Telemetry Row (5 KPI Cards)
# -----------------------------------------------------------------------------
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100
best_f1_idx = metrics_df["F1-Score"].idxmax()
best_f1_model = metrics_df.loc[best_f1_idx, "Model"]
best_f1_val = metrics_df["F1-Score"].max()

q1, q2, q3, q4, q5 = st.columns(5)
render_kpi_card(q1, "Clean Repository", f"{total_clean:,}", f"{dup_count:,} duplicates filtered", accent_border=PRIMARY_VIOLET)
render_kpi_card(q2, "Threat Ratio (Spam)", f"{spam_pct:.1f}%", f"{spam_count:,} spam messages", accent_border=DANGER_RED)
render_kpi_card(q3, "Clean Baseline (Ham)", f"{100-spam_pct:.1f}%", f"{ham_count:,} verified ham", accent_border=SUCCESS_GREEN)
render_kpi_card(q4, "Active Classifiers", "3 Models", "NB, LogReg, Linear SVM", accent_border=CYAN_ACCENT)
render_kpi_card(q5, "Peak Defense F1", f"{best_f1_val:.2%}", best_f1_model, accent_border="#EC4899")

st.write("")

# -----------------------------------------------------------------------------
# 2. Live Threat Scanner Console
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
        <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">
            ⚡ Live Threat Scanner Console
        </h3>
        <span class="status-pill active">
            Real-Time Inference Engine
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

preset_cols = st.columns(4)
demo_presets = [
    ("🎁 Lottery Scam", "WINNER!! As a valued customer you have won a £1,000 cash prize reward! Call 09061701461 to claim. Valid 12 hours."),
    ("🏦 Bank Phish", "URGENT! Your debit card has been suspended due to suspicious activity. Visit https://secure-bank-verify.com to re-activate."),
    ("☕ Team Lunch", "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too."),
    ("📅 Project Sync", "Ok sounds good, I'll review the pull request and update the documentation before our sync at 3 PM."),
]

for idx, (btn_label, msg_text) in enumerate(demo_presets):
    if preset_cols[idx].button(btn_label, use_container_width=True):
        st.session_state["portal_scanner_input"] = msg_text

default_scanner_val = st.session_state.get(
    "portal_scanner_input",
    "WINNER!! You have won a £1,000 cash prize! Text CLAIM to 88088 now to collect your reward."
)

quick_input = st.text_input(
    "Enter an SMS message to test instantaneous consensus classification:",
    value=default_scanner_val
)

if quick_input.strip():
    pred_res = predictor.predict_single(quick_input)
    if pred_res.get("is_valid", False):
        models_data = pred_res["models"]
        is_spam_consensus = (pred_res["consensus"] == "SPAM")

        # Animated Verdict Banner
        banner_class = "verdict-banner-spam" if is_spam_consensus else "verdict-banner-ham"
        banner_color = DANGER_RED if is_spam_consensus else SUCCESS_GREEN
        banner_icon = "🚨" if is_spam_consensus else "🟢"
        banner_title = "CRITICAL THREAT DETECTED: SPAM" if is_spam_consensus else "VERIFIED SECURE: LEGITIMATE HAM"

        st.markdown(
            f"""
            <div class="{banner_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: {banner_color};">
                            Three-Model Ensemble Consensus
                        </div>
                        <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                            {banner_icon} {banner_title}
                        </div>
                        <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 2px;">
                            Consensus Agreement: <strong>{pred_res['consensus_detail']}</strong> (Spam Votes: {pred_res['spam_votes']}/3)
                        </div>
                    </div>
                    <div>
                        <span class="status-pill {'threat' if is_spam_consensus else 'success'}" style="font-size: 0.85rem; padding: 6px 14px;">
                            {pred_res['consensus']}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3 Model Telemetry Cards
        qc1, qc2, qc3 = st.columns(3)

        def render_portal_model_box(col, m_data, accent_color):
            with col:
                is_m_spam = (m_data["pred"] == 1)
                st.markdown(
                    f"""
                    <div class="cyber-card" style="border-left: 4px solid {accent_color}; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC;">{m_data['name']}</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: {'#F87171' if is_m_spam else '#34D399'};">
                                {'🚨 SPAM' if is_m_spam else '🟢 HAM'}
                            </span>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.82rem; color: #94A3B8;">
                            Spam Probability: <strong style="color: {'#F87171' if is_m_spam else '#F8FAFC'};">{m_data['prob_spam']:.2%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(float(m_data["prob_spam"]))

        render_portal_model_box(qc1, models_data["naive_bayes"], PRIMARY_VIOLET)
        render_portal_model_box(qc2, models_data["logistic_regression"], CYAN_ACCENT)
        render_portal_model_box(qc3, models_data["linear_svm"], SUCCESS_GREEN)

st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Platform Module Directory Cards
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">
            🧭 Platform Intelligence Modules
        </h3>
        <span style="font-size: 0.85rem; color: #94A3B8;">
            Select a module from the sidebar navigation
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #38BDF8; font-size: 1.1rem;">📊 1. Global System Dashboard</h4>
                <span class="status-pill">Telemetry</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Executive visual analytics of dataset distributions, class imbalance, message character length curves,
                and high-level model comparison benchmarks across Naive Bayes, Logistic Regression, and Linear SVM.
            </p>
        </div>

        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #8B5CF6; font-size: 1.1rem;">🔍 2. Interactive Message Analyzer</h4>
                <span class="status-pill active">Inference</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Real-time threat inspection workspace with 3-model consensus voting, raw output probabilities,
                heuristic threat indicators (URLs, shortcodes, urgency), and token-level TF-IDF explainability.
            </p>
        </div>

        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #34D399; font-size: 1.1rem;">🔬 3. NLP Intelligence & N-Grams</h4>
                <span class="status-pill">Linguistics</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Interactive educational trace through all 7 pipeline stages, suspicious promotional term forensics,
                and comparative unigram, bigram, and trigram dataset distributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    st.markdown(
        """
        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #F59E0B; font-size: 1.1rem;">📈 4. Model Evaluation & ROC/PR Lab</h4>
                <span class="status-pill">Evaluation</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Rigorous statistical assessment on the held-out test split (1,035 messages) with side-by-side
                Confusion Matrices, ROC curves, Precision-Recall curves, 5-Fold Cross-Validation, and n-gram experiments.
            </p>
        </div>

        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #EC4899; font-size: 1.1rem;">📁 5. Bulk SMS Analysis & Reports</h4>
                <span class="status-pill">Batch Scans</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                High-throughput batch CSV processing engine with intelligent column mapping, consensus voting,
                interactive multi-criteria filtering, and one-click downloadable audit reports.
            </p>
        </div>

        <div class="cyber-card" style="border-left: 4px solid #F87171;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #F87171; font-size: 1.1rem;">🛡️ 6. Robustness & Adversarial Defense</h4>
                <span class="status-pill threat">Defense Lab</span>
            </div>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Adversarial attack playground testing 7 evasion vectors (leetspeak, spacing, delimiters),
                sub-word character TF-IDF evaluation, and automated rule-based SMS normalization recovery.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
