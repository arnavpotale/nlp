"""
SMS Shield 4.0 - Page 2: Message Analyzer
Author: SMS Shield NLP Project

Polished real-time classification interface with 3-model consensus,
side-by-side probability indicators, heuristic pattern analysis, and token influence.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    render_header,
    render_sidebar_status,
    get_predictor,
)
from src.ui_theme import (
    CARD_BG,
    BORDER_COLOR,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_plotly_theme,
)

st.set_page_config(page_title="SMS Shield | Message Analyzer", page_icon="🔍", layout="wide")
render_header(
    "Message Analyzer",
    "Real-time SMS classification with multi-model consensus, probability indicators, and token influence."
)
render_sidebar_status()

# Load unified predictor
try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Quick Presets
# -----------------------------------------------------------------------------
presets = {
    "Lottery Scam": "WINNER!! As a valued customer you have won a £1,000 cash prize reward! Call 09061701461 to claim. Valid 12 hours.",
    "Bank Phish": "URGENT! Your debit card has been suspended due to suspicious activity. Visit https://secure-bank-verify.com to re-activate.",
    "Friendly Lunch": "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too.",
    "Casual Chat": "Ok lar... Joking wif u oni... I will see you at home in 20 minutes!",
}

st.caption("Load a test preset:")
p_cols = st.columns(len(presets))
for idx, (label, text_val) in enumerate(presets.items()):
    if p_cols[idx].button(label, use_container_width=True):
        st.session_state["analyzer_input"] = text_val

default_text = st.session_state.get(
    "analyzer_input",
    "Congratulations! You have won a free prize. Click here to claim your £500 cash reward now."
)

# -----------------------------------------------------------------------------
# 2. Input Message Form
# -----------------------------------------------------------------------------
user_text = st.text_area(
    "Message text:",
    value=default_text,
    height=90,
    placeholder="Type or paste any SMS message to analyze..."
)

analyze_clicked = st.button("Analyze Message", type="primary", use_container_width=True)

if not user_text or not user_text.strip():
    st.warning("Please enter an SMS message to analyze.")
    st.stop()

# Run Prediction
result = predictor.predict_single(user_text)

if not result.get("is_valid", False):
    st.warning(f"{result.get('error', 'Invalid input')}")
    st.stop()

models_data = result["models"]
heuristics = result["heuristics"]

# -----------------------------------------------------------------------------
# 3. Main Result Panel
# -----------------------------------------------------------------------------
is_spam = (result["consensus"] == "SPAM")
box_class = "verdict-box-spam" if is_spam else "verdict-box-ham"
pill_class = "saas-pill-spam" if is_spam else "saas-pill-ham"
verdict_title = "Spam Detected" if is_spam else "Legitimate Message (Ham)"
detail_text = (
    f"{result['consensus_detail']} • {result['spam_votes']} of 3 models flagged as spam"
    if is_spam
    else f"{result['consensus_detail']} • {result['ham_votes']} of 3 models verified as legitimate"
)

st.markdown(
    f"""
    <div class="{box_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: {'#F87171' if is_spam else '#34D399'}; letter-spacing: 0.04em;">
                    Consensus Verdict
                </div>
                <div style="font-size: 1.35rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">
                    {verdict_title}
                </div>
                <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 4px;">
                    {detail_text}
                </div>
            </div>
            <span class="saas-pill {pill_class}">
                {result['consensus']}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 4. Side-by-Side Model Comparison
# -----------------------------------------------------------------------------
st.markdown("#### Model Breakdown")
col_nb, col_lr, col_svc = st.columns(3)

def render_model_box(col, m_info):
    with col:
        m_is_spam = (m_info["pred"] == 1)
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.92rem; font-weight: 600; color: #F1F5F9;">{m_info['name']}</span>
                    <span class="saas-pill {'saas-pill-spam' if m_is_spam else 'saas-pill-ham'}" style="font-size: 0.72rem;">
                        {'Spam' if m_is_spam else 'Ham'}
                    </span>
                </div>
                <div style="margin-top: 14px; font-size: 0.82rem; color: #9CA3AF;">
                    Spam Probability: <strong style="color: {'#F87171' if m_is_spam else '#F1F5F9'}; font-size: 0.95rem;">{m_info['prob_spam']:.1%}</strong>
                </div>
                <div style="font-size: 0.78rem; color: #6B7280; margin-top: 2px;">
                    Ham Probability: {m_info['prob_ham']:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(m_info["prob_spam"]))

render_model_box(col_nb, models_data["naive_bayes"])
render_model_box(col_lr, models_data["logistic_regression"])
render_model_box(col_svc, models_data["linear_svm"])

st.caption("Note: Percentages reflect raw statistical model probabilities derived from training priors, not calibrated real-world risk scores.")

st.write("")

# -----------------------------------------------------------------------------
# 5. Heuristic Indicators & Token Influence
# -----------------------------------------------------------------------------
exp_c1, exp_c2 = st.columns([1, 1.2])

with exp_c1:
    st.markdown("#### Pattern Indicators")
    st.caption("Rule-based checks for common SMS spam patterns.")

    signals = []
    if heuristics["suspicious_keywords"]:
        signals.append(("Keywords", ", ".join(heuristics["suspicious_keywords"])))
    if heuristics["has_url"]:
        signals.append(("Web Links", ", ".join(heuristics["urls_found"])))
    if heuristics["has_phone_number"]:
        signals.append(("Phone Numbers", ", ".join(heuristics["phones_found"])))
    if heuristics["has_currency"]:
        signals.append(("Currency", ", ".join(heuristics["currencies_found"])))
    if heuristics["uppercase_words"]:
        signals.append(("ALL-CAPS", ", ".join(heuristics["uppercase_words"])))

    if signals:
        for cat, val in signals:
            st.markdown(
                f"""
                <div class="saas-card" style="padding: 10px 14px; margin-bottom: 8px;">
                    <div style="font-size: 0.74rem; font-weight: 500; text-transform: uppercase; color: #9CA3AF;">{cat}</div>
                    <div style="font-size: 0.86rem; color: #F1F5F9; margin-top: 2px; font-family: monospace;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="saas-card" style="padding: 14px 16px;">
                <span style="color: #34D399; font-size: 0.88rem; font-weight: 500;">✓ Clean Message</span>
                <div style="color: #9CA3AF; font-size: 0.82rem; margin-top: 2px;">
                    No suspicious keywords, URLs, phone shortcodes, or urgency indicators detected.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with exp_c2:
    st.markdown("#### Token Influence")
    st.caption("Vocabulary tokens that contributed most toward Spam or Ham.")
    feat_df = result.get("feature_explanation", pd.DataFrame())
    
    if not feat_df.empty:
        top_tokens = feat_df.head(6).copy()
        
        fig_tokens = go.Figure()
        fig_tokens.add_trace(
            go.Bar(
                y=top_tokens["Token"],
                x=top_tokens["LR Weight"],
                orientation="h",
                marker_color=[DANGER_RED if w > 0 else SECONDARY_ACCENT for w in top_tokens["LR Weight"]],
                hovertemplate="Token: <b>%{y}</b><br>Weight: %{x:.3f}<extra></extra>"
            )
        )
        apply_plotly_theme(fig_tokens, height=220)
        fig_tokens.update_layout(
            margin=dict(l=60, r=15, t=10, b=25),
            xaxis=dict(title="Weight (Positive = Spam, Negative = Ham)"),
            yaxis=dict(title="", autorange="reversed")
        )
        st.plotly_chart(fig_tokens, use_container_width=True)
    else:
        st.info("No active vocabulary tokens remained after punctuation and stopword removal.")
