"""
SMS Shield 4.0 - Page 2: Threat Analysis Workspace
Author: SMS Shield NLP Project

Interactive real-time classification interface featuring 3-model consensus,
raw statistical model probabilities, heuristic signals, and token-level TF-IDF weights.
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
    render_kpi_card,
    get_predictor,
)
from src.ui_theme import (
    CARD_BG,
    BORDER_COLOR,
    PRIMARY_VIOLET,
    CYAN_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_MUTED,
    apply_plotly_theme,
)

st.set_page_config(page_title="Threat Analyzer | SMS Shield 4.0", page_icon="🔍", layout="wide")
render_header(
    "Threat Analysis Workspace",
    "Inspect SMS payloads in real time with 3-model consensus voting, probability meters, and token explainability."
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
    "🎁 Lottery Scam (Spam)": "WINNER!! As a valued customer you have won a £1,000 cash prize reward! Call 09061701461 to claim. Valid 12 hours.",
    "🏦 Urgent Bank Phish (Spam)": "URGENT! Your debit card has been suspended due to suspicious activity. Visit https://secure-bank-verify.com to re-activate.",
    "☕ Friendly Lunch (Ham)": "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too.",
    "💬 Casual Chat (Ham)": "Ok lar... Joking wif u oni... I'll see you at home in 20 minutes!",
}

st.markdown("##### ⚡ Quick Threat Presets (Click to Test)")
cols = st.columns(len(presets))
for idx, (label, text_val) in enumerate(presets.items()):
    if cols[idx].button(label, use_container_width=True):
        st.session_state["analyzer_input"] = text_val

default_text = st.session_state.get(
    "analyzer_input",
    "Congratulations! You have won a free prize. Click here to claim your £500 cash reward now."
)

# -----------------------------------------------------------------------------
# 2. Message Input Form
# -----------------------------------------------------------------------------
user_text = st.text_area(
    "SMS Message Payload:",
    value=default_text,
    height=100,
    placeholder="Type or paste any SMS text message here for threat inspection..."
)

analyze_clicked = st.button("🔍 Scan & Inspect Message Payload", type="primary", use_container_width=True)

if not user_text or not user_text.strip():
    st.warning("⚠️ Please enter a non-empty SMS message above.")
    st.stop()

# Run Prediction
result = predictor.predict_single(user_text)

if not result.get("is_valid", False):
    st.warning(f"⚠️ {result.get('error', 'Invalid input')}")
    st.stop()

models_data = result["models"]
heuristics = result["heuristics"]

# -----------------------------------------------------------------------------
# 3. Message Structural Telemetry Row
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
render_kpi_card(m1, "Payload Length", f"{heuristics['char_count']} chars", "Character Count", accent_border=CYAN_ACCENT)
render_kpi_card(m2, "Token Count", f"{heuristics['word_count']} words", "Word Token Units", accent_border=PRIMARY_VIOLET)
render_kpi_card(m3, "Length Profile", heuristics["length_category"], "Density Profile", accent_border="#EC4899")
render_kpi_card(m4, "Heuristic Risk", heuristics["heuristic_risk_level"], "Lexical Signals", accent_border=DANGER_RED if "High" in heuristics["heuristic_risk_level"] else SUCCESS_GREEN)

st.write("")

# -----------------------------------------------------------------------------
# 4. Consensus Verdict Banner
# -----------------------------------------------------------------------------
is_spam_consensus = (result["consensus"] == "SPAM")
banner_class = "verdict-banner-spam" if is_spam_consensus else "verdict-banner-ham"
banner_color = DANGER_RED if is_spam_consensus else SUCCESS_GREEN
banner_icon = "🚨" if is_spam_consensus else "🟢"
banner_title = "CRITICAL THREAT DETECTED: SPAM" if is_spam_consensus else "VERIFIED SECURE: LEGITIMATE HAM"

st.markdown(
    f"""
    <div class="{banner_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: {banner_color};">
                    Ensemble Decision Engine
                </div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                    {banner_icon} {banner_title}
                </div>
                <div style="font-size: 0.9rem; color: #E2E8F0; margin-top: 4px;">
                    Voting Alignment: <strong>{result['consensus_detail']}</strong> | Spam Votes: <strong>{result['spam_votes']}/3</strong> | Ham Votes: <strong>{result['ham_votes']}/3</strong>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-pill {'threat' if is_spam_consensus else 'success'}" style="font-size: 0.9rem; padding: 8px 16px;">
                    {result['consensus']}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 5. Three Individual Model Verdict Cards
# -----------------------------------------------------------------------------
st.markdown("### 🤖 Multi-Algorithm Inference Telemetry")
col_nb, col_lr, col_svc = st.columns(3)

def render_model_telemetry_card(col, model_info, accent_color):
    with col:
        is_spam = (model_info["pred"] == 1)
        st.markdown(
            f"""
            <div class="cyber-card" style="border-top: 3px solid {accent_color}; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">{model_info['name']}</span>
                    <span class="status-pill {'threat' if is_spam else 'success'}" style="font-size: 0.76rem;">
                        {'🚨 SPAM' if is_spam else '🟢 HAM'}
                    </span>
                </div>
                <div style="margin-top: 14px; font-size: 0.86rem; color: #94A3B8;">
                    Spam Probability: <strong style="font-size: 1.1rem; color: {'#F87171' if is_spam else '#F8FAFC'};">{model_info['prob_spam']:.2%}</strong>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-top: 2px;">
                    Ham Probability: {model_info['prob_ham']:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(model_info["prob_spam"]))

render_model_telemetry_card(col_nb, models_data["naive_bayes"], PRIMARY_VIOLET)
render_model_telemetry_card(col_lr, models_data["logistic_regression"], CYAN_ACCENT)
render_model_telemetry_card(col_svc, models_data["linear_svm"], SUCCESS_GREEN)

st.markdown(
    """
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid #26334D; border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; color: #94A3B8; margin-top: 10px;">
        ⚠️ <strong>Statistical Output Probability Notice</strong>: The confidence percentages above represent raw mathematical posteriors/margins derived from training data distributions. They indicate internal model confidence rather than real-world frequency of occurrence.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------------------------------
# 6. Heuristic Signals & Token TF-IDF Feature Contributions
# -----------------------------------------------------------------------------
exp_c1, exp_c2 = st.columns([1, 1.2])

with exp_c1:
    st.markdown("### 🚩 Heuristic Indicator Scan")
    st.caption("Deterministic lexical patterns commonly exploited in SMS phishing campaigns.")

    signals_found = []
    if heuristics["suspicious_keywords"]:
        signals_found.append(f"**Promotional Keywords:** `{', '.join(heuristics['suspicious_keywords'])}`")
    if heuristics["has_url"]:
        signals_found.append(f"**Web Link / URL Detected:** `{', '.join(heuristics['urls_found'])}`")
    if heuristics["has_phone_number"]:
        signals_found.append(f"**Phone / Premium Shortcode:** `{', '.join(heuristics['phones_found'])}`")
    if heuristics["has_currency"]:
        signals_found.append(f"**Currency Symbols:** `{', '.join(heuristics['currencies_found'])}`")
    if heuristics["uppercase_words"]:
        signals_found.append(f"**High-Urgency ALL-CAPS Words:** `{', '.join(heuristics['uppercase_words'])}`")

    if signals_found:
        for s in signals_found:
            st.markdown(
                f"""
                <div style="background: #151E33; border: 1px solid #26334D; border-left: 3px solid #F59E0B; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.88rem;">
                    {s}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style="background: #151E33; border: 1px solid #26334D; border-left: 3px solid #10B981; border-radius: 8px; padding: 12px 14px;">
                <span style="color: #34D399; font-weight: 600;">✅ Clean Payload</span><br>
                <span style="color: #94A3B8; font-size: 0.85rem;">No predatory shortcodes, high-pressure urgency words, or suspicious hyperlinks detected.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

with exp_c2:
    st.markdown("### 🔬 Token TF-IDF Feature Influence")
    st.caption("Active vocabulary tokens and their learned weights pushing classification toward Spam or Ham.")
    feat_df = result.get("feature_explanation", pd.DataFrame())
    if not feat_df.empty:
        # Create interactive horizontal bar chart for top token contributions
        top_tokens = feat_df.head(8).copy()
        
        fig_tokens = go.Figure()
        fig_tokens.add_trace(
            go.Bar(
                y=top_tokens["Token"],
                x=top_tokens["LR Weight"],
                orientation="h",
                marker_color=[DANGER_RED if w > 0 else CYAN_ACCENT for w in top_tokens["LR Weight"]],
                hovertemplate="Token: <b>%{y}</b><br>Weight: %{x:.3f}<extra></extra>"
            )
        )
        apply_plotly_theme(fig_tokens, height=260)
        fig_tokens.update_layout(
            margin=dict(l=60, r=20, t=20, b=20),
            xaxis=dict(title="Logistic Regression Weight (Positive = Spam, Negative = Ham)"),
            yaxis=dict(title="", autorange="reversed")
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

        st.dataframe(
            feat_df[["Token", "TF-IDF Value", "LR Weight", "SVM Weight", "Direction"]].head(8),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No vocabulary tokens remained after punctuation and stopword filtering.")
