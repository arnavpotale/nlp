"""
SMS Shield 2.0 - Page 2: Message Analyzer
Author: SMS Shield NLP Project

Interactive real-time classification interface with 3-model predictions,
confidence probabilities, consensus voting, heuristic signals, and token contributions.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_predictor


st.set_page_config(page_title="Message Analyzer | SMS Shield 2.0", page_icon="🔍", layout="wide")
render_header("Page 2: Interactive Message Analyzer", "Analyze SMS messages with 3-model consensus, probability scores, and heuristic signals.")
render_sidebar_status()

# Load unified predictor
try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Quick Presets
# -----------------------------------------------------------------------------
presets = {
    "🎁 Lottery Scam (Spam)": "WINNER!! As a valued customer you have been selected for a £1,000 cash prize reward! Call 09061701461 to claim. Valid 12 hours.",
    "🏦 Urgent Bank Phish (Spam)": "URGENT! Your debit card has been suspended due to suspicious activity. Visit https://secure-bank-verify.com to re-activate.",
    "☕ Friendly Lunch (Ham)": "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too.",
    "💬 Casual Chat (Ham)": "Ok lar... Joking wif u oni... I'll see you at home in 20 minutes!",
}

st.markdown("##### ⚡ Quick Test Presets (Click to Load)")
cols = st.columns(len(presets))
for idx, (label, text_val) in enumerate(presets.items()):
    if cols[idx].button(label, use_container_width=True):
        st.session_state["analyzer_input"] = text_val

default_text = st.session_state.get(
    "analyzer_input",
    "Congratulations! You have won a free prize. Click here to claim your £500 cash reward now."
)

# -----------------------------------------------------------------------------
# Message Input Form
# -----------------------------------------------------------------------------
user_text = st.text_area(
    "Enter SMS message text to inspect:",
    value=default_text,
    height=110,
    placeholder="Type or paste any SMS message here..."
)

analyze_clicked = st.button("🔍 Analyze Message Now", type="primary", use_container_width=True)

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
# Metadata Metrics Row
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Character Count", f"{heuristics['char_count']} chars")
with m2:
    st.metric("Word Count", f"{heuristics['word_count']} words")
with m3:
    st.metric("Length Profile", heuristics["length_category"])
with m4:
    st.metric("Heuristic Alert Level", heuristics["heuristic_risk_level"])

st.write("")

# -----------------------------------------------------------------------------
# Consensus Banner
# -----------------------------------------------------------------------------
is_spam_consensus = (result["consensus"] == "SPAM")
consensus_class = "verdict-spam" if is_spam_consensus else "verdict-ham"
consensus_icon = "🚨" if is_spam_consensus else "🟢"

st.markdown(
    f"""
    <div style="background-color: {'#fef2f2' if is_spam_consensus else '#f0fdf4'}; border: 2px solid {'#ef4444' if is_spam_consensus else '#22c55e'}; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: {'#991b1b' if is_spam_consensus else '#166534'}; font-weight: 700;">Three-Model Final Consensus</span>
                <h2 style="margin: 4px 0; color: {'#7f1d1d' if is_spam_consensus else '#14532d'}; font-size: 1.8rem; font-weight: 800;">
                    {consensus_icon} {result['consensus']} ({result['consensus_detail']})
                </h2>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.95rem; font-weight: 600; color: #475569;">Spam Votes: <strong>{result['spam_votes']}/3</strong></span><br>
                <span style="font-size: 0.95rem; font-weight: 600; color: #475569;">Ham Votes: <strong>{result['ham_votes']}/3</strong></span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Three Individual Model Cards
# -----------------------------------------------------------------------------
st.markdown("### 🤖 Individual Model Verdicts & Output Probabilities")
col_nb, col_lr, col_svc = st.columns(3)

def render_model_card(col, model_info, border_color):
    with col:
        is_spam = (model_info["pred"] == 1)
        badge_style = "verdict-spam" if is_spam else "verdict-ham"
        badge_icon = "🚨" if is_spam else "🟢"
        
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 2px solid {border_color}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); margin-bottom: 12px;">
                <h4 style="margin: 0 0 10px 0; color: #0f172a; font-size: 1.05rem;">{model_info['name']}</h4>
                <div class="{badge_style}" style="margin-bottom: 14px;">
                    {badge_icon} {model_info['label']}
                </div>
                <p style="margin: 4px 0; font-size: 0.9rem; color: #334155;">
                    <strong>Spam Probability:</strong> <span style="font-weight: 700; color: {'#dc2626' if is_spam else '#0f172a'};">{model_info['prob_spam']:.2%}</span>
                </p>
                <p style="margin: 4px 0; font-size: 0.9rem; color: #334155;">
                    <strong>Ham Probability:</strong> {model_info['prob_ham']:.2%}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(model_info["prob_spam"]), text=f"Spam Probability: {model_info['prob_spam']:.1%}")

render_model_card(col_nb, models_data["naive_bayes"], "#2563eb")
render_model_card(col_lr, models_data["logistic_regression"], "#059669")
render_model_card(col_svc, models_data["linear_svm"], "#7c3aed")

st.info(
    "⚠️ **Confidence Probability Notice**: The percentages above are **raw statistical model output probabilities** "
    "calculated directly from TF-IDF feature distributions and training priors. They are **not calibrated real-world risk scores**."
)

st.divider()

# -----------------------------------------------------------------------------
# Explainable Indicators & Feature Contributions
# -----------------------------------------------------------------------------
exp_c1, exp_c2 = st.columns([1, 1.2])

with exp_c1:
    st.markdown("### 🚩 Heuristic Indicator Signals")
    st.caption("Deterministic lexical & structural patterns commonly found in SMS spam.")

    signals_found = []
    if heuristics["suspicious_keywords"]:
        signals_found.append(f"**Promotional Keywords:** `{', '.join(heuristics['suspicious_keywords'])}`")
    if heuristics["has_url"]:
        signals_found.append(f"**Web Link / URL Detected:** `{', '.join(heuristics['urls_found'])}`")
    if heuristics["has_phone_number"]:
        signals_found.append(f"**Phone Number / Shortcode:** `{', '.join(heuristics['phones_found'])}`")
    if heuristics["has_currency"]:
        signals_found.append(f"**Currency Symbols:** `{', '.join(heuristics['currencies_found'])}`")
    if heuristics["uppercase_words"]:
        signals_found.append(f"**ALL-CAPS Words:** `{', '.join(heuristics['uppercase_words'])}`")

    if signals_found:
        for s in signals_found:
            st.markdown(f"- {s}")
    else:
        st.success("✅ No suspicious heuristic patterns or phone shortcodes detected.")

    st.caption("ℹ️ *Heuristic signals are illustrative indicators, not mathematical proof of malicious intent.*")

with exp_c2:
    st.markdown("### 🔬 Token TF-IDF Feature Contributions")
    feat_df = result.get("feature_explanation", pd.DataFrame())
    if not feat_df.empty:
        st.dataframe(
            feat_df[["Token", "TF-IDF Value", "LR Weight", "SVM Weight", "Direction"]].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No vocabulary tokens remained after punctuation and stopword removal.")
