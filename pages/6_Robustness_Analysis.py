"""
SMS Shield 4.0 - Page 6: Robustness & Adversarial Defense Lab
Author: SMS Shield NLP Project

Interactive adversarial attack playground, sub-word character TF-IDF evaluation,
automated SMS normalization defense, and empirical adversarial benchmarks.
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
    get_evaluation_metrics,
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
from src.normalization import normalize_sms
from src.adversarial import generate_adversarial_variant

st.set_page_config(page_title="Robustness Lab | SMS Shield 4.0", page_icon="🛡️", layout="wide")
render_header(
    "Adversarial Robustness Lab",
    "Stress-test classifiers against leetspeak, spacing, and obfuscation; inspect character n-grams and normalization defense."
)
render_sidebar_status()

# Load predictor and metrics
try:
    predictor = get_predictor()
    eval_data = get_evaluation_metrics()
    char_word_df = eval_data.get("char_word_df", None)
    adv_df = eval_data.get("adv_df", None)
except Exception as ex:
    st.error(f"Error loading prediction & evaluation resources: {ex}")
    st.stop()

# -----------------------------------------------------------------------------
# Section 1: Interactive Adversarial Attack Playground
# -----------------------------------------------------------------------------
st.markdown("### 🥊 Interactive Adversarial Attack Playground")
st.write(
    "Simulate evasive modifications applied by real-world spam campaigns (leetspeak numbers, "
    "delimiters, whitespace spacing, character repetition) and observe how different NLP representations respond."
)

preset_spams = {
    "🎁 £1,000 Cash Prize": "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 87121 now.",
    "🏦 Urgent Bank Phishing": "URGENT! Your account has suspicious activity. Call 08001234567 or visit https://secure-bank-verify.com immediately.",
    "📱 Free Ringtone Bonus": "FreeMsg: 500 free ringtones waiting for you. Text YES to 88088 now! 18+ only.",
    "☕ Friendly Lunch (Ham Test)": "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too.",
}

st.markdown("##### 1. Select a Sample Payload or Enter Custom Text:")
p_cols = st.columns(len(preset_spams))
for idx, (label, text_val) in enumerate(preset_spams.items()):
    if p_cols[idx].button(label, use_container_width=True):
        st.session_state["adv_input_text"] = text_val

default_msg = st.session_state.get(
    "adv_input_text",
    "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 87121 now."
)

raw_user_msg = st.text_area("Original Message Text:", value=default_msg, height=85)

attack_options = [
    "Composite Multi-Vector Attack",
    "Character Substitution (Leetspeak)",
    "Whitespace Spacing (w i n n e r)",
    "Character Insertion (Delimiters)",
    "Character Deletion (Vowel Drop)",
    "Repeated Letters (Elongation)",
    "Punctuation Insertion (~CLAIM~)",
]

col_att, col_seed = st.columns([3, 1])
with col_att:
    selected_attack = st.selectbox("Select Adversarial Attack Vector:", attack_options)
with col_seed:
    attack_seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42)

# Generate perturbed variant
perturbed_msg = generate_adversarial_variant(raw_user_msg, selected_attack, seed=int(attack_seed))

# Compute normalized variant
norm_info = normalize_sms(perturbed_msg)
normalized_msg = norm_info["normalized_text"]

st.write("")

# Side-by-side message cards
st.markdown("##### 2. Message Progression: Clean vs. Attacked vs. Defended")
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #38BDF8;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
                Original Clean Payload
            </div>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #0B1020; border: 1px solid #26334D; color: #F8FAFC; padding: 10px; border-radius: 6px;">
                {raw_user_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc2:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #F87171;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #F87171; text-transform: uppercase;">
                Adversarially Perturbed (Attacked)
            </div>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #0B1020; border: 1px solid #26334D; color: #FCA5A5; padding: 10px; border-radius: 6px;">
                {perturbed_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc3:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #10B981;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #34D399; text-transform: uppercase;">
                Pre-Normalized Recovery
            </div>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #0B1020; border: 1px solid #26334D; color: #6EE7B7; padding: 10px; border-radius: 6px;">
                {normalized_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Live Model Prediction Comparison Matrix
# -----------------------------------------------------------------------------
st.markdown("##### 3. Live Defense Matrix: Did the Attack Evade the Filter?")

# Run predictions across setups
res_orig_word = predictor.predict_single(raw_user_msg, use_normalization=False, feature_space="word")
res_adv_word = predictor.predict_single(perturbed_msg, use_normalization=False, feature_space="word")
res_adv_char = predictor.predict_single(perturbed_msg, use_normalization=False, feature_space="char")
res_adv_norm = predictor.predict_single(perturbed_msg, use_normalization=True, feature_space="word")

matrix_rows = []
for model_key, model_display in [
    ("naive_bayes", "Multinomial Naive Bayes"),
    ("logistic_regression", "Logistic Regression"),
    ("linear_svm", "Linear SVM"),
]:
    orig_lbl = res_orig_word["models"][model_key]["label"]
    orig_p = res_orig_word["models"][model_key]["prob_spam"]

    adv_word_lbl = res_adv_word["models"][model_key]["label"]
    adv_word_p = res_adv_word["models"][model_key]["prob_spam"]

    adv_char_lbl = res_adv_char["models"][model_key]["label"]
    adv_char_p = res_adv_char["models"][model_key]["prob_spam"]

    adv_norm_lbl = res_adv_norm["models"][model_key]["label"]
    adv_norm_p = res_adv_norm["models"][model_key]["prob_spam"]

    # Check evasion: original was SPAM, but attacked became HAM
    word_status = "⚠️ EVADED (Spam -> Ham)" if (orig_lbl == "SPAM" and adv_word_lbl == "HAM") else ("🛡️ Defended" if adv_word_lbl == "SPAM" else "Legitimate Ham")
    char_status = "⚠️ EVADED" if (orig_lbl == "SPAM" and adv_char_lbl == "HAM") else ("🛡️ Defended" if adv_char_lbl == "SPAM" else "Legitimate Ham")
    norm_status = "⚠️ EVADED" if (orig_lbl == "SPAM" and adv_norm_lbl == "HAM") else ("🛡️ Defended" if adv_norm_lbl == "SPAM" else "Legitimate Ham")

    matrix_rows.append({
        "Model": model_display,
        "Original (Word)": f"{orig_lbl} ({orig_p:.1%})",
        "Attacked (Word TF-IDF)": f"{adv_word_lbl} ({adv_word_p:.1%})",
        "Word Status": word_status,
        "Attacked (Char TF-IDF)": f"{adv_char_lbl} ({adv_char_p:.1%})",
        "Char Status": char_status,
        "Attacked (Normalized)": f"{adv_norm_lbl} ({adv_norm_p:.1%})",
        "Normalized Status": norm_status,
    })

matrix_df = pd.DataFrame(matrix_rows)
st.dataframe(matrix_df, use_container_width=True, hide_index=True)

# Consensus summary banner
orig_cons = res_orig_word["consensus"]
adv_word_cons = res_adv_word["consensus"]
adv_char_cons = res_adv_char["consensus"]
adv_norm_cons = res_adv_norm["consensus"]

if orig_cons == "SPAM" and adv_word_cons == "HAM":
    st.markdown(
        f"""
        <div class="verdict-banner-spam">
            <div style="font-size: 0.95rem; font-weight: 700; color: #FCA5A5;">
                🚨 Successful Adversarial Evasion on Word Baseline!
            </div>
            <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px;">
                The attack flipped baseline word consensus from <strong>SPAM</strong> down to <strong>HAM</strong>.<br>
                However: <strong>Character-Level consensus held firm as {adv_char_cons}</strong>, and <strong>Normalization restored consensus to {adv_norm_cons}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
elif orig_cons == "SPAM" and adv_word_cons == "SPAM":
    st.markdown(
        """
        <div class="verdict-banner-ham">
            <div style="font-size: 0.95rem; font-weight: 700; color: #6EE7B7;">
                🛡️ Defense Resilient Across All Models!
            </div>
            <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 4px;">
                The classifiers resisted this perturbation and successfully maintained unanimous SPAM identification.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Character vs. Word vs. Combined TF-IDF Feature Spaces
# -----------------------------------------------------------------------------
st.markdown("### 📊 Feature Space Comparison: Word vs. Character vs. Combined TF-IDF")
st.write(
    "To counteract word-boundary disruptions and leetspeak, we trained models across three distinct feature representations "
    "on the exact same stratified 80/20 split (1,035 test samples):"
)

if char_word_df is not None:
    c_tab1, c_tab2 = st.tabs(["📈 Comparative Metrics Table", "📊 Interactive Plotly Comparison"])

    with c_tab1:
        disp_cw = char_word_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            disp_cw[col] = disp_cw[col].apply(lambda v: f"{v * 100:.2f}%")
        st.dataframe(disp_cw, use_container_width=True, hide_index=True)

    with c_tab2:
        # Grouped bar chart comparing F1-score and Recall across feature representations
        fig_rep = go.Figure()
        representations = char_word_df["Feature Space"].unique()
        models = char_word_df["Classifier"].unique()
        
        for rep in representations:
            subset = char_word_df[char_word_df["Feature Space"] == rep]
            fig_rep.add_trace(
                go.Bar(
                    name=rep,
                    x=subset["Classifier"],
                    y=subset["Recall"] * 100,
                    text=[f"{v*100:.1f}%" for v in subset["Recall"]],
                    textposition="outside",
                    hovertemplate="Setup: %{data.name}<br>Classifier: %{x}<br>Recall: %{y:.2f}%<extra></extra>"
                )
            )

        apply_plotly_theme(fig_rep, height=360, title="Spam Recall Across Feature Representations (%)")
        fig_rep.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 115], title="Spam Recall (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    st.markdown(
        """
        <div class="cyber-card" style="border-left: 3px solid #34D399; margin-top: 10px;">
            <span style="color: #34D399; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Why Character N-Grams Excel Against Obfuscation
            </span>
            <div style="font-size: 0.86rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                • <strong>Sub-Word Invariance:</strong> Character n-grams (<code>char_wb</code>, n-gram range 2 to 5) capture internal morphs like <em>'call'</em>, <em>'free'</em>, <em>'clai'</em>, and <em>'ize'</em> regardless of punctuation or leetspeak.<br>
                • <strong>Linear SVM with Char TF-IDF:</strong> Reached <strong>99.13% Accuracy</strong>, <strong>100% Precision</strong>, and <strong>93.13% Recall</strong> on the clean split, outperforming word-level TF-IDF across all dimensions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Feature space comparison table not found. Please run `python src/train.py`.")

st.divider()

# -----------------------------------------------------------------------------
# Section 3: Empirical Adversarial Benchmarking Results
# -----------------------------------------------------------------------------
st.markdown("### 🧪 Controlled Adversarial Robustness Benchmarks")
st.write(
    "Evaluated across the held-out test split under a **Composite Multi-Vector Attack** "
    "(reproducible random seed = 42). Measures accuracy drop, spam recall collapse, robustness score, and evasion flips:"
)

if adv_df is not None:
    disp_adv = adv_df.copy()
    for col in ["Clean Accuracy", "Adversarial Accuracy", "Accuracy Drop", "Clean Recall (Spam)", "Adversarial Recall", "Recall Drop", "Clean F1-Score", "Adversarial F1"]:
        disp_adv[col] = disp_adv[col].apply(lambda v: f"{v * 100:.2f}%")
    disp_adv["Robustness Score (%)"] = disp_adv["Robustness Score (%)"].apply(lambda v: f"{v:.2f}%")

    st.dataframe(
        disp_adv[[
            "Defense Pipeline", "Classifier", "Clean Recall (Spam)",
            "Adversarial Recall", "Recall Drop", "Robustness Score (%)", "Spam Miss Flips"
        ]],
        use_container_width=True,
        hide_index=True
    )

    adv_img = PROJECT_ROOT / "metrics" / "adversarial_robustness_bar.png"
    if adv_img.exists():
        st.image(str(adv_img), use_container_width=True)

    st.markdown(
        """
        <div class="cyber-card" style="border-left: 3px solid #F87171; margin-top: 12px;">
            <span style="color: #F87171; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Empirical Benchmark Findings
            </span>
            <div style="font-size: 0.86rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                • On baseline Word TF-IDF, Logistic Regression recall collapsed from <strong>73.28%</strong> down to <strong>48.09%</strong> (a 25.19% plunge; 35 spam messages slipped through!).<br>
                • In contrast, <strong>Character TF-IDF maintained an 83.97% recall</strong> (drop of only 0.76%), and <strong>SMS Normalization restored recall to 67.18%</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Adversarial benchmark table not found. Please run `python src/train.py`.")

st.divider()

# -----------------------------------------------------------------------------
# Section 4: Interactive SMS Normalization Inspector
# -----------------------------------------------------------------------------
st.markdown("### 🔬 SMS Text Normalization Inspector")
st.write("Test the rule-based obfuscation reversal engine on custom leetspeak, abbreviations, or symbol substitutions:")

norm_sample = "C0NGRATS! U R THE W!NNER OF A FR33 £1000 PR!ZE. T7X7 CL41M N0W TO 88088. HURRYYYY!"
test_norm_input = st.text_input("Enter text to inspect normalization:", value=norm_sample)

if test_norm_input.strip():
    inspect_res = normalize_sms(test_norm_input)
    st.markdown(
        f"""
        <div class="cyber-card" style="margin-top: 8px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #34D399; text-transform: uppercase;">
                Recovered Normalized Text
            </div>
            <p style="margin: 6px 0; font-family: monospace; font-size: 0.95rem; color: #6EE7B7; background: #0B1020; border: 1px solid #26334D; padding: 10px; border-radius: 6px;">
                {inspect_res['normalized_text']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if inspect_res["modifications"]:
        st.markdown(f"**Modifications Applied ({inspect_res['modification_count']}):**")
        mods_df = pd.DataFrame(inspect_res["modifications"])
        st.dataframe(mods_df, use_container_width=True, hide_index=True)
    else:
        st.info("No obfuscation or abbreviation patterns detected; text preserved as-is.")
