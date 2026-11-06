"""
SMS Shield 3.0 - Page 6: Robustness Analysis & Adversarial Defense
Author: SMS Shield NLP Project

Interactive adversarial attack playground, sub-word character TF-IDF evaluation,
SMS text normalization inspector, and empirical robustness benchmarks.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_predictor, get_evaluation_metrics
from src.normalization import normalize_sms
from src.adversarial import generate_adversarial_variant


st.set_page_config(page_title="Robustness Analysis | SMS Shield 3.0", page_icon="🛡️", layout="wide")
render_header("Page 6: Adversarial Robustness & Evasion Defense", "Stress-test classifiers against leetspeak, spacing, and obfuscation; inspect character n-grams and normalization defense.")
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

st.markdown("##### 1. Select a Sample Message or Enter Custom Text:")
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
st.markdown("##### 2. Message Comparison (Original vs. Attacked vs. Normalized)")
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #2563eb;">
            <strong style="color: #1e3a8a;">Original Clean Message</strong>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #f8fafc; padding: 10px; border-radius: 6px;">
                {raw_user_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc2:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #dc2626;">
            <strong style="color: #991b1b;">Adversarially Perturbed (Attacked)</strong>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #fef2f2; padding: 10px; border-radius: 6px;">
                {perturbed_msg}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc3:
    st.markdown(
        f"""
        <div class="cyber-card" style="border-left: 4px solid #059669;">
            <strong style="color: #065f46;">Pre-Normalized Recovery</strong>
            <p style="margin: 8px 0; font-family: monospace; font-size: 0.88rem; background: #f0fdf4; padding: 10px; border-radius: 6px;">
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
    word_status = "⚠️ EVADED (Spam -> Ham)" if (orig_lbl == "SPAM" and adv_word_lbl == "HAM") else ("✅ Defended" if adv_word_lbl == "SPAM" else "Legitimate Ham")
    char_status = "⚠️ EVADED" if (orig_lbl == "SPAM" and adv_char_lbl == "HAM") else ("✅ Defended" if adv_char_lbl == "SPAM" else "Legitimate Ham")
    norm_status = "⚠️ EVADED" if (orig_lbl == "SPAM" and adv_norm_lbl == "HAM") else ("✅ Defended" if adv_norm_lbl == "SPAM" else "Legitimate Ham")

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
    st.error(
        f"🚨 **Successful Adversarial Evasion on Word Baseline!** The attack flipped consensus from **SPAM** down to **HAM**. "
        f"However: Character-Level consensus held as **{adv_char_cons}**, and Normalization restored consensus to **{adv_norm_cons}**."
    )
elif orig_cons == "SPAM" and adv_word_cons == "SPAM":
    st.success("🛡️ **Defense Resilient!** The models resisted this perturbation and maintained unanimous SPAM classification.")

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
    c_tab1, c_tab2 = st.tabs(["📈 Comparative Metrics Table", "📊 Visualization Chart"])

    with c_tab1:
        disp_cw = char_word_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            disp_cw[col] = disp_cw[col].apply(lambda v: f"{v * 100:.2f}%")
        st.dataframe(disp_cw, use_container_width=True, hide_index=True)

    with c_tab2:
        cw_img = PROJECT_ROOT / "metrics" / "char_word_comparison_bar.png"
        if cw_img.exists():
            st.image(str(cw_img), use_container_width=True)

    st.markdown("##### 💡 Technical Insights on Sub-Word Character N-Grams:")
    st.markdown(
        """
        - **Why Character N-Grams Excel:** Character n-grams (`char_wb`, range 2 to 5) capture sub-word fragments (e.g. `'call'`, `'free'`, `'clai'`, `'ize'`) regardless of token boundaries, typos, or leetspeak substitutions.
        - **Linear SVM with Char TF-IDF:** Achieved **99.13% Accuracy**, **100% Precision**, and **93.13% Recall** on the clean test split, outperforming word-level TF-IDF across all metrics.
        - **Combined Word + Char:** Delivers near-identical peak recall (90.84%) and precision (100%), offering a hybrid representation.
        """
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
    "(reproducible random seed = 42). Measures accuracy drop, spam recall drop, robustness score, and spam miss flips:"
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

    st.warning(
        "📌 **Empirical Benchmark Takeaway**: On the baseline word-level pipeline, Logistic Regression recall collapsed from **73.28%** "
        "down to **48.09%** (a 25.19% drop; 35 spam messages slipped through!). "
        "In contrast, **Character TF-IDF maintained an 83.97% recall** (drop of only 0.76%), and **Pre-Normalization restored recall to 67.18%**."
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
    st.markdown(f"**Normalized Output:**\n`{inspect_res['normalized_text']}`")

    if inspect_res["modifications"]:
        st.markdown(f"**Modifications Applied ({inspect_res['modification_count']}):**")
        mods_df = pd.DataFrame(inspect_res["modifications"])
        st.dataframe(mods_df, use_container_width=True, hide_index=True)
    else:
        st.info("No obfuscation or abbreviation patterns detected; text preserved as-is.")
