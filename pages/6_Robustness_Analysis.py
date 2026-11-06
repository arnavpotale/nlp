"""
SMS Shield 4.0 - Page 6: Robustness Lab
Author: SMS Shield NLP Project

Clean adversarial stress-testing interface: Evaluate classifier robustness against
leetspeak, spacing, and obfuscation; inspect character n-grams and normalization defense.
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
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_plotly_theme,
)
from src.normalization import normalize_sms
from src.adversarial import generate_adversarial_variant

st.set_page_config(page_title="SMS Shield | Robustness Lab", page_icon="🛡️", layout="wide")
render_header(
    "Robustness Lab",
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
# 1. Adversarial Attack Playground
# -----------------------------------------------------------------------------
st.markdown("#### Attack Simulation")
st.caption("Simulate evasive modifications (leetspeak, whitespace, delimiters) to observe model responses.")

preset_spams = {
    "Cash Prize Scam": "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 87121 now.",
    "Bank Phishing": "URGENT! Your account has suspicious activity. Call 08001234567 or visit https://secure-bank-verify.com immediately.",
    "Ringtone Bonus": "FreeMsg: 500 free ringtones waiting for you. Text YES to 88088 now! 18+ only.",
    "Friendly Lunch": "Hey! Are we still on for lunch at 1 PM today? Let me know if you want to invite Sarah too.",
}

st.caption("Select a sample message:")
p_cols = st.columns(len(preset_spams))
for idx, (label, text_val) in enumerate(preset_spams.items()):
    if p_cols[idx].button(label, use_container_width=True):
        st.session_state["adv_input_text"] = text_val

default_msg = st.session_state.get(
    "adv_input_text",
    "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 87121 now."
)

raw_user_msg = st.text_area("Original message:", value=default_msg, height=75)

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
    selected_attack = st.selectbox("Adversarial attack vector:", attack_options)
with col_seed:
    attack_seed = st.number_input("Random seed:", min_value=1, max_value=9999, value=42)

perturbed_msg = generate_adversarial_variant(raw_user_msg, selected_attack, seed=int(attack_seed))
norm_info = normalize_sms(perturbed_msg)
normalized_msg = norm_info["normalized_text"]

st.write("")

# Message Comparison (Original vs Attacked vs Normalized)
st.caption("Message Progression:")
mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #60A5FA;">Original Clean</div>
            <div style="font-family: monospace; font-size: 0.84rem; color: #F1F5F9; margin-top: 6px; word-break: break-word;">{raw_user_msg}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc2:
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #F87171;">Attacked Variant</div>
            <div style="font-family: monospace; font-size: 0.84rem; color: #F87171; margin-top: 6px; word-break: break-word;">{perturbed_msg}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mc3:
    st.markdown(
        f"""
        <div class="saas-card" style="padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #34D399;">Normalized Recovery</div>
            <div style="font-family: monospace; font-size: 0.84rem; color: #34D399; margin-top: 6px; word-break: break-word;">{normalized_msg}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# -----------------------------------------------------------------------------
# 2. Live Defense Matrix
# -----------------------------------------------------------------------------
st.markdown("#### Live Defense Comparison")
st.caption("Evaluating whether the attack evaded Word TF-IDF, Character TF-IDF, or Normalized pipelines.")

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

    word_status = "Evaded" if (orig_lbl == "SPAM" and adv_word_lbl == "HAM") else ("Defended" if adv_word_lbl == "SPAM" else "Legitimate")
    char_status = "Evaded" if (orig_lbl == "SPAM" and adv_char_lbl == "HAM") else ("Defended" if adv_char_lbl == "SPAM" else "Legitimate")
    norm_status = "Evaded" if (orig_lbl == "SPAM" and adv_norm_lbl == "HAM") else ("Defended" if adv_norm_lbl == "SPAM" else "Legitimate")

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

orig_cons = res_orig_word["consensus"]
adv_word_cons = res_adv_word["consensus"]
adv_char_cons = res_adv_char["consensus"]
adv_norm_cons = res_adv_norm["consensus"]

if orig_cons == "SPAM" and adv_word_cons == "HAM":
    st.markdown(
        f"""
        <div class="verdict-box-spam" style="padding: 12px 16px;">
            <span style="font-weight: 600; color: #F87171;">Adversarial Evasion on Word Baseline:</span>
            <span style="color: #9CA3AF; font-size: 0.85rem; margin-left: 6px;">
                Flipped from SPAM to HAM on word TF-IDF. However, Character TF-IDF held as <strong>{adv_char_cons}</strong> and Normalization restored to <strong>{adv_norm_cons}</strong>.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
elif orig_cons == "SPAM" and adv_word_cons == "SPAM":
    st.markdown(
        """
        <div class="verdict-box-ham" style="padding: 12px 16px;">
            <span style="font-weight: 600; color: #34D399;">Defense Resilient:</span>
            <span style="color: #9CA3AF; font-size: 0.85rem; margin-left: 6px;">
                Models maintained unanimous SPAM classification despite perturbation.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# -----------------------------------------------------------------------------
# 3. Feature Space Comparison
# -----------------------------------------------------------------------------
st.markdown("#### Feature Space Comparison")
st.caption("Word-level vs. sub-word character (2,5) vs. combined representations on the held-out test split.")

if char_word_df is not None:
    t_table, t_chart = st.tabs(["Metrics Table", "Recall Comparison"])

    with t_table:
        disp_cw = char_word_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            disp_cw[col] = disp_cw[col].apply(lambda v: f"{v * 100:.2f}%")
        st.dataframe(disp_cw, use_container_width=True, hide_index=True)

    with t_chart:
        fig_rep = go.Figure()
        representations = char_word_df["Feature Space"].unique()
        
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

        apply_plotly_theme(fig_rep, height=300, title="Spam Recall Across Representations (%)")
        fig_rep.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 115], title="Spam Recall (%)"),
        )
        st.plotly_chart(fig_rep, use_container_width=True)

st.write("")

# -----------------------------------------------------------------------------
# 4. Empirical Benchmarks & Normalization Inspector
# -----------------------------------------------------------------------------
st.markdown("#### Empirical Robustness Benchmarks")
st.caption("Measured under a Composite Multi-Vector Attack (reproducible seed = 42).")

if adv_df is not None:
    disp_adv = adv_df.copy()
    for col in ["Clean Recall (Spam)", "Adversarial Recall", "Recall Drop"]:
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

with st.expander("SMS Text Normalization Inspector"):
    st.caption("Test the rule-based obfuscation reversal engine on custom leetspeak, abbreviations, or symbol substitutions.")
    norm_sample = "C0NGRATS! U R THE W!NNER OF A FR33 £1000 PR!ZE. T7X7 CL41M N0W TO 88088. HURRYYYY!"
    test_norm_input = st.text_input("Text to normalize:", value=norm_sample)

    if test_norm_input.strip():
        inspect_res = normalize_sms(test_norm_input)
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 10px 14px; margin-top: 6px;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #34D399;">Normalized Output</div>
                <div style="font-family: monospace; font-size: 0.88rem; color: #F1F5F9; margin-top: 4px;">{inspect_res['normalized_text']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if inspect_res["modifications"]:
            st.caption(f"Applied rules ({inspect_res['modification_count']}):")
            mods_df = pd.DataFrame(inspect_res["modifications"])
            st.dataframe(mods_df, use_container_width=True, hide_index=True)
        else:
            st.info("No modifications required; text preserved as-is.")
