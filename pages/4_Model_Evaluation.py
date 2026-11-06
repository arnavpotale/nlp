"""
SMS Shield 4.0 - Page 4: Model Performance & Diagnostic Lab
Author: SMS Shield NLP Project

Comprehensive statistical validation across held-out test split (n=1,035):
Multi-algorithm metrics, side-by-side Confusion Matrices, ROC/PR curves,
5-Fold Stratified Cross-Validation, and TF-IDF n-gram capacity experiments.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    render_header,
    render_sidebar_status,
    render_kpi_card,
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
)

st.set_page_config(page_title="Model Performance | SMS Shield 4.0", page_icon="📈", layout="wide")
render_header(
    "Model Performance & Diagnostic Lab",
    "Rigorous statistical validation across held-out test split, 5-fold cross-validation, and ROC/PR curves."
)
render_sidebar_status()

# Load evaluation data
try:
    eval_data = get_evaluation_metrics()
    metrics_df = eval_data["metrics_df"]
    cv_df = eval_data["cv_df"]
    exp_df = eval_data["experiments_df"]
    baseline_df = eval_data["baseline_df"]
except Exception as e:
    st.error(f"Error loading evaluation metrics: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. High-Level Model Cards
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
for idx, (col, row) in enumerate(zip([c1, c2, c3], metrics_df.itertuples())):
    border_accent = PRIMARY_VIOLET if "Naive" in row.Model else (CYAN_ACCENT if "Logistic" in row.Model else SUCCESS_GREEN)
    with col:
        st.markdown(
            f"""
            <div class="cyber-card" style="border-top: 3px solid {border_accent}; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">{row.Model}</span>
                    <span class="status-pill active" style="font-size: 0.74rem;">TEST SET</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px;">
                    <div>
                        <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">Accuracy</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">{row.Accuracy:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">Spam F1-Score</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8;">{row._5:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">Precision</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #34D399;">{row.Precision:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">Recall</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #F87171;">{row.Recall:.2%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# 2. Main Quantitative Comparison Table
# -----------------------------------------------------------------------------
st.markdown("### 📋 Test Set Evaluation Metrics (Held-Out Split: 1,035 Messages)")
st.write(
    "All three algorithms were evaluated on the identical stratified test set (904 legitimate Ham, 131 Spam). "
    "TF-IDF feature extraction was fitted strictly on the 4,136 training messages to enforce **zero data leakage**."
)

disp_metrics = metrics_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    disp_metrics[col] = disp_metrics[col].apply(lambda v: f"{v:.4f} ({v * 100:.2f}%)")

disp_metrics = disp_metrics.rename(
    columns={
        "TN": "True Negatives (TN)",
        "FP": "False Positives (FP)",
        "FN": "False Negatives (FN)",
        "TP": "True Positives (TP)",
    }
)

st.dataframe(disp_metrics, use_container_width=True, hide_index=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Confusion Matrices Side-by-Side (All Three Models)
# -----------------------------------------------------------------------------
st.markdown("### 🔲 Confusion Matrix Diagnostics (Actual vs. Predicted)")
st.write("Visualizing false alarms (False Positives) versus spam slip-throughs (False Negatives):")

cm_col1, cm_col2, cm_col3 = st.columns(3)
metrics_dir = PROJECT_ROOT / "metrics"

with cm_col1:
    st.markdown("##### 1. Multinomial Naive Bayes")
    nb_cm_img = metrics_dir / "confusion_matrix_nb.png"
    if nb_cm_img.exists():
        st.image(str(nb_cm_img), use_container_width=True)
    st.markdown(
        """
        <div class="cyber-card" style="font-size: 0.82rem; color: #CBD5E1; padding: 10px 14px;">
            <strong>TN:</strong> 903 &nbsp;|&nbsp; <strong>FP (False Alarms):</strong> 1 &nbsp;|&nbsp; <strong>FN:</strong> 33 &nbsp;|&nbsp; <strong>TP:</strong> 98
        </div>
        """,
        unsafe_allow_html=True
    )

with cm_col2:
    st.markdown("##### 2. Logistic Regression")
    lr_cm_img = metrics_dir / "confusion_matrix_lr.png"
    if lr_cm_img.exists():
        st.image(str(lr_cm_img), use_container_width=True)
    st.markdown(
        """
        <div class="cyber-card" style="font-size: 0.82rem; color: #CBD5E1; padding: 10px 14px;">
            <strong>TN:</strong> 901 &nbsp;|&nbsp; <strong>FP (False Alarms):</strong> 3 &nbsp;|&nbsp; <strong>FN:</strong> 35 &nbsp;|&nbsp; <strong>TP:</strong> 96
        </div>
        """,
        unsafe_allow_html=True
    )

with cm_col3:
    st.markdown("##### 3. Linear SVM")
    svc_cm_img = metrics_dir / "confusion_matrix_svc.png"
    if svc_cm_img.exists():
        st.image(str(svc_cm_img), use_container_width=True)
    st.markdown(
        """
        <div class="cyber-card" style="font-size: 0.82rem; color: #CBD5E1; padding: 10px 14px;">
            <strong>TN:</strong> 902 &nbsp;|&nbsp; <strong>FP (False Alarms):</strong> 2 &nbsp;|&nbsp; <strong>FN:</strong> 17 &nbsp;|&nbsp; <strong>TP:</strong> 114
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="cyber-card" style="border-left: 3px solid #8B5CF6; margin-top: 10px;">
        <span style="color: #A78BFA; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
            Operational Security Analysis
        </span>
        <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px; line-height: 1.5;">
            <strong>Naive Bayes</strong> generates only <strong>1 false alarm</strong> across 904 legitimate messages (98.99% Precision).
            However, <strong>Linear SVM</strong> cuts spam slip-throughs nearly in half (only 17 false negatives vs 33 for NB), achieving an <strong>87.02% recall</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------------------------------
# 4. Diagnostic Curves: ROC & Precision-Recall Curves
# -----------------------------------------------------------------------------
st.markdown("### 📉 Discrimination Curves: ROC & Precision-Recall")
roc_col, pr_col = st.columns(2)

with roc_col:
    st.markdown("##### Receiver Operating Characteristic (ROC)")
    roc_img = metrics_dir / "roc_curves.png"
    if roc_img.exists():
        st.image(str(roc_img), use_container_width=True)
    st.caption("All three models demonstrate exceptional discriminative capability (AUC > 0.983).")

with pr_col:
    st.markdown("##### Precision-Recall (Spam Minority Class)")
    pr_img = metrics_dir / "pr_curves.png"
    if pr_img.exists():
        st.image(str(pr_img), use_container_width=True)
    st.caption("Precision-Recall curves emphasize performance on the imbalanced minority positive class (Spam).")

st.divider()

# -----------------------------------------------------------------------------
# 5. 5-Fold Stratified Cross-Validation
# -----------------------------------------------------------------------------
st.markdown("### 🔁 5-Fold Stratified Cross-Validation Stability")
st.write(
    "Stratified 5-fold cross-validation proves that model performance is statistically resilient "
    "and not dependent on an advantageous single train/test partition."
)

if cv_df is not None:
    disp_cv = cv_df.copy()
    disp_cv["Mean Accuracy"] = disp_cv.apply(lambda r: f"{r['CV_Accuracy_Mean']:.4f} ± {r['CV_Accuracy_Std']:.4f}", axis=1)
    disp_cv["Mean Precision"] = disp_cv.apply(lambda r: f"{r['CV_Precision_Mean']:.4f} ± {r['CV_Precision_Std']:.4f}", axis=1)
    disp_cv["Mean Recall"] = disp_cv.apply(lambda r: f"{r['CV_Recall_Mean']:.4f} ± {r['CV_Recall_Std']:.4f}", axis=1)
    disp_cv["Mean F1-Score"] = disp_cv.apply(lambda r: f"{r['CV_F1_Mean']:.4f} ± {r['CV_F1_Std']:.4f}", axis=1)

    st.dataframe(
        disp_cv[["Model", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1-Score"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Cross-validation records not found.")

st.divider()

# -----------------------------------------------------------------------------
# 6. Experimental TF-IDF N-gram Configurations Comparison
# -----------------------------------------------------------------------------
st.markdown("### 🔬 TF-IDF Feature Representation Experiments")
st.write(
    "Evaluating three distinct TF-IDF vocabulary structures across all models on the identical test split:"
)

if exp_df is not None:
    exp_display = exp_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        exp_display[col] = exp_display[col].apply(lambda v: f"{v * 100:.2f}%")

    st.dataframe(exp_display, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="cyber-card" style="border-left: 3px solid #38BDF8; margin-top: 12px;">
            <span style="color: #38BDF8; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Architectural Findings: Feature Capacity vs. Generalization
            </span>
            <div style="font-size: 0.86rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                • <strong>Config A vs. B (Unigrams vs. Unigrams + Bigrams):</strong> Incorporating bi-grams (<code>ngram_range=(1,2)</code>) boosted Naive Bayes precision to 100% and recall to 76.34% by capturing phrases like <em>'claim call'</em>.<br>
                • <strong>Config B vs. C (Trigrams with 10k Vocab):</strong> Doubling vocabulary capacity to 10,000 features and adding tri-grams <strong>degraded</strong> Naive Bayes recall significantly (from 76.34% down to 67.94%).<br>
                • <strong>Why did added complexity harm performance?</strong> Short SMS messages (~15 words) have sparse trigram occurrences. The huge feature space introduces severe sparsity, leading to parameter dilution.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("TF-IDF configuration experiment records not found.")

st.divider()

# -----------------------------------------------------------------------------
# 7. Historical Baseline Verification
# -----------------------------------------------------------------------------
with st.expander("🏛️ Historical Baseline Determinism Verification"):
    st.write(
        "Historical baseline metrics recorded during initial development. "
        "Notice that Naive Bayes and Logistic Regression results match the baseline precisely, "
        "confirming deterministic evaluation."
    )
    if baseline_df is not None:
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)
    else:
        st.write("Baseline CSV not found.")
