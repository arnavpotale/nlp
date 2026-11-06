"""
SMS Shield 2.0 - Page 4: Model Evaluation
Author: SMS Shield NLP Project

Comprehensive comparative analysis of Multinomial Naive Bayes, Logistic Regression,
and Linear Support Vector Machine across:
- Quantitative metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Confusion Matrices (counts and percentages)
- ROC Curves & Precision-Recall Curves
- 5-Fold Stratified Cross-Validation
- TF-IDF N-gram & Feature Capacity Experiments (Configs A, B, C)
- Historical Baseline Comparison
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_evaluation_metrics


st.set_page_config(page_title="Model Evaluation | SMS Shield 2.0", page_icon="📈", layout="wide")
render_header("Page 4: Machine Learning Model Evaluation & Comparative Diagnostics", "Rigorous statistical assessment of 3 models across hold-out test split, cross-validation, and ROC/PR curves.")
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
# 1. Main Quantitative Comparison Table
# -----------------------------------------------------------------------------
st.markdown("### 📋 Test Set Evaluation Metrics (Held-Out Split: 1,035 Messages)")
st.write(
    "All models were evaluated on the exact same stratified test set (904 legitimate Ham, 131 Spam). "
    "TF-IDF feature extraction was fitted strictly on the 4,136 training messages to prevent data leakage."
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

# -----------------------------------------------------------------------------
# 2. Confusion Matrices Side-by-Side (All Three Models)
# -----------------------------------------------------------------------------
st.markdown("### 🔲 Confusion Matrices (Actual vs. Predicted)")
st.write("Visualizing the classification balance and operational error distribution across all 3 models:")

cm_col1, cm_col2, cm_col3 = st.columns(3)
metrics_dir = PROJECT_ROOT / "metrics"

with cm_col1:
    st.markdown("##### 1. Multinomial Naive Bayes")
    nb_cm_img = metrics_dir / "confusion_matrix_nb.png"
    if nb_cm_img.exists():
        st.image(str(nb_cm_img), use_container_width=True)
    st.caption("**TN:** 903 | **FP (False Alarms):** 1 | **FN:** 33 | **TP:** 98")

with cm_col2:
    st.markdown("##### 2. Logistic Regression")
    lr_cm_img = metrics_dir / "confusion_matrix_lr.png"
    if lr_cm_img.exists():
        st.image(str(lr_cm_img), use_container_width=True)
    st.caption("**TN:** 901 | **FP (False Alarms):** 3 | **FN:** 35 | **TP:** 96")

with cm_col3:
    st.markdown("##### 3. Linear SVM")
    svc_cm_img = metrics_dir / "confusion_matrix_svc.png"
    if svc_cm_img.exists():
        st.image(str(svc_cm_img), use_container_width=True)
    st.caption("**TN:** 902 | **FP (False Alarms):** 2 | **FN:** 17 | **TP:** 114")

# Operational insight callout
st.info(
    "🚨 **Critical Engineering Takeaway**: While **Naive Bayes** produced only **1 false alarm** across 904 legitimate messages, "
    "**Linear SVM** caught **114 out of 131 spam messages** (Recall = 87.02%, compared to 74.81% for NB), reducing spam slip-throughs by almost half (17 vs. 33 missed)."
)

st.divider()

# -----------------------------------------------------------------------------
# 3. Diagnostic Curves: ROC & Precision-Recall Curves
# -----------------------------------------------------------------------------
st.markdown("### 📉 Diagnostic Threshold Curves")
roc_col, pr_col = st.columns(2)

with roc_col:
    st.markdown("##### Receiver Operating Characteristic (ROC) Curves")
    roc_img = metrics_dir / "roc_curves.png"
    if roc_img.exists():
        st.image(str(roc_img), use_container_width=True)
    st.caption("All three models demonstrate exceptional discriminative capability (AUC > 0.983).")

with pr_col:
    st.markdown("##### Precision-Recall Curves (Spam Class)")
    pr_img = metrics_dir / "pr_curves.png"
    if pr_img.exists():
        st.image(str(pr_img), use_container_width=True)
    st.caption("Precision-Recall curves highlight performance specifically on the minority positive class (Spam).")

st.divider()

# -----------------------------------------------------------------------------
# 4. 5-Fold Stratified Cross-Validation
# -----------------------------------------------------------------------------
st.markdown("### 🔁 5-Fold Stratified Cross-Validation")
st.write(
    "Cross-validation confirms that the high test metrics are statistically robust "
    "and not an artifact of a lucky single train/test split."
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
# 5. Experimental TF-IDF N-gram Configurations Comparison
# -----------------------------------------------------------------------------
st.markdown("### 🔬 TF-IDF Feature Representation Experiments")
st.write(
    "To understand feature representation impact, three distinct TF-IDF configurations were evaluated "
    "across all models on the identical stratified test split:"
)

if exp_df is not None:
    exp_display = exp_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        exp_display[col] = exp_display[col].apply(lambda v: f"{v * 100:.2f}%")

    st.dataframe(exp_display, use_container_width=True, hide_index=True)

    st.markdown("##### 📝 Experimental Findings & Discussion")
    st.markdown(
        """
        - **Config A vs. Config B (Unigrams vs. Unigrams + Bigrams):** Adding bigrams (`ngram_range=(1,2)`) provides phrases like `'claim call'` and `'cash prize'`, resulting in a modest boost in Naive Bayes precision (reaching 100%) and Recall (76.34%).
        - **Config B vs. Config C (Adding Trigrams with 10,000 features):** Expanding to trigrams and doubling vocabulary capacity to 10,000 features **degraded** Naive Bayes recall significantly (from 76.34% down to 67.94%).
        - **Why did complexity reduce performance?** SMS messages are brief (~15 words). Trigrams occur very infrequently, introducing extreme sparsity without sufficient statistical support, leading to over-regularization and parameter dilution.
        - **Conclusion:** More complex feature representations do not necessarily yield superior accuracy in short-text NLP tasks.
        """
    )
else:
    st.info("TF-IDF configuration experiment records not found.")

st.divider()

# -----------------------------------------------------------------------------
# 6. Historical Baseline Verification
# -----------------------------------------------------------------------------
with st.expander("🏛️ View Historical Baseline Verification"):
    st.write(
        "Below are the verified historical baseline metrics recorded during the original project version. "
        "Notice that Naive Bayes and Logistic Regression results match the baseline perfectly, "
        "confirming reproducible, deterministic evaluation."
    )
    if baseline_df is not None:
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)
    else:
        st.write("Baseline CSV not found.")
