"""
SMS Shield 4.0 - Page 4: Model Evaluation
Author: SMS Shield NLP Project

Comprehensive statistical validation across the held-out test split (n=1,035):
Quantitative metrics, confusion matrices, ROC/PR curves, cross-validation, and n-gram experiments.
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
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

st.set_page_config(page_title="SMS Shield | Model Evaluation", page_icon="📈", layout="wide")
render_header(
    "Model Evaluation",
    "Rigorous statistical assessment on held-out test split (1,035 messages), confusion matrices, and ROC/PR curves."
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
# 1. Model Summary Cards
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
for col, row in zip([c1, c2, c3], metrics_df.itertuples()):
    with col:
        st.markdown(
            f"""
            <div class="saas-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 0.95rem; color: #F1F5F9;">{row.Model}</span>
                    <span class="saas-pill">Test Set</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px;">
                    <div>
                        <div style="font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">Accuracy</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #F1F5F9;">{row.Accuracy:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">F1-Score</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #8B7CF6;">{row._5:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">Precision</div>
                        <div style="font-size: 1.05rem; font-weight: 500; color: #34D399;">{row.Precision:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">Recall</div>
                        <div style="font-size: 1.05rem; font-weight: 500; color: #F87171;">{row.Recall:.2%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# 2. Main Metrics Table
# -----------------------------------------------------------------------------
st.markdown("#### Test Split Metrics (1,035 Messages)")
st.caption("Stratified test split with 904 legitimate messages and 131 spam messages. Fitted strictly on training data to prevent data leakage.")

disp_metrics = metrics_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    disp_metrics[col] = disp_metrics[col].apply(lambda v: f"{v * 100:.2f}%")

disp_metrics = disp_metrics.rename(
    columns={
        "TN": "True Neg (TN)",
        "FP": "False Pos (FP)",
        "FN": "False Neg (FN)",
        "TP": "True Pos (TP)",
    }
)
st.dataframe(disp_metrics, use_container_width=True, hide_index=True)

st.write("")

# -----------------------------------------------------------------------------
# 3. Confusion Matrices
# -----------------------------------------------------------------------------
st.markdown("#### Confusion Matrices")
st.caption("Visualizing classification distribution, false positives (false alarms), and false negatives (missed spam).")

cm_col1, cm_col2, cm_col3 = st.columns(3)
metrics_dir = PROJECT_ROOT / "metrics"

with cm_col1:
    st.caption("Multinomial Naive Bayes")
    nb_cm_img = metrics_dir / "confusion_matrix_nb.png"
    if nb_cm_img.exists():
        st.image(str(nb_cm_img), use_container_width=True)
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #9CA3AF; text-align: center; margin-top: 4px;">
            TN: 903 • FP: 1 • FN: 33 • TP: 98
        </div>
        """,
        unsafe_allow_html=True
    )

with cm_col2:
    st.caption("Logistic Regression")
    lr_cm_img = metrics_dir / "confusion_matrix_lr.png"
    if lr_cm_img.exists():
        st.image(str(lr_cm_img), use_container_width=True)
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #9CA3AF; text-align: center; margin-top: 4px;">
            TN: 901 • FP: 3 • FN: 35 • TP: 96
        </div>
        """,
        unsafe_allow_html=True
    )

with cm_col3:
    st.caption("Linear SVM")
    svc_cm_img = metrics_dir / "confusion_matrix_svc.png"
    if svc_cm_img.exists():
        st.image(str(svc_cm_img), use_container_width=True)
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #9CA3AF; text-align: center; margin-top: 4px;">
            TN: 902 • FP: 2 • FN: 17 • TP: 114
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# -----------------------------------------------------------------------------
# 4. Diagnostic Curves
# -----------------------------------------------------------------------------
st.markdown("#### Diagnostic Curves")

roc_col, pr_col = st.columns(2)

with roc_col:
    st.caption("Receiver Operating Characteristic (ROC)")
    roc_img = metrics_dir / "roc_curves.png"
    if roc_img.exists():
        st.image(str(roc_img), use_container_width=True)

with pr_col:
    st.caption("Precision-Recall (Spam Class)")
    pr_img = metrics_dir / "pr_curves.png"
    if pr_img.exists():
        st.image(str(pr_img), use_container_width=True)

st.write("")

# -----------------------------------------------------------------------------
# 5. Validation & Experiments Tabs
# -----------------------------------------------------------------------------
st.markdown("#### Extended Experiments & Validation")

tab_cv, tab_exp, tab_base = st.tabs(["5-Fold Cross-Validation", "Feature Space Experiments", "Historical Baseline"])

with tab_cv:
    if cv_df is not None:
        disp_cv = cv_df.copy()
        disp_cv["Accuracy"] = disp_cv.apply(lambda r: f"{r['CV_Accuracy_Mean']:.3f} ± {r['CV_Accuracy_Std']:.3f}", axis=1)
        disp_cv["Precision"] = disp_cv.apply(lambda r: f"{r['CV_Precision_Mean']:.3f} ± {r['CV_Precision_Std']:.3f}", axis=1)
        disp_cv["Recall"] = disp_cv.apply(lambda r: f"{r['CV_Recall_Mean']:.3f} ± {r['CV_Recall_Std']:.3f}", axis=1)
        disp_cv["F1-Score"] = disp_cv.apply(lambda r: f"{r['CV_F1_Mean']:.3f} ± {r['CV_F1_Std']:.3f}", axis=1)

        st.dataframe(
            disp_cv[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Cross-validation records not found.")

with tab_exp:
    if exp_df is not None:
        exp_disp = exp_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            exp_disp[col] = exp_disp[col].apply(lambda v: f"{v * 100:.2f}%")
        st.dataframe(exp_disp, use_container_width=True, hide_index=True)

        st.caption(
            "Note: Expanding vocabulary to 10,000 features with trigrams increased sparsity and reduced recall on short SMS texts, "
            "confirming that higher n-gram order does not automatically improve short-text NLP performance."
        )
    else:
        st.info("Experiment records not found.")

with tab_base:
    if baseline_df is not None:
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)
    else:
        st.info("Baseline records not found.")
