"""
SMS Shield 2.0 - Page 1: Dashboard
Author: SMS Shield NLP Project

Displays global dataset metrics, class imbalance, model performance summaries,
and length distributions dynamically loaded from dataset and evaluation files.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_dataset, get_evaluation_metrics
from src.evaluate import plot_model_comparison_bar


st.set_page_config(page_title="Dashboard | SMS Shield 2.0", page_icon="📊", layout="wide")
render_header("Page 1: Global System Dashboard", "Executive overview of SMS dataset distribution, model metrics, and telemetry.")
render_sidebar_status()

# Load data dynamically
try:
    raw_df, clean_df, duplicates_count = get_dataset()
    eval_data = get_evaluation_metrics()
    metrics_df = eval_data["metrics_df"]
except Exception as e:
    st.error(f"Error loading dashboard resources: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Metric KPI Cards
# -----------------------------------------------------------------------------
total_raw = len(raw_df)
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100

# Identify best performing model by F1-Score
best_f1_idx = metrics_df["F1-Score"].idxmax()
best_model_name = metrics_df.loc[best_f1_idx, "Model"]
best_f1_score = metrics_df.loc[best_f1_idx, "F1-Score"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Total Messages", f"{total_clean:,}", f"{duplicates_count:,} deduplicated", delta_color="inverse")
with c2:
    st.metric("Spam Messages", f"{spam_count:,}", f"{spam_pct:.1f}% prevalence")
with c3:
    st.metric("Legitimate (Ham)", f"{ham_count:,}", f"{100 - spam_pct:.1f}% of total")
with c4:
    st.metric("Trained Models", "3 Models", "NB, LR, Linear SVM")
with c5:
    st.metric("Best F1-Score", f"{best_f1_score:.2%}", best_model_name)
with c6:
    best_prec = metrics_df["Precision"].max()
    best_prec_model = metrics_df.loc[metrics_df["Precision"].idxmax(), "Model"]
    st.metric("Peak Precision", f"{best_prec:.2%}", best_prec_model)

st.divider()

# -----------------------------------------------------------------------------
# Visualizations: Row 1 (Class Distribution & Length Analysis)
# -----------------------------------------------------------------------------
st.markdown("### 📊 Dataset Exploration & Class Imbalance")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("##### Class Distribution (Ham vs. Spam)")
    fig_dist, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    counts = clean_df["label"].value_counts()
    colors = {"ham": "#2563eb", "spam": "#dc2626"}
    sns.barplot(
        x=counts.index.str.upper(),
        y=counts.values,
        hue=counts.index,
        palette=colors,
        legend=False,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )
    ax.set_ylabel("Message Count", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Class", fontsize=9.5, fontweight="bold")
    ax.set_title("UCI SMS Spam Collection (Deduplicated)", fontsize=11, fontweight="bold")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h):,} ({h/total_clean:.1%})",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points"
        )
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)

with chart_col2:
    st.markdown("##### SMS Character Length Distribution")
    fig_len, ax2 = plt.subplots(figsize=(6, 3.8), dpi=150)
    sns.histplot(
        data=clean_df,
        x="char_length",
        hue="label",
        bins=40,
        palette={"ham": "#2563eb", "spam": "#dc2626"},
        kde=True,
        element="step",
        ax=ax2
    )
    ax2.set_xlim(0, 260)
    ax2.set_xlabel("Characters per Message", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=9.5, fontweight="bold")
    ax2.set_title("Ham (Median ~52 chars) vs. Spam (Median ~149 chars)", fontsize=11, fontweight="bold")
    ax2.legend(title="Class", labels=["Spam", "Ham"])
    plt.tight_layout()
    st.pyplot(fig_len)
    plt.close(fig_len)

st.divider()

# -----------------------------------------------------------------------------
# Visualizations: Row 2 (Comparative Model Performance Summary)
# -----------------------------------------------------------------------------
st.markdown("### 🏆 Three-Model Comparative Performance")
c_bar, c_table = st.columns([1.5, 1])

with c_bar:
    fig_comp = plot_model_comparison_bar(metrics_df)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

with c_table:
    st.markdown("##### Quantitative Model Summary")
    disp_df = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        disp_df[col] = disp_df[col].apply(lambda v: f"{v * 100:.2f}%")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.info(
        "💡 **Key Operational Insight**: **Multinomial Naive Bayes** maintains peak precision (98.99%), "
        "minimizing false alarms. **Linear SVM** delivers the highest recall (87.02%) and overall F1-score (92.31%), "
        "making them complementary algorithms for a multi-model consensus system."
    )

