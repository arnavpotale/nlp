"""
SMS Shield 4.0 - Page 1: Dashboard
Author: SMS Shield NLP Project

Executive Threat Intelligence Overview: Dataset distributions, class imbalance,
interactive length histograms, and 3-model comparative performance benchmarks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    render_header,
    render_sidebar_status,
    render_kpi_card,
    get_dataset,
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
    create_donut_chart,
    create_comparative_bar_chart,
)

st.set_page_config(page_title="Dashboard | SMS Shield 4.0", page_icon="📊", layout="wide")
render_header(
    "Threat Intelligence Overview",
    "Dataset distribution telemetry, lexical length profiles, and cross-algorithm comparative benchmarks."
)
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
# 1. Metric KPI Cards Row
# -----------------------------------------------------------------------------
total_raw = len(raw_df)
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100

best_f1_idx = metrics_df["F1-Score"].idxmax()
best_model_name = metrics_df.loc[best_f1_idx, "Model"]
best_f1_score = metrics_df.loc[best_f1_idx, "F1-Score"]
best_prec = metrics_df["Precision"].max()
best_prec_model = metrics_df.loc[metrics_df["Precision"].idxmax(), "Model"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
render_kpi_card(c1, "Total Corpus", f"{total_clean:,}", f"{duplicates_count:,} duplicates dropped", accent_border=PRIMARY_VIOLET)
render_kpi_card(c2, "Malicious (Spam)", f"{spam_count:,}", f"{spam_pct:.1f}% ratio", accent_border=DANGER_RED)
render_kpi_card(c3, "Legitimate (Ham)", f"{ham_count:,}", f"{100 - spam_pct:.1f}% ratio", accent_border=SUCCESS_GREEN)
render_kpi_card(c4, "Active Models", "3 Models", "NB, LR, Linear SVM", accent_border=CYAN_ACCENT)
render_kpi_card(c5, "Peak F1-Score", f"{best_f1_score:.2%}", best_model_name, accent_border="#EC4899")
render_kpi_card(c6, "Peak Precision", f"{best_prec:.2%}", best_prec_model, accent_border="#F59E0B")

st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Visualizations: Row 1 (Class Distribution & Character Length Distribution)
# -----------------------------------------------------------------------------
st.markdown("### 📊 Dataset Exploration & Class Imbalance")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("##### 🍩 Label Imbalance Distribution (Ham vs. Spam)")
    donut_fig = create_donut_chart(
        labels=["Legitimate (Ham)", "Malicious (Spam)"],
        values=[ham_count, spam_count],
        colors=[CYAN_ACCENT, DANGER_RED],
        title="",
        center_text=f"{total_clean:,} MSGS",
        height=350,
    )
    st.plotly_chart(donut_fig, use_container_width=True)

with chart_col2:
    st.markdown("##### 📈 Character Length Profiling (Ham vs. Spam)")
    len_fig = go.Figure()
    
    # Ham messages
    ham_lengths = clean_df[clean_df["label"] == "ham"]["char_length"]
    len_fig.add_trace(
        go.Histogram(
            x=ham_lengths,
            name="Ham (Legitimate)",
            marker_color=CYAN_ACCENT,
            opacity=0.75,
            nbinsx=45,
            hovertemplate="Length: %{x} chars<br>Count: %{y}<extra></extra>"
        )
    )
    
    # Spam messages
    spam_lengths = clean_df[clean_df["label"] == "spam"]["char_length"]
    len_fig.add_trace(
        go.Histogram(
            x=spam_lengths,
            name="Spam (Malicious)",
            marker_color=DANGER_RED,
            opacity=0.75,
            nbinsx=45,
            hovertemplate="Length: %{x} chars<br>Count: %{y}<extra></extra>"
        )
    )
    
    apply_plotly_theme(len_fig, height=350)
    len_fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Character Length per Message", range=[0, 250]),
        yaxis=dict(title="Frequency"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(len_fig, use_container_width=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Visualizations: Row 2 (Comparative Model Performance Summary)
# -----------------------------------------------------------------------------
st.markdown("### 🏆 Three-Model Comparative Performance Benchmarks")
c_bar, c_table = st.columns([1.5, 1])

with c_bar:
    st.markdown("##### 📊 Cross-Metric Performance Comparison")
    comp_fig = create_comparative_bar_chart(
        metrics_df,
        model_col="Model",
        metric_cols=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        colors=[PRIMARY_VIOLET, CYAN_ACCENT, SUCCESS_GREEN, "#F59E0B", "#EC4899"],
        height=380,
    )
    st.plotly_chart(comp_fig, use_container_width=True)

with c_table:
    st.markdown("##### 📋 Quantitative Metric Matrix")
    disp_df = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        disp_df[col] = disp_df[col].apply(lambda v: f"{v * 100:.2f}%")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="cyber-card" style="border-left: 3px solid #8B5CF6; margin-top: 12px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #C4B5FD; text-transform: uppercase;">
                Operational Security Insights
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                • <strong>Multinomial Naive Bayes</strong> achieves highest precision (<strong>98.99%</strong>), yielding near-zero false alarms.<br>
                • <strong>Linear SVM</strong> delivers highest spam recall (<strong>87.02%</strong>) and peak F1-score (<strong>92.31%</strong>), minimizing evasions.<br>
                • <em>Ensemble consensus combines both strengths for production filtering.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
