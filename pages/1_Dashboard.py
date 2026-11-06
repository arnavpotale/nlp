"""
SMS Shield 4.0 - Page 1: Dashboard
Author: SMS Shield NLP Project

Clean, minimalist dashboard displaying dataset distributions, class imbalance,
message length profiling, and quantitative model performance benchmarks.
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
    get_dataset,
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
    create_donut_chart,
    create_comparative_bar_chart,
)

st.set_page_config(page_title="SMS Shield | Dashboard", page_icon="📊", layout="wide")
render_header(
    "Dashboard",
    "Dataset telemetry, label distributions, and comparative machine learning benchmarks."
)
render_sidebar_status()

# Load cached data
try:
    raw_df, clean_df, duplicates_count = get_dataset()
    eval_data = get_evaluation_metrics()
    metrics_df = eval_data["metrics_df"]
except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Summary KPI Cards
# -----------------------------------------------------------------------------
total_clean = len(clean_df)
spam_count = int((clean_df["label"] == "spam").sum())
ham_count = int((clean_df["label"] == "ham").sum())
spam_pct = (spam_count / total_clean) * 100

best_f1_idx = metrics_df["F1-Score"].idxmax()
best_model_name = metrics_df.loc[best_f1_idx, "Model"]
best_f1_score = metrics_df.loc[best_f1_idx, "F1-Score"]

c1, c2, c3, c4 = st.columns(4)
render_kpi_card(c1, "Total Corpus", f"{total_clean:,}", f"{duplicates_count:,} duplicates dropped")
render_kpi_card(c2, "Spam Messages", f"{spam_count:,}", f"{spam_pct:.1f}% ratio")
render_kpi_card(c3, "Legitimate (Ham)", f"{ham_count:,}", f"{100 - spam_pct:.1f}% ratio")
render_kpi_card(c4, "Top F1-Score", f"{best_f1_score:.2%}", best_model_name)

st.write("")

# -----------------------------------------------------------------------------
# 2. Dataset Distributions Row
# -----------------------------------------------------------------------------
st.markdown("#### Dataset Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.caption("Class Distribution (Ham vs. Spam)")
    donut_fig = create_donut_chart(
        labels=["Legitimate (Ham)", "Spam"],
        values=[ham_count, spam_count],
        colors=[SECONDARY_ACCENT, DANGER_RED],
        center_text=f"{total_clean:,}",
        height=280,
    )
    st.plotly_chart(donut_fig, use_container_width=True)

with chart_col2:
    st.caption("Message Character Length Distribution")
    len_fig = go.Figure()
    
    ham_lengths = clean_df[clean_df["label"] == "ham"]["char_length"]
    spam_lengths = clean_df[clean_df["label"] == "spam"]["char_length"]
    
    len_fig.add_trace(
        go.Histogram(
            x=ham_lengths,
            name="Ham",
            marker_color=SECONDARY_ACCENT,
            opacity=0.7,
            nbinsx=40,
            hovertemplate="Length: %{x} chars<br>Count: %{y}<extra></extra>"
        )
    )
    len_fig.add_trace(
        go.Histogram(
            x=spam_lengths,
            name="Spam",
            marker_color=DANGER_RED,
            opacity=0.7,
            nbinsx=40,
            hovertemplate="Length: %{x} chars<br>Count: %{y}<extra></extra>"
        )
    )
    
    apply_plotly_theme(len_fig, height=280)
    len_fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Characters per Message", range=[0, 240]),
        yaxis=dict(title="Count"),
    )
    st.plotly_chart(len_fig, use_container_width=True)

st.write("")

# -----------------------------------------------------------------------------
# 3. Model Performance Comparison Row
# -----------------------------------------------------------------------------
st.markdown("#### Classifier Performance")

c_bar, c_table = st.columns([1.3, 1])

with c_bar:
    st.caption("Comparative Metrics Across Models")
    comp_fig = create_comparative_bar_chart(
        metrics_df,
        model_col="Model",
        metric_cols=["Accuracy", "Precision", "Recall", "F1-Score"],
        colors=[PRIMARY_ACCENT, SECONDARY_ACCENT, SUCCESS_GREEN, "#FBBF24"],
        height=320,
    )
    st.plotly_chart(comp_fig, use_container_width=True)

with c_table:
    st.caption("Quantitative Evaluation Table")
    disp_df = metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        disp_df[col] = disp_df[col].apply(lambda v: f"{v * 100:.2f}%")
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="saas-card" style="padding: 12px 14px; margin-top: 10px;">
            <div style="font-size: 0.78rem; font-weight: 600; color: #F1F5F9;">
                Model Trade-offs
            </div>
            <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 4px; line-height: 1.45;">
                • <strong>Naive Bayes:</strong> Highest precision (98.99%), practically zero false alarms.<br>
                • <strong>Linear SVM:</strong> Highest spam recall (87.02%) and peak F1-score (92.31%).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
