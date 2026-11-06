"""
SMS Shield 4.0 - Page 5: Bulk Analysis
Author: SMS Shield NLP Project

Clean batch CSV classification engine: Upload arbitrary CSVs of SMS messages,
execute 3-model consensus classification, filter by agreement, and export audit reports.
"""

import sys
from pathlib import Path
from datetime import datetime
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
)
from src.bulk_analysis import parse_and_validate_csv, run_bulk_classification, generate_sample_bulk_csv

st.set_page_config(page_title="SMS Shield | Bulk Analysis", page_icon="📁", layout="wide")
render_header(
    "Bulk Analysis",
    "Batch classify SMS messages from uploaded CSV datasets with multi-model consensus and downloadable reports."
)
render_sidebar_status()

# Load predictor
try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Ingestion Experience
# -----------------------------------------------------------------------------
st.markdown("#### Ingest Dataset")
st.caption("Upload a CSV file containing an SMS text column, or test with our sample dataset.")

c_upload, c_sample = st.columns([2, 1])

with c_sample:
    sample_csv_path = PROJECT_ROOT / "data" / "sample_bulk_sms.csv"
    if not sample_csv_path.exists():
        generate_sample_bulk_csv(sample_csv_path)

    sample_df_preview = pd.read_csv(sample_csv_path)
    sample_csv_bytes = sample_df_preview.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Sample CSV",
        data=sample_csv_bytes,
        file_name="sample_bulk_sms.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("Load 25-Message Sample", use_container_width=True):
        st.session_state["bulk_csv_data"] = sample_df_preview
        st.session_state["bulk_csv_name"] = "sample_bulk_sms.csv"

with c_upload:
    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            parsed_df, candidate_cols = parse_and_validate_csv(uploaded_file)
            st.session_state["bulk_csv_data"] = parsed_df
            st.session_state["bulk_csv_name"] = uploaded_file.name
        except Exception as ex:
            st.error(f"CSV parsing error: {ex}")

active_df = st.session_state.get("bulk_csv_data", None)

if active_df is None:
    st.info("Upload a CSV file or load the sample dataset to proceed.")
    st.stop()
else:
    st.write("")

    # -----------------------------------------------------------------------------
    # 2. Column Selection & Run
    # -----------------------------------------------------------------------------
    col_names = list(active_df.columns)
    default_idx = 0
    for i, col in enumerate(col_names):
        if any(k in col.lower() for k in ["text", "message", "sms", "body", "content"]):
            default_idx = i
            break

    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        msg_col = st.selectbox(
            f"Select message column ({len(active_df):,} rows in {st.session_state.get('bulk_csv_name', 'file')}):",
            col_names,
            index=default_idx
        )
    with c_btn:
        st.write("")
        st.write("")
        run_batch = st.button("Run Batch Classification", type="primary", use_container_width=True)

    if run_batch or "bulk_results_df" in st.session_state:
        if run_batch:
            with st.spinner(f"Classifying {len(active_df):,} messages across three models..."):
                try:
                    results_df, summary = run_bulk_classification(active_df, msg_col, predictor=predictor)
                    st.session_state["bulk_results_df"] = results_df
                    st.session_state["bulk_summary"] = summary
                except Exception as ex:
                    st.error(f"Classification error: {ex}")
                    st.stop()

        results_df = st.session_state["bulk_results_df"]
        summary = st.session_state["bulk_summary"]

        st.write("")

        # -------------------------------------------------------------------------
        # 3. Compact Summary Metrics
        # -------------------------------------------------------------------------
        st.markdown("#### Batch Summary")

        k1, k2, k3, k4 = st.columns(4)
        render_kpi_card(k1, "Total Processed", f"{summary['total_messages']:,}", "Analyzed messages")
        render_kpi_card(k2, "Consensus Spam", f"{summary['spam_count']:,}", f"{summary['spam_percentage']:.1f}% of total")
        render_kpi_card(k3, "Consensus Ham", f"{summary['ham_count']:,}", f"{100 - summary['spam_percentage']:.1f}% of total")
        render_kpi_card(k4, "Unanimous Agreement", f"{summary['unanimous_percentage']:.1f}%", f"{summary['unanimous_consensus_count']} messages (3/3)")

        st.write("")

        # -------------------------------------------------------------------------
        # 4. Minimal Visualizations
        # -------------------------------------------------------------------------
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.caption("Batch Label Breakdown")
            donut_batch = create_donut_chart(
                labels=["Legitimate (Ham)", "Spam"],
                values=[summary["ham_count"], summary["spam_count"]],
                colors=[SECONDARY_ACCENT, DANGER_RED],
                center_text=f"{summary['total_messages']:,}",
                height=260
            )
            st.plotly_chart(donut_batch, use_container_width=True)

        with ch_col2:
            st.caption("Per-Model Spam Counts")
            fig_models = go.Figure()
            m_names = ["Naive Bayes", "Log. Reg.", "Linear SVM"]
            m_counts = [summary["nb_spam_count"], summary["lr_spam_count"], summary["svm_spam_count"]]
            fig_models.add_trace(
                go.Bar(
                    x=m_names,
                    y=m_counts,
                    marker_color=[PRIMARY_ACCENT, SECONDARY_ACCENT, SUCCESS_GREEN],
                    text=[str(c) for c in m_counts],
                    textposition="outside",
                    hovertemplate="Model: <b>%{x}</b><br>Spam: %{y}<extra></extra>"
                )
            )
            apply_plotly_theme(fig_models, height=260)
            fig_models.update_layout(yaxis=dict(title="Spam Count"))
            st.plotly_chart(fig_models, use_container_width=True)

        st.write("")

        # -------------------------------------------------------------------------
        # 5. Search & Filter Controls
        # -------------------------------------------------------------------------
        st.markdown("#### Message Explorer")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            consensus_filter = st.selectbox("Consensus Verdict:", ["All", "SPAM Only", "HAM Only"])
        with f_col2:
            agreement_filter = st.selectbox("Model Agreement:", ["All Messages", "Unanimous (3/3)", "Divided (2/1)"])
        with f_col3:
            search_query = st.text_input("Filter by keyword:", placeholder="Type a keyword to filter...")

        filtered_df = results_df.copy()

        if consensus_filter == "SPAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "SPAM"]
        elif consensus_filter == "HAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "HAM"]

        if agreement_filter == "Unanimous (3/3)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["3/3", "0/3"])]
        elif agreement_filter == "Divided (2/1)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["2/3", "1/3"])]

        if search_query.strip():
            filtered_df = filtered_df[filtered_df["Message"].str.contains(search_query.strip(), case=False, na=False)]

        st.caption(f"Showing {len(filtered_df):,} of {len(results_df):,} messages")

        display_cols = [
            "Message_ID", "Message", "Consensus_Prediction", "Spam_Votes",
            "NB_Prediction", "NB_Spam_Probability",
            "LR_Prediction", "LR_Spam_Probability",
            "SVM_Prediction", "SVM_Spam_Probability"
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.write("")

        # -------------------------------------------------------------------------
        # 6. Download Button
        # -------------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Full Results (CSV)",
            data=export_csv,
            file_name=f"sms_shield_bulk_{timestamp}.csv",
            mime="text/csv",
            type="primary"
        )
