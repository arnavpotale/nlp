"""
SMS Shield 4.0 - Page 5: Bulk Threat Analysis Engine
Author: SMS Shield NLP Project

Batch CSV processing engine: Upload arbitrary CSVs of SMS messages, execute 3-model
consensus classification, inspect agreement distributions, filter, and export audit reports.
"""

import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
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
    PRIMARY_VIOLET,
    CYAN_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_MUTED,
    apply_plotly_theme,
    create_donut_chart,
)
from src.bulk_analysis import parse_and_validate_csv, run_bulk_classification, generate_sample_bulk_csv

st.set_page_config(page_title="Bulk Analysis | SMS Shield 4.0", page_icon="📁", layout="wide")
render_header(
    "Bulk Threat Analysis Engine",
    "High-throughput batch CSV inspection, multi-algorithm agreement telemetry, and downloadable classification reports."
)
render_sidebar_status()

# Load predictor
try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. File Upload & Pre-Packaged Sample Loader
# -----------------------------------------------------------------------------
st.markdown("### 📤 Ingest Batch SMS Dataset (CSV)")
st.write(
    "Upload any CSV file containing a column of SMS messages, or immediately run "
    "a test scan with our pre-packaged **25-message evaluation dataset**."
)

c_upload, c_sample = st.columns([2, 1])

with c_sample:
    st.markdown("##### ⚡ Quick Test Sample")
    sample_csv_path = PROJECT_ROOT / "data" / "sample_bulk_sms.csv"
    if not sample_csv_path.exists():
        generate_sample_bulk_csv(sample_csv_path)

    sample_df_preview = pd.read_csv(sample_csv_path)
    sample_csv_bytes = sample_df_preview.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample CSV",
        data=sample_csv_bytes,
        file_name="sample_bulk_sms.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🚀 Load Pre-Packaged Sample Dataset", use_container_width=True, type="secondary"):
        st.session_state["bulk_csv_data"] = sample_df_preview
        st.session_state["bulk_csv_name"] = "sample_bulk_sms.csv"

with c_upload:
    uploaded_file = st.file_uploader("Choose a CSV file for threat scanning:", type=["csv"])
    if uploaded_file is not None:
        try:
            parsed_df, candidate_cols = parse_and_validate_csv(uploaded_file)
            st.session_state["bulk_csv_data"] = parsed_df
            st.session_state["bulk_csv_name"] = uploaded_file.name
        except Exception as ex:
            st.error(f"Failed to parse uploaded CSV: {ex}")

# Check active dataframe in session state
active_df = st.session_state.get("bulk_csv_data", None)

if active_df is None:
    st.info("👆 Please upload a CSV file or click 'Load Pre-Packaged Sample Dataset' to begin.")
    st.stop()
else:
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # 2. Column Selection & Execution Settings
    # -----------------------------------------------------------------------------
    st.markdown(f"### ⚙️ Scan Configuration (`{st.session_state.get('bulk_csv_name', 'dataset')}`)")
    st.caption(f"Loaded **{len(active_df):,} rows** and **{len(active_df.columns)} columns**.")

    col_names = list(active_df.columns)
    default_idx = 0
    for i, col in enumerate(col_names):
        if any(k in col.lower() for k in ["text", "message", "sms", "body", "content"]):
            default_idx = i
            break

    msg_col = st.selectbox("Select message payload column:", col_names, index=default_idx)

    run_batch = st.button("🚀 Execute 3-Model Batch Classification", type="primary", use_container_width=True)

    if run_batch or "bulk_results_df" in st.session_state:
        if run_batch:
            with st.spinner(f"Classifying {len(active_df):,} messages across Naive Bayes, Logistic Regression, and Linear SVM..."):
                try:
                    results_df, summary = run_bulk_classification(active_df, msg_col, predictor=predictor)
                    st.session_state["bulk_results_df"] = results_df
                    st.session_state["bulk_summary"] = summary
                except Exception as ex:
                    st.error(f"Batch processing error: {ex}")
                    st.stop()

        results_df = st.session_state["bulk_results_df"]
        summary = st.session_state["bulk_summary"]

        # -------------------------------------------------------------------------
        # 3. Batch Telemetry Row (5 KPI Cards)
        # -------------------------------------------------------------------------
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Batch Threat Telemetry")

        k1, k2, k3, k4, k5 = st.columns(5)
        render_kpi_card(k1, "Total Ingested", f"{summary['total_messages']:,}", "Analyzed Messages", accent_border=PRIMARY_VIOLET)
        render_kpi_card(k2, "Consensus Spam", f"{summary['spam_count']:,}", f"{summary['spam_percentage']:.1f}% Threat Ratio", accent_border=DANGER_RED)
        render_kpi_card(k3, "Consensus Ham", f"{summary['ham_count']:,}", f"{100 - summary['spam_percentage']:.1f}% Clean Ratio", accent_border=SUCCESS_GREEN)
        render_kpi_card(k4, "Unanimous Consensus", f"{summary['unanimous_percentage']:.1f}%", f"{summary['unanimous_consensus_count']} msgs (3/3)", accent_border=CYAN_ACCENT)
        render_kpi_card(k5, "Blank / Empty Rows", f"{summary['empty_count']:,}", "Skipped Payloads", accent_border="#64748B")

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------------------------------
        # 4. Interactive Plotly Batch Visualizations
        # -------------------------------------------------------------------------
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("##### 🍩 Batch Threat vs. Clean Ratio")
            donut_batch = create_donut_chart(
                labels=["Legitimate (Ham)", "Malicious (Spam)"],
                values=[summary["ham_count"], summary["spam_count"]],
                colors=[CYAN_ACCENT, DANGER_RED],
                title="",
                center_text=f"{summary['total_messages']:,} MSGS",
                height=320
            )
            st.plotly_chart(donut_batch, use_container_width=True)

        with chart_c2:
            st.markdown("##### 🤖 Model Detection Comparison")
            fig_models = go.Figure()
            m_names = ["Naive Bayes", "Log. Regression", "Linear SVM"]
            m_counts = [summary["nb_spam_count"], summary["lr_spam_count"], summary["svm_spam_count"]]
            fig_models.add_trace(
                go.Bar(
                    x=m_names,
                    y=m_counts,
                    marker_color=[PRIMARY_VIOLET, CYAN_ACCENT, SUCCESS_GREEN],
                    text=[f"{c} flagged" for c in m_counts],
                    textposition="outside",
                    hovertemplate="Model: <b>%{x}</b><br>Spam Count: %{y}<extra></extra>"
                )
            )
            apply_plotly_theme(fig_models, height=320)
            fig_models.update_layout(yaxis=dict(title="Spam Detections"))
            st.plotly_chart(fig_models, use_container_width=True)

        st.divider()

        # -------------------------------------------------------------------------
        # 5. Multi-Criteria Filtering & Interactive Explorer
        # -------------------------------------------------------------------------
        st.markdown("### 🔍 Search & Filter Batch Telemetry")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            consensus_filter = st.selectbox("Consensus Verdict:", ["All", "SPAM Only", "HAM Only"])
        with f_col2:
            agreement_filter = st.selectbox("Model Agreement:", ["All Messages", "Unanimous Consensus (3/3)", "Divided Consensus (2/1)"])
        with f_col3:
            search_query = st.text_input("Search messages by keyword:", placeholder="e.g. claim, call, free, account...")

        filtered_df = results_df.copy()

        if consensus_filter == "SPAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "SPAM"]
        elif consensus_filter == "HAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "HAM"]

        if agreement_filter == "Unanimous Consensus (3/3)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["3/3", "0/3"])]
        elif agreement_filter == "Divided Consensus (2/1)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["2/3", "1/3"])]

        if search_query.strip():
            filtered_df = filtered_df[filtered_df["Message"].str.contains(search_query.strip(), case=False, na=False)]

        st.caption(f"Displaying **{len(filtered_df):,}** of **{len(results_df):,}** records:")

        display_cols = [
            "Message_ID", "Message", "Consensus_Prediction", "Spam_Votes",
            "NB_Prediction", "NB_Spam_Probability",
            "LR_Prediction", "LR_Spam_Probability",
            "SVM_Prediction", "SVM_Spam_Probability"
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------------------------------
        # 6. Downloadable Export Button
        # -------------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full Batch Prediction Audit Report (CSV)",
            data=export_csv,
            file_name=f"sms_shield_bulk_predictions_{timestamp}.csv",
            mime="text/csv",
            type="primary"
        )
