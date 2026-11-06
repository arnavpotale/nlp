"""
SMS Shield 2.0 - Page 5: Bulk SMS Analysis
Author: SMS Shield NLP Project

Upload batch CSV files of SMS messages, classify each with all three models,
inspect consensus distributions, filter results, and export downloadable reports.
"""

import sys
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_predictor
from src.bulk_analysis import parse_and_validate_csv, run_bulk_classification, generate_sample_bulk_csv


st.set_page_config(page_title="Bulk SMS Analysis | SMS Shield 2.0", page_icon="📁", layout="wide")
render_header("Page 5: Bulk SMS Classification & Batch Reporting", "Classify batches of SMS messages from uploaded CSV files with multi-model consensus and export reports.")
render_sidebar_status()

# Load predictor
try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# File Upload & Sample Loader
# -----------------------------------------------------------------------------
st.markdown("### 📤 Upload SMS Dataset (CSV)")
st.write(
    "Upload any CSV file containing a column of SMS text. "
    "Or click below to immediately test with our pre-packaged **25-message sample dataset**."
)

c_upload, c_sample = st.columns([2, 1])

with c_sample:
    st.markdown("##### ⚡ Don't have a CSV handy?")
    sample_csv_path = PROJECT_ROOT / "data" / "sample_bulk_sms.csv"
    if not sample_csv_path.exists():
        generate_sample_bulk_csv(sample_csv_path)

    sample_df_preview = pd.read_csv(sample_csv_path)
    sample_csv_bytes = sample_df_preview.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample Test CSV",
        data=sample_csv_bytes,
        file_name="sample_bulk_sms.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🚀 Load Pre-Packaged Sample Dataset", use_container_width=True, type="secondary"):
        st.session_state["bulk_csv_data"] = sample_df_preview
        st.session_state["bulk_csv_name"] = "sample_bulk_sms.csv"

with c_upload:
    uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
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
    st.divider()

    # -----------------------------------------------------------------------------
    # Column Selection & Execution
    # -----------------------------------------------------------------------------
    st.markdown(f"### ⚙️ Processing Settings (`{st.session_state.get('bulk_csv_name', 'dataset')}`)")
    st.write(f"Loaded **{len(active_df):,} rows** and **{len(active_df.columns)} columns**.")

    # Guess text column
    col_names = list(active_df.columns)
    default_idx = 0
    for i, col in enumerate(col_names):
        if any(k in col.lower() for k in ["text", "message", "sms", "body", "content"]):
            default_idx = i
            break

    msg_col = st.selectbox("Select the column containing SMS messages:", col_names, index=default_idx)

    # Action button
    run_batch = st.button("🚀 Run 3-Model Batch Classification", type="primary", use_container_width=True)

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
        # Batch Results Summary KPIs
        # -------------------------------------------------------------------------
        st.markdown("### 📊 Batch Classification Summary")

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Processed", f"{summary['total_messages']:,}")
        with k2:
            st.metric("Consensus Spam", f"{summary['spam_count']:,}", f"{summary['spam_percentage']:.1f}%")
        with k3:
            st.metric("Consensus Ham", f"{summary['ham_count']:,}", f"{100 - summary['spam_percentage']:.1f}%")
        with k4:
            st.metric("Unanimous Agreement", f"{summary['unanimous_percentage']:.1f}%", f"{summary['unanimous_consensus_count']} msgs")
        with k5:
            st.metric("Empty / Blank Rows", f"{summary['empty_count']:,}")

        # Charts: Distribution & Model Agreement Breakdown
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("##### Batch Spam vs. Ham Distribution")
            fig_batch, ax = plt.subplots(figsize=(5.5, 3.4), dpi=150)
            labels = ["HAM", "SPAM"]
            counts = [summary["ham_count"], summary["spam_count"]]
            colors = ["#2563eb", "#dc2626"]
            sns.barplot(x=labels, y=counts, palette=colors, edgecolor="black", ax=ax)
            ax.set_ylabel("Message Count", fontsize=9.5, fontweight="bold")
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
            plt.tight_layout()
            st.pyplot(fig_batch)
            plt.close(fig_batch)

        with chart_c2:
            st.markdown("##### Per-Model Spam Detections")
            fig_models, ax2 = plt.subplots(figsize=(5.5, 3.4), dpi=150)
            m_names = ["Naive Bayes", "Log. Regression", "Linear SVM"]
            m_counts = [summary["nb_spam_count"], summary["lr_spam_count"], summary["svm_spam_count"]]
            sns.barplot(x=m_names, y=m_counts, palette=["#2563eb", "#059669", "#7c3aed"], edgecolor="black", ax=ax2)
            ax2.set_ylabel("Spam Flagged", fontsize=9.5, fontweight="bold")
            for p in ax2.patches:
                h = p.get_height()
                ax2.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
            plt.tight_layout()
            st.pyplot(fig_models)
            plt.close(fig_models)

        st.divider()

        # -------------------------------------------------------------------------
        # Filtering & Interactive Explorer
        # -------------------------------------------------------------------------
        st.markdown("### 🔍 Search & Filter Batch Results")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            consensus_filter = st.selectbox("Consensus Category:", ["All", "SPAM Only", "HAM Only"])
        with f_col2:
            agreement_filter = st.selectbox("Model Agreement:", ["All Messages", "Unanimous Consensus Only (3/3)", "Divided Consensus Only (2/1)"])
        with f_col3:
            search_query = st.text_input("Filter messages by keyword:", placeholder="e.g. claim, call, free...")

        filtered_df = results_df.copy()

        if consensus_filter == "SPAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "SPAM"]
        elif consensus_filter == "HAM Only":
            filtered_df = filtered_df[filtered_df["Consensus_Prediction"] == "HAM"]

        if agreement_filter == "Unanimous Consensus Only (3/3)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["3/3", "0/3"])]
        elif agreement_filter == "Divided Consensus Only (2/1)":
            filtered_df = filtered_df[filtered_df["Spam_Votes"].isin(["2/3", "1/3"])]

        if search_query.strip():
            filtered_df = filtered_df[filtered_df["Message"].str.contains(search_query.strip(), case=False, na=False)]

        st.write(f"Showing **{len(filtered_df):,}** of **{len(results_df):,}** messages:")

        # Display columns
        display_cols = [
            "Message_ID", "Message", "Consensus_Prediction", "Spam_Votes",
            "NB_Prediction", "NB_Spam_Probability",
            "LR_Prediction", "LR_Spam_Probability",
            "SVM_Prediction", "SVM_Spam_Probability"
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.divider()

        # -------------------------------------------------------------------------
        # Export Download Button
        # -------------------------------------------------------------------------
        st.markdown("### 💾 Export Classification Report")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full Batch Predictions (CSV)",
            data=export_csv,
            file_name=f"sms_shield_bulk_predictions_{timestamp}.csv",
            mime="text/csv",
            type="primary"
        )
