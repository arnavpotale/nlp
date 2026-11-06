"""
SMS Shield 4.0 - Shared UI Components & Minimalist Layout Module
Author: SMS Shield NLP Project

Provides unified SaaS styling, clean page headers, refined sidebar navigation,
standardized KPI cards, and cached dataset/model loaders.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import streamlit as st

from src.ui_theme import (
    inject_theme_css,
    BG_DARK,
    CARD_BG,
    BORDER_COLOR,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
METRICS_DIR = PROJECT_ROOT / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"


def inject_custom_css():
    """Injects refined minimalist SaaS CSS stylesheet."""
    inject_theme_css()


def render_header(page_title: str, page_description: str):
    """Renders a clean, minimalist SaaS page header without noisy borders or heavy gradients."""
    inject_theme_css()
    st.markdown(
        f"""
        <div class="saas-header">
            <h1 class="saas-header-title">
                {page_title}
            </h1>
            <div class="saas-header-subtitle">
                {page_description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_status():
    """Renders a clean, uncluttered sidebar with subtle branding and status."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #292F3D; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 1.15rem; font-weight: 600; color: #F1F5F9; letter-spacing: -0.01em;">
                        SMS Shield
                    </span>
                    <span class="saas-pill">v4.0</span>
                </div>
                <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 3px;">
                    NLP Threat Detection Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        models_exist = (
            (MODELS_DIR / "tfidf_vectorizer.joblib").exists() and
            (MODELS_DIR / "char_tfidf_vectorizer.joblib").exists() and
            (MODELS_DIR / "linear_svc.joblib").exists()
        )

        if models_exist:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #9CA3AF; margin-bottom: 18px;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #34D399; display: inline-block;"></span>
                    <span>3 Active Models (NB, LR, SVM)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #FBBF24; margin-bottom: 18px;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #FBBF24; display: inline-block;"></span>
                    <span>Models Incomplete — Retrain</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em; margin-bottom: 8px;">
                Platform Navigation
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #9CA3AF; line-height: 1.8;">
                <a href="/" target="_self" style="color: #F1F5F9; text-decoration: none;">• Overview</a><br>
                <a href="/Dashboard" target="_self" style="color: #9CA3AF; text-decoration: none;">• 1. Dashboard</a><br>
                <a href="/Message_Analyzer" target="_self" style="color: #9CA3AF; text-decoration: none;">• 2. Message Analyzer</a><br>
                <a href="/NLP_Analysis" target="_self" style="color: #9CA3AF; text-decoration: none;">• 3. NLP Analysis</a><br>
                <a href="/Model_Evaluation" target="_self" style="color: #9CA3AF; text-decoration: none;">• 4. Model Evaluation</a><br>
                <a href="/Bulk_Analysis" target="_self" style="color: #9CA3AF; text-decoration: none;">• 5. Bulk Analysis</a><br>
                <a href="/Robustness_Analysis" target="_self" style="color: #9CA3AF; text-decoration: none;">• 6. Robustness Lab</a>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        if st.button("🔄 Retrain Models", use_container_width=True):
            with st.spinner("Retraining models..."):
                try:
                    from src.train import run_training_pipeline
                    run_training_pipeline()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Retrained successfully.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Retraining error: {ex}")

        st.markdown(
            """
            <div style="border-top: 1px solid #292F3D; margin-top: 20px; padding-top: 12px; font-size: 0.72rem; color: #6B7280;">
                Zero Data Leakage Pipeline<br>
                Stratified 80/20 Test Split (n=1,035)
            </div>
            """,
            unsafe_allow_html=True
        )


def render_kpi_card(
    col,
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    accent_border: Optional[str] = None
):
    """Renders a clean, minimalist metric card into a Streamlit column."""
    with col:
        sub_html = f"<div class='saas-metric-sub'>{delta}</div>" if delta else ""
        st.markdown(
            f"""
            <div class="saas-metric-card">
                <div>
                    <div class="saas-metric-label">{label}</div>
                    <div class="saas-metric-value">{value}</div>
                </div>
                {sub_html}
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Loads and caches raw and deduplicated dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    try:
        raw_df = pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"], quoting=3, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"], quoting=3, encoding="latin-1")

    raw_df["label"] = raw_df["label"].astype(str).str.strip().str.lower()
    raw_df = raw_df[raw_df["label"].isin(["ham", "spam"])].reset_index(drop=True)

    duplicates_count = int(raw_df.duplicated(subset=["message"]).sum())
    clean_df = raw_df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    clean_df["char_length"] = clean_df["message"].apply(len)
    clean_df["word_count"] = clean_df["message"].apply(lambda s: len(s.split()))

    return raw_df, clean_df, duplicates_count


@st.cache_resource
def get_predictor():
    """Loads and caches the unified 3-model predictor."""
    from src.prediction import SMSShieldPredictor
    predictor = SMSShieldPredictor().load()
    return predictor


@st.cache_data
def get_evaluation_metrics() -> Dict[str, Any]:
    """Loads evaluation CSV files."""
    eval_csv = METRICS_DIR / "evaluation_metrics.csv"
    cv_csv = METRICS_DIR / "cross_val_metrics.csv"
    exp_csv = METRICS_DIR / "tfidf_experiments.csv"
    baseline_csv = METRICS_DIR / "baseline_metrics.csv"
    char_word_csv = METRICS_DIR / "char_word_comparison.csv"
    adv_csv = METRICS_DIR / "adversarial_evaluation.csv"

    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation metrics at {eval_csv.resolve()}. Run `python src/train.py`.")

    return {
        "metrics_df": pd.read_csv(eval_csv),
        "cv_df": pd.read_csv(cv_csv) if cv_csv.exists() else None,
        "experiments_df": pd.read_csv(exp_csv) if exp_csv.exists() else None,
        "baseline_df": pd.read_csv(baseline_csv) if baseline_csv.exists() else None,
        "char_word_df": pd.read_csv(char_word_csv) if char_word_csv.exists() else None,
        "adv_df": pd.read_csv(adv_csv) if adv_csv.exists() else None,
    }
