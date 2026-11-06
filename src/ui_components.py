"""
SMS Shield 4.0 - Shared UI Components & Cybersecurity Styling Module
Author: SMS Shield NLP Project

Provides consistent visual identity, custom SaaS CSS styling, navigation headers,
status telemetry, cached artifact loaders, and UI layout primitives.
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
    PRIMARY_VIOLET,
    CYAN_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_MUTED,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
METRICS_DIR = PROJECT_ROOT / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"


def inject_custom_css():
    """Injects professional cybersecurity-themed CSS styles."""
    inject_theme_css()


def render_header(page_title: str, page_description: str):
    """Renders a consistent cybersecurity SaaS hero banner across all pages."""
    inject_theme_css()
    st.markdown(
        f"""
        <div class="cyber-hero">
            <div class="cyber-hero-title-row">
                <div>
                    <h1 class="cyber-title">
                        <span>🛡️ SMS SHIELD 4.0</span>
                    </h1>
                    <div class="cyber-subtitle">
                        AI-Powered SMS Threat Detection & Intelligence Platform
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="status-pill active">
                        <span class="status-dot online"></span> ENGINE: ONLINE
                    </span>
                    <span class="status-pill violet" style="margin-left: 6px;">
                        v4.0 PRO
                    </span>
                </div>
            </div>
            <div class="cyber-telemetry-row">
                <span class="status-pill">Word & Char TF-IDF</span>
                <span class="status-pill">SMS Normalization Engine</span>
                <span class="status-pill">Adversarial Lab</span>
                <span class="status-pill">Zero Data Leakage</span>
                <span class="status-pill success" style="margin-left: auto;">
                    {page_title}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if page_description:
        st.markdown(
            f"""
            <div style="color: #94A3B8; font-size: 0.95rem; margin-top: -12px; margin-bottom: 20px;">
                💡 <em>{page_description}</em>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_status():
    """Renders persistent project controls, live telemetry, and model status in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 14px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                    🛡️ SMS SHIELD <span style="font-size: 0.75rem; background: #8B5CF6; color: white; padding: 2px 7px; border-radius: 9999px;">v4.0</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 3px;">
                    Threat Detection Platform
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
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; font-weight: 700; color: #34D399; text-transform: uppercase;">
                        <span class="status-dot online"></span> Threat Engine Active
                    </div>
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 6px; line-height: 1.4;">
                        • Multinomial Naive Bayes<br>
                        • Logistic Regression (L2)<br>
                        • Linear SVM (Dual Solver)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #F59E0B;">
                        ⚠️ Models Incomplete
                    </div>
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 4px;">
                        Trigger model training to initialize artifacts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.5px; margin-bottom: 8px;">
                Platform Modules
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            - 📊 **Dashboard**: Threat Overview & Imbalance
            - 🔍 **Message Analyzer**: Live 3-Model Consensus
            - 🔬 **NLP Intelligence**: 7-Stage Pipeline & N-Grams
            - 📈 **Model Performance**: ROC/PR Curves & Splits
            - 📁 **Bulk Analysis**: Batch CSV Scans & Reports
            - 🛡️ **Robustness Lab**: Adversarial Evasion Testing
            """
        )

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

        if st.button("🔄 Retrain Defense Models", use_container_width=True, type="secondary"):
            with st.spinner("Retraining all models & regenerating curves..."):
                try:
                    from src.train import run_training_pipeline
                    run_training_pipeline()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Retrained successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Retraining failed: {ex}")

        st.markdown(
            """
            <div style="border-top: 1px solid #1E293B; margin-top: 18px; padding-top: 12px; font-size: 0.74rem; color: #64748B;">
                🔒 Zero Data Leakage Enforced<br>
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
    """Renders a styled high-contrast cyber KPI card into a Streamlit column."""
    with col:
        border_style = f"border-top: 3px solid {accent_border};" if accent_border else ""
        delta_html = f"<div class='cyber-kpi-delta'>{delta}</div>" if delta else ""
        st.markdown(
            f"""
            <div class="cyber-kpi-card" style="{border_style}">
                <div>
                    <div class="cyber-kpi-label">{label}</div>
                    <div class="cyber-kpi-value">{value}</div>
                </div>
                {delta_html}
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
