"""
SMS Shield 2.0 - Shared UI Components & Cybersecurity Styling Module
Author: SMS Shield NLP Project

Provides consistent visual identity, custom CSS styling, navigation headers,
and cached artifact loaders across the multi-page Streamlit application.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
METRICS_DIR = PROJECT_ROOT / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"


def inject_custom_css():
    """Injects professional cybersecurity-themed CSS styles."""
    st.markdown(
        """
        <style>
            /* Base layout & typography */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            /* Professional Header Banner */
            .shield-header {
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e3a8a 100%);
                border: 1px solid #312e81;
                border-radius: 14px;
                padding: 24px 28px;
                color: #ffffff;
                margin-bottom: 24px;
                box-shadow: 0 4px 20px rgba(15, 23, 42, 0.15);
            }
            .shield-title {
                font-size: 2.1rem;
                font-weight: 800;
                letter-spacing: -0.5px;
                margin: 0;
                background: linear-gradient(90deg, #60a5fa, #a78bfa, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .shield-subtitle {
                font-size: 1.02rem;
                font-weight: 500;
                color: #cbd5e1;
                margin: 6px 0 0 0;
            }
            .shield-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
            }
            .shield-badge {
                font-size: 0.76rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 3px 9px;
                color: #e2e8f0;
            }
            
            /* Card Containers */
            .cyber-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 18px 22px;
                margin-bottom: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                transition: all 0.2s ease;
            }
            .cyber-card:hover {
                border-color: #6366f1;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
            }
            
            /* Verdict Badges */
            .verdict-spam {
                background-color: #fef2f2;
                border: 1.5px solid #ef4444;
                color: #991b1b;
                font-weight: 700;
                font-size: 1.1rem;
                padding: 6px 14px;
                border-radius: 8px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .verdict-ham {
                background-color: #f0fdf4;
                border: 1.5px solid #22c55e;
                color: #166534;
                font-weight: 700;
                font-size: 1.1rem;
                padding: 6px 14px;
                border-radius: 8px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            
            /* Code / token styling */
            code {
                font-family: 'JetBrains Mono', monospace !important;
                background-color: #f1f5f9;
                color: #0f172a;
                padding: 2px 6px;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(page_title: str, page_description: str):
    """Renders a consistent hero banner on every page."""
    inject_custom_css()
    st.markdown(
        f"""
        <div class="shield-header">
            <div class="shield-title">
                🛡️ SMS SHIELD 2.0
            </div>
            <div class="shield-subtitle">
                Intelligent SMS Spam Detection & NLP Analytics Platform
            </div>
            <div class="shield-badges">
                <span class="shield-badge">Three-Model Consensus</span>
                <span class="shield-badge">Explainable NLP</span>
                <span class="shield-badge">Zero Data Leakage</span>
                <span class="shield-badge">Local Execution</span>
                <span class="shield-badge" style="background: rgba(99, 102, 241, 0.3); border-color: #818cf8; color: #e0e7ff;">{page_title}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if page_description:
        st.markdown(f"*{page_description}*")
        st.write("")


def render_sidebar_status():
    """Renders persistent project controls and health status in the sidebar."""
    with st.sidebar:
        st.markdown("### 🛡️ System Monitor")
        models_exist = (
            (MODELS_DIR / "tfidf_vectorizer.joblib").exists() and
            (MODELS_DIR / "multinomial_nb.joblib").exists() and
            (MODELS_DIR / "logistic_regression.joblib").exists() and
            (MODELS_DIR / "linear_svc.joblib").exists()
        )
        if models_exist:
            st.success("🟢 **Models Active**: 3/3 Loaded\n- Multinomial Naive Bayes\n- Logistic Regression\n- Linear SVM")
        else:
            st.warning("⚠️ **Models Incomplete**: Run Training")

        st.markdown("---")
        st.markdown("**Navigation Pages**")
        st.markdown(
            """
            1. **📊 Dashboard**: Global Dataset & Model KPIs
            2. **🔍 Message Analyzer**: 3-Model Live Classification
            3. **🔬 NLP Analysis**: Step-by-Step Transformations & N-Grams
            4. **📈 Model Evaluation**: ROC, PR Curves & Experiments
            5. **📁 Bulk SMS Analysis**: Batch CSV Inference & Reports
            """
        )
        st.markdown("---")

        if st.button("🔄 Retrain All Models", use_container_width=True):
            with st.spinner("Retraining all models & regenerating evaluation curves..."):
                try:
                    from src.train import run_training_pipeline
                    run_training_pipeline()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Retrained successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Retraining failed: {ex}")

        st.caption("SMS Shield 2.0 • Academic NLP Mini-Project")


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

    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation metrics at {eval_csv.resolve()}. Run `python src/train.py`.")

    return {
        "metrics_df": pd.read_csv(eval_csv),
        "cv_df": pd.read_csv(cv_csv) if cv_csv.exists() else None,
        "experiments_df": pd.read_csv(exp_csv) if exp_csv.exists() else None,
        "baseline_df": pd.read_csv(baseline_csv) if baseline_csv.exists() else None,
    }
