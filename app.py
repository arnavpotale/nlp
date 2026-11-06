"""
SMS Shield — NLP Spam Classification Dashboard
Title: Implementation and Comparative Analysis of NLP-Based SMS Spam Classification Techniques
Technology Stack: Python, Streamlit, scikit-learn, NLTK, pandas, NumPy, Matplotlib, Seaborn, Joblib
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="SMS Shield — NLP Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Ensure local imports work reliably
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_text, preprocess_with_steps
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_model_comparison_bar


# -----------------------------------------------------------------------------
# Cached Loaders: Models, Vectorizer, Dataset, and Evaluation Metrics
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_vectorizer():
    """
    Loads persisted vectorizer and trained models from the models/ directory.
    Returns (vectorizer, nb_model, lr_model) or raises FileNotFoundError.
    """
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    nb_path = MODELS_DIR / "multinomial_nb.joblib"
    lr_path = MODELS_DIR / "logistic_regression.joblib"

    missing = [str(p.name) for p in [vec_path, nb_path, lr_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifact(s): {', '.join(missing)}. "
            "Please run the training script: `python src/train.py`."
        )

    vectorizer = joblib.load(vec_path)
    nb_model = joblib.load(nb_path)
    lr_model = joblib.load(lr_path)
    return vectorizer, nb_model, lr_model


@st.cache_data
def load_and_inspect_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Loads raw UCI dataset and returns (raw_df, deduplicated_df, duplicates_count).
    Provides clear error messages if dataset is missing.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH.resolve()}'. "
            "Please ensure 'SMSSpamCollection' is placed inside the 'data/' folder."
        )

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


@st.cache_data
def load_evaluation_metrics() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads actual saved evaluation results and classification reports."""
    metrics_csv = METRICS_DIR / "evaluation_metrics.csv"
    nb_rep_csv = METRICS_DIR / "classification_report_nb.csv"
    lr_rep_csv = METRICS_DIR / "classification_report_lr.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_csv.resolve()}'. "
            "Please execute `python src/train.py` to generate actual evaluation records."
        )

    metrics_df = pd.read_csv(metrics_csv)
    nb_rep_df = pd.read_csv(nb_rep_csv, index_col=0) if nb_rep_csv.exists() else None
    lr_rep_df = pd.read_csv(lr_rep_csv, index_col=0) if lr_rep_csv.exists() else None

    return metrics_df, nb_rep_df, lr_rep_df


# -----------------------------------------------------------------------------
# Main Application Header and Sidebar
# -----------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                🛡️ SMS Shield
            </h1>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.95;">
                Implementation and Comparative Analysis of NLP-Based SMS Spam Classification Techniques
            </p>
            <p style="margin: 8px 0 0 0; font-size: 0.85rem; opacity: 0.8;">
                Algorithms: <strong>Multinomial Naive Bayes</strong> & <strong>Logistic Regression</strong> | Dataset: <strong>UCI SMS Spam Collection</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Project Control Panel")
        st.markdown(
            """
            **NLP Mini-Project Highlights**
            - **Text Cleaning**: Lowercase, punctuation removal
            - **Tokenization**: NLTK `word_tokenize`
            - **Stopwords**: NLTK English stopword filter
            - **Stemming**: NLTK Porter Stemmer
            - **Feature Extraction**: TF-IDF (Max 5,000 features)
            - **Evaluation Split**: 80/20 Stratified (`random_state=42`)
            - **Data Leakage Defense**: TF-IDF fit strictly on train split
            """
        )
        st.divider()

        # Training status check
        models_ready = (
            (MODELS_DIR / "tfidf_vectorizer.joblib").exists() and
            (MODELS_DIR / "multinomial_nb.joblib").exists() and
            (MODELS_DIR / "logistic_regression.joblib").exists()
        )

        if models_ready:
            st.success("✅ Models & Vectorizer Loaded")
        else:
            st.warning("⚠️ Models not found. Click below to train.")

        if st.button("🚀 Re-train Models Now", use_container_width=True):
            with st.spinner("Training models and computing metrics..."):
                try:
                    from src.train import run_training_pipeline
                    run_training_pipeline()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Training complete! Artifacts updated.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Error during training: {err}")

        st.divider()
        st.caption("SMS Shield NLP Project • Local Execution Only")


# -----------------------------------------------------------------------------
# Section 1: Dataset Overview & Class Distribution
# -----------------------------------------------------------------------------
def render_dataset_overview(raw_df: pd.DataFrame, clean_df: pd.DataFrame, duplicates_count: int):
    st.subheader("1. Dataset Overview and Class Distribution")
    st.write(
        "The **UCI SMS Spam Collection** contains SMS messages tagged as either **ham** (legitimate) "
        "or **spam** (unwanted promotional/phishing messages). Below is the exploratory analysis "
        "and deduplication breakdown."
    )

    # Metric KPI Cards
    total_raw = len(raw_df)
    total_clean = len(clean_df)
    ham_count = (clean_df["label"] == "ham").sum()
    spam_count = (clean_df["label"] == "spam").sum()
    ham_pct = (ham_count / total_clean) * 100
    spam_pct = (spam_count / total_clean) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Raw Messages", f"{total_raw:,}")
    with col2:
        st.metric("Duplicate Messages Removed", f"{duplicates_count:,}")
    with col3:
        st.metric("Clean Unique Messages", f"{total_clean:,}")
    with col4:
        st.metric("Spam Prevalence", f"{spam_pct:.1f}%", f"{ham_pct:.1f}% Ham")

    st.markdown("---")

    # Two column layout: Class Distribution Plots & Character Length Analysis
    col_chart, col_stats = st.columns([1, 1])

    with col_chart:
        st.markdown("##### 📊 Class Distribution (Ham vs. Spam)")
        fig_dist, ax = plt.subplots(figsize=(6, 4), dpi=150)
        counts = clean_df["label"].value_counts()
        palette = {"ham": "#2563eb", "spam": "#dc2626"}
        sns.barplot(
            x=counts.index.str.upper(),
            y=counts.values,
            hue=counts.index,
            palette=palette,
            legend=False,
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )
        ax.set_ylabel("Message Count", fontsize=10, fontweight="bold")
        ax.set_xlabel("Class", fontsize=10, fontweight="bold")
        ax.set_title("Deduplicated Class Counts", fontsize=11, fontweight="bold")
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(
                f"{int(h):,} ({h/total_clean:.1%})",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points"
            )
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    with col_stats:
        st.markdown("##### 📏 Message Length & Character Distribution")
        fig_len, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
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
        ax2.set_xlim(0, 300)
        ax2.set_xlabel("Character Count per SMS", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax2.set_title("SMS Character Length: Ham vs Spam", fontsize=11, fontweight="bold")
        ax2.legend(title="Label", labels=["Spam", "Ham"])
        plt.tight_layout()
        st.pyplot(fig_len)
        plt.close(fig_len)

    # Key takeaway alert
    st.info(
        "💡 **Data Observation**: Legitimate (Ham) messages show a wide variance in length but average around ~70 characters. "
        "Spam messages cluster near the carrier limit of ~130–160 characters, maximizing advertising and promotional content."
    )

    # Sample data table explorer
    with st.expander("🔎 Browse Sample Dataset Messages"):
        filter_choice = st.radio(
            "Filter messages by category:",
            ["All", "Ham (Legitimate)", "Spam (Unwanted)"],
            horizontal=True
        )
        if filter_choice == "Ham (Legitimate)":
            sample_df = clean_df[clean_df["label"] == "ham"]
        elif filter_choice == "Spam (Unwanted)":
            sample_df = clean_df[clean_df["label"] == "spam"]
        else:
            sample_df = clean_df

        st.dataframe(
            sample_df[["label", "message", "char_length", "word_count"]].head(15),
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# Section 2: Live SMS Message Prediction (Both Models)
# -----------------------------------------------------------------------------
def render_prediction_section(vectorizer, nb_model, lr_model):
    st.subheader("2. SMS Message Prediction Using Both Models")
    st.write(
        "Type or paste any SMS text below to observe how **Multinomial Naive Bayes** and "
        "**Logistic Regression** classify the message and output model probabilities."
    )

    # Quick test sample presets
    preset_samples = {
        "Lottery Scam (Spam)": (
            "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! "
            "To claim call 09061701461. Claim code KL341. Valid 12 hours only."
        ),
        "Friendly Lunch (Ham)": (
            "Hey, are we still meeting for lunch at 1 PM today? Let me know if you want to invite Sarah too."
        ),
        "Urgent Account Alert (Spam)": (
            "URGENT! Your mobile account has won a £1,000 cash bonus. Call 08712460324 to claim before midnight. T&Cs apply."
        ),
        "Short Casual Chat (Ham)": (
            "Ok lar... Joking wif u oni... I'll see you at home later!"
        ),
    }

    st.markdown("**Quick Preset Examples (click to populate):**")
    btn_cols = st.columns(len(preset_samples))
    selected_sample = None

    for i, (name, sample_text) in enumerate(preset_samples.items()):
        if btn_cols[i].button(name, use_container_width=True):
            st.session_state["sms_input_text"] = sample_text

    default_val = st.session_state.get(
        "sms_input_text",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question!"
    )

    user_input = st.text_area(
        "Enter SMS message text to analyze:",
        value=default_val,
        height=100,
        placeholder="Type an SMS message here..."
    )

    # Defensive check: empty message handling
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter a non-empty SMS message above to run classification.")
        return

    # NLP Preprocessing breakdown
    step_results = preprocess_with_steps(user_input)
    clean_text = step_results["final_preprocessed"]

    # Expandable step-by-step NLP inspection
    with st.expander("🛠️ View Step-by-Step NLP Preprocessing Transformations", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**1. Original Raw Input:**\n`{step_results['raw']}`")
            st.markdown(f"**2. Lowercased:**\n`{step_results['lowercased']}`")
            st.markdown(f"**3. Punctuation Removed:**\n`{step_results['no_punctuation']}`")
        with c2:
            st.markdown(f"**4. Tokens Identified ({len(step_results['tokens'])}):**\n`{step_results['tokens']}`")
            st.markdown(f"**5. Stopwords Removed ({len(step_results['filtered_tokens'])} left):**\n`{step_results['filtered_tokens']}`")
            st.markdown(f"**6. Final Stemmed String:**\n`{clean_text if clean_text else '(No informative tokens remaining)'}`")

    if not clean_text:
        st.warning(
            "⚠️ Notice: All tokens in the entered message were removed during stopword and punctuation filtering. "
            "Classification will be based on baseline model priors."
        )

    # TF-IDF Transform
    input_vec = vectorizer.transform([clean_text])

    # Model Predictions & Probabilities
    nb_pred = int(nb_model.predict(input_vec)[0])
    nb_proba = nb_model.predict_proba(input_vec)[0]  # [P(ham), P(spam)]

    lr_pred = int(lr_model.predict(input_vec)[0])
    lr_proba = lr_model.predict_proba(input_vec)[0]  # [P(ham), P(spam)]

    # Display side-by-side results
    st.markdown("#### 🎯 Classification Verdicts")
    res_col1, res_col2 = st.columns(2)

    # Helper function for rendering model card
    def render_model_result(col, title, pred, proba, border_color):
        with col:
            is_spam = (pred == 1)
            verdict_badge = "🚨 SPAM (Unwanted)" if is_spam else "🟢 HAM (Legitimate)"
            badge_bg = "#fee2e2" if is_spam else "#dcfce7"
            badge_text = "#991b1b" if is_spam else "#166534"

            st.markdown(
                f"""
                <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 16px; background-color: #f8fafc; margin-bottom: 12px;">
                    <h4 style="margin: 0 0 10px 0; color: #1e293b;">{title}</h4>
                    <div style="display: inline-block; background-color: {badge_bg}; color: {badge_text}; font-weight: bold; font-size: 1.15rem; padding: 6px 14px; border-radius: 8px; margin-bottom: 12px;">
                        {verdict_badge}
                    </div>
                    <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Model Probability (Spam):</strong> {proba[1]:.2%}</p>
                    <p style="margin: 4px 0; font-size: 0.95rem;"><strong>Model Probability (Ham):</strong> {proba[0]:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Progress bar for visual spam probability
            st.progress(float(proba[1]), text=f"Spam Probability: {proba[1]:.2%}")

    render_model_result(res_col1, "Model 1: Multinomial Naive Bayes", nb_pred, nb_proba, "#2563eb")
    render_model_result(res_col2, "Model 2: Logistic Regression", lr_pred, lr_proba, "#059669")

    # Required explicit probability disclaimer
    st.info(
        "⚠️ **Statistical Confidence Notice**: The percentages displayed above represent "
        "**raw statistical model output probabilities** computed directly from the TF-IDF feature distributions "
        "and training priors, **rather than calibrated real-world risk scores**."
    )

    # Agreement banner
    if nb_pred == lr_pred:
        verdict_str = "SPAM" if nb_pred == 1 else "HAM"
        st.success(f"🤝 **Consensus**: Both models unanimously agree that this message is **{verdict_str}**.")
    else:
        st.warning(
            "⚠️ **Model Divergence**: The models disagree on this message. "
            f"Naive Bayes predicted **{'SPAM' if nb_pred == 1 else 'HAM'}** while "
            f"Logistic Regression predicted **{'SPAM' if lr_pred == 1 else 'HAM'}**. "
            "This can occur with subtle promotional or borderline messages."
        )


# -----------------------------------------------------------------------------
# Section 3: Performance Comparison Table & Bar Chart
# -----------------------------------------------------------------------------
def render_performance_comparison(metrics_df: pd.DataFrame):
    st.subheader("3. Performance Comparison Table and Bar Chart")
    st.write(
        "Both models were trained on 80% (4,136 messages) and evaluated on a held-out test split "
        "of 20% (1,035 messages) using stratified sampling (`random_state=42`)."
    )

    # Format dataframe for display
    display_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f} ({x * 100:.2f}%)")

    display_df = display_df.rename(
        columns={
            "TN": "True Negatives (TN)",
            "FP": "False Positives (FP)",
            "FN": "False Negatives (FN)",
            "TP": "True Positives (TP)"
        }
    )

    # Table display
    st.markdown("##### 📋 Quantitative Metric Comparison")
    st.dataframe(display_df, use_container_width=True)

    # Grouped Bar Chart
    st.markdown("##### 📊 Comparative Performance Bar Chart")
    chart_col1, chart_col2 = st.columns([2.5, 1])

    with chart_col1:
        fig_comp = plot_model_comparison_bar(metrics_df)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

    with chart_col2:
        st.markdown("##### 📌 Key Analysis & Trade-offs")
        st.markdown(
            """
            - **Precision Priority**: In spam filtering, **Precision** is paramount. A false positive blocks a legitimate message (e.g. medical reminder, banking OTP, personal SMS).
            - **Multinomial Naive Bayes** achieved **~98.99% Precision**, producing only **1 False Positive** across 904 legitimate test messages.
            - **Logistic Regression** achieved **~96.97% Precision** with **3 False Positives**.
            - **Recall**: Both models caught ~73–75% of spam messages without aggressive threshold tuning, demonstrating strong baseline performance.
            """
        )


# -----------------------------------------------------------------------------
# Section 4: Confusion Matrices & Classification Reports
# -----------------------------------------------------------------------------
def render_confusion_matrices_and_reports(
    metrics_df: pd.DataFrame,
    nb_rep_df: Optional[pd.DataFrame],
    lr_rep_df: Optional[pd.DataFrame]
):
    st.subheader("4. Confusion Matrices and Classification Reports")
    st.write(
        "A confusion matrix illustrates the exact count of correct versus incorrect predictions, "
        "highlighting the operational trade-off between **False Positives** (Type I Error) and "
        "**False Negatives** (Type II Error)."
    )

    cm_col1, cm_col2 = st.columns(2)

    # Model A: Naive Bayes Confusion Matrix
    nb_row = metrics_df[metrics_df["Model"].str.contains("Naive Bayes")].iloc[0]
    nb_cm = np.array([
        [int(nb_row["TN"]), int(nb_row["FP"])],
        [int(nb_row["FN"]), int(nb_row["TP"])]
    ])

    with cm_col1:
        st.markdown("##### Multinomial Naive Bayes")
        fig_nb = plot_confusion_matrix(nb_cm, "Multinomial Naive Bayes", cmap="Blues")
        st.pyplot(fig_nb)
        plt.close(fig_nb)
        st.caption(
            f"**True Negatives:** {nb_row['TN']} | **False Positives:** {nb_row['FP']} | "
            f"**False Negatives:** {nb_row['FN']} | **True Positives:** {nb_row['TP']}"
        )

    # Model B: Logistic Regression Confusion Matrix
    lr_row = metrics_df[metrics_df["Model"].str.contains("Logistic")].iloc[0]
    lr_cm = np.array([
        [int(lr_row["TN"]), int(lr_row["FP"])],
        [int(lr_row["FN"]), int(lr_row["TP"])]
    ])

    with cm_col2:
        st.markdown("##### Logistic Regression")
        fig_lr = plot_confusion_matrix(lr_cm, "Logistic Regression", cmap="Greens")
        st.pyplot(fig_lr)
        plt.close(fig_lr)
        st.caption(
            f"**True Negatives:** {lr_row['TN']} | **False Positives:** {lr_row['FP']} | "
            f"**False Negatives:** {lr_row['FN']} | **True Positives:** {lr_row['TP']}"
        )

    st.markdown("---")

    # Classification Reports
    st.markdown("##### 📄 Detailed Per-Class Classification Reports")
    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        st.markdown("**Multinomial Naive Bayes Report**")
        if nb_rep_df is not None:
            st.dataframe(nb_rep_df.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("Report CSV not available. Please re-run training.")

    with rep_col2:
        st.markdown("**Logistic Regression Report**")
        if lr_rep_df is not None:
            st.dataframe(lr_rep_df.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("Report CSV not available. Please re-run training.")

    # Educational breakdown of error types
    st.markdown("##### 💡 Real-World Spam Classification Implications")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.error(
            "🚨 **False Positive (Type I Error)**: A legitimate SMS is wrongly classified as Spam.\n\n"
            "- **Consequence**: The user misses a vital notification (e.g. two-factor SMS OTP, delivery status, emergency alert).\n"
            "- **Target**: Minimize FP at all costs (high precision required)."
        )
    with exp_col2:
        st.warning(
            "⚠️ **False Negative (Type II Error)**: A spam SMS slips past the filter into the inbox.\n\n"
            "- **Consequence**: Minor annoyance or potential phishing risk, but no critical legitimate communications are lost.\n"
            "- **Target**: Keep FN reasonably low without sacrificing precision."
        )


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main():
    render_header()
    render_sidebar()

    # Load resources defensively
    try:
        raw_df, clean_df, duplicates_count = load_and_inspect_dataset()
        vectorizer, nb_model, lr_model = load_models_and_vectorizer()
        metrics_df, nb_rep_df, lr_rep_df = load_evaluation_metrics()
    except FileNotFoundError as fnf_err:
        st.error(f"⚠️ {fnf_err}")
        st.info("Click 'Re-train Models Now' in the sidebar or run `python src/train.py` from your terminal.")
        return
    except Exception as err:
        st.error(f"Unexpected error loading application resources: {err}")
        return

    # Tabs for the four required sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Dataset Overview & Distribution",
        "🔍 2. Live SMS Prediction",
        "📈 3. Performance Comparison",
        "🔲 4. Confusion Matrices & Reports"
    ])

    with tab1:
        render_dataset_overview(raw_df, clean_df, duplicates_count)

    with tab2:
        render_prediction_section(vectorizer, nb_model, lr_model)

    with tab3:
        render_performance_comparison(metrics_df)

    with tab4:
        render_confusion_matrices_and_reports(metrics_df, nb_rep_df, lr_rep_df)


if __name__ == "__main__":
    main()
