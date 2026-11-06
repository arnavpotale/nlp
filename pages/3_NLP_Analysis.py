"""
SMS Shield 4.0 - Page 3: NLP Analysis
Author: SMS Shield NLP Project

Clear, educational walkthrough of the Natural Language Processing pipeline:
Sequential preprocessing stages, before-and-after comparisons, keyword analysis, and n-grams.
"""

import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from nltk.util import ngrams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    render_header,
    render_sidebar_status,
    get_dataset,
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
)
from src.preprocessing import preprocess_with_steps
from src.explainability import get_global_top_features

st.set_page_config(page_title="SMS Shield | NLP Analysis", page_icon="🔬", layout="wide")
render_header(
    "NLP Analysis",
    "Inspect step-by-step text preprocessing, suspicious vocabulary patterns, and n-gram distributions."
)
render_sidebar_status()

# Load data and predictor
try:
    raw_df, clean_df, _ = get_dataset()
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Step-by-Step Pipeline Walkthrough
# -----------------------------------------------------------------------------
st.markdown("#### Preprocessing Pipeline")
st.caption("Enter any SMS message to trace its transformation through the 6 NLP preprocessing steps.")

sample_demo = "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 88088 now."
user_msg = st.text_input("Message to trace:", value=sample_demo)

if user_msg.strip():
    steps = preprocess_with_steps(user_msg)
    clean_text = steps["final_preprocessed"]

    # Clean 2-column layout for steps 1-6
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.markdown(
            f"""
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 1 • Raw Ingestion</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #F1F5F9; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D;">{steps['raw']}</div>
            </div>
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 2 • Lowercase Conversion</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #F1F5F9; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D;">{steps['lowercased']}</div>
            </div>
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 3 • Punctuation Removal</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #F1F5F9; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D;">{steps['no_punctuation']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s_col2:
        st.markdown(
            f"""
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 4 • Word Tokenization ({len(steps['tokens'])} tokens)</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #F1F5F9; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D; word-break: break-all;">{str(steps['tokens'])}</div>
            </div>
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 5 • Stopword Removal ({len(steps['filtered_tokens'])} tokens)</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #F1F5F9; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D; word-break: break-all;">{str(steps['filtered_tokens'])}</div>
            </div>
            <div class="saas-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #8B7CF6;">Step 6 • Porter Stemming</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #34D399; background: #141720; padding: 8px 10px; border-radius: 4px; margin-top: 6px; border: 1px solid #292F3D; word-break: break-all;">{steps['final_preprocessed']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Step 7: TF-IDF Feature Representation Details"):
        if clean_text:
            tfidf_vec = predictor.vectorizer.transform([clean_text])
            feature_names = np.array(predictor.vectorizer.get_feature_names_out())
            non_zero_idx = tfidf_vec.nonzero()[1]

            if len(non_zero_idx) > 0:
                active_features = [
                    {"Feature": feature_names[i], "TF-IDF Weight": round(float(tfidf_vec[0, i]), 4)}
                    for i in non_zero_idx
                ]
                active_df = pd.DataFrame(active_features).sort_values(by="TF-IDF Weight", ascending=False)
                st.dataframe(active_df, use_container_width=True, hide_index=True)
            else:
                st.info("No tokens matched the TF-IDF vocabulary.")
        else:
            st.info("No tokens remaining after stopword filtering.")

st.write("")

# -----------------------------------------------------------------------------
# 2. Suspicious Keywords Analysis
# -----------------------------------------------------------------------------
st.markdown("#### High-Frequency Spam Keywords")
st.caption("Common promotional terms observed in the dataset. Note: these are descriptive indicators, not definitive proof of spam.")

keywords_data = [
    {"Keyword": "FREE", "Occurrences": 195, "Primary Context": "Promotional offers & bonuses"},
    {"Keyword": "CLAIM", "Occurrences": 113, "Primary Context": "Prize collection instructions"},
    {"Keyword": "PRIZE", "Occurrences": 92, "Primary Context": "Reward and contest notifications"},
    {"Keyword": "WINNER", "Occurrences": 63, "Primary Context": "Lottery and contest notices"},
    {"Keyword": "CASH", "Occurrences": 62, "Primary Context": "Monetary rewards and loans"},
    {"Keyword": "URGENT", "Occurrences": 57, "Primary Context": "High-pressure action demands"},
]
kw_df = pd.DataFrame(keywords_data)
st.dataframe(kw_df, use_container_width=True, hide_index=True)

st.write("")

# -----------------------------------------------------------------------------
# 3. Dataset-Wide N-Gram Distributions
# -----------------------------------------------------------------------------
st.markdown("#### Dataset N-Gram Distributions")
st.caption("Comparing multi-word sequences across legitimate (Ham) and promotional (Spam) messages.")

@st.cache_data
def compute_dataset_ngrams(df: pd.DataFrame):
    """Extracts top n-grams for Ham and Spam subsets."""
    spam_clean = df[df["label"] == "spam"]["clean_message"].dropna().tolist()
    ham_clean = df[df["label"] == "ham"]["clean_message"].dropna().tolist()

    return {
        "spam_uni": Counter([w for s in spam_clean for w in s.split()]).most_common(10),
        "ham_uni": Counter([w for s in ham_clean for w in s.split()]).most_common(10),
        "spam_bi": Counter([" ".join(bg) for s in spam_clean for bg in ngrams(s.split(), 2)]).most_common(10),
        "ham_bi": Counter([" ".join(bg) for s in ham_clean for bg in ngrams(s.split(), 2)]).most_common(10),
        "spam_tri": Counter([" ".join(tg) for s in spam_clean for tg in ngrams(s.split(), 3)]).most_common(10),
        "ham_tri": Counter([" ".join(tg) for s in ham_clean for tg in ngrams(s.split(), 3)]).most_common(10),
    }

if "clean_message" not in clean_df.columns:
    from src.preprocessing import preprocess_text
    clean_df["clean_message"] = clean_df["message"].apply(preprocess_text)

ngram_data = compute_dataset_ngrams(clean_df)

def render_clean_ngrams(spam_items, ham_items, title_suffix):
    s_terms, s_counts = zip(*spam_items)
    h_terms, h_counts = zip(*ham_items)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Spam {title_suffix}", f"Ham {title_suffix}"),
        horizontal_spacing=0.14
    )
    fig.add_trace(
        go.Bar(
            y=list(s_terms),
            x=list(s_counts),
            orientation="h",
            marker_color=DANGER_RED,
            name="Spam",
            hovertemplate="Phrase: <b>%{y}</b><br>Count: %{x}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=list(h_terms),
            x=list(h_counts),
            orientation="h",
            marker_color=SECONDARY_ACCENT,
            name="Ham",
            hovertemplate="Phrase: <b>%{y}</b><br>Count: %{x}<extra></extra>"
        ),
        row=1, col=2
    )
    apply_plotly_theme(fig, height=300)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return fig

t_uni, t_bi, t_tri = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

with t_uni:
    fig_u = render_clean_ngrams(ngram_data["spam_uni"], ngram_data["ham_uni"], "Unigrams")
    st.plotly_chart(fig_u, use_container_width=True)

with t_bi:
    fig_b = render_clean_ngrams(ngram_data["spam_bi"], ngram_data["ham_bi"], "Bigrams")
    st.plotly_chart(fig_b, use_container_width=True)

with t_tri:
    fig_t = render_clean_ngrams(ngram_data["spam_tri"], ngram_data["ham_tri"], "Trigrams")
    st.plotly_chart(fig_t, use_container_width=True)

st.write("")

# -----------------------------------------------------------------------------
# 4. Global Feature Weights
# -----------------------------------------------------------------------------
st.markdown("#### Global TF-IDF Feature Weights")
st.caption("Vocabulary features with the strongest weights in the linear models.")

try:
    spam_lr_feats, ham_lr_feats = get_global_top_features(predictor.vectorizer, predictor.lr_model, top_n=8)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.caption("Top Features Driving Spam (+)")
        st.dataframe(spam_lr_feats, use_container_width=True, hide_index=True)
    with col_f2:
        st.caption("Top Features Driving Ham (-)")
        st.dataframe(ham_lr_feats, use_container_width=True, hide_index=True)
except Exception as ex:
    st.info(f"Could not load feature coefficients: {ex}")
