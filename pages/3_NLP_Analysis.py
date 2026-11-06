"""
SMS Shield 2.0 - Page 3: NLP Analysis
Author: SMS Shield NLP Project

Detailed educational walkthrough of the Natural Language Processing pipeline:
Step-by-step transformations, tokenization, stemming, TF-IDF representations,
token frequency distributions, and n-gram comparisons (unigrams, bigrams, trigrams).
"""

import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from nltk.util import ngrams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import render_header, render_sidebar_status, get_dataset, get_predictor
from src.preprocessing import preprocess_with_steps, get_stopwords_set
from src.explainability import get_global_top_features


st.set_page_config(page_title="NLP Analysis | SMS Shield 2.0", page_icon="🔬", layout="wide")
render_header("Page 3: Step-by-Step NLP Pipeline & N-Gram Analytics", "Interactive demonstration of text normalization, stemming, vocabulary distributions, and n-grams.")
render_sidebar_status()

# Load data and predictor
try:
    raw_df, clean_df, _ = get_dataset()
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Part 1: Step-by-Step Pipeline Demonstration for User Message
# -----------------------------------------------------------------------------
st.markdown("### 🛠️ Interactive Pipeline Step-by-Step Walkthrough")
st.write("Enter an SMS message to trace its exact transformations through all 7 stages of the NLP workflow.")

sample_demo = "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 88088 now."
user_msg = st.text_input("Enter SMS text for step-by-step trace:", value=sample_demo)

if user_msg.strip():
    steps = preprocess_with_steps(user_msg)
    clean_text = steps["final_preprocessed"]

    # Visual pipeline progression cards
    st.markdown("##### 🔄 Sequential Transformations")
    
    col_step1, col_step2 = st.columns(2)
    with col_step1:
        st.markdown(
            f"""
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 1: Original Message</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px;">{steps['raw']}</p>
                <small style="color: #64748b;">Raw user string with punctuation, capitalization, and formatting.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 2: Lowercase Conversion</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px;">{steps['lowercased']}</p>
                <small style="color: #64748b;">Standardizes casing so 'Free', 'FREE', and 'free' become identical tokens.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 3: Punctuation Removal</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px;">{steps['no_punctuation']}</p>
                <small style="color: #64748b;">Punctuation characters replaced with spaces to preserve word boundaries.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step2:
        st.markdown(
            f"""
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 4: Word Tokenization ({len(steps['tokens'])} tokens)</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px; word-break: break-all;">{str(steps['tokens'])}</p>
                <small style="color: #64748b;">NLTK tokenizes the stream into discrete word units.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 5: Stopword Removal ({len(steps['filtered_tokens'])} tokens remain)</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px; word-break: break-all;">{str(steps['filtered_tokens'])}</p>
                <small style="color: #64748b;">High-frequency English stopwords (e.g. 'as', 'a', 'you', 'have') eliminated.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #6366f1; font-weight: 700;">STAGE 6: Porter Stemming & Normalization</span>
                <p style="margin: 6px 0; font-family: monospace; background: #f8fafc; padding: 8px; border-radius: 6px; word-break: break-all;">{steps['final_preprocessed']}</p>
                <small style="color: #64748b;">Tokens stemmed to root bases (e.g. 'valued' -> 'valu', 'winner' -> 'winner').</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Step 7: TF-IDF Vectorization
    st.markdown("##### 📐 STAGE 7: TF-IDF Vector Representation")
    if clean_text:
        tfidf_vec = predictor.vectorizer.transform([clean_text])
        feature_names = np.array(predictor.vectorizer.get_feature_names_out())
        non_zero_idx = tfidf_vec.nonzero()[1]

        if len(non_zero_idx) > 0:
            active_features = [
                {"Vocabulary Feature": feature_names[i], "TF-IDF Weight": round(float(tfidf_vec[0, i]), 4)}
                for i in non_zero_idx
            ]
            active_df = pd.DataFrame(active_features).sort_values(by="TF-IDF Weight", ascending=False)
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.info("No tokens matched the TF-IDF vocabulary.")
    else:
        st.warning("No tokens remaining after stopword removal.")

st.divider()

# -----------------------------------------------------------------------------
# Part 2: Suspicious Keywords Pedagogical Spotlight
# -----------------------------------------------------------------------------
st.markdown("### 🚨 Illustrative Suspicious Terms Spotlight")
st.write(
    "In SMS spam, certain high-pressure marketing terms recur with extreme frequency. "
    "Below are the primary trigger words identified across the UCI dataset:"
)

badge_cols = st.columns(6)
suspicious_spotlight = [
    ("FREE", "195 occurrences", "#dc2626"),
    ("CLAIM", "113 occurrences", "#ea580c"),
    ("WINNER", "63 occurrences", "#d97706"),
    ("URGENT", "57 occurrences", "#b45309"),
    ("PRIZE", "92 occurrences", "#7c2d12"),
    ("CASH", "62 occurrences", "#991b1b"),
]

for idx, (kw, occ, color) in enumerate(suspicious_spotlight):
    badge_cols[idx].markdown(
        f"""
        <div style="background-color: #fff1f2; border: 1.5px solid {color}; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 1.15rem; font-weight: 800; color: {color};">{kw}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">{occ}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    "⚠️ **Pedagogical Note**: The presence of words like 'FREE' or 'WINNER' serves as an **illustrative indicator** "
    "of commercial or promotional style, **not mathematical proof of malicious intent**. "
    "Legitimate conversational messages may occasionally contain these words."
)

st.divider()

# -----------------------------------------------------------------------------
# Part 3: Dataset-Wide N-Gram Analytics (Unigrams, Bigrams, Trigrams)
# -----------------------------------------------------------------------------
st.markdown("### 📚 Dataset-Wide N-Gram Frequency Analysis")
st.write(
    "Comparing n-gram distributions across legitimate (`ham`) versus promotional/fraudulent (`spam`) messages "
    "reveals how multi-word phrases provide rich semantic context for classification."
)

ngram_tab1, ngram_tab2, ngram_tab3 = st.tabs(["Unigrams (Single Words)", "Bigrams (Two-Word Pairs)", "Trigrams (Three-Word Sequences)"])

@st.cache_data
def compute_dataset_ngrams(df: pd.DataFrame):
    """Extracts top n-grams for Ham and Spam subsets."""
    spam_clean = df[df["label"] == "spam"]["clean_message"].dropna().tolist()
    ham_clean = df[df["label"] == "ham"]["clean_message"].dropna().tolist()

    # Unigrams
    spam_uni = Counter([w for s in spam_clean for w in s.split()])
    ham_uni = Counter([w for s in ham_clean for w in s.split()])

    # Bigrams
    spam_bi = Counter([" ".join(bg) for s in spam_clean for bg in ngrams(s.split(), 2)])
    ham_bi = Counter([" ".join(bg) for s in ham_clean for bg in ngrams(s.split(), 2)])

    # Trigrams
    spam_tri = Counter([" ".join(tg) for s in spam_clean for tg in ngrams(s.split(), 3)])
    ham_tri = Counter([" ".join(tg) for s in ham_clean for tg in ngrams(s.split(), 3)])

    return {
        "spam_uni": spam_uni.most_common(12),
        "ham_uni": ham_uni.most_common(12),
        "spam_bi": spam_bi.most_common(12),
        "ham_bi": ham_bi.most_common(12),
        "spam_tri": spam_tri.most_common(12),
        "ham_tri": ham_tri.most_common(12),
    }

# Compute or load ngrams from clean_df
if "clean_message" not in clean_df.columns:
    from src.preprocessing import preprocess_text
    clean_df["clean_message"] = clean_df["message"].apply(preprocess_text)

ngram_data = compute_dataset_ngrams(clean_df)

def plot_ngram_comparison(spam_items, ham_items, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    # Spam plot
    s_terms, s_counts = zip(*spam_items)
    sns.barplot(x=list(s_counts), y=list(s_terms), color="#dc2626", ax=ax1, edgecolor="black")
    ax1.set_title(f"Top Spam {title_suffix}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Frequency Count", fontsize=9.5, fontweight="bold")

    # Ham plot
    h_terms, h_counts = zip(*ham_items)
    sns.barplot(x=list(h_counts), y=list(h_terms), color="#2563eb", ax=ax2, edgecolor="black")
    ax2.set_title(f"Top Ham {title_suffix}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Frequency Count", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    return fig

with ngram_tab1:
    fig_uni = plot_ngram_comparison(ngram_data["spam_uni"], ngram_data["ham_uni"], "Unigrams")
    st.pyplot(fig_uni)
    plt.close(fig_uni)

with ngram_tab2:
    fig_bi = plot_ngram_comparison(ngram_data["spam_bi"], ngram_data["ham_bi"], "Bigrams")
    st.pyplot(fig_bi)
    plt.close(fig_bi)

with ngram_tab3:
    fig_tri = plot_ngram_comparison(ngram_data["spam_tri"], ngram_data["ham_tri"], "Trigrams")
    st.pyplot(fig_tri)
    plt.close(fig_tri)

st.divider()

# -----------------------------------------------------------------------------
# Part 4: Global Influential TF-IDF Features (Linear Models)
# -----------------------------------------------------------------------------
st.markdown("### ⚖️ Top Global Influential TF-IDF Features")
st.write(
    "Linear models assign positive coefficients to features that increase the likelihood of Spam, "
    "and negative coefficients to features that predict Ham."
)

c_feat1, c_feat2 = st.columns(2)
try:
    spam_lr_feats, ham_lr_feats = get_global_top_features(predictor.vectorizer, predictor.lr_model, top_n=10)
    with c_feat1:
        st.markdown("##### 🚨 Top 10 Features Driving Spam (Positive LR Weights)")
        st.dataframe(spam_lr_feats, use_container_width=True, hide_index=True)
    with c_feat2:
        st.markdown("##### 🟢 Top 10 Features Driving Ham (Negative LR Weights)")
        st.dataframe(ham_lr_feats, use_container_width=True, hide_index=True)
except Exception as ex:
    st.info(f"Feature coefficients could not be rendered: {ex}")

