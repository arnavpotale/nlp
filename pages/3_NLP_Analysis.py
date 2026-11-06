"""
SMS Shield 4.0 - Page 3: NLP Intelligence & Linguistic Forensics
Author: SMS Shield NLP Project

Comprehensive educational walkthrough of the Natural Language Processing workflow:
Sequential 7-stage transformations, tokenization, stemming, TF-IDF vectorization,
forensic keyword spotting, and interactive comparative n-gram analytics.
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
    PRIMARY_VIOLET,
    CYAN_ACCENT,
    SUCCESS_GREEN,
    DANGER_RED,
    TEXT_PRIMARY,
    TEXT_MUTED,
    apply_plotly_theme,
)
from src.preprocessing import preprocess_with_steps
from src.explainability import get_global_top_features

st.set_page_config(page_title="NLP Intelligence | SMS Shield 4.0", page_icon="🔬", layout="wide")
render_header(
    "NLP Intelligence & Linguistic Forensics",
    "Interactive inspection of text normalization, morphological stemming, TF-IDF feature projection, and n-grams."
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
# Part 1: Step-by-Step Pipeline Demonstration for User Message
# -----------------------------------------------------------------------------
st.markdown("### 🛠️ Sequential 7-Stage NLP Transformation Pipeline")
st.write("Enter an SMS message to trace its exact lexical mutations across each layer of the NLP architecture:")

sample_demo = "WINNER!! As a valued network customer you have won a £1,000 cash prize reward! Text CLAIM to 88088 now."
user_msg = st.text_input("Enter SMS text for step-by-step trace:", value=sample_demo)

if user_msg.strip():
    steps = preprocess_with_steps(user_msg)
    clean_text = steps["final_preprocessed"]

    # Visual pipeline progression cards
    st.markdown("##### 🔄 Pipeline Execution Flow")
    
    col_step1, col_step2 = st.columns(2)
    with col_step1:
        st.markdown(
            f"""
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 1: Ingestion & Raw Text</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #F8FAFC; padding: 10px; border-radius: 6px;">{steps['raw']}</p>
                <small style="color: #94A3B8;">Raw string retaining all capitalization, punctuation marks, emojis, and digits.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 2: Case Standardization</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #38BDF8; padding: 10px; border-radius: 6px;">{steps['lowercased']}</p>
                <small style="color: #94A3B8;">Converts string to lowercase so 'PRIZE', 'Prize', and 'prize' map to identical tokens.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 3: Delimiter & Punctuation Strip</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #F8FAFC; padding: 10px; border-radius: 6px;">{steps['no_punctuation']}</p>
                <small style="color: #94A3B8;">Punctuation characters replaced with whitespace, isolating word boundaries cleanly.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step2:
        st.markdown(
            f"""
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 4: Lexical Tokenization ({len(steps['tokens'])} tokens)</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #38BDF8; padding: 10px; border-radius: 6px; word-break: break-all;">{str(steps['tokens'])}</p>
                <small style="color: #94A3B8;">NLTK tokenizes the character stream into discrete linguistic elements.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 5: Stopword Filtering ({len(steps['filtered_tokens'])} tokens retained)</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #F8FAFC; padding: 10px; border-radius: 6px; word-break: break-all;">{str(steps['filtered_tokens'])}</p>
                <small style="color: #94A3B8;">High-frequency non-discriminative English terms (e.g. 'as', 'a', 'you', 'to') pruned.</small>
            </div>
            <div class="cyber-card">
                <span style="color: #A78BFA; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">STAGE 6: Porter Stemming & Normalization</span>
                <p style="margin: 8px 0; font-family: monospace; background: #0B1020; border: 1px solid #26334D; color: #10B981; padding: 10px; border-radius: 6px; word-break: break-all;">{steps['final_preprocessed']}</p>
                <small style="color: #94A3B8;">Morphological reduction to root stems (e.g. 'valued' → 'valu', 'winner' → 'winner').</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Step 7: TF-IDF Vectorization
    st.markdown("##### 📐 STAGE 7: High-Dimensional TF-IDF Projection")
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
st.markdown("### 🚨 High-Frequency Threat Keyword Spotter")
st.write(
    "In fraudulent SMS traffic, high-pressure promotional vocabulary recurs with abnormal statistical frequency. "
    "Below are the primary trigger words identified across the UCI dataset:"
)

badge_cols = st.columns(6)
suspicious_spotlight = [
    ("FREE", "195 occurrences", DANGER_RED),
    ("CLAIM", "113 occurrences", "#EA580C"),
    ("WINNER", "63 occurrences", "#D97706"),
    ("URGENT", "57 occurrences", "#B45309"),
    ("PRIZE", "92 occurrences", "#991B1B"),
    ("CASH", "62 occurrences", "#7C2D12"),
]

for idx, (kw, occ, color) in enumerate(suspicious_spotlight):
    badge_cols[idx].markdown(
        f"""
        <div class="cyber-card" style="border-top: 3px solid {color}; text-align: center; padding: 12px 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">{kw}</div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">{occ}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid #26334D; border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">
        ⚠️ <strong>Pedagogical Integrity Note</strong>: Keyword indicators serve as contextual indicators of commercial tone, <strong>not absolute mathematical proof of malice</strong>. Legitimate peer-to-peer conversations occasionally contain 'free' or 'cash' without fraudulent intent.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------------------------------
# Part 3: Dataset-Wide N-Gram Analytics (Unigrams, Bigrams, Trigrams)
# -----------------------------------------------------------------------------
st.markdown("### 📚 Dataset-Wide N-Gram Frequency Intelligence")
st.write(
    "Analyzing n-gram collocations across legitimate (`ham`) versus promotional (`spam`) traffic "
    "demonstrates how multi-word phrases supply discriminative semantic context."
)

ngram_tab1, ngram_tab2, ngram_tab3 = st.tabs(["Unigrams (Single Words)", "Bigrams (Two-Word Collocations)", "Trigrams (Three-Word Sequences)"])

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

# Ensure clean_message exists
if "clean_message" not in clean_df.columns:
    from src.preprocessing import preprocess_text
    clean_df["clean_message"] = clean_df["message"].apply(preprocess_text)

ngram_data = compute_dataset_ngrams(clean_df)

def render_plotly_ngrams(spam_items, ham_items, title_suffix):
    """Renders dual interactive Plotly horizontal bar charts for Spam vs Ham n-grams."""
    s_terms, s_counts = zip(*spam_items)
    h_terms, h_counts = zip(*ham_items)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top Malicious (Spam) {title_suffix}", f"Top Legitimate (Ham) {title_suffix}"),
        horizontal_spacing=0.12
    )

    # Spam bar
    fig.add_trace(
        go.Bar(
            y=list(s_terms),
            x=list(s_counts),
            orientation="h",
            marker_color=DANGER_RED,
            name="Spam",
            hovertemplate="Phrase: <b>%{y}</b><br>Frequency: %{x}<extra></extra>"
        ),
        row=1, col=1
    )

    # Ham bar
    fig.add_trace(
        go.Bar(
            y=list(h_terms),
            x=list(h_counts),
            orientation="h",
            marker_color=CYAN_ACCENT,
            name="Ham",
            hovertemplate="Phrase: <b>%{y}</b><br>Frequency: %{x}<extra></extra>"
        ),
        row=1, col=2
    )

    apply_plotly_theme(fig, height=360)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return fig

with ngram_tab1:
    fig_uni = render_plotly_ngrams(ngram_data["spam_uni"], ngram_data["ham_uni"], "Unigrams")
    st.plotly_chart(fig_uni, use_container_width=True)

with ngram_tab2:
    fig_bi = render_plotly_ngrams(ngram_data["spam_bi"], ngram_data["ham_bi"], "Bigrams")
    st.plotly_chart(fig_bi, use_container_width=True)

with ngram_tab3:
    fig_tri = render_plotly_ngrams(ngram_data["spam_tri"], ngram_data["ham_tri"], "Trigrams")
    st.plotly_chart(fig_tri, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Part 4: Global Influential TF-IDF Features (Linear Models)
# -----------------------------------------------------------------------------
st.markdown("### ⚖️ Top Global Influential TF-IDF Coefficients")
st.write(
    "Linear models assign positive coefficients to vocabulary features that push predictions toward Spam, "
    "and negative coefficients to features that indicate legitimate Ham."
)

c_feat1, c_feat2 = st.columns(2)
try:
    spam_lr_feats, ham_lr_feats = get_global_top_features(predictor.vectorizer, predictor.lr_model, top_n=10)
    
    with c_feat1:
        st.markdown("##### 🚨 Top Features Driving Spam (+ Weights)")
        fig_spam_feat = go.Figure()
        fig_spam_feat.add_trace(
            go.Bar(
                y=spam_lr_feats["Feature"],
                x=spam_lr_feats["Coefficient"],
                orientation="h",
                marker_color=DANGER_RED,
                hovertemplate="Feature: <b>%{y}</b><br>Coefficient: +%{x:.3f}<extra></extra>"
            )
        )
        apply_plotly_theme(fig_spam_feat, height=300)
        fig_spam_feat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_spam_feat, use_container_width=True)
        st.dataframe(spam_lr_feats, use_container_width=True, hide_index=True)

    with c_feat2:
        st.markdown("##### 🟢 Top Features Driving Ham (- Weights)")
        fig_ham_feat = go.Figure()
        fig_ham_feat.add_trace(
            go.Bar(
                y=ham_lr_feats["Feature"],
                x=ham_lr_feats["Coefficient"],
                orientation="h",
                marker_color=CYAN_ACCENT,
                hovertemplate="Feature: <b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>"
            )
        )
        apply_plotly_theme(fig_ham_feat, height=300)
        fig_ham_feat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_ham_feat, use_container_width=True)
        st.dataframe(ham_lr_feats, use_container_width=True, hide_index=True)

except Exception as ex:
    st.info(f"Feature coefficients could not be rendered: {ex}")
