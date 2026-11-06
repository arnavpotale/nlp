"""
SMS Shield 2.0 - NLP Explainability & Heuristic Signal Analysis Module
Author: SMS Shield NLP Project

This module provides dual-layer explainability for SMS spam predictions:
1. Model-Based Interpretability:
   - Logistic Regression and Linear SVM feature weights (coefficients).
   - Multinomial Naive Bayes log-likelihood ratios: log(P(w|Spam) / P(w|Ham)).
   - Per-token feature contribution for individual messages.
2. Rule-Based / Heuristic Indicators:
   - High-risk promotional keywords (e.g., FREE, WINNER, CLAIM, URGENT, CASH).
   - Structural signals (URLs, telephone shortcodes, currency symbols, all-caps ratio).
   - Message character length comparisons.

IMPORTANT: Rule-based indicators are explicitly presented as descriptive heuristics,
completely distinguished from statistical ML model predictions.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


# Curated illustrative keywords observed frequently in promotional / fraudulent SMS
SUSPICIOUS_KEYWORD_PATTERNS = [
    r"\bfree\b",
    r"\bwinner\b",
    r"\bwin\b",
    r"\bwon\b",
    r"\bclaim\b",
    r"\burgent\b",
    r"\bprize\b",
    r"\bcash\b",
    r"\baward\b",
    r"\bcall\b",
    r"\btxt\b",
    r"\btext\b",
    r"\bbonus\b",
    r"\bguaranteed\b",
    r"\bselected\b",
    r"\bcongratulations\b",
    r"\bcongrats\b",
    r"\bentry\b",
    r"\brefund\b",
    r"\baccount\b",
    r"\bcredit\b",
    r"\bvalid 12 hours\b",
    r"\bapply now\b",
    r"\bstop\b",
]

# Regex patterns for deterministic signals
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+", re.IGNORECASE)
PHONE_SHORTCODE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,6}\b")
CURRENCY_PATTERN = re.compile(r"[£$€]|(?:rs\.?|inr|usd|gbp)\b", re.IGNORECASE)


def extract_heuristic_signals(raw_text: str) -> Dict[str, Any]:
    """
    Extracts structural, lexical, and deterministic heuristic signals from a message.

    Parameters
    ----------
    raw_text : str
        The raw SMS message string.

    Returns
    -------
    dict
        Dictionary containing extracted deterministic signals.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "uppercase_words": [],
            "uppercase_ratio": 0.0,
            "suspicious_keywords": [],
            "has_url": False,
            "urls_found": [],
            "has_phone_number": False,
            "phones_found": [],
            "has_currency": False,
            "currencies_found": [],
            "length_category": "Empty",
            "heuristic_risk_level": "None",
        }

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)

    # Uppercase tokens (words with >= 2 chars entirely capitalized, excluding standalone numbers)
    uppercase_words = [w for w in words if w.isupper() and len(w) >= 2 and w.isalpha()]
    uppercase_ratio = len(uppercase_words) / max(word_count, 1)

    # Keyword matching
    lower_text = raw_text.lower()
    found_keywords = []
    for pattern in SUSPICIOUS_KEYWORD_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            found_keywords.extend(matches)
    found_keywords = list(dict.fromkeys(found_keywords))  # deduplicate preserving order

    # URL detection
    urls_found = URL_PATTERN.findall(raw_text)

    # Phone/Shortcode detection (filter for sequences with >= 5 digits)
    potential_phones = PHONE_SHORTCODE_PATTERN.findall(raw_text)
    phones_found = [p for p in potential_phones if sum(c.isdigit() for c in p) >= 5]

    # Currency symbols
    currencies_found = CURRENCY_PATTERN.findall(raw_text)

    # Length category comparison against UCI dataset distributions
    # (Ham dataset median ~52 chars; Spam median ~149 chars)
    if char_count > 120:
        length_category = "Long (Typical Spam Profile)"
    elif char_count > 70:
        length_category = "Medium Length"
    else:
        length_category = "Short (Typical Ham Profile)"

    # Descriptive heuristic risk score (0 to 5) purely for UI display
    heuristic_score = 0
    if found_keywords:
        heuristic_score += min(len(found_keywords), 2)
    if urls_found:
        heuristic_score += 1
    if phones_found:
        heuristic_score += 1
    if currencies_found:
        heuristic_score += 1
    if uppercase_ratio > 0.20:
        heuristic_score += 1

    if heuristic_score >= 3:
        heuristic_risk_level = "High Heuristic Indicators"
    elif heuristic_score >= 1:
        heuristic_risk_level = "Moderate Heuristic Indicators"
    else:
        heuristic_risk_level = "Low / Neutral Heuristic Indicators"

    return {
        "char_count": char_count,
        "word_count": word_count,
        "uppercase_words": uppercase_words,
        "uppercase_ratio": uppercase_ratio,
        "suspicious_keywords": found_keywords,
        "has_url": bool(urls_found),
        "urls_found": urls_found,
        "has_phone_number": bool(phones_found),
        "phones_found": phones_found,
        "has_currency": bool(currencies_found),
        "currencies_found": currencies_found,
        "length_category": length_category,
        "heuristic_score": heuristic_score,
        "heuristic_risk_level": heuristic_risk_level,
    }


def explain_message_features(
    clean_text: str,
    vectorizer,
    lr_model,
    nb_model=None,
    svc_model=None
) -> pd.DataFrame:
    """
    Computes per-token model contributions for the active message.

    Parameters
    ----------
    clean_text : str
        The preprocessed SMS string.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    lr_model : LogisticRegression
        Fitted Logistic Regression model.
    nb_model : MultinomialNB, optional
        Fitted Multinomial Naive Bayes model.
    svc_model : LinearSVC, optional
        Fitted Linear SVM model.

    Returns
    -------
    pd.DataFrame
        DataFrame listing tokens, their TF-IDF weight, LR coefficient, and directional influence.
    """
    if not clean_text or not clean_text.strip():
        return pd.DataFrame()

    feature_names = np.array(vectorizer.get_feature_names_out())
    tokens = clean_text.split()
    unique_tokens = list(set(tokens))

    # Match tokens in vectorizer vocabulary
    vocab = vectorizer.vocabulary_
    matched_tokens = [t for t in unique_tokens if t in vocab]

    if not matched_tokens:
        return pd.DataFrame()

    indices = [vocab[t] for t in matched_tokens]

    # Transform current text
    tfidf_vec = vectorizer.transform([clean_text])
    tfidf_dense = tfidf_vec.toarray()[0]

    lr_coefs = lr_model.coef_[0] if hasattr(lr_model, "coef_") else None
    svc_coefs = svc_model.coef_[0] if (svc_model is not None and hasattr(svc_model, "coef_")) else None

    # Naive Bayes log probability ratio: log(P(w|Spam)) - log(P(w|Ham))
    nb_log_ratios = None
    if nb_model is not None and hasattr(nb_model, "feature_log_prob_"):
        # class 1 (Spam) - class 0 (Ham)
        nb_log_ratios = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]

    records = []
    for t in matched_tokens:
        idx = vocab[t]
        weight = float(tfidf_dense[idx])
        lr_weight = float(lr_coefs[idx]) if lr_coefs is not None else 0.0
        svc_weight = float(svc_coefs[idx]) if svc_coefs is not None else 0.0
        nb_ratio = float(nb_log_ratios[idx]) if nb_log_ratios is not None else 0.0

        # Overall influence direction for Logistic Regression
        lr_contribution = weight * lr_weight
        direction = "Spam (+)" if lr_contribution > 0 else ("Ham (-)" if lr_contribution < 0 else "Neutral")

        records.append({
            "Token": t,
            "TF-IDF Value": round(weight, 4),
            "LR Weight": round(lr_weight, 4),
            "SVM Weight": round(svc_weight, 4),
            "NB Log Ratio": round(nb_ratio, 4),
            "LR Contribution": round(lr_contribution, 4),
            "Direction": direction,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="LR Contribution", ascending=False).reset_index(drop=True)
    return df


def get_global_top_features(vectorizer, model, top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves global top positive (Spam) and top negative (Ham) features for linear models.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    model : LogisticRegression or LinearSVC
        Fitted linear classifier with coef_ attribute.
    top_n : int
        Number of top features to return.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        DataFrames of top spam features and top ham features.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_spam_idx = np.argsort(coefs)[-top_n:][::-1]
    top_ham_idx = np.argsort(coefs)[:top_n]

    spam_df = pd.DataFrame({
        "Feature": feature_names[top_spam_idx],
        "Coefficient": coefs[top_spam_idx]
    })

    ham_df = pd.DataFrame({
        "Feature": feature_names[top_ham_idx],
        "Coefficient": coefs[top_ham_idx]
    })

    return spam_df, ham_df

