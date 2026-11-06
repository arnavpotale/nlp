"""
SMS Shield - NLP Preprocessing Module
Author: SMS Shield NLP Project

This module implements a standard and consistent Natural Language Processing (NLP)
preprocessing pipeline for SMS messages. Preprocessing ensures that text data
is transformed into a normalized, clean format suitable for TF-IDF vectorization
and machine learning classification.

Pipeline Steps:
1. Lowercase Conversion: Standardizes text capitalization.
2. Punctuation Removal: Strips punctuation symbols while keeping word boundaries.
3. Tokenization: Splits text strings into individual word tokens.
4. Stopword Removal: Removes frequent, uninformative words (e.g., 'the', 'is', 'at').
5. Porter Stemming: Reduces words to their base or root form (e.g., 'winning' -> 'win').
"""

import os
import re
import string
from pathlib import Path
from typing import Dict, List, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Configure NLTK data search paths:
# Prioritize local project 'nltk_data' directory for self-contained, offline execution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_NLTK_DATA = PROJECT_ROOT / "nltk_data"

if LOCAL_NLTK_DATA.exists():
    nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))

# Ensure required NLTK resources are available
def _ensure_nltk_resources():
    required_packages = ["punkt", "punkt_tab", "stopwords"]
    for package in required_packages:
        try:
            if package == "stopwords":
                stopwords.words("english")
            elif package in ["punkt", "punkt_tab"]:
                word_tokenize("test sentence")
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

_ensure_nltk_resources()

# Initialize stemmer and stopword list once for performance
_STEMMER = PorterStemmer()
try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback minimal stop words if NLTK corpus cannot be read
    _STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "don", "should", "now"
    }


def preprocess_text(text: str) -> str:
    """
    Transforms raw input SMS text through the complete 5-step NLP pipeline:
    1. Lowercase conversion
    2. Punctuation removal
    3. Tokenization
    4. Stopword removal
    5. Porter stemming

    Parameters
    ----------
    text : str
        The raw SMS message string.

    Returns
    -------
    str
        Space-separated string of cleaned, stemmed tokens.
    """
    # Defensive check for None, empty, or non-string input
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase conversion
    lowercased = text.lower()

    # Step 2: Punctuation removal
    # Replace punctuation characters with spaces to preserve word boundaries
    punct_pattern = re.compile(r"[" + re.escape(string.punctuation) + r"]")
    no_punct = punct_pattern.sub(" ", lowercased)

    # Step 3: Tokenization
    try:
        tokens = word_tokenize(no_punct)
    except Exception:
        # Whitespace-based fallback if tokenizer fails
        tokens = no_punct.split()

    # Step 4: Stopword removal & Step 5: Porter Stemming
    cleaned_tokens: List[str] = []
    for token in tokens:
        # Keep alphanumeric tokens only (drops residual standalone symbols)
        if token.isalnum() and token not in _STOP_WORDS:
            stemmed = _STEMMER.stem(token)
            if stemmed:
                cleaned_tokens.append(stemmed)

    return " ".join(cleaned_tokens)


def preprocess_with_steps(text: str) -> Dict[str, Any]:
    """
    Executes the NLP pipeline while preserving the output of each intermediate step.
    Useful for educational displays, UI inspection, and debugging.

    Parameters
    ----------
    text : str
        The raw SMS message string.

    Returns
    -------
    dict
        Dictionary containing representations at each stage of the pipeline.
    """
    if not isinstance(text, str):
        text = str(text or "")

    # Step 1: Lowercase
    lowercased = text.lower()

    # Step 2: Punctuation removal
    punct_pattern = re.compile(r"[" + re.escape(string.punctuation) + r"]")
    no_punct = punct_pattern.sub(" ", lowercased)

    # Step 3: Tokenization
    try:
        tokens = word_tokenize(no_punct)
    except Exception:
        tokens = no_punct.split()

    # Step 4: Stopword filtering
    non_stopwords = [t for t in tokens if t.isalnum() and t not in _STOP_WORDS]

    # Step 5: Porter Stemming
    stemmed_tokens = [_STEMMER.stem(t) for t in non_stopwords if _STEMMER.stem(t)]

    return {
        "raw": text,
        "lowercased": lowercased,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "filtered_tokens": non_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "final_preprocessed": " ".join(stemmed_tokens),
    }


def get_stopwords_set() -> set:
    """Returns the set of active English stopwords."""
    return set(_STOP_WORDS)
