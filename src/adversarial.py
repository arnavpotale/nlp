"""
SMS Shield 3.0 - Adversarial Perturbation & Robustness Benchmarking Module
Author: SMS Shield NLP Project

This module provides tools to simulate controlled real-world adversarial evasions
against SMS spam filters, and evaluate model robustness under adversarial conditions:
1. Character Substitution (Leetspeak: e->3, i->!, a->@, s->$, o->0)
2. Character Insertion (delimiters: f.r.e.e, w-i-n-n-e-r)
3. Character Deletion (vowel dropping: claim -> clm, prize -> prz)
4. Repeated Letters (elongation: free -> frrreeee, winner -> winnnerrr)
5. Whitespace Insertion (spaced letters: w i n n e r, f r e e)
6. Punctuation Insertion (decorative symbols: ~CLAIM~, *FREE*)
7. Composite Multi-Vector Attack

Includes benchmark functions measuring:
- Original Accuracy & Recall
- Adversarial Accuracy & Recall
- Detection Degradation
- Robustness Percentage (% unchanged predictions)
- Prediction Flips (Spam classified as Ham)
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


LEET_SUBSTITUTIONS = {
    "a": "@",
    "A": "4",
    "e": "3",
    "E": "3",
    "i": "!",
    "I": "1",
    "o": "0",
    "O": "0",
    "s": "$",
    "S": "$",
    "t": "7",
}

VOWELS = set("aeiouAEIOU")


def perturb_char_substitution(text: str, rng: Optional[random.Random] = None, rate: float = 0.5) -> str:
    """
    Substitutes standard letters with typical leetspeak characters and symbols.
    Example: 'WINNER prize reward' -> 'W!NN3R pr!z3 r3w@rd'
    """
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    perturbed = []
    for w in words:
        # Don't alter phone numbers, URLs, or currency
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 3 and rng.random() < rate:
            new_chars = [LEET_SUBSTITUTIONS.get(ch, ch) if rng.random() < 0.7 else ch for ch in w]
            perturbed.append("".join(new_chars))
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def perturb_char_insertion(text: str, rng: Optional[random.Random] = None, rate: float = 0.4) -> str:
    """
    Inserts internal delimiters (dots, hyphens, underscores) within words.
    Example: 'free winner claim' -> 'f.r.e.e w-i-n-n-e-r c.l.a.i.m'
    """
    if rng is None:
        rng = random.Random(42)

    delims = [".", "-", "_", "*"]
    words = text.split()
    perturbed = []
    for w in words:
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            d = rng.choice(delims)
            perturbed.append(d.join(list(w)))
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def perturb_char_deletion(text: str, rng: Optional[random.Random] = None, rate: float = 0.4) -> str:
    """
    Simulates SMS vowel dropping and typo deletions inside words.
    Example: 'claim your prize reward' -> 'clm your prz rwrd'
    """
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    perturbed = []
    for w in words:
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            # Drop interior vowels
            first, mid, last = w[0], w[1:-1], w[-1]
            new_mid = "".join(ch for ch in mid if ch not in VOWELS or rng.random() > 0.8)
            perturbed.append(first + (new_mid if new_mid else mid) + last)
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def perturb_repeated_letters(text: str, rng: Optional[random.Random] = None, rate: float = 0.4) -> str:
    """
    Simulates deliberate character elongation.
    Example: 'free cash prize' -> 'frrreeee caaash prizze'
    """
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    perturbed = []
    for w in words:
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 3 and w.isalpha() and rng.random() < rate:
            new_w = "".join(ch * (rng.randint(3, 5)) if ch.isalpha() and rng.random() < 0.5 else ch for ch in w)
            perturbed.append(new_w)
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def perturb_whitespace_insertion(text: str, rng: Optional[random.Random] = None, rate: float = 0.4) -> str:
    """
    Inserts spaces between individual letters to break tokenization.
    Example: 'WINNER of free prize' -> 'W I N N E R of f r e e prize'
    """
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    perturbed = []
    for w in words:
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            perturbed.append(" ".join(list(w)))
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def perturb_punctuation_insertion(text: str, rng: Optional[random.Random] = None, rate: float = 0.4) -> str:
    """
    Injects decorative punctuation symbols around and inside words.
    Example: 'WINNER CLAIM FREE' -> '~WINNER~ !CLAIM! *FREE*'
    """
    if rng is None:
        rng = random.Random(42)

    symbols = ["~", "*", "!", "#", "^"]
    words = text.split()
    perturbed = []
    for w in words:
        if re.match(r"^(?:[£$€]?\d+|\d{5,}|https?://)", w):
            perturbed.append(w)
            continue

        if len(w) >= 3 and w.isalpha() and rng.random() < rate:
            sym = rng.choice(symbols)
            perturbed.append(f"{sym}{w}{sym}")
        else:
            perturbed.append(w)

    return " ".join(perturbed)


def generate_adversarial_variant(text: str, attack_type: str, seed: int = 42) -> str:
    """
    Generates a perturbed adversarial version of a message according to the chosen attack type.
    """
    rng = random.Random(seed)

    if attack_type == "Character Substitution (Leetspeak)":
        return perturb_char_substitution(text, rng, rate=0.6)
    elif attack_type == "Character Insertion (Delimiters)":
        return perturb_char_insertion(text, rng, rate=0.5)
    elif attack_type == "Character Deletion (Vowel Drop)":
        return perturb_char_deletion(text, rng, rate=0.5)
    elif attack_type == "Repeated Letters (Elongation)":
        return perturb_repeated_letters(text, rng, rate=0.5)
    elif attack_type == "Whitespace Spacing (w i n n e r)":
        return perturb_whitespace_insertion(text, rng, rate=0.5)
    elif attack_type == "Punctuation Insertion (~CLAIM~)":
        return perturb_punctuation_insertion(text, rng, rate=0.5)
    elif attack_type == "Composite Multi-Vector Attack":
        t = perturb_char_substitution(text, rng, rate=0.4)
        t = perturb_punctuation_insertion(t, rng, rate=0.3)
        t = perturb_whitespace_insertion(t, rng, rate=0.25)
        return t
    else:
        return text


def evaluate_adversarial_robustness(
    test_df: pd.DataFrame,
    word_pipeline_dict: Dict[str, Any],
    char_pipeline_dict: Dict[str, Any],
    comb_pipeline_dict: Dict[str, Any],
    normalize_fn=None,
    attack_type: str = "Composite Multi-Vector Attack",
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluates model resilience under clean versus adversarial conditions.
    Strictly conducted on the held-out test split with zero train leakage.

    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame with columns ['message', 'clean_message', 'target'].
    word_pipeline_dict : dict
        Mapping model_name -> fitted word-level model.
    char_pipeline_dict : dict
        Mapping model_name -> fitted char-level model.
    comb_pipeline_dict : dict
        Mapping model_name -> fitted combined model.
    normalize_fn : callable, optional
        Function returning normalized text.
    attack_type : str
        The type of adversarial perturbation to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Detailed comparative metrics table.
    """
    rng = random.Random(seed)
    y_true = test_df["target"].to_numpy()

    # Generate adversarial messages
    raw_messages = test_df["message"].tolist()
    adv_messages = [generate_adversarial_variant(m, attack_type, seed=seed + i) for i, m in enumerate(raw_messages)]

    # Normalized variants
    norm_messages = [normalize_fn(m)["normalized_text"] if normalize_fn else m for m in adv_messages]

    records = []

    # Helper evaluator
    def evaluate_setup(setup_name, model_name, clf, vec, is_normalized=False):
        from src.preprocessing import preprocess_text

        # 1. Clean evaluation
        clean_texts = test_df["clean_message"].tolist()
        X_clean = vec.transform(clean_texts)
        y_pred_clean = clf.predict(X_clean)

        clean_acc = accuracy_score(y_true, y_pred_clean)
        clean_rec = recall_score(y_true, y_pred_clean, zero_division=0)
        clean_f1 = f1_score(y_true, y_pred_clean, zero_division=0)

        # 2. Adversarial evaluation
        eval_texts = norm_messages if is_normalized else adv_messages
        clean_adv = [preprocess_text(t) for t in eval_texts]
        X_adv = vec.transform(clean_adv)
        y_pred_adv = clf.predict(X_adv)

        adv_acc = accuracy_score(y_true, y_pred_adv)
        adv_rec = recall_score(y_true, y_pred_adv, zero_division=0)
        adv_f1 = f1_score(y_true, y_pred_adv, zero_division=0)

        # Metrics
        acc_degradation = clean_acc - adv_acc
        rec_degradation = clean_rec - adv_rec
        robustness_pct = np.mean(y_pred_clean == y_pred_adv) * 100

        # Prediction flips on spam: spam that was correctly caught on clean, but missed on adversarial
        spam_mask = (y_true == 1)
        flips = int(np.sum((y_pred_clean[spam_mask] == 1) & (y_pred_adv[spam_mask] == 0)))

        records.append({
            "Defense Pipeline": setup_name,
            "Classifier": model_name,
            "Attack Type": attack_type,
            "Clean Accuracy": clean_acc,
            "Adversarial Accuracy": adv_acc,
            "Accuracy Drop": acc_degradation,
            "Clean Recall (Spam)": clean_rec,
            "Adversarial Recall": adv_rec,
            "Recall Drop": rec_degradation,
            "Clean F1-Score": clean_f1,
            "Adversarial F1": adv_f1,
            "Robustness Score (%)": robustness_pct,
            "Spam Miss Flips": flips,
        })

    # Evaluate all combinations
    setups = [
        ("Baseline (Word TF-IDF)", word_pipeline_dict, False),
        ("Sub-word (Char TF-IDF)", char_pipeline_dict, False),
        ("Hybrid (Word + Char)", comb_pipeline_dict, False),
        ("Normalized + Word TF-IDF", word_pipeline_dict, True),
    ]

    for setup_name, p_dict, is_norm in setups:
        vec = p_dict["vectorizer"]
        for m_name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]:
            clf = p_dict[m_name]
            evaluate_setup(setup_name, m_name, clf, vec, is_normalized=is_norm)

    return pd.DataFrame(records)
