"""
SMS Shield - System Verification and Test Suite
Author: SMS Shield NLP Project

Tests pipeline components:
1. NLP preprocessing behavior across standard text, edge cases, punctuation, and empty input.
2. Model loading, inference, and probability outputs.
3. Metric integrity and verification against saved evaluation_metrics.csv.
4. Error handling for missing files.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.preprocessing import preprocess_text, preprocess_with_steps
from src.evaluate import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent


def test_preprocessing():
    print("[TEST] Testing NLP preprocessing...")

    # Test 1: Standard spam sentence
    raw = "WINNER!! Claim £1,000 cash prize now by calling 09061701461!"
    cleaned = preprocess_text(raw)
    assert "winner" in cleaned, "Expected 'winner' in cleaned text"
    assert "prize" in cleaned, "Expected 'prize' in cleaned text"
    assert "!" not in cleaned, "Punctuation was not removed"
    assert "£" not in cleaned, "Currency symbol was not removed"
    print("  ✓ Standard text cleaning passed.")

    # Test 2: Empty input & whitespace
    assert preprocess_text("") == ""
    assert preprocess_text("   \n\t  ") == ""
    assert preprocess_text(None) == ""
    print("  ✓ Empty and whitespace input handling passed.")

    # Test 3: Punctuation only
    assert preprocess_text("!@#$%^&*()_+=-{}[]:;'\"<>,.?/") == ""
    print("  ✓ Punctuation-only input handling passed.")

    # Test 4: Stopwords only
    # E.g. "it is what it is" -> all are stopwords
    res = preprocess_text("it is what it is")
    assert res == "", f"Expected empty string for stopwords-only, got '{res}'"
    print("  ✓ Stopwords-only input handling passed.")

    # Test 5: Step-by-step pipeline inspection
    steps = preprocess_with_steps(raw)
    assert "raw" in steps and "final_preprocessed" in steps
    print("  ✓ Step-by-step inspection dictionary passed.")


def test_model_inference():
    print("\n[TEST] Testing Model Inference and Probability Predictions...")

    vec_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
    nb_path = PROJECT_ROOT / "models" / "multinomial_nb.joblib"
    lr_path = PROJECT_ROOT / "models" / "logistic_regression.joblib"

    assert vec_path.exists(), "TF-IDF vectorizer artifact missing"
    assert nb_path.exists(), "Naive Bayes artifact missing"
    assert lr_path.exists(), "Logistic Regression artifact missing"

    vec = joblib.load(vec_path)
    nb = joblib.load(nb_path)
    lr = joblib.load(lr_path)

    # Obvious Spam
    spam_sample = "URGENT! You have won a £2,000 cash prize! Text CLAIM to 88088 now."
    spam_clean = preprocess_text(spam_sample)
    X_spam = vec.transform([spam_clean])

    nb_pred = nb.predict(X_spam)[0]
    lr_pred = lr.predict(X_spam)[0]
    nb_proba = nb.predict_proba(X_spam)[0]
    lr_proba = lr.predict_proba(X_spam)[0]

    assert nb_pred == 1, "Naive Bayes failed on obvious spam"
    assert lr_pred == 1, "Logistic Regression failed on obvious spam"
    assert 0.0 <= nb_proba[1] <= 1.0, "Invalid probability range"
    assert 0.0 <= lr_proba[1] <= 1.0, "Invalid probability range"
    print(f"  ✓ Spam detection passed (NB prob: {nb_proba[1]:.2%}, LR prob: {lr_proba[1]:.2%}).")

    # Obvious Ham
    ham_sample = "Hey mom, I will be home by 6 PM for dinner. Love you!"
    ham_clean = preprocess_text(ham_sample)
    X_ham = vec.transform([ham_clean])

    nb_ham_pred = nb.predict(X_ham)[0]
    lr_ham_pred = lr.predict(X_ham)[0]

    assert nb_ham_pred == 0, "Naive Bayes failed on obvious ham"
    assert lr_ham_pred == 0, "Logistic Regression failed on obvious ham"
    print("  ✓ Ham detection passed for both models.")


def test_evaluation_metrics_integrity():
    print("\n[TEST] Testing Metrics File Integrity...")
    metrics_path = PROJECT_ROOT / "metrics" / "evaluation_metrics.csv"
    assert metrics_path.exists(), "evaluation_metrics.csv is missing"

    df = pd.read_csv(metrics_path)
    assert len(df) == 2, "Expected 2 models in evaluation metrics"
    assert "Accuracy" in df.columns
    assert "Precision" in df.columns
    assert "Recall" in df.columns
    assert "F1-Score" in df.columns

    for _, row in df.iterrows():
        total_eval = row["TN"] + row["FP"] + row["FN"] + row["TP"]
        assert total_eval == 1035, f"Expected 1,035 test samples, got {total_eval}"
        assert 0.90 <= row["Accuracy"] <= 1.0, f"Unexpected accuracy: {row['Accuracy']}"
        assert 0.90 <= row["Precision"] <= 1.0, f"Unexpected precision: {row['Precision']}"

    print("  ✓ Metrics integrity verified (1,035 test samples total).")


if __name__ == "__main__":
    print("========================================")
    print(" RUNNING SMS SHIELD VERIFICATION SUITE")
    print("========================================")
    test_preprocessing()
    test_model_inference()
    test_evaluation_metrics_integrity()
    print("========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! ✅")
    print("========================================")
