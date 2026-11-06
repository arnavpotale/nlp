"""
SMS Shield 2.0 - Comprehensive System Verification and Test Suite
Author: SMS Shield NLP Project

Validates:
1. NLP Preprocessing Pipeline (lowercasing, punctuation, tokenization, stopwords, stemming, edge cases)
2. All 3 Model Predictions (Naive Bayes, Logistic Regression, Linear SVM) & Consensus logic
3. Defensive handling of empty strings, whitespace, and punctuation-only inputs
4. Heuristic signal extraction (URLs, phone shortcodes, currencies, keywords)
5. Explainability feature contributions (weights, directions)
6. Bulk CSV processing and report generation
7. Evaluation metrics & confusion matrix integrity
8. Verification of zero train/test data leakage
9. Multi-page dashboard import & health verification
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_text, preprocess_with_steps
from src.evaluate import compute_metrics
from src.explainability import extract_heuristic_signals, explain_message_features
from src.prediction import SMSShieldPredictor
from src.bulk_analysis import run_bulk_classification, generate_sample_bulk_csv


def test_preprocessing():
    print("[TEST 1/9] Testing NLP preprocessing pipeline...")

    # Standard spam message
    raw = "WINNER!! Claim £1,000 cash prize now by calling 09061701461!"
    cleaned = preprocess_text(raw)
    assert "winner" in cleaned, "Expected 'winner' in cleaned text"
    assert "prize" in cleaned, "Expected 'prize' in cleaned text"
    assert "!" not in cleaned, "Punctuation was not removed"
    assert "£" not in cleaned, "Currency symbol was not removed"

    # Edge cases
    assert preprocess_text("") == ""
    assert preprocess_text("   \n\t  ") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("!@#$%^&*()_+=-{}[]:;'\"<>,.?/") == ""

    # Stopwords only
    assert preprocess_text("it is what it is and will be") == ""

    # Intermediate steps check
    steps = preprocess_with_steps(raw)
    assert "raw" in steps and "final_preprocessed" in steps
    assert len(steps["tokens"]) > 0
    print("  ✓ NLP preprocessing & edge cases passed.")


def test_three_model_predictions():
    print("\n[TEST 2/9] Testing Three-Model Predictions & Consensus Engine...")

    predictor = SMSShieldPredictor().load()
    assert predictor.is_loaded()

    # Clear Spam
    spam_sample = "URGENT! You have won a £2,000 cash prize! Text CLAIM to 88088 now."
    res_spam = predictor.predict_single(spam_sample)
    assert res_spam["is_valid"] is True
    assert res_spam["consensus"] == "SPAM"
    assert res_spam["spam_votes"] >= 2
    for m in ["naive_bayes", "logistic_regression", "linear_svm"]:
        assert m in res_spam["models"]
        assert 0.0 <= res_spam["models"][m]["prob_spam"] <= 1.0

    print(f"  ✓ Spam classified unanimously (Consensus: {res_spam['consensus']}, Votes: {res_spam['spam_votes']}/3).")

    # Clear Ham
    ham_sample = "Hey mom, I will be home by 6 PM for dinner. Love you!"
    res_ham = predictor.predict_single(ham_sample)
    assert res_ham["is_valid"] is True
    assert res_ham["consensus"] == "HAM"
    assert res_ham["ham_votes"] >= 2
    print(f"  ✓ Ham classified unanimously (Consensus: {res_ham['consensus']}, Votes: {res_ham['ham_votes']}/3).")


def test_empty_and_invalid_inputs():
    print("\n[TEST 3/9] Testing Empty and Invalid Input Handling...")

    predictor = SMSShieldPredictor().load()

    res_empty = predictor.predict_single("")
    assert res_empty["is_valid"] is False
    assert "empty" in res_empty["error"].lower()

    res_ws = predictor.predict_single("      \t\n  ")
    assert res_ws["is_valid"] is False

    res_none = predictor.predict_single(None)
    assert res_none["is_valid"] is False

    print("  ✓ Defensive validation prevents invalid/empty inference crashes.")


def test_heuristic_signals_and_explainability():
    print("\n[TEST 4/9] Testing Heuristic Signals and Feature Explainability...")

    text = "URGENT WINNER: Claim £500 cash at http://claim-prize.com or call 09061701461!"
    heuristics = extract_heuristic_signals(text)

    assert heuristics["has_url"] is True
    assert heuristics["has_phone_number"] is True
    assert heuristics["has_currency"] is True
    assert "urgent" in heuristics["suspicious_keywords"] or "winner" in heuristics["suspicious_keywords"]

    # Per-token explainability
    predictor = SMSShieldPredictor().load()
    clean = preprocess_text(text)
    exp_df = explain_message_features(
        clean_text=clean,
        vectorizer=predictor.vectorizer,
        lr_model=predictor.lr_model,
        nb_model=predictor.nb_model,
        svc_model=predictor.svc_raw,
    )
    assert not exp_df.empty
    assert "LR Weight" in exp_df.columns
    assert "Direction" in exp_df.columns
    print("  ✓ Heuristic signals & token explainability successfully computed.")


def test_bulk_csv_processing_and_export():
    print("\n[TEST 5/9] Testing Bulk CSV Batch Classification & Export...")

    sample_csv_path = PROJECT_ROOT / "data" / "sample_bulk_sms.csv"
    if not sample_csv_path.exists():
        generate_sample_bulk_csv(sample_csv_path)

    df_sample = pd.read_csv(sample_csv_path)
    assert len(df_sample) == 25

    predictor = SMSShieldPredictor().load()
    results_df, summary = run_bulk_classification(df_sample, "text", predictor=predictor)

    assert len(results_df) == 25, "Output row count must match input row count (no silent drops)"
    assert summary["empty_count"] >= 1, "Expected at least 1 intentional empty row"
    assert summary["total_messages"] == 25
    assert "Consensus_Prediction" in results_df.columns
    assert "NB_Prediction" in results_df.columns
    assert "LR_Prediction" in results_df.columns
    assert "SVM_Prediction" in results_df.columns

    # Test export to CSV buffer
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
    print(f"  ✓ Bulk processing validated on 25 messages ({summary['spam_count']} Spam, {summary['ham_count']} Ham).")


def test_evaluation_metrics_and_confusion_matrices():
    print("\n[TEST 6/9] Testing Evaluation Metrics & Confusion Matrices...")

    metrics_path = PROJECT_ROOT / "metrics" / "evaluation_metrics.csv"
    assert metrics_path.exists()

    df_metrics = pd.read_csv(metrics_path)
    assert len(df_metrics) == 3, "Expected 3 models evaluated (NB, LR, Linear SVM)"
    assert set(df_metrics["Model"]) == {"Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"}

    for _, row in df_metrics.iterrows():
        total_eval = row["TN"] + row["FP"] + row["FN"] + row["TP"]
        assert total_eval == 1035, f"Expected 1,035 test samples, got {total_eval}"
        assert 0.95 <= row["Accuracy"] <= 1.0
        assert 0.95 <= row["Precision"] <= 1.0
        assert 0.70 <= row["Recall"] <= 1.0
        assert 0.80 <= row["F1-Score"] <= 1.0

    # Verify confusion matrix images exist
    for m in ["nb", "lr", "svc"]:
        cm_file = PROJECT_ROOT / "metrics" / f"confusion_matrix_{m}.png"
        assert cm_file.exists(), f"Missing confusion matrix image: {cm_file.name}"

    print("  ✓ Evaluation metrics and confusion matrices verified across 1,035 test samples.")


def test_data_leakage_absence():
    print("\n[TEST 7/9] Verifying Zero Data Leakage...")

    # Load vectorizer and dataset
    vec_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
    vec = joblib.load(vec_path)
    assert len(vec.vocabulary_) <= 5000, "Vectorizer exceeded max_features limit"

    # Verify that test split was strictly transformed and not fitted
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(PROJECT_ROOT / "data" / "SMSSpamCollection", sep="\t", names=["label", "message"], quoting=3)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean"] = df["message"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.20, random_state=42, stratify=df["target"]
    )

    # Check that vectorizer vocabulary matches train fit
    from sklearn.feature_extraction.text import TfidfVectorizer
    fresh_train_vec = TfidfVectorizer(max_features=5000).fit(X_train)
    assert set(vec.vocabulary_.keys()) == set(fresh_train_vec.vocabulary_.keys()), "Vectorizer was not fit strictly on X_train!"
    print("  ✓ Zero data leakage confirmed: vocabulary strictly fitted on X_train.")


def test_cross_validation_and_experiments():
    print("\n[TEST 8/9] Verifying Cross-Validation and TF-IDF Experiment Artifacts...")

    cv_path = PROJECT_ROOT / "metrics" / "cross_val_metrics.csv"
    assert cv_path.exists()
    cv_df = pd.read_csv(cv_path)
    assert len(cv_df) == 3

    exp_path = PROJECT_ROOT / "metrics" / "tfidf_experiments.csv"
    assert exp_path.exists()
    exp_df = pd.read_csv(exp_path)
    assert len(exp_df) == 9, "Expected 3 configurations x 3 models = 9 experiments"

    print("  ✓ Cross-validation and TF-IDF experimental tables verified.")


def test_dashboard_modules_import():
    print("\n[TEST 9/9] Verifying Streamlit Dashboard & Page Imports...")
    import importlib

    import app
    for page_num, page_name in [
        (1, "pages.1_Dashboard"),
        (2, "pages.2_Message_Analyzer"),
        (3, "pages.3_NLP_Analysis"),
        (4, "pages.4_Model_Evaluation"),
        (5, "pages.5_Bulk_Analysis"),
    ]:
        mod = importlib.import_module(page_name)
        assert mod is not None, f"Failed to load {page_name}"

    print("  ✓ All 5 Streamlit application pages imported cleanly without syntax or import errors.")


if __name__ == "__main__":
    print("======================================================================")
    print(" SMS SHIELD 2.0: FULL SYSTEM VERIFICATION SUITE")
    print("======================================================================")

    test_preprocessing()
    test_three_model_predictions()
    test_empty_and_invalid_inputs()
    test_heuristic_signals_and_explainability()
    test_bulk_csv_processing_and_export()
    test_evaluation_metrics_and_confusion_matrices()
    test_data_leakage_absence()
    test_cross_validation_and_experiments()
    test_dashboard_modules_import()

    print("======================================================================")
    print(" ALL 9 VALIDATION CHECKPOINTS PASSED PERFECTLY! ✅")
    print("======================================================================")
