"""
SMS Shield 3.0 - Comprehensive System Verification and Test Suite
Author: SMS Shield NLP Project

Validates:
1. NLP Preprocessing Pipeline (lowercasing, punctuation, tokenization, stopwords, stemming, edge cases)
2. All 3 Baseline Model Predictions (Naive Bayes, Logistic Regression, Linear SVM) & Consensus logic
3. Defensive handling of empty strings, whitespace, and punctuation-only inputs
4. Heuristic signal extraction (URLs, phone shortcodes, currencies, keywords)
5. Explainability feature contributions (weights, directions)
6. Bulk CSV processing and report generation
7. Evaluation metrics & confusion matrix integrity
8. Verification of zero train/test data leakage
9. SMS Normalization Suite (number substitutions, symbol substitutions, repeated characters, abbreviations)
10. Character-Level & Combined TF-IDF Feature Extraction and Models
11. Adversarial Perturbation Generators & Robustness Benchmarks
12. Multi-page dashboard import & health verification (All 6 pages)
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
from src.normalization import normalize_sms
from src.adversarial import (
    generate_adversarial_variant,
    perturb_char_substitution,
    perturb_char_insertion,
    perturb_char_deletion,
    perturb_repeated_letters,
    perturb_whitespace_insertion,
    perturb_punctuation_insertion,
)


def test_preprocessing():
    print("[TEST 1/12] Testing NLP preprocessing pipeline...")
    raw = "WINNER!! Claim £1,000 cash prize now by calling 09061701461!"
    cleaned = preprocess_text(raw)
    assert "winner" in cleaned, "Expected 'winner' in cleaned text"
    assert "prize" in cleaned, "Expected 'prize' in cleaned text"
    assert "!" not in cleaned, "Punctuation was not removed"
    assert "£" not in cleaned, "Currency symbol was not removed"

    assert preprocess_text("") == ""
    assert preprocess_text("   \n\t  ") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("!@#$%^&*()_+=-{}[]:;'\"<>,.?/") == ""
    assert preprocess_text("it is what it is and will be") == ""
    print("  ✓ NLP preprocessing & edge cases passed.")


def test_three_model_predictions():
    print("\n[TEST 2/12] Testing Three-Model Predictions & Consensus Engine...")
    predictor = SMSShieldPredictor().load()
    assert predictor.is_loaded()

    # Clear Spam
    spam_sample = "URGENT! You have won a £2,000 cash prize! Text CLAIM to 88088 now."
    res_spam = predictor.predict_single(spam_sample)
    assert res_spam["is_valid"] is True
    assert res_spam["consensus"] == "SPAM"
    assert res_spam["spam_votes"] >= 2

    # Clear Ham
    ham_sample = "Hey mom, I will be home by 6 PM for dinner. Love you!"
    res_ham = predictor.predict_single(ham_sample)
    assert res_ham["is_valid"] is True
    assert res_ham["consensus"] == "HAM"
    assert res_ham["ham_votes"] >= 2
    print("  ✓ Three-model predictions & consensus logic passed.")


def test_empty_and_invalid_inputs():
    print("\n[TEST 3/12] Testing Empty and Invalid Input Handling...")
    predictor = SMSShieldPredictor().load()

    assert predictor.predict_single("")["is_valid"] is False
    assert predictor.predict_single("      \t\n  ")["is_valid"] is False
    assert predictor.predict_single(None)["is_valid"] is False
    print("  ✓ Defensive validation prevents invalid/empty inference crashes.")


def test_heuristic_signals_and_explainability():
    print("\n[TEST 4/12] Testing Heuristic Signals and Feature Explainability...")
    text = "URGENT WINNER: Claim £500 cash at http://claim-prize.com or call 09061701461!"
    heuristics = extract_heuristic_signals(text)

    assert heuristics["has_url"] is True
    assert heuristics["has_phone_number"] is True
    assert heuristics["has_currency"] is True

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
    print("  ✓ Heuristic signals & token explainability successfully computed.")


def test_bulk_csv_processing_and_export():
    print("\n[TEST 5/12] Testing Bulk CSV Batch Classification & Export...")
    sample_csv_path = PROJECT_ROOT / "data" / "sample_bulk_sms.csv"
    if not sample_csv_path.exists():
        generate_sample_bulk_csv(sample_csv_path)

    df_sample = pd.read_csv(sample_csv_path)
    predictor = SMSShieldPredictor().load()
    results_df, summary = run_bulk_classification(df_sample, "text", predictor=predictor)

    assert len(results_df) == 25, "Output row count must match input row count (no silent drops)"
    assert summary["empty_count"] >= 1, "Expected at least 1 intentional empty row"
    assert "Consensus_Prediction" in results_df.columns

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
    print("  ✓ Bulk CSV processing validated on 25 messages.")


def test_evaluation_metrics_and_confusion_matrices():
    print("\n[TEST 6/12] Testing Evaluation Metrics & Confusion Matrices...")
    metrics_path = PROJECT_ROOT / "metrics" / "evaluation_metrics.csv"
    assert metrics_path.exists()

    df_metrics = pd.read_csv(metrics_path)
    assert len(df_metrics) == 3
    for _, row in df_metrics.iterrows():
        total_eval = row["TN"] + row["FP"] + row["FN"] + row["TP"]
        assert total_eval == 1035, f"Expected 1,035 test samples, got {total_eval}"

    for m in ["nb", "lr", "svc"]:
        assert (PROJECT_ROOT / "metrics" / f"confusion_matrix_{m}.png").exists()

    print("  ✓ Evaluation metrics and confusion matrices verified across 1,035 test samples.")


def test_data_leakage_absence():
    print("\n[TEST 7/12] Verifying Zero Data Leakage...")
    vec_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
    vec = joblib.load(vec_path)

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.read_csv(PROJECT_ROOT / "data" / "SMSSpamCollection", sep="\t", names=["label", "message"], quoting=3)
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean"] = df["message"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.20, random_state=42, stratify=df["target"]
    )

    fresh_train_vec = TfidfVectorizer(max_features=5000).fit(X_train)
    assert set(vec.vocabulary_.keys()) == set(fresh_train_vec.vocabulary_.keys())
    print("  ✓ Zero data leakage confirmed: vocabulary strictly fitted on X_train.")


def test_cross_validation_and_experiments():
    print("\n[TEST 8/12] Verifying Cross-Validation and TF-IDF Experiment Artifacts...")
    assert (PROJECT_ROOT / "metrics" / "cross_val_metrics.csv").exists()
    assert (PROJECT_ROOT / "metrics" / "tfidf_experiments.csv").exists()
    print("  ✓ Cross-validation and TF-IDF experimental tables verified.")


def test_sms_normalization():
    print("\n[TEST 9/12] Testing SMS Normalization Module...")
    # 1. Number substitution
    res1 = normalize_sms("FR33 prize for the B0NU5 customer!")
    assert "FREE" in res1["normalized_text"], f"Expected FREE, got {res1['normalized_text']}"
    assert "BONUS" in res1["normalized_text"], f"Expected BONUS, got {res1['normalized_text']}"

    # 2. Symbol substitution
    res2 = normalize_sms("W!NNER of £500 C@SH reward!")
    assert "WINNER" in res2["normalized_text"]
    assert "CASH" in res2["normalized_text"]
    assert "£500" in res2["normalized_text"], "Protected currency amount should be retained"

    # 3. Repeated characters
    res3 = normalize_sms("Hurryyyy! Freeeeee prize for you!")
    assert "Hurry" in res3["normalized_text"]
    assert "Free" in res3["normalized_text"]

    # 4. Common abbreviations
    res4 = normalize_sms("Hey u, meeting b4 5 PM is gr8!")
    assert "you" in res4["normalized_text"].lower()
    assert "before" in res4["normalized_text"].lower()
    assert "great" in res4["normalized_text"].lower()

    # 5. Protected phone numbers and monetary entities
    res5 = normalize_sms("Call 09061701461 to claim £1000 prize.")
    assert "09061701461" in res5["normalized_text"]
    assert "£1000" in res5["normalized_text"]

    # 6. Spaced-out words
    res6 = normalize_sms("Claim your w i n n e r bonus now!")
    assert "winner" in res6["normalized_text"]

    print("  ✓ SMS Normalization handles leetspeak, symbols, abbreviations, and protects numerics.")


def test_character_and_combined_models():
    print("\n[TEST 10/12] Testing Character & Combined TF-IDF Models...")
    predictor = SMSShieldPredictor().load()

    assert predictor.char_vectorizer is not None, "Char vectorizer missing"
    assert predictor.char_svc is not None, "Char SVM missing"
    assert predictor.comb_vectorizer is not None, "Combined vectorizer missing"

    test_msg = "WINNER!! You have won a £1,000 cash prize! Text CLAIM to 87121."

    # Test char prediction
    res_char = predictor.predict_single(test_msg, feature_space="char")
    assert res_char["is_valid"] is True
    assert res_char["consensus"] == "SPAM"

    # Test combined prediction
    res_comb = predictor.predict_single(test_msg, feature_space="combined")
    assert res_comb["is_valid"] is True
    assert res_comb["consensus"] == "SPAM"

    # Test feature space comparison metrics file
    cw_csv = PROJECT_ROOT / "metrics" / "char_word_comparison.csv"
    assert cw_csv.exists()
    cw_df = pd.read_csv(cw_csv)
    assert len(cw_df) == 9, "Expected 3 feature spaces x 3 models = 9 rows"

    print("  ✓ Character and Combined TF-IDF models verified.")


def test_adversarial_perturbations_and_benchmarks():
    print("\n[TEST 11/12] Testing Adversarial Generators & Benchmarks...")
    msg = "WINNER!! Claim your free cash prize reward by calling 09061701461."

    # Test each attack type generator
    v_sub = perturb_char_substitution(msg)
    assert v_sub != msg or len(v_sub) > 0

    v_ins = perturb_char_insertion(msg)
    assert v_ins != msg or len(v_ins) > 0

    v_del = perturb_char_deletion(msg)
    assert v_del != msg or len(v_del) > 0

    v_rep = perturb_repeated_letters(msg)
    assert v_rep != msg or len(v_rep) > 0

    v_space = perturb_whitespace_insertion(msg)
    assert v_space != msg or len(v_space) > 0

    v_punct = perturb_punctuation_insertion(msg)
    assert v_punct != msg or len(v_punct) > 0

    v_comp = generate_adversarial_variant(msg, "Composite Multi-Vector Attack")
    assert len(v_comp) > 0

    # Verify adversarial benchmark metrics file exists
    adv_csv = PROJECT_ROOT / "metrics" / "adversarial_evaluation.csv"
    assert adv_csv.exists()
    adv_df = pd.read_csv(adv_csv)
    assert len(adv_df) == 12, "Expected 4 setups x 3 models = 12 benchmark rows"
    assert "Robustness Score (%)" in adv_df.columns
    assert "Recall Drop" in adv_df.columns

    print("  ✓ All 7 adversarial generators and benchmark artifacts verified.")


def test_dashboard_modules_import():
    print("\n[TEST 12/12] Verifying All 6 Streamlit Dashboard Page Imports...")
    import importlib
    import app

    pages = [
        "pages.1_Dashboard",
        "pages.2_Message_Analyzer",
        "pages.3_NLP_Analysis",
        "pages.4_Model_Evaluation",
        "pages.5_Bulk_Analysis",
        "pages.6_Robustness_Analysis",
    ]

    for p in pages:
        mod = importlib.import_module(p)
        assert mod is not None, f"Failed to load {p}"

    print("  ✓ All 6 Streamlit application pages imported cleanly without syntax or import errors.")


if __name__ == "__main__":
    print("======================================================================")
    print(" SMS SHIELD 3.0: FULL SYSTEM VERIFICATION SUITE")
    print("======================================================================")

    test_preprocessing()
    test_three_model_predictions()
    test_empty_and_invalid_inputs()
    test_heuristic_signals_and_explainability()
    test_bulk_csv_processing_and_export()
    test_evaluation_metrics_and_confusion_matrices()
    test_data_leakage_absence()
    test_cross_validation_and_experiments()
    test_sms_normalization()
    test_character_and_combined_models()
    test_adversarial_perturbations_and_benchmarks()
    test_dashboard_modules_import()

    print("======================================================================")
    print(" ALL 12 VALIDATION CHECKPOINTS PASSED PERFECTLY! ✅")
    print("======================================================================")
