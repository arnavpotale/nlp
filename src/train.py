"""
SMS Shield 2.0 - Comprehensive Training & Experimental Pipeline
Author: SMS Shield NLP Project

This script executes the complete machine learning engineering pipeline:
1. Loads the UCI SMS Spam Collection dataset from data/SMSSpamCollection.
2. Removes duplicate messages.
3. Applies NLP preprocessing (lowercasing, punctuation stripping, tokenization,
   stopword removal, Porter stemming).
4. Performs stratified 80/20 train/test split with random_state=42.
5. Evaluates three TF-IDF feature configurations (Unigrams, Bigrams, Trigrams)
   to evaluate feature representation impact.
6. Fits the main TF-IDF vectorizer (max_features=5000) strictly on training data
   to guarantee zero data leakage.
7. Trains three classification models:
   - Model 1: Multinomial Naive Bayes (MultinomialNB)
   - Model 2: Logistic Regression (LogisticRegression)
   - Model 3: Linear Support Vector Machine (LinearSVC + CalibratedClassifierCV)
8. Evaluates all three models on test data and performs 5-fold cross-validation.
9. Exports all models, evaluation metrics, diagnostic curves (ROC, PR, CM, Bar),
   and reports to models/ and metrics/.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_precision_recall_curves,
)
from src.preprocessing import preprocess_text
from src.bulk_analysis import generate_sample_bulk_csv


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads and validates the UCI dataset from local storage."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset file not found at: '{data_path.resolve()}'.\n"
            "Please ensure 'SMSSpamCollection' is located in the 'data/' directory."
        )

    print(f"[INFO] Loading dataset from: {data_path.resolve()}")
    try:
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="latin-1")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df


def run_tfidf_experiments(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates three distinct TF-IDF feature configurations across all three models:
    - Config A: Unigrams only, max_features=5000
    - Config B: Unigrams and bigrams, max_features=5000
    - Config C: Unigrams, bigrams, and trigrams, max_features=10000
    """
    print("\n" + "=" * 70)
    print(" EXPERIMENT: TF-IDF N-GRAM & FEATURE CAPACITY ANALYSIS")
    print("=" * 70)

    configs = [
        ("Config A (Unigrams, max=5000)", (1, 1), 5000),
        ("Config B (Unigram+Bigram, max=5000)", (1, 2), 5000),
        ("Config C (Unigram+Bigram+Trigram, max=10000)", (1, 3), 10000),
    ]

    model_defs = [
        ("Multinomial Naive Bayes", MultinomialNB()),
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        ("Linear SVM", LinearSVC(random_state=42, max_iter=2000)),
    ]

    exp_records = []
    for cfg_name, ngrams, max_feat in configs:
        print(f"\nEvaluating {cfg_name}...")
        for model_name, clf in model_defs:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=ngrams, max_features=max_feat)),
                ("clf", clf),
            ])
            # Zero data leakage: pipeline fits strictly on training split
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            exp_records.append({
                "Configuration": cfg_name,
                "Model": model_name,
                "N-Gram Range": str(ngrams),
                "Max Features": max_feat,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
            })
            print(f"  - {model_name:24s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    return pd.DataFrame(exp_records)


def run_cross_validation(X: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Runs 5-fold Stratified Cross-Validation on the full deduplicated dataset."""
    print("\n" + "=" * 70)
    print(" 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]

    cv_records = []
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", clf),
        ])
        scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        record = {
            "Model": name,
            "CV_Accuracy_Mean": float(scores["test_accuracy"].mean()),
            "CV_Accuracy_Std": float(scores["test_accuracy"].std()),
            "CV_Precision_Mean": float(scores["test_precision"].mean()),
            "CV_Precision_Std": float(scores["test_precision"].std()),
            "CV_Recall_Mean": float(scores["test_recall"].mean()),
            "CV_Recall_Std": float(scores["test_recall"].std()),
            "CV_F1_Mean": float(scores["test_f1"].mean()),
            "CV_F1_Std": float(scores["test_f1"].std()),
        }
        cv_records.append(record)
        print(f"{name:24s} -> Acc: {record['CV_Accuracy_Mean']:.4f} ± {record['CV_Accuracy_Std']:.4f} | Prec: {record['CV_Precision_Mean']:.4f} | Rec: {record['CV_Recall_Mean']:.4f} | F1: {record['CV_F1_Mean']:.4f}")

    return pd.DataFrame(cv_records)


def run_training_pipeline():
    """Executes the complete training, evaluation, and artifact saving process."""
    data_path = PROJECT_ROOT / "data" / "SMSSpamCollection"
    models_dir = PROJECT_ROOT / "models"
    metrics_dir = PROJECT_ROOT / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" SMS SHIELD 2.0: ADVANCED NLP CLASSIFICATION PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df = load_dataset(data_path)
    initial_rows = len(df)
    print(f"[STEP 1] Loaded raw dataset: {initial_rows:,} records.")

    # 2. Remove duplicates
    duplicate_count = df.duplicated(subset=["message"]).sum()
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    cleaned_rows = len(df)
    print(f"[STEP 2] Removed {duplicate_count:,} duplicates. Unique records: {cleaned_rows:,}.")

    # Save class distribution plot
    plot_class_distribution(df, label_col="label", save_path=metrics_dir / "class_distribution.png")

    # Target mapping
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. NLP Preprocessing
    print("\n[STEP 3] Running 5-step NLP preprocessing pipeline on all messages...")
    df["clean_message"] = df["message"].apply(preprocess_text)

    # 4. Stratified 80/20 train/test split
    print("\n[STEP 4] Splitting dataset (80% train / 20% test, stratified, random_state=42)...")
    X = df["clean_message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"         Training set: {len(X_train):,} samples ({y_train.sum():,} spam, {len(y_train)-y_train.sum():,} ham)")
    print(f"         Testing set:  {len(X_test):,} samples ({y_test.sum():,} spam, {len(y_test)-y_test.sum():,} ham)")

    # 5. TF-IDF N-gram Experiments
    exp_df = run_tfidf_experiments(X_train, X_test, y_train, y_test)
    exp_csv_path = metrics_dir / "tfidf_experiments.csv"
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"         Saved TF-IDF experiments table to: {exp_csv_path.resolve()}")

    # 6. Fit main TF-IDF vectorizer (Config A: max_features=5000)
    print("\n[STEP 5] Fitting main TF-IDF vectorizer (strictly on train split)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    # 7. Train Models
    print("\n[STEP 6] Training three machine learning classifiers...")
    # Model 1: Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_vec, y_train)

    # Model 3: Linear Support Vector Machine (Raw + Calibrated)
    svc_raw = LinearSVC(random_state=42, max_iter=2000)
    svc_raw.fit(X_train_vec, y_train)

    svc_calibrated = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=5)
    svc_calibrated.fit(X_train_vec, y_train)

    # 8. Evaluation
    print("\n[STEP 7] Evaluating models on test split (1,035 samples)...")
    y_test_arr = y_test.to_numpy()

    # Predictions & scores
    y_pred_nb = nb_model.predict(X_test_vec)
    y_scores_nb = nb_model.predict_proba(X_test_vec)[:, 1]

    y_pred_lr = lr_model.predict(X_test_vec)
    y_scores_lr = lr_model.predict_proba(X_test_vec)[:, 1]

    y_pred_svc = svc_raw.predict(X_test_vec)
    y_scores_svc_dec = svc_raw.decision_function(X_test_vec)
    y_scores_svc_cal = svc_calibrated.predict_proba(X_test_vec)[:, 1]

    metrics_nb = compute_metrics(y_test_arr, y_pred_nb, y_scores=y_scores_nb, model_name="Multinomial Naive Bayes")
    metrics_lr = compute_metrics(y_test_arr, y_pred_lr, y_scores=y_scores_lr, model_name="Logistic Regression")
    metrics_svc = compute_metrics(y_test_arr, y_pred_svc, y_scores=y_scores_svc_dec, model_name="Linear SVM")

    results = [
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy": metrics_nb["accuracy"],
            "Precision": metrics_nb["precision"],
            "Recall": metrics_nb["recall"],
            "F1-Score": metrics_nb["f1_score"],
            "ROC-AUC": metrics_nb["roc_auc"],
            "TN": metrics_nb["true_negatives"],
            "FP": metrics_nb["false_positives"],
            "FN": metrics_nb["false_negatives"],
            "TP": metrics_nb["true_positives"],
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": metrics_lr["accuracy"],
            "Precision": metrics_lr["precision"],
            "Recall": metrics_lr["recall"],
            "F1-Score": metrics_lr["f1_score"],
            "ROC-AUC": metrics_lr["roc_auc"],
            "TN": metrics_lr["true_negatives"],
            "FP": metrics_lr["false_positives"],
            "FN": metrics_lr["false_negatives"],
            "TP": metrics_lr["true_positives"],
        },
        {
            "Model": "Linear SVM",
            "Accuracy": metrics_svc["accuracy"],
            "Precision": metrics_svc["precision"],
            "Recall": metrics_svc["recall"],
            "F1-Score": metrics_svc["f1_score"],
            "ROC-AUC": metrics_svc["roc_auc"],
            "TN": metrics_svc["true_negatives"],
            "FP": metrics_svc["false_positives"],
            "FN": metrics_svc["false_negatives"],
            "TP": metrics_svc["true_positives"],
        },
    ]

    metrics_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" THREE-MODEL EVALUATION SUMMARY")
    print("=" * 70)
    print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    # 9. Cross-Validation
    cv_df = run_cross_validation(X, y)
    cv_df.to_csv(metrics_dir / "cross_val_metrics.csv", index=False)

    # 10. Save Artifacts
    print("\n[STEP 8] Persisting trained model artifacts and diagnostic charts...")
    joblib.dump(tfidf_vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, models_dir / "multinomial_nb.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(svc_raw, models_dir / "linear_svc.joblib")
    joblib.dump(svc_calibrated, models_dir / "calibrated_linear_svc.joblib")

    # Save metrics CSV
    metrics_csv_path = metrics_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save classification reports
    pd.DataFrame(metrics_nb["classification_report"]).transpose().to_csv(metrics_dir / "classification_report_nb.csv")
    pd.DataFrame(metrics_lr["classification_report"]).transpose().to_csv(metrics_dir / "classification_report_lr.csv")
    pd.DataFrame(metrics_svc["classification_report"]).transpose().to_csv(metrics_dir / "classification_report_svc.csv")

    # Confusion matrix plots
    plot_confusion_matrix(metrics_nb["confusion_matrix"], "Multinomial Naive Bayes", metrics_dir / "confusion_matrix_nb.png", "Blues")
    plot_confusion_matrix(metrics_lr["confusion_matrix"], "Logistic Regression", metrics_dir / "confusion_matrix_lr.png", "Greens")
    plot_confusion_matrix(metrics_svc["confusion_matrix"], "Linear SVM", metrics_dir / "confusion_matrix_svc.png", "Purples")

    # Comparison bar chart
    plot_model_comparison_bar(metrics_df, metrics_dir / "model_comparison_bar.png")

    # ROC Curves
    fpr_nb, tpr_nb, _ = roc_curve(y_test_arr, y_scores_nb)
    fpr_lr, tpr_lr, _ = roc_curve(y_test_arr, y_scores_lr)
    fpr_svc, tpr_svc, _ = roc_curve(y_test_arr, y_scores_svc_dec)

    roc_dict = {
        "Multinomial Naive Bayes": (fpr_nb, tpr_nb, metrics_nb["roc_auc"]),
        "Logistic Regression": (fpr_lr, tpr_lr, metrics_lr["roc_auc"]),
        "Linear SVM": (fpr_svc, tpr_svc, metrics_svc["roc_auc"]),
    }
    plot_roc_curves(roc_dict, metrics_dir / "roc_curves.png")

    # Precision-Recall Curves
    p_nb, r_nb, _ = precision_recall_curve(y_test_arr, y_scores_nb)
    p_lr, r_lr, _ = precision_recall_curve(y_test_arr, y_scores_lr)
    p_svc, r_svc, _ = precision_recall_curve(y_test_arr, y_scores_svc_dec)

    pr_dict = {
        "Multinomial Naive Bayes": (p_nb, r_nb, average_precision_score(y_test_arr, y_scores_nb)),
        "Logistic Regression": (p_lr, r_lr, average_precision_score(y_test_arr, y_scores_lr)),
        "Linear SVM": (p_svc, r_svc, average_precision_score(y_test_arr, y_scores_svc_dec)),
    }
    plot_precision_recall_curves(pr_dict, metrics_dir / "pr_curves.png")

    # Generate sample bulk test CSV
    generate_sample_bulk_csv(PROJECT_ROOT / "data" / "sample_bulk_sms.csv")

    print("=" * 70)
    print(" SMS SHIELD 2.0: ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
