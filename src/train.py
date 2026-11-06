"""
SMS Shield - Training and Evaluation Pipeline
Author: SMS Shield NLP Project

This script executes the complete machine learning pipeline:
1. Loads the UCI SMS Spam Collection dataset from the data/ directory.
2. Inspects structure and drops duplicate records.
3. Applies NLP preprocessing (lowercasing, punctuation removal, tokenization,
   stopword removal, Porter stemming).
4. Splits dataset into 80% training and 20% testing sets using stratification
   and random_state=42.
5. Fits a TF-IDF vectorizer (max_features=5000) strictly on training data to prevent
   data leakage, then transforms test data.
6. Trains:
   - Multinomial Naive Bayes
   - Logistic Regression
7. Evaluates both models on Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
8. Persists trained models and vectorizer via Joblib and exports evaluation metrics & plots.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison_bar,
)
from src.preprocessing import preprocess_text


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Loads and validates the UCI SMS Spam Collection dataset.
    Provides clear, informative error messages if file is missing.

    Parameters
    ----------
    data_path : Path
        Path to the SMSSpamCollection file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['label', 'message'].
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset file not found at: '{data_path.resolve()}'.\n"
            "Please ensure 'SMSSpamCollection' is located inside the 'data/' directory.\n"
            "You can download it from: https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
        )

    print(f"[INFO] Loading dataset from: {data_path.resolve()}")
    try:
        # The UCI dataset is tab-separated with no header. quoting=3 prevents quoting issues.
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback encoding if utf-8 encounters rare non-standard characters
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="latin-1")

    # Basic schema validation
    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("[ERROR] Dataset missing expected columns: ['label', 'message']")

    # Filter out any unexpected non-binary labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

    return df


def run_training_pipeline():
    """Executes the end-to-end training, evaluation, and artifact saving process."""
    data_path = PROJECT_ROOT / "data" / "SMSSpamCollection"
    models_dir = PROJECT_ROOT / "models"
    metrics_dir = PROJECT_ROOT / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" SMS SHIELD: NLP SPAM CLASSIFICATION TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df = load_dataset(data_path)
    initial_rows = len(df)
    print(f"[STEP 1] Loaded raw dataset: {initial_rows:,} records.")
    print("         Class breakdown before deduplication:")
    for label, count in df["label"].value_counts().items():
        print(f"           - {label.upper():4s}: {count:,} ({count / initial_rows:.2%})")

    # 2. Remove duplicate messages
    duplicate_count = df.duplicated(subset=["message"]).sum()
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    cleaned_rows = len(df)
    print(f"\n[STEP 2] Removed {duplicate_count:,} duplicate messages.")
    print(f"         Cleaned dataset size: {cleaned_rows:,} unique messages.")
    print("         Class breakdown after deduplication:")
    for label, count in df["label"].value_counts().items():
        print(f"           - {label.upper():4s}: {count:,} ({count / cleaned_rows:.2%})")

    # Save class distribution plot
    plot_class_distribution(df, label_col="label", save_path=metrics_dir / "class_distribution.png")

    # Map labels: ham -> 0, spam -> 1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. Perform NLP preprocessing
    print("\n[STEP 3] Performing NLP preprocessing on all messages...")
    print("         Pipeline: Lowercase -> Remove Punctuation -> Tokenize -> Stopwords -> Porter Stemming")
    df["clean_message"] = df["message"].apply(preprocess_text)
    print("         NLP preprocessing complete.")

    # 4. Stratified 80/20 train/test split
    print("\n[STEP 4] Splitting dataset: 80% train / 20% test (stratified, random_state=42)...")
    X = df["clean_message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"         Training set:   {len(X_train):,} samples ({y_train.sum():,} spam, {len(y_train) - y_train.sum():,} ham)")
    print(f"         Testing set:    {len(X_test):,} samples ({y_test.sum():,} spam, {len(y_test) - y_test.sum():,} ham)")

    # 5. TF-IDF vectorization (fit strictly on train to avoid data leakage)
    print("\n[STEP 5] Implementing TF-IDF vectorization (max_features=5000)...")
    print("         [NO DATA LEAKAGE] Fitting TF-IDF exclusively on training split...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
    print(f"         Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,} features.")

    # 6. Train models
    print("\n[STEP 6] Training machine learning models...")
    # Model A: Multinomial Naive Bayes
    print("         - Training Multinomial Naive Bayes (MultinomialNB)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Model B: Logistic Regression
    print("         - Training Logistic Regression (random_state=42)...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_vec, y_train)

    # 7. Evaluate models
    print("\n[STEP 7] Evaluating models on test split...")
    y_pred_nb = nb_model.predict(X_test_vec)
    y_pred_lr = lr_model.predict(X_test_vec)

    metrics_nb = compute_metrics(y_test.to_numpy(), y_pred_nb, model_name="Multinomial Naive Bayes")
    metrics_lr = compute_metrics(y_test.to_numpy(), y_pred_lr, model_name="Logistic Regression")

    # Display comparison
    results = [
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy": metrics_nb["accuracy"],
            "Precision": metrics_nb["precision"],
            "Recall": metrics_nb["recall"],
            "F1-Score": metrics_nb["f1_score"],
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
            "TN": metrics_lr["true_negatives"],
            "FP": metrics_lr["false_positives"],
            "FN": metrics_lr["false_negatives"],
            "TP": metrics_lr["true_positives"],
        },
    ]

    metrics_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" PERFORMANCE EVALUATION SUMMARY")
    print("=" * 70)
    print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))
    print("-" * 70)
    print(f"Multinomial Naive Bayes Confusion Matrix: TN={metrics_nb['true_negatives']}, FP={metrics_nb['false_positives']}, FN={metrics_nb['false_negatives']}, TP={metrics_nb['true_positives']}")
    print(f"Logistic Regression     Confusion Matrix: TN={metrics_lr['true_negatives']}, FP={metrics_lr['false_positives']}, FN={metrics_lr['false_negatives']}, TP={metrics_lr['true_positives']}")

    # 8. Save artifacts
    print("\n[STEP 8] Persisting trained models, vectorizer, and evaluation artifacts...")

    # Save models and vectorizer with Joblib
    joblib.dump(tfidf_vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, models_dir / "multinomial_nb.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    print(f"         Saved models to: {models_dir.resolve()}")

    # Save evaluation metrics CSV (actual measured numbers, not hardcoded)
    metrics_csv_path = metrics_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"         Saved evaluation CSV: {metrics_csv_path.resolve()}")

    # Save classification reports as CSV
    pd.DataFrame(metrics_nb["classification_report"]).transpose().to_csv(
        metrics_dir / "classification_report_nb.csv"
    )
    pd.DataFrame(metrics_lr["classification_report"]).transpose().to_csv(
        metrics_dir / "classification_report_lr.csv"
    )

    # Save confusion matrix plots
    plot_confusion_matrix(
        metrics_nb["confusion_matrix"],
        model_name="Multinomial Naive Bayes",
        save_path=metrics_dir / "confusion_matrix_nb.png",
        cmap="Blues",
    )
    plot_confusion_matrix(
        metrics_lr["confusion_matrix"],
        model_name="Logistic Regression",
        save_path=metrics_dir / "confusion_matrix_lr.png",
        cmap="Greens",
    )

    # Save model comparison bar chart
    plot_model_comparison_bar(
        metrics_df,
        save_path=metrics_dir / "model_comparison_bar.png"
    )
    print(f"         Saved visual plots to: {metrics_dir.resolve()}")

    print("=" * 70)
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
