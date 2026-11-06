"""
SMS Shield 3.0 - Comprehensive Training, Feature Space & Adversarial Pipeline
Author: SMS Shield NLP Project

This script executes the complete machine learning engineering pipeline for SMS Shield 3.0:
1. Loads and deduplicates the UCI SMS Spam Collection dataset.
2. Applies consistent 5-step NLP preprocessing.
3. Performs stratified 80/20 train/test split (random_state=42).
4. Feature Spaces Trained (Strictly on Train Split):
   - Feature Space 1: Word-Level TF-IDF (1,2 n-grams, max 5,000 features)
   - Feature Space 2: Character-Level TF-IDF (char_wb 2,5 n-grams, max 5,000 features)
   - Feature Space 3: Combined Word + Character TF-IDF (FeatureUnion, max 10,000 features)
5. Classifiers Trained for each Feature Space:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (Raw + CalibratedClassifierCV)
6. Comparative Evaluation:
   - Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC across all spaces.
   - Saves metrics/char_word_comparison.csv and plot.
7. Adversarial Robustness Benchmarking:
   - Evaluates resilience against leetspeak, spacing, deletions, and composite attacks.
   - Saves metrics/adversarial_evaluation.csv and plot.
8. Persists all model artifacts to models/ and metrics/.
"""

import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
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
from src.normalization import normalize_sms
from src.adversarial import evaluate_adversarial_robustness
from src.bulk_analysis import generate_sample_bulk_csv


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads and validates the UCI dataset from local storage."""
    if not data_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset file not found at: '{data_path.resolve()}'.")

    print(f"[INFO] Loading dataset from: {data_path.resolve()}")
    try:
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, sep="\t", names=["label", "message"], quoting=3, encoding="latin-1")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df


def plot_char_word_comparison_bar(comparison_df: pd.DataFrame, save_path: Path):
    """Generates a bar chart comparing Word vs Char vs Combined TF-IDF."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Melt dataframe for seaborn barplot
    df_melt = comparison_df.melt(
        id_vars=["Feature Space", "Classifier"],
        value_vars=["Accuracy", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    palette = ["#2563eb", "#059669", "#7c3aed"]
    sns.barplot(
        data=df_melt,
        x="Classifier",
        y="Score",
        hue="Feature Space",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    ax.set_ylim(0.65, 1.05)
    ax.set_title("Feature Space Comparison: Word vs. Character vs. Combined TF-IDF", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Score", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Classifier", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Feature Representation", loc="lower right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_robustness_bar(adv_df: pd.DataFrame, save_path: Path):
    """Plots robustness percentage and recall degradation across defenses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Subplot 1: Robustness Score (%)
    palette = ["#dc2626", "#2563eb", "#059669", "#7c3aed"]
    sns.barplot(
        data=adv_df,
        x="Classifier",
        y="Robustness Score (%)",
        hue="Defense Pipeline",
        palette=palette,
        edgecolor="black",
        ax=ax1
    )
    ax1.set_ylim(60, 105)
    ax1.set_title("Adversarial Robustness Score (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predictions Unchanged (%)", fontsize=9.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(title="Pipeline", loc="lower left", fontsize=8)

    # Subplot 2: Adversarial Recall
    sns.barplot(
        data=adv_df,
        x="Classifier",
        y="Adversarial Recall",
        hue="Defense Pipeline",
        palette=palette,
        edgecolor="black",
        ax=ax2
    )
    ax2.set_ylim(0.3, 1.05)
    ax2.set_title("Spam Recall Under Adversarial Attack", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Recall Score", fontsize=9.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(title="Pipeline", loc="lower left", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_training_pipeline():
    """Executes the complete SMS Shield 3.0 training, evaluation & adversarial pipeline."""
    data_path = PROJECT_ROOT / "data" / "SMSSpamCollection"
    models_dir = PROJECT_ROOT / "models"
    metrics_dir = PROJECT_ROOT / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" SMS SHIELD 3.0: ADVANCED NLP & ADVERSARIAL ROBUSTNESS PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df = load_dataset(data_path)
    print(f"[STEP 1] Loaded raw dataset: {len(df):,} records.")

    # 2. Remove duplicates
    duplicate_count = df.duplicated(subset=["message"]).sum()
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    cleaned_rows = len(df)
    print(f"[STEP 2] Removed {duplicate_count:,} duplicates. Unique records: {cleaned_rows:,}.")

    # Save class distribution
    plot_class_distribution(df, label_col="label", save_path=metrics_dir / "class_distribution.png")
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. NLP Preprocessing
    print("\n[STEP 3] Running 5-step NLP preprocessing on all messages...")
    df["clean_message"] = df["message"].apply(preprocess_text)

    # 4. Stratified 80/20 train/test split
    print("\n[STEP 4] Stratified 80/20 train/test split (random_state=42)...")
    X = df["clean_message"]
    y = df["target"]

    train_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42, stratify=y
    )

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    X_train = train_df["clean_message"]
    y_train = train_df["target"]
    X_test = test_df["clean_message"]
    y_test = test_df["target"]

    print(f"         Training set: {len(X_train):,} samples ({y_train.sum():,} spam, {len(y_train)-y_train.sum():,} ham)")
    print(f"         Testing set:  {len(X_test):,} samples ({y_test.sum():,} spam, {len(y_test)-y_test.sum():,} ham)")

    # -------------------------------------------------------------------------
    # 5. Train Feature Space 1: Word-Level TF-IDF (1,2)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Training Feature Space 1: Word-Level TF-IDF...")
    word_vec = TfidfVectorizer(max_features=5000)
    X_train_word = word_vec.fit_transform(X_train)
    X_test_word = word_vec.transform(X_test)

    word_nb = MultinomialNB().fit(X_train_word, y_train)
    word_lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_word, y_train)
    word_svc_raw = LinearSVC(random_state=42, max_iter=2000).fit(X_train_word, y_train)
    word_svc_cal = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=5).fit(X_train_word, y_train)

    # -------------------------------------------------------------------------
    # 6. Train Feature Space 2: Character-Level TF-IDF (char_wb, (2,5))
    # -------------------------------------------------------------------------
    print("[STEP 6] Training Feature Space 2: Character-Level TF-IDF (char_wb (2,5))...")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)
    X_train_char = char_vec.fit_transform(X_train)
    X_test_char = char_vec.transform(X_test)

    char_nb = MultinomialNB().fit(X_train_char, y_train)
    char_lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_char, y_train)
    char_svc_raw = LinearSVC(random_state=42, max_iter=2000).fit(X_train_char, y_train)
    char_svc_cal = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=5).fit(X_train_char, y_train)

    # -------------------------------------------------------------------------
    # 7. Train Feature Space 3: Combined Word + Character TF-IDF
    # -------------------------------------------------------------------------
    print("[STEP 7] Training Feature Space 3: Combined Word + Character TF-IDF...")
    comb_vec = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)),
    ])
    X_train_comb = comb_vec.fit_transform(X_train)
    X_test_comb = comb_vec.transform(X_test)

    comb_nb = MultinomialNB().fit(X_train_comb, y_train)
    comb_lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_comb, y_train)
    comb_svc_raw = LinearSVC(random_state=42, max_iter=2000).fit(X_train_comb, y_train)
    comb_svc_cal = CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000), cv=5).fit(X_train_comb, y_train)

    # -------------------------------------------------------------------------
    # 8. Feature Space Comparative Evaluation (Word vs Char vs Combined)
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Benchmarking Feature Spaces on Held-Out Test Set (1,035 samples)...")
    comp_records = []

    eval_pairs = [
        ("Word TF-IDF", word_vec, X_test_word, word_nb, word_lr, word_svc_raw),
        ("Char TF-IDF (2,5)", char_vec, X_test_char, char_nb, char_lr, char_svc_raw),
        ("Combined Word+Char", comb_vec, X_test_comb, comb_nb, comb_lr, comb_svc_raw),
    ]

    y_test_arr = y_test.to_numpy()

    for fs_name, vec, X_eval, m_nb, m_lr, m_svc in eval_pairs:
        for m_name, clf in [("Multinomial Naive Bayes", m_nb), ("Logistic Regression", m_lr), ("Linear SVM", m_svc)]:
            y_pred = clf.predict(X_eval)
            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(X_eval)[:, 1]
            else:
                scores = clf.decision_function(X_eval)

            acc = float(accuracy_score(y_test_arr, y_pred))
            prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
            rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
            f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))
            auc_val = float(roc_auc_score(y_test_arr, scores))

            comp_records.append({
                "Feature Space": fs_name,
                "Classifier": m_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc_val,
            })

    comparison_df = pd.DataFrame(comp_records)
    comparison_csv_path = metrics_dir / "char_word_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    plot_char_word_comparison_bar(comparison_df, metrics_dir / "char_word_comparison_bar.png")
    print(f"         Saved feature space comparison to: {comparison_csv_path.resolve()}")
    print(comparison_df[["Feature Space", "Classifier", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 9. Main Model Evaluation & Baseline Alignment
    # -------------------------------------------------------------------------
    y_pred_nb = word_nb.predict(X_test_word)
    y_scores_nb = word_nb.predict_proba(X_test_word)[:, 1]

    y_pred_lr = word_lr.predict(X_test_word)
    y_scores_lr = word_lr.predict_proba(X_test_word)[:, 1]

    y_pred_svc = word_svc_raw.predict(X_test_word)
    y_scores_svc_dec = word_svc_raw.decision_function(X_test_word)

    metrics_nb = compute_metrics(y_test_arr, y_pred_nb, y_scores=y_scores_nb, model_name="Multinomial Naive Bayes")
    metrics_lr = compute_metrics(y_test_arr, y_pred_lr, y_scores=y_scores_lr, model_name="Logistic Regression")
    metrics_svc = compute_metrics(y_test_arr, y_pred_svc, y_scores=y_scores_svc_dec, model_name="Linear SVM")

    eval_df = pd.DataFrame([
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy": metrics_nb["accuracy"], "Precision": metrics_nb["precision"],
            "Recall": metrics_nb["recall"], "F1-Score": metrics_nb["f1_score"],
            "ROC-AUC": metrics_nb["roc_auc"], "TN": metrics_nb["true_negatives"],
            "FP": metrics_nb["false_positives"], "FN": metrics_nb["false_negatives"],
            "TP": metrics_nb["true_positives"],
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": metrics_lr["accuracy"], "Precision": metrics_lr["precision"],
            "Recall": metrics_lr["recall"], "F1-Score": metrics_lr["f1_score"],
            "ROC-AUC": metrics_lr["roc_auc"], "TN": metrics_lr["true_negatives"],
            "FP": metrics_lr["false_positives"], "FN": metrics_lr["false_negatives"],
            "TP": metrics_lr["true_positives"],
        },
        {
            "Model": "Linear SVM",
            "Accuracy": metrics_svc["accuracy"], "Precision": metrics_svc["precision"],
            "Recall": metrics_svc["recall"], "F1-Score": metrics_svc["f1_score"],
            "ROC-AUC": metrics_svc["roc_auc"], "TN": metrics_svc["true_negatives"],
            "FP": metrics_svc["false_positives"], "FN": metrics_svc["false_negatives"],
            "TP": metrics_svc["true_positives"],
        },
    ])
    eval_df.to_csv(metrics_dir / "evaluation_metrics.csv", index=False)

    # Confusion matrix plots
    plot_confusion_matrix(metrics_nb["confusion_matrix"], "Multinomial Naive Bayes", metrics_dir / "confusion_matrix_nb.png", "Blues")
    plot_confusion_matrix(metrics_lr["confusion_matrix"], "Logistic Regression", metrics_dir / "confusion_matrix_lr.png", "Greens")
    plot_confusion_matrix(metrics_svc["confusion_matrix"], "Linear SVM", metrics_dir / "confusion_matrix_svc.png", "Purples")
    plot_model_comparison_bar(eval_df, metrics_dir / "model_comparison_bar.png")

    # ROC & PR curves
    fpr_nb, tpr_nb, _ = roc_curve(y_test_arr, y_scores_nb)
    fpr_lr, tpr_lr, _ = roc_curve(y_test_arr, y_scores_lr)
    fpr_svc, tpr_svc, _ = roc_curve(y_test_arr, y_scores_svc_dec)
    plot_roc_curves({
        "Multinomial Naive Bayes": (fpr_nb, tpr_nb, metrics_nb["roc_auc"]),
        "Logistic Regression": (fpr_lr, tpr_lr, metrics_lr["roc_auc"]),
        "Linear SVM": (fpr_svc, tpr_svc, metrics_svc["roc_auc"]),
    }, metrics_dir / "roc_curves.png")

    p_nb, r_nb, _ = precision_recall_curve(y_test_arr, y_scores_nb)
    p_lr, r_lr, _ = precision_recall_curve(y_test_arr, y_scores_lr)
    p_svc, r_svc, _ = precision_recall_curve(y_test_arr, y_scores_svc_dec)
    plot_precision_recall_curves({
        "Multinomial Naive Bayes": (p_nb, r_nb, average_precision_score(y_test_arr, y_scores_nb)),
        "Logistic Regression": (p_lr, r_lr, average_precision_score(y_test_arr, y_scores_lr)),
        "Linear SVM": (p_svc, r_svc, average_precision_score(y_test_arr, y_scores_svc_dec)),
    }, metrics_dir / "pr_curves.png")

    # -------------------------------------------------------------------------
    # 10. Adversarial Robustness Benchmarking
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Executing Controlled Adversarial Robustness Benchmarks...")
    word_p_dict = {
        "vectorizer": word_vec,
        "Multinomial Naive Bayes": word_nb,
        "Logistic Regression": word_lr,
        "Linear SVM": word_svc_raw,
    }
    char_p_dict = {
        "vectorizer": char_vec,
        "Multinomial Naive Bayes": char_nb,
        "Logistic Regression": char_lr,
        "Linear SVM": char_svc_raw,
    }
    comb_p_dict = {
        "vectorizer": comb_vec,
        "Multinomial Naive Bayes": comb_nb,
        "Logistic Regression": comb_lr,
        "Linear SVM": comb_svc_raw,
    }

    adv_df = evaluate_adversarial_robustness(
        test_df=test_df,
        word_pipeline_dict=word_p_dict,
        char_pipeline_dict=char_p_dict,
        comb_pipeline_dict=comb_p_dict,
        normalize_fn=normalize_sms,
        attack_type="Composite Multi-Vector Attack",
        seed=42
    )

    adv_csv_path = metrics_dir / "adversarial_evaluation.csv"
    adv_df.to_csv(adv_csv_path, index=False)
    plot_adversarial_robustness_bar(adv_df, metrics_dir / "adversarial_robustness_bar.png")
    print(f"         Saved adversarial evaluation to: {adv_csv_path.resolve()}")

    # -------------------------------------------------------------------------
    # 11. Save All Model Artifacts
    # -------------------------------------------------------------------------
    print("\n[STEP 10] Persisting all model artifacts...")
    # Word models
    joblib.dump(word_vec, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(word_nb, models_dir / "multinomial_nb.joblib")
    joblib.dump(word_lr, models_dir / "logistic_regression.joblib")
    joblib.dump(word_svc_raw, models_dir / "linear_svc.joblib")
    joblib.dump(word_svc_cal, models_dir / "calibrated_linear_svc.joblib")

    # Character models
    joblib.dump(char_vec, models_dir / "char_tfidf_vectorizer.joblib")
    joblib.dump(char_nb, models_dir / "char_nb.joblib")
    joblib.dump(char_lr, models_dir / "char_lr.joblib")
    joblib.dump(char_svc_raw, models_dir / "char_svc.joblib")
    joblib.dump(char_svc_cal, models_dir / "char_svc_cal.joblib")

    # Combined models
    joblib.dump(comb_vec, models_dir / "combined_tfidf_vectorizer.joblib")
    joblib.dump(comb_nb, models_dir / "comb_nb.joblib")
    joblib.dump(comb_lr, models_dir / "comb_lr.joblib")
    joblib.dump(comb_svc_raw, models_dir / "comb_svc.joblib")

    # Sample dataset
    generate_sample_bulk_csv(PROJECT_ROOT / "data" / "sample_bulk_sms.csv")

    print("=" * 70)
    print(" SMS SHIELD 3.0: TRAINING & ADVERSARIAL BENCHMARK COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
