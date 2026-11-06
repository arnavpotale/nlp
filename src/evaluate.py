"""
SMS Shield 2.0 - Advanced Model Evaluation & Visualization Suite
Author: SMS Shield NLP Project

This module provides comprehensive evaluation metrics, diagnostic curves,
and publication-grade visualization plots for:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (LinearSVC)

Capabilities:
- Metric computation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Reports, ROC-AUC).
- Confusion Matrix Heatmaps.
- Multi-Model Grouped Bar Charts.
- ROC Curves with Area Under the Curve (AUC).
- Precision-Recall Curves with Average Precision (AP).
- 5-Fold Stratified Cross-Validation reporting.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for binary SMS spam classification.
    Optionally computes ROC-AUC if probability / decision scores are provided.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 = Ham, 1 = Spam).
    y_pred : np.ndarray
        Predicted binary labels (0 = Ham, 1 = Spam).
    y_scores : np.ndarray, optional
        Predicted spam probabilities or continuous decision scores.
    model_name : str
        Display name of the model.

    Returns
    -------
    dict
        Dictionary containing metric values, counts, and classification report.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    roc_auc = None
    if y_scores is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            roc_auc = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
    cmap: str = "Blues"
) -> plt.Figure:
    """Generates an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6), dpi=300)

    total = np.sum(cm)
    group_counts = [f"{val:,}" for val in cm.flatten()]
    group_percentages = [f"{val / total:.2%}" for val in cm.flatten()]
    labels = [
        f"TN\n{group_counts[0]}\n({group_percentages[0]})",
        f"FP\n{group_counts[1]}\n({group_percentages[1]})",
        f"FN\n{group_counts[2]}\n({group_percentages[2]})",
        f"TP\n{group_counts[3]}\n({group_percentages[3]})",
    ]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        ax=ax,
        annot_kws={"size": 10.5, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10.5, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score
    across all evaluated models.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    df_melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    palette = ["#2563eb", "#059669", "#7c3aed"]  # Blue, Green, Purple
    barplot = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette[:len(metrics_df["Model"].unique())],
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    ax.set_ylim(0.0, 1.12)
    ax.set_title(
        "Comparative Model Performance: NB vs. LR vs. Linear SVM",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_ylabel("Metric Score", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Model", frameon=True, loc="upper right")

    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_roc_curves(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots Receiver Operating Characteristic (ROC) curves for multiple models.

    Parameters
    ----------
    model_curves : dict
        Mapping from model name to (fpr, tpr, roc_auc).
    save_path : Optional[Path]
        File path to save the generated figure.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

    colors = ["#2563eb", "#059669", "#7c3aed"]
    for idx, (name, (fpr, tpr, roc_auc)) in enumerate(model_curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            linewidth=2.2,
            color=color
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10.5, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
    model_pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plots Precision-Recall (PR) curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

    colors = ["#2563eb", "#059669", "#7c3aed"]
    for idx, (name, (precision, recall, avg_prec)) in enumerate(model_pr_curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {avg_prec:.4f})",
            linewidth=2.2,
            color=color
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=10.5, fontweight="bold")
    ax.set_title("Precision-Recall Curves (Focus on Spam Class)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Visualizes dataset class distribution (Ham vs Spam)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    counts = df[label_col].value_counts()
    colors = ["#2563eb", "#dc2626"]  # Blue for Ham, Red for Spam
    labels = [f"{idx.upper()} ({cnt:,})" for idx, cnt in zip(counts.index, counts.values)]

    sns.barplot(
        x=counts.index.str.upper(),
        y=counts.values,
        hue=counts.index.str.upper(),
        palette=colors,
        legend=False,
        ax=ax1,
        edgecolor="black"
    )
    ax1.set_title("Class Frequency", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Message Count", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Class", fontsize=10, fontweight="bold")
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 2),
            textcoords="offset points",
        )

    ax2.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops={"fontsize": 10, "weight": "bold"}
    )
    ax2.set_title("Class Percentage Ratio", fontsize=12, fontweight="bold")

    plt.suptitle("UCI SMS Spam Collection - Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
