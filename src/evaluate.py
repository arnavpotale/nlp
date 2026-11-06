"""
SMS Shield - Model Evaluation and Visualization Module
Author: SMS Shield NLP Project

This module provides functions to evaluate machine learning classification models,
compute key performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix),
and generate publication-quality visual plots using Matplotlib and Seaborn.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments & scripts
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
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for binary SMS spam classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 = Ham, 1 = Spam).
    y_pred : np.ndarray
        Predicted binary labels (0 = Ham, 1 = Spam).
    model_name : str
        Human-readable name of the evaluated model.

    Returns
    -------
    dict
        Dictionary containing Accuracy, Precision, Recall, F1-Score,
        Confusion Matrix, and detailed classification report dictionary.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Ham", "Spam"],
        output_dict=True,
        zero_division=0
    )

    # Detailed confusion matrix counts
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
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
    """
    Generates an annotated confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix array [[TN, FP], [FN, TP]].
    model_name : str
        Model title for the plot.
    save_path : Optional[Path]
        File path to save the generated figure.
    cmap : str
        Color map palette for the heatmap.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Annotations displaying both raw counts and percentage of total
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
        xticklabels=["Predicted Ham (0)", "Predicted Spam (1)"],
        yticklabels=["Actual Ham (0)", "Actual Spam (1)"],
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
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
    Creates a grouped bar chart comparing Accuracy, Precision, Recall, and F1-score
    across the two models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'].
    save_path : Optional[Path]
        File path to save the generated plot.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    # Melt dataframe into long format for seaborn barplot
    df_melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    palette = ["#1f77b4", "#ff7f0e"]
    barplot = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    ax.set_ylim(0.0, 1.10)
    ax.set_title(
        "Comparative Model Performance: Multinomial NB vs Logistic Regression",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model", frameon=True, loc="upper right")

    # Add numeric score annotations above bars
    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

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
    """
    Visualizes the dataset class distribution (Ham vs Spam).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    label_col : str
        Column name containing class labels ('ham' / 'spam').
    save_path : Optional[Path]
        File path to save the generated plot.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    counts = df[label_col].value_counts()
    colors = ["#2ca02c", "#d62728"]  # Green for Ham, Red for Spam
    labels = [f"{idx.upper()} ({cnt:,})" for idx, cnt in zip(counts.index, counts.values)]

    # Subplot 1: Bar Chart
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
    ax1.set_ylabel("Message Count", fontsize=10)
    ax1.set_xlabel("Class", fontsize=10)
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

    # Subplot 2: Donut Chart
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
