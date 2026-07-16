"""
Evaluation metric utilities.

This module contains helper functions for:
- threshold optimization
- performance evaluation
- ROC/PR curve computation
- metric summarization
"""

# IMPORTS
from imports import * 

# Threshold Optimization
def compute_youden_threshold(
    y_true,
    y_probability,
):

    """
    Compute the optimal probability threshold using Youden's Index.
    
    Returns
    -------
    float
        Optimal classification threshold.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_probability
    )

    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)

    return thresholds[best_index]

# Cross-Validation Metrics
def compute_fold_metrics(
    y_true,
    y_probability,
):
    
    """
    Compute fold-level ROC-AUC and PR-AUC scores.
    
    Returns
    -------
    dict
        Dictionary containing fold performance metrics.
    """
    
    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            y_probability
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability
        )
    }

    return metrics

# Holdout Evaluation
def evaluate_predictions(
    y_true,
    y_prediction,
    y_probability
):
    
    """
    Evaluate holdout predictions using multiple classification metrics.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics,
        confusion matrix and predictions.
    """

    results = {
        "accuracy": accuracy_score(
            y_true,
            y_prediction
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_probability
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_prediction
        ),

        "precision": precision_score(
            y_true,
            y_prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_prediction,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_prediction
        ),

        "predictions": y_prediction
    }

    return results

# ROC / PR Curve Data
def compute_curve_data(
    y_true,
    y_probability
):
    
    """
    Compute ROC and Precision-Recall curve coordinates.

    Returns
    -------
    dict
        ROC and PR curve data.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability
    )

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_probability
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall
    }

# Metric Reporting
def summarize_metrics(
    metrics
):
    
    """
    Print evaluation metrics in a readable format.
    """

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

# Result Export
def metrics_to_dict(
    model_name,
    threshold,
    metrics
):
    
    """
    Convert evaluation metrics into a dictionary for reporting or export.
    """

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }

__all__ = [
    "compute_youden_threshold",
    "compute_fold_metrics",
    "evaluate_predictions",
    "compute_curve_data",
    "summarize_metrics",
    "metrics_to_dict",
]