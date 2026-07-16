"""
Holdout evaluation utilities.

This module trains the final model on the complete training set,
evaluates it on the independent holdout set, computes evaluation
metrics and ROC/PR curves, and prepares the final model package
for downstream explainability analysis.
"""

# IMPORTS
from imports import *
from config import *
from metrics import *
from shap_analysis import *
from feature_selection import (
    correlation_clustering_feature_selection,
)

# TRAIN FINAL MODEL
def train_final_model(
    search,
    X_train,
    y_train,
):
    
    """
    Train the final optimized pipeline on the complete training dataset.

    Parameters
    ----------
    search : RandomizedSearchCV
        Hyperparameter search object.

    X_train : pandas.DataFrame
        Training feature matrix.

    y_train : pandas.Series
        Training labels.

    Returns
    -------
    tuple
        Trained pipeline, representative features, and
        correlation-filtered training data.
    """

    representative_features = (
        correlation_clustering_feature_selection(
            X_train,
            correlation_threshold=CORRELATION_THRESHOLD,
        )

    )

    X_train_corr = X_train[
        representative_features
    ]

    search.fit(
        X_train_corr,
        y_train,
    )

    final_model = search.best_estimator_

    return (
        final_model,
        representative_features,
        X_train_corr,
    )

# HOLDOUT PREDICTION
def predict_holdout(
    final_model,
    X_hold,
    representative_features,
):
    
    """
    Generate prediction probabilities for the independent holdout set.

    Parameters
    ----------
    final_model : sklearn.pipeline.Pipeline
        Trained pipeline.

    X_hold : pandas.DataFrame
        Holdout feature matrix.

    representative_features : list
        Features retained after correlation filtering.

    Returns
    -------
    tuple
        Predicted probabilities and correlation-filtered holdout data.
    """

    X_hold_corr = X_hold[
        representative_features
    ]

    y_probability = (
        final_model
        .predict_proba(
            X_hold_corr
        )[:, 1]
    )

    return (
        y_probability,
        X_hold_corr,
    )

# APPLY THRESHOLD
def apply_threshold(
    y_probability,
    threshold,
):
    
    """
    Convert predicted probabilities into binary class labels using
    the selected decision threshold.
    """

    y_prediction = (
        y_probability >= threshold).astype(int)

    return y_prediction

# HOLDOUT METRICS
def evaluate_holdout(
    y_true,
    y_prediction,
    y_probability,
):
    
    """
    Compute classification performance metrics on the holdout set.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    metrics = evaluate_predictions(
        y_true=y_true,
        y_prediction=y_prediction,
        y_probability=y_probability,
    )

    return metrics


# HOLDOUT CURVES
def compute_holdout_curves(
    y_true,
    y_probability,
):
    
    """
    Compute ROC and Precision-Recall curve data for the holdout set.

    Returns
    -------
    dict
        ROC curve, PR curve, and corresponding AUC values.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability,
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_probability,
        )
    )

    curves = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }

    return curves

# HOLDOUT STORAGE
def create_holdout_result(
    final_model,
    metrics,
    curves,
    threshold,
    y_probability,
    y_prediction,
    X_train_final,
    X_hold_final,
    representative_features,
):
    
    """
    Create a dictionary containing all holdout evaluation results.

    Returns
    -------
    dict
        Complete holdout evaluation results.
    """

    result = {
        "final_model": final_model,
        "metrics": metrics,
        "curves": curves,
        "threshold": threshold,
        "y_probability": y_probability,
        "y_prediction": y_prediction,
        "X_train_final": X_train_final,
        "X_hold_final": X_hold_final,
        "representative_features": representative_features,
    }

    return result

# RUN HOLDOUT EVALUATION
def run_holdout(
    search,
    X_train,
    y_train,
    X_hold,
    y_hold,
    threshold,
):
    
    """
    Execute the complete holdout evaluation workflow.

    This function trains the final model, generates predictions on the
    independent holdout dataset, computes evaluation metrics, prepares
    the final selected features, and stores all outputs required for
    downstream analyses.

    Returns
    -------
    dict
        Complete holdout evaluation results.
    """

    final_model, representative_features, X_train_corr = train_final_model(
        search=search,
        X_train=X_train,
        y_train=y_train,
    )

    y_probability, X_hold_corr = predict_holdout(
        final_model=final_model,
        X_hold=X_hold,
        representative_features=representative_features,
    )

    y_prediction = apply_threshold(
        y_probability=y_probability,
        threshold=threshold,
    )

    metrics = evaluate_holdout(
        y_true=y_hold,
        y_prediction=y_prediction,
        y_probability=y_probability,
    )

    curves = compute_holdout_curves(
        y_true=y_hold,
        y_probability=y_probability,
    )

    X_train_selected = prepare_tree_input(
        final_model,
        X_train_corr,
    )

    X_hold_selected = prepare_tree_input(        
        final_model,
        X_hold_corr,
    )

    holdout_result = create_holdout_result(
        final_model=final_model,
        metrics=metrics,
        curves=curves,
        threshold=threshold,
        y_probability=y_probability,
        y_prediction=y_prediction,
        X_train_final=X_train_selected,
        X_hold_final=X_hold_selected,
        representative_features=representative_features,
    )
    
    return holdout_result