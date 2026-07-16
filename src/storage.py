"""
Storage utilities for nested cross-validation.

This module provides helper functions to initialize
the dictionaries used throughout the machine learning
workflow.
"""
# IMPORTS
from imports import np

# Global Storage
def initialize_global_storage():

    """
    Initialize the project-level storage dictionary.

    Returns
    -------
    dict
        Global container used to store results from all evaluated models.
    """

    return {
        "plot_data": [],
        "plot_data_pr": [],
        "all_model_curves": {},
        "stability_tables": {},
        "stability_matrices": {},
        "model_feature_freq": {},
        "results_all_models": {},
        "stable_features_all_models": {},
        "kuncheva_scores": {}
    }

# Model Storage
def initialize_model_storage(n_samples):

    """
    Initialize storage for one machine learning model.

    Parameters
    ----------
    n_samples : int
        Number of training samples.

    Returns
    -------
    dict
        Model-specific storage dictionary.
    """
    
    return {
        "feature_importance": [],
        "feature_rank": [],
        "selected_features": [],
        "n_candidate_features": None,
        "roc_auc": [],
        "pr_auc": [],
        "thresholds": [],
        "oof_predictions": np.zeros(n_samples),
        "oof_probabilities": np.zeros(n_samples),
        "folds": []
    }

# Fold Storage
def initialize_fold_result():

    """
    Initialize storage for one outer cross-validation fold.

    Returns
    -------
    dict
        Fold-specific result dictionary.
    """

    return {
        "fold": None,
        "best_pipeline": None,
        "selected_features": None,
        "importance": None,
        "rank": None,
        "importance_df": None,
        "rank_df": None,
        "roc_auc": None,
        "pr_auc": None,
        "threshold": None,
        "y_pred": None,
        "y_probability": None,
        "y_true": None,
        "test_index": None,
    }