"""
Hyperparameter search spaces for all models.

This module defines the hyperparameter distributions used for
RandomizedSearchCV during nested cross-validation.
"""
# Hyperparameter Search Spaces
from imports import *
from config import *
from pipelines import (
    pipe_lr,
    pipe_en_lr,
    pipe_brf,
    pipe_lgbm,
    pipe_cat,
    pipe_eec,
)
# Common Feature Selection Parameters
COMMON_PARAMS = {
    "select__k":[5, 10, 15, 20, 25],
    "lasso__estimator__C":[0.1]
}

param_lr = {
    **COMMON_PARAMS,
    "clf__C":np.logspace(-3, 2, 15)
}

param_en_lr = {
    **COMMON_PARAMS,
    "clf__C":np.logspace(-3, 2, 15),
    "clf__l1_ratio":np.linspace(0.1, 0.9, 9)
}

param_brf = {
    **COMMON_PARAMS,
    "clf__n_estimators":[200, 300, 500],
    "clf__max_depth":[2, 3, 4, 5],
    "clf__min_samples_split":[2, 5, 8],
    "clf__min_samples_leaf":[1, 3, 5],
    "clf__max_features":["sqrt", 0.3, 0.5]
}

param_eec = {
    **COMMON_PARAMS,
    'clf__n_estimators': [8, 12, 16, 20],      
    'clf__replacement': [False],               
    'clf__sampling_strategy': ['auto'],       
    'clf__estimator__max_depth': [1, 2, 3, 4],  
    'clf__estimator__min_samples_leaf': [1, 2, 3],
    'clf__estimator__min_samples_split': [2, 4, 6],
    'clf__estimator__criterion': ['gini', 'entropy'],
} 

param_lgbm = {
    **COMMON_PARAMS,
    "clf__num_leaves": [3, 5, 7, 15],
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.005, 0.01, 0.05],
    "clf__n_estimators": [50, 100, 200],
    "clf__subsample": [0.7, 0.9, 1],
    "clf__colsample_bytree": [0.6, 0.8, 1],
    "clf__min_child_samples": [5, 10, 20]
}

param_cat = {
    **COMMON_PARAMS,
    "clf__depth": [3, 4, 5],
    "clf__learning_rate": [0.01, 0.03, 0.1],
    "clf__iterations": [100, 200, 400],
    "clf__l2_leaf_reg": [1, 3, 5, 7],
    "clf__border_count": [32, 64, 128]
}

def build_search(
    pipeline,
    params,
    n_iter
):
    
    """
    Create a RandomizedSearchCV object for a given pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Machine learning pipeline.

    params : dict
        Hyperparameter search space.

    n_iter : int
        Number of parameter combinations to evaluate.

    Returns
    -------
    RandomizedSearchCV
        Configured randomized hyperparameter search object.
    """

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=n_iter,
        scoring=SCORING_METRIC,
        cv=INNER_CV,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

# Model Search Objects
searches = [
    ("CAT", build_search(pipe_cat, param_cat, 35)),
    ("EN-LR", build_search(pipe_en_lr, param_en_lr, 60)),
    ("LR", build_search(pipe_lr, param_lr, 80)),
    ("LGBM", build_search(pipe_lgbm, param_lgbm, 60)),
    ("BRF", build_search(pipe_brf, param_brf, 40)),
    ("EEC", build_search(pipe_eec, param_eec, 50))
]

__all__ = [
    "COMMON_PARAMS",
    "param_lr",
    "param_en_lr",
    "param_brf",
    "param_lgbm",
    "param_cat",
    "param_eec",
    "build_search",
    "searches"
]