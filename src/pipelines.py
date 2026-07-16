"""
Pipeline definitions for all machine learning models used in the study.

Each pipeline follows the same preprocessing workflow:
Correlation Filtering (outside pipeline)
→ Scaling (when required)
→ Mutual Information feature selection
→ LASSO feature selection
→ Classifier
"""
# IMPORTS
from imports import *
from config import *
from feature_selection import (
    mi_score,
    feature_selector
)

# Pipeline Construction
def build_pipeline(model, scale=False):

    """
    Build a standardized machine-learning pipeline.

    Parameters
    ----------
    model : estimator
        Final classifier.

    scale : bool
        Whether StandardScaler should be applied.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.extend([
        (
            "select",
            SelectKBest(
                score_func=mi_score
            )
        ),
        
        (
            "lasso",
            feature_selector
        ),

        (
            "clf",
            model
        )
    ])

    return Pipeline(steps)

# Logistic Regression
pipe_lr = build_pipeline(
    LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    scale=True
)

# Elastic Net
pipe_en_lr = build_pipeline(
    LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=10000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    scale=True
)

# Balanced Random Forest
pipe_brf = build_pipeline(
    BalancedRandomForestClassifier(
        random_state=RANDOM_STATE
    )
)

# LightGBM
pipe_lgbm = build_pipeline(
    LGBMClassifier(
        deterministic=True,
        force_col_wise=True,
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1
    )
)

# CatBoost
pipe_cat = build_pipeline(
    CatBoostClassifier(
        verbose=0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_state=RANDOM_STATE
    )
)

# Easy Ensemble Classifier
pipe_eec = build_pipeline(
    EasyEnsembleClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3
        ),
        n_estimators=8,
        random_state=RANDOM_STATE
    )
)

# Pipeline Registry
PIPELINES = {
    "LR": pipe_lr,
    "EN-LR": pipe_en_lr,
    "BRF": pipe_brf,
    "LGBM": pipe_lgbm,
    "CAT": pipe_cat,
    "EEC": pipe_eec
}

__all__ = [
    "build_pipeline",
    "pipe_lr",
    "pipe_en_lr",
    "pipe_brf",
    "pipe_lgbm",
    "pipe_cat",
    "pipe_eec",
    "PIPELINES"
]