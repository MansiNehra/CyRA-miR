"""
Common imports used throughout the project.

This module centralizes all third-party and standard library imports
to maintain consistency and simplify dependency management across
the project.
"""
# Standard Library
import os
import random
import warnings

from collections import Counter
from itertools import (combinations, chain)

# Numerical Computing
import numpy as np
import pandas as pd

# Visualization
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

# Scientific Computing
from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
)

from scipy.spatial.distance import squareform

# Scikit-Learn Pipeline
from sklearn.pipeline import Pipeline

# Model Selection
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    SelectFromModel,
)

# SkLearn Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Gradient Boosting
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Imbalanced Learning
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)

# Metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Model Saving
import joblib

# Explainability
import shap