"""
Project-wide configuration settings.

This module defines constants, random seed initialization,
cross-validation settings, file paths, and visualization
parameters used throughout the project.
"""

# PROJECT CONFIGURATION
from imports import *

# WARNING CONFIGURATION
warnings.filterwarnings("ignore")

# RANDOM SEED
SEED = 42
RANDOM_STATE = SEED

# REPRODUCIBILITY SETTINGS
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(SEED)
np.random.seed(SEED)

# PROJECT CONSTANTS
TEST_SIZE = 0.20
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
CORRELATION_THRESHOLD = 0.70
STABLE_FEATURE_THRESHOLD = 0.80
SCORING_METRIC = "average_precision"
SAVE_DPI = 600

# CROSS-VALIDATION CONFIGURATION
INNER_CV = StratifiedKFold(
    n_splits=N_INNER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

OUTER_CV = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# PROJECT PATHS
DATA_PATH = "data/cyclophosphamide_miRNA_dataset.csv"
RESULTS_DIR = "results"
MODEL_SAVE_DIR = "Saved_Models"
SHAP_SAVE_DIR = "SHAP"
FIGURE_SAVE_DIR = "Figures"

# VISUALIZATION
COLORS = {
    "EEC": "#004488",       # Easy Ensemble Classifier
    "BRF": "#D62728",       # Balanced Random Forest
    "CAT": "#2CA02C",       # CatBoost
    "LGBM": "#9467BD",      # LightGBM
    "LR": "#8C564B",        # Logistic Regression
    "EN-LR": "#7F7F7F"      # Elastic-Net Logistic Regression
}

__all__ = [
    "SEED",
    "RANDOM_STATE",
    "TEST_SIZE",
    "N_OUTER_FOLDS",
    "N_INNER_FOLDS",
    "CORRELATION_THRESHOLD",
    "STABLE_FEATURE_THRESHOLD",
    "SCORING_METRIC",
    "SAVE_DPI",
    "INNER_CV",
    "OUTER_CV",
    "DATA_PATH",
    "MODEL_SAVE_DIR",
    "SHAP_SAVE_DIR",
    "FIGURE_SAVE_DIR",
    "COLORS",
]