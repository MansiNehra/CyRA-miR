"""
SHAP explainability utilities.

This module creates SHAP explainers, computes SHAP values,
generates summary visualizations, and stores explainability
artifacts for trained machine learning models.
"""

# IMPORTS
from imports import *
from config import *
from save_load import *

# CREATE SHAP EXPLAINER
def create_explainer(
    classifier,
    X_train,
):
    """
    Create an appropriate SHAP explainer for the trained classifier.

    Tree-based models use TreeExplainer, whereas all remaining
    classifiers use the permutation explainer.

    Returns
    -------
    tuple
        SHAP explainer and tree-model indicator.
    """
    
    is_tree_model = isinstance(
        classifier,
        (
            LGBMClassifier,
            CatBoostClassifier,
        ),
    )
    
    if is_tree_model:
        explainer = shap.TreeExplainer(
            classifier
        )

    else:
        explainer = shap.Explainer(
            classifier.predict_proba,
            X_train,
            algorithm="permutation",
        )
    
    return (
        explainer,
        is_tree_model,
    )

# PREPARE TREE INPUT
def prepare_tree_input(
    pipeline,
    X,
):
    
    """
    Apply all preprocessing steps before SHAP computation.

    The function sequentially applies scaling,
    mutual information feature selection,
    and LASSO feature selection so that the
    transformed dataset exactly matches the
    classifier input space.
    """

    X_transformed = X.copy()
    feature_names = list(
        X.columns
    )

    if "scaler" in pipeline.named_steps:
        X_transformed = (
            pipeline.named_steps["scaler"]
            .transform(X_transformed)
        )

    if "select" in pipeline.named_steps:
        selector = pipeline.named_steps["select"]
        support = selector.get_support()
        feature_names = list(
            np.array(feature_names)[support]
        )

        X_transformed = selector.transform(
            X_transformed
        )

    if "lasso" in pipeline.named_steps:
        selector = pipeline.named_steps["lasso"]
        support = selector.get_support()
        feature_names = list(
            np.array(feature_names)[support]
        )

        X_transformed = selector.transform(
            X_transformed
        )

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    return X_transformed

# COMPUTE SHAP VALUES
def compute_shap_values(
    explainer,
    X_hold,
    is_tree_model,
):
    
    """
    Compute SHAP values for the holdout dataset.

    Tree models use TreeExplainer whereas all
    remaining classifiers use the permutation
    explainer.

    Returns
    -------
    tuple
        SHAP values and plotting dataset.
    """

    # Tree Models
    if is_tree_model:
        X_plot = X_hold.copy()
        shap_values = explainer.shap_values(X_plot)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "values"):
            shap_values = shap_values.values
        return shap_values, X_plot

    # Linear / Other Models
    else:
        X_plot = X_hold
        explanation = explainer(
            X_plot,
        )

        shap_values = (
            explanation[:, :, 1]
            .values
        )

        return (
            shap_values,
            X_plot,
        )
    
# SAVE SHAP VALUES
def save_shap_values(
    shap_values,
    filepath,
):

    """
    Save computed SHAP values to disk.
    """

    save_object(
        obj=shap_values,
        filepath=filepath,
    )

# SHAP BEESWARM
def plot_shap_beeswarm(
    shap_values,
    X_hold,
    filepath,
):
    
    """
    Generate and save the SHAP beeswarm summary plot.
    """

    shap.summary_plot(
        shap_values,
        X_hold,
        show=False,
    )

    plt.savefig(
        filepath,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close()

# SHAP BAR PLOT
def plot_shap_bar(
    shap_values,
    X_hold,
    filepath,
):
    
    """
    Generate and save the SHAP feature importance bar plot.
    """

    shap.summary_plot(
        shap_values,
        X_hold,
        plot_type="bar",
        show=False,
    )

    plt.savefig(
        filepath,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close()

# RUN SHAP ANALYSIS
def run_shap_analysis(
    package,
    save_directory,
):
    
    """
    Execute the complete SHAP explainability workflow.

    This function creates the SHAP explainer,
    computes SHAP values,
    saves SHAP outputs,
    and generates summary plots.

    Returns
    -------
    dict
        SHAP explainer, SHAP values, and plotting dataset.
    """

    create_directory(
        save_directory,
    )

    pipeline = package["final_model"]
    classifier = pipeline.named_steps["clf"]
    
    X_train = package[
        "X_train_final"
    ]

    X_hold = package[
        "X_hold_final"
    ]

    model_name = package[
        "model_name"
    ]

    explainer, is_tree_model = create_explainer(
        classifier=classifier,
        X_train=X_train,
    )

    shap_values, X_plot= compute_shap_values(
        explainer=explainer,
        X_hold=X_hold,
        is_tree_model=is_tree_model,
    )

    save_shap_values(
        shap_values=shap_values,
        filepath=os.path.join(
            save_directory,
            f"{model_name}_SHAP_VALUES.pkl",
        ),
    )

    plot_shap_beeswarm(
        shap_values=shap_values,
        X_hold=X_plot,
        filepath=os.path.join(
            save_directory,
            f"{model_name}_Beeswarm.png",
        ),
    )

    plot_shap_bar(
        shap_values=shap_values,
        X_hold=X_plot,
        filepath=os.path.join(
            save_directory,
            f"{model_name}_Bar.png",
        ),
    )
    
    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "X_plot": X_plot,
    }

__all__ = [
    "create_explainer",
    "prepare_tree_input",
    "compute_shap_values",
    "save_shap_values",
    "plot_shap_beeswarm",
    "plot_shap_bar",
    "run_shap_analysis",
]