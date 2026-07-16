"""
Utility functions for saving, loading, and packaging trained models.

This module centralizes helper functions used to serialize model
artifacts, create project directories, and package all information
required for holdout evaluation and SHAP analysis.
"""

# IMPORTS
from imports import *

# SAVE OBJECT
def save_object(
    obj,
    filepath,
):
    
    """
    Save a Python object to disk using Joblib.

    Parameters
    ----------
    obj : object
        Python object to serialize.

    filepath : str
        Destination file path.
    """

    joblib.dump(
        obj,
        filepath,
    )

# LOAD OBJECT
def load_object(
    filepath,
):
    
    """
    Load a serialized Python object from disk.

    Parameters
    ----------
    filepath : str
        Path to the serialized object.

    Returns
    -------
    object
        Loaded Python object.
    """

    obj = joblib.load(
        filepath,
    )

    return obj

# BUILD MODEL PACKAGE
def build_model_package(
    final_model,
    X_train,
    X_hold,
    y_hold,
    y_probability,
    threshold,
    metrics,
    representative_features,
    model_name,
):
    
    """
    Build a dictionary containing all information required to
    reproduce holdout evaluation and SHAP analysis.

    Parameters
    ----------
    final_model : sklearn.pipeline.Pipeline
        Trained final pipeline.

    X_train : pd.DataFrame
        Final processed training data.

    X_hold : pd.DataFrame
        Final processed holdout data.

    y_hold : pd.Series
        Holdout labels.

    y_probability : np.ndarray
        Predicted probabilities for the holdout set.

    threshold : float
        Classification threshold.

    metrics : dict
        Holdout performance metrics.

    representative_features : list
        Features retained after correlation filtering.

    model_name : str
        Name of the trained model.

    Returns
    -------
    dict
        Model package containing all required artifacts.
    """

    package = {
        "model_name": model_name,
        "final_model": final_model,
        "X_train_final": X_train,
        "X_hold_final": X_hold,
        "y_hold": y_hold,
        "y_proba": y_probability,
        "threshold": threshold,
        "metrics": metrics,
        "representative_features": representative_features,
    }

    return package

# CREATE DIRECTORY
def create_directory(
    directory,
):
    
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : str
        Directory path.
    """

    os.makedirs(
        directory,
        exist_ok=True,
    )

# BUILD MODEL FILEPATH
def build_model_filepath(
    save_directory,
    model_name,
):
    
    """
    Build the output filepath for a serialized model.

    Parameters
    ----------
    save_directory : str
        Directory where the model will be stored.

    model_name : str
        Model identifier.

    Returns
    -------
    str
        Full filepath of the model package.
    """

    return os.path.join(
        save_directory,
        f"{model_name}_complete.pkl",
    )

# SAVE COMPLETE MODEL
def save_complete_model(
    package,
    save_directory,
    model_name,
):
    
    """
    Save the complete trained model package.

    Parameters
    ----------
    package : dict
        Complete model package.

    save_directory : str
        Output directory.

    model_name : str
        Model identifier.

    Returns
    -------
    str
        Filepath of the saved model package.
    """

    create_directory(
        save_directory,
    )

    filepath = os.path.join(
        save_directory,
        f"{model_name}_complete.pkl",
    )

    save_object(
        obj=package,
        filepath=filepath
    )

    return filepath

__all__ = [
    "save_object",
    "load_object",
    "create_directory",
    "build_model_filepath",
    "build_model_package",
    "save_complete_model",
]
