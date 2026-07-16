"""
Utilities for feature stability analysis.

This module computes:
- Feature selection frequency
- Mean feature importance
- Mean feature ranking
- Stability scores
- Kuncheva stability index
- Feature stability matrices
"""

# IMPORTS
from imports import *

# Feature Frequency
def compute_feature_frequency(selected_features):

    """Compute the feature-selection frequency across 
    cross-validation folds."""

    feature_counts = Counter(
        feature
        for fold in selected_features
        for feature in fold
    )

    total_folds = len(selected_features)

    frequency = {
        feature: count / total_folds
        for feature, count in feature_counts.items()
    }

    return frequency

# Feature Importance
def compute_mean_importance(
    importance_tables
):

    merged = pd.concat(
        importance_tables,
        axis=0
    )

    summary = (
        merged
        .groupby("feature")["importance"]
        .mean()
    )

    return summary

# Feature Rank
def compute_mean_rank(
    rank_tables
):

    merged = pd.concat(
        rank_tables,
        axis=0
    )

    summary = (
        merged
        .groupby("feature")["rank"]
        .mean()
    )

    return summary

# Stability Table
def build_stability_table(
    frequency,
    importance,
    rank
):
    
    """
    Combine feature frequency, importance and rank into a 
    unified stability table.
    """

    frequency_df = pd.DataFrame({
        "feature": list(frequency.keys()),
        "frequency": list(frequency.values()),
    })
    
    importance_df = (
        importance
        .rename("importance_mean")
        .reset_index()
    )

    rank_df = (
        rank
        .rename("rank_mean")
        .reset_index()
    )

    df = frequency_df.merge(
        importance_df,
        on="feature",
        how="left",
    )

    df = df.merge(
        rank_df,
        on="feature",
        how="left",
    )

    df = df.fillna(0)

    df["importance_mean_norm"] = (
        df["importance_mean"]/df["importance_mean"].max()
    )
    
    df["rank_score"] = 1 - (df["rank_mean"]/df["rank_mean"].max()
    )
    
    df["stability_score"] = (
        0.5 * df["frequency"]
        + 0.3 * df["importance_mean_norm"]
        + 0.2 * df["rank_score"]
    )
    
    df = df.sort_values(
        "stability_score",
        ascending=False
    )
    
    df = df.drop(
    
    columns=[
        "importance_mean_norm",
        "rank_score",
    ])
    
    return df

# Stability Summary
def compute_stability(
    model_storage,
):
    
    """
    Compute stability statistics for one model.

    Returns
    -------
    dict
        Stability table and mean Kuncheva Index.
    """

    frequency = compute_feature_frequency(
        model_storage["selected_features"]
    )

    importance = compute_mean_importance(
        model_storage["feature_importance"]
    )

    rank = compute_mean_rank(
        model_storage["feature_rank"]
    )

    stability_table = build_stability_table(
        frequency=frequency,
        importance=importance,
        rank=rank
    )

    total_features = model_storage["n_candidate_features"]

    mean_kuncheva = (
        compute_mean_kuncheva(
            selected_features=model_storage[
                "selected_features"
            ],
            total_features=total_features,
        )
    )
    
    return {
        "stability_table": stability_table,
        "mean_kuncheva": mean_kuncheva,
    }

# Stability Matrix
def compute_stability_matrix(
    selected_features,
):
    
    """
    Create a binary feature-selection matrix across folds.
    """

    all_features = sorted(
        set(
            chain.from_iterable(
                selected_features
            )
        )
    )
 
    stability_matrix = pd.DataFrame(
        0,
        index=all_features,
        columns=[
            f"Fold_{i+1}"
            for i in range(
                len(selected_features)
            )
        ],
    )

    for fold_number, features in enumerate(
        selected_features,
        start=1,
    ):
        
        for feature in features:
            stability_matrix.loc[
                feature,
                f"Fold_{fold_number}"
            ] = 1

    return stability_matrix

# Kuncheva Index
def compute_kuncheva_index(
    A,
    B,
    total_features,
):
    A = set(A)
    B = set(B)
    k = len(A)
    r = len(
        A & B
    )

    expected = (
        k ** 2/total_features
    )

    return (
        (r - expected)/(k - expected)
    )

def compute_mean_kuncheva(
    selected_features,
    total_features,
):
    
    """
    Compute the mean Kuncheva Index across all fold pairs.
    """

    kunchevas = []
    for A, B in combinations(
        selected_features,
        2,
    ):
        
        ki = compute_kuncheva_index(
            A,
            B,
            total_features,
        )
    
        kunchevas.append(ki)
    
    return np.mean(kunchevas)