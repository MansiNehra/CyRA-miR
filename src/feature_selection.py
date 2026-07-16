"""
Feature selection utilities.

This module provides correlation-based feature filtering,
Mutual Information, and LASSO-based feature selection
for identifying informative miRNA biomarkers.
"""

# IMPORTS
from imports import *
from config import *

# Correlation Clustering
def correlation_clustering_feature_selection(
    X_train,
    correlation_threshold=CORRELATION_THRESHOLD
):
    
    corr_matrix = X_train.corr().abs()  
    
    distance_matrix = 1 - corr_matrix
       
    condensed_distance = squareform(
        distance_matrix.values
    )

    Z = linkage(
        condensed_distance,
        method="average"
    )

    cluster_labels = fcluster(
        Z,
        t=1 - correlation_threshold,
        criterion="distance"
    )

    feature_clusters = pd.DataFrame({   
        "feature": X_train.columns,
        "cluster": cluster_labels
    })

    representative_features = []

    for cluster_id in np.unique(cluster_labels):
        cluster_features = feature_clusters[
            feature_clusters["cluster"] == cluster_id
        ]["feature"].tolist()

        if len(cluster_features) == 1:
            representative_features.append(
                cluster_features[0]
            )

            continue
        
        variances = X_train[
            cluster_features
        ].var()

        representative = variances.idxmax()
        representative_features.append(
            representative
        )

    return representative_features

# Mutual Information
def mi_score(X, y):
    return mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE,
        n_neighbors=5
    )

# LASSO Feature Selector
feature_selector = SelectFromModel(
    LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
)

__all__ = [
    "correlation_clustering_feature_selection",
    "mi_score",
    "feature_selector"
]