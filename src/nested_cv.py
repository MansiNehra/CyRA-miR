"""
Nested cross-validation workflow.

This module performs:
- Outer cross-validation
- Correlation-based feature filtering
- Hyperparameter optimization
- Feature extraction
- Performance evaluation
- Stability analysis
"""
# IMPORTS
from imports import *
from config import *
from feature_selection import (
    correlation_clustering_feature_selection,
)
from metrics import *
from storage import *
from stability import *

def run_outer_fold(
    train_index,
    test_index,    
    X,
    y,
    search,):
    
    """
    Execute one outer cross-validation fold.
    
    Performs:
    - correlation-based feature filtering
    - hyperparameter optimization
    - returns the best fitted pipeline and fold-specific datasets.
    """
    
    X_train_fold = X.iloc[train_index].copy()    
    X_test_fold = X.iloc[test_index].copy()
    y_train_fold = y.iloc[train_index].copy()
    y_test_fold = y.iloc[test_index].copy()
    representative_features = (
        correlation_clustering_feature_selection(
            X_train_fold,
            correlation_threshold=CORRELATION_THRESHOLD,
        )
    )
    
    X_train_fold = X_train_fold[
        representative_features
    ]

    X_test_fold = X_test_fold[
        representative_features
    ]
    
    search.fit(
        X_train_fold,
        y_train_fold
    )

    best_pipeline = search.best_estimator_
    
    return {

        "best_pipeline": best_pipeline,
        "X_train_fold": X_train_fold,
        "X_test_fold": X_test_fold,
        "y_train_fold": y_train_fold,
        "y_test_fold": y_test_fold,
        "representative_features": representative_features,

    }

def extract_feature_information(    
    best_pipeline,
    X_train_fold):

    """
    Extract selected features, mutual information rankings,
    and LASSO feature importance from the optimized pipeline.
    """
    
    mi_step = best_pipeline.named_steps["select"]
    mi_idx = mi_step.get_support(indices=True)
    mi_features = X_train_fold.columns[mi_idx]
    lasso_step = best_pipeline.named_steps["lasso"]
    lasso_idx = lasso_step.get_support(indices=True)
    final_features = mi_features[lasso_idx]
    lasso_model = lasso_step.estimator_
    importance = np.abs(
        lasso_model.coef_
    ).flatten()
    mi_scores = mi_step.scores_
    
    rank_df = (
    
        pd.DataFrame({
            "feature": X_train_fold.columns,
            "mi_score": mi_scores,
        })
        .sort_values(
            "mi_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    
    rank_df["rank"] = rank_df.index + 1
    rank_df = rank_df[
        ["feature", "rank"]
    ]
    
    importance_df = pd.DataFrame({
        "feature": mi_features,
        "importance": importance,    
    })
    
    print("Final features:", len(final_features))

    return {
        "mi_features": mi_features,
        "final_features": final_features,
        "importance_df": importance_df,
        "rank_df": rank_df,
    }

# RUN SINGLE MODEL
def run_single_model(
    model_name,
    search,
    X_train,
    y_train):

    """
    Run complete nested cross-validation for a single model.
    
    Returns
    -------
    dict
        Model-specific storage containing
        fold results, selected features,
        metrics and out-of-fold predictions.
    """
    
    model_storage = initialize_model_storage(
        len(X_train)
    )
    
    for fold_number, (train_index, test_index) in enumerate(
        OUTER_CV.split(
            X_train,
            y_train,
        ),
        start=1,
    ):
    
        print("\n" + "=" * 60)
        print(f"MODEL: {model_name} | FOLD: {fold_number}")
        print("=" * 60)

        fold_result = run_outer_fold(
            train_index=train_index,
            test_index=test_index,
            X=X_train,
            y=y_train,
            search=search,
        )

        if model_storage["n_candidate_features"] is None:
            model_storage["n_candidate_features"] = len(
                fold_result["representative_features"]
           )

        feature_result = extract_feature_information(
            best_pipeline=fold_result["best_pipeline"],
            X_train_fold=fold_result["X_train_fold"],
        )
        
        model_storage["feature_importance"].append(
            feature_result["importance_df"]
        )

        model_storage["feature_rank"].append(
            feature_result["rank_df"]
        )

        model_storage["selected_features"].append(
            list(feature_result["final_features"])
        )

        print("Feature information stored." )

        best_pipeline = fold_result["best_pipeline"]
        X_test_fold = fold_result["X_test_fold"]
        y_test_fold = fold_result["y_test_fold"]
        y_proba = best_pipeline.predict_proba(
            X_test_fold
        )[:, 1]
        
        threshold = compute_youden_threshold(
            y_true=y_test_fold,
            y_probability=y_proba,
        )

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        metrics = compute_fold_metrics(
            y_true=y_test_fold,
            y_probability=y_proba,
        )

        model_storage["thresholds"].append(
            threshold
        )

        model_storage["roc_auc"].append(
            metrics["roc_auc"]
        )

        model_storage["pr_auc"].append(
            metrics["pr_auc"]
        )

        model_storage["oof_probabilities"][
            test_index] = y_proba
        
        model_storage["oof_predictions"][
            test_index] = y_pred
        
        fold_storage = initialize_fold_result()
        
        fold_storage["fold"] = fold_number
        
        fold_storage["best_pipeline"] = (
            fold_result["best_pipeline"]
        )

        fold_storage["selected_features"] = (
            list(feature_result["final_features"])
        )

        fold_storage["importance_df"] = (
            feature_result["importance_df"]
        )

        fold_storage["rank_df"] = (
            feature_result["rank_df"]
        )

        fold_storage["roc_auc"] = (
            metrics["roc_auc"]
        )

        fold_storage["pr_auc"] = (
            metrics["pr_auc"]
        )

        fold_storage["threshold"] = (
            threshold
        )

        fold_storage["y_pred"] = (
            y_pred
        )

        fold_storage["y_probability"] = (
            y_proba
        )

        fold_storage["y_true"] = (
            y_test_fold
        )
        
        fold_storage["test_index"] = (
            test_index
        )

        model_storage["folds"].append(
            fold_storage
        )

    return model_storage

def run_nested_cv(
    searches,
    X_train,
    y_train,
):

    """
    Run nested cross-validation for all candidate models.
    
    Computes:

    - out-of-fold predictions
    - ROC/PR metrics
    - feature stability
    - feature frequencies
    - performance summaries
    """

    global_storage = initialize_global_storage()

    for model_name, search in searches:

        print("\n" + "=" * 60)
        print(f"RUNNING MODEL: {model_name}")
        print("=" * 60)

        model_storage = run_single_model(
            model_name=model_name,
            search=search,
            X_train=X_train,
            y_train=y_train,
            )

        global_storage["plot_data"].append(
            pd.DataFrame({
                "model": model_name,
                "roc_auc": model_storage["roc_auc"],
            })
        )

        global_storage["plot_data_pr"].append(
            pd.DataFrame({
                "model": model_name,
                "pr_auc": model_storage["pr_auc"],
            })
        )

        y_probability = model_storage[
            "oof_probabilities"
        ]

        y_prediction = model_storage[
            "oof_predictions"
        ]

        overall_metrics = compute_fold_metrics(
            y_true=y_train,
            y_probability=y_probability,
        )

        curve_data = compute_curve_data(
            y_true=y_train,
            y_probability=y_probability,
        )

        global_storage["all_model_curves"][
            model_name
        ] = {
            **curve_data,
            "roc_auc": overall_metrics["roc_auc"],
            "pr_auc": overall_metrics["pr_auc"],
        }
    
        stability_results = compute_stability(
            model_storage
        )

        stability_matrix = compute_stability_matrix(
            model_storage["selected_features"]
        )

        feature_frequency = compute_feature_frequency(
            model_storage["selected_features"]
        )

        print(f"Completed: {model_name}")

        global_storage["stability_tables"][
            model_name
        ] = stability_results[
            "stability_table"
        ]

        global_storage["kuncheva_scores"][
            model_name
        ] = stability_results[
            "mean_kuncheva"
        ]

        global_storage["model_feature_freq"][
            model_name] = feature_frequency

        global_storage["stability_matrices"][
            model_name] = stability_matrix

        global_storage["results_all_models"][
            model_name
        ] = {
            "model_storage": model_storage,
            "overall_metrics": overall_metrics,
        }

        print(f"Mean ROC-AUC : {np.mean(model_storage['roc_auc']):.3f}")
        print(f"Mean PR-AUC  : {np.mean(model_storage['pr_auc']):.3f}")
        print(f"OOF ROC-AUC : {overall_metrics['roc_auc']:.3f}")
        print(f"OOF PR-AUC : {overall_metrics['pr_auc']:.3f}")

    return global_storage
        