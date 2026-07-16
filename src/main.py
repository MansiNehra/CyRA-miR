# IMPORTS
from imports import *
from config import *
from data_loader import load_data
from hyperparameters import searches
from nested_cv import run_nested_cv
from holdout import run_holdout
from save_load import (
    build_model_package,
    save_complete_model,
)
from shap_analysis import *
from visualization import *

# MAIN
def main():

    """
    Execute the complete miRNA drug-response prediction workflow.

    Workflow
    --------
    1. Load dataset.
    2. Perform nested cross-validation.
    3. Train final models on the complete training set.
    4. Evaluate each model on the independent holdout set.
    5. Save trained models.
    6. Perform SHAP analysis.
    7. Generate all performance and visualizations.
    """

    print("=" * 70)
    print("miRNA DRUG RESPONSE PREDICTION")
    print("=" * 70)

    # LOAD DATA
    print("\nLoading dataset...\n")
    (
        X,
        y,
        X_train,
        X_hold,
        y_train,
        y_hold,
        sample_ids,
    ) = load_data(DATA_PATH)

    # NESTED CROSS VALIDATION
    print("\nRunning Nested Cross Validation...\n")
    global_storage = run_nested_cv(
        searches=searches,
        X_train=X_train,
        y_train=y_train,
    )

    # HOLDOUT STORAGE
    holdout_results = {}
    
    # HOLDOUT EVALUATION
    print("\nRunning Holdout Evaluation...\n")

    # Evaluate Each Candidate Model
    for model_name, search in searches:
        print("\n" + "=" * 70)
        print(f"FINAL MODEL : {model_name}")
        print("=" * 70)
        model_storage = (
            global_storage[
                "results_all_models"
            ][model_name][
                "model_storage"
            ]
        )

        threshold = np.median(
            model_storage["thresholds"]
        )

        print(
            f"Median Threshold : {threshold:.4f}"
        )

        holdout_result = run_holdout(
            search=search,
            X_train=X_train,
            y_train=y_train,
            X_hold=X_hold,
            y_hold=y_hold,
            threshold=threshold,
        )

        # Build Model Package
        package = build_model_package(
            final_model=holdout_result[
                "final_model"
            ],

            X_train=holdout_result[
                "X_train_final"
            ],

            X_hold=holdout_result[
                "X_hold_final"
            ],

            y_hold=y_hold,

            y_probability=holdout_result[
                "y_probability"
            ],

            threshold=holdout_result[
                "threshold"
            ],

            metrics=holdout_result[
                "metrics"
            ],

            representative_features=
            holdout_result[
                "representative_features"
            ],

            model_name=model_name,
        )

        # Save Final Model
        save_complete_model(
            package=package,
            save_directory=MODEL_SAVE_DIR,
            model_name=model_name,
        )

        # Run SHAP Analysis
        run_shap_analysis(
            package=package,
            save_directory=SHAP_SAVE_DIR,
        )

        # Store Holdout Results
        holdout_results[model_name] = {
            "holdout": holdout_result,
            "package": package,
        }

    # VISUALIZATION
    print("\nGenerating Figures...\n")
    plot_raincloud(
        plot_data=global_storage["plot_data"],
        plot_data_pr=global_storage["plot_data_pr"],
        save_path="Nested_ROC_PR_curves"
    )

    plot_oof_curves(
        all_model_curves=global_storage["all_model_curves"],
        y_train=y_train,
        colors=COLORS,
        save_path="OOF_ROC_PR_curves",
    )

    plot_holdout_curves(
        holdout_curves={
            model: result["holdout"]["curves"]
            for model, result in holdout_results.items()
        },
        y_hold=y_hold,
        colors=COLORS,
    )

    plot_feature_frequency_heatmap(
        model_feature_freq=global_storage["model_feature_freq"],
        save_path="Models_Frequency_Heatmap",
    )

    plot_stability_heatmap(
        stability_tables=global_storage["stability_tables"],
        best_model="EEC",
        save_path="EEC_Stability_Heatmap",
    )

    # FINISHED
    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)

# ENTRY POINT

if __name__ == "__main__":
    main()