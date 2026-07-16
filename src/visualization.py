"""
Visualization Utilities

This module contains all visualization functions used 
in the project, including:

- ROC and Precision-Recall curves
- Feature frequency heatmaps
- Stability heatmaps
- Raincloud plots
- SHAP visualizations

All figures generated in the manuscript originate from
this module.
""" 

# VISUALIZATION
from imports import *
from config import *

# BUILD FEATURE FREQUENCY MATRIX
def build_feature_frequency_matrix(
    model_feature_freq,
):
    
    """
    Construct a feature-frequency matrix across all models.

    Parameters
    ----------
    model_feature_freq : dict
        Dictionary containing feature selection frequencies for
        each model.

    Returns
    -------
    pd.DataFrame
        Feature-frequency matrix with models as columns and
        features as rows.
    """

    # Collect feature frequencies from every model
    frequency_series = {}

    for model, freq in model_feature_freq.items():

        # If DataFrame
        if isinstance(
            freq,
            pd.DataFrame,
        ):

            frequency_series[model] = (
                freq
                .set_index("feature")["frequency"]
            )

        # If Dictionary
        elif isinstance(
            freq,
            dict,
        ):

            frequency_series[model] = pd.Series(
                freq,
            )

        else:
            raise TypeError(
                f"Unsupported type: {type(freq)}"
            )

    # Merge all models into a single matrix
    plot_df = pd.concat(
        frequency_series,
        axis=1,
    )

    plot_df = plot_df.fillna(
        0,
    )

    # Compute mean frequency to rank features
    plot_df["Mean"] = plot_df.mean(
        axis=1,
    )

    plot_df = plot_df.sort_values(
        "Mean",
        ascending=False,
    )

    # Remove helper column after sorting
    plot_df = plot_df.drop(
        columns="Mean",
    )

    return plot_df

# BUILD_ANNOTATION_MATRIX
def build_annotation_matrix(
    plot_df,
):
    
    """
    Create annotation labels for heatmap cells.

    Zero values are replaced with "-" while non-zero values
    are formatted to two decimal places.

    Parameters
    ----------
    plot_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    # Create formatted annotations for display
    annot_df = plot_df.copy().astype(str)

    for r in range(plot_df.shape[0]):
        
        for c in range(plot_df.shape[1]):
            value = plot_df.iloc[r, c]

            if value == 0:
                annot_df.iloc[r, c] = "-"

            else:

                annot_df.iloc[r, c] = (
                    f"{value:.2f}"
                )

    return annot_df

# ORDER MODELS BY ROC
def order_models_by_roc(
    all_model_curves,
):
    
    """
    Order models according to ROC-AUC performance.

    Parameters
    ----------
    all_model_curves : dict

    Returns
    -------
    list
        Model names sorted by descending ROC-AUC.
    """

    ordered_models = sorted(
        all_model_curves.keys(),
        key=lambda model:
        all_model_curves[
            model
        ][
            "roc_auc"
        ],
        reverse=True,
    )

    return ordered_models

# CREATE ROC PR FIGURE
def create_roc_pr_figure(
):
    
    """
    Create the ROC and Precision-Recall figure canvas.

    Returns
    -------
    tuple
        Figure object and two subplot axes.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    ax1, ax2 = axes

    return (
        fig,
        ax1,
        ax2,
    )

# FORMAT ROC PR AXES
def format_roc_pr_axes(
    axes,
):
    
    """
    Apply consistent formatting to ROC and PR axes.

    Parameters
    ----------
    axes : iterable
        Collection of matplotlib axes.
    """

    for ax in axes:

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.tick_params(
            labelsize=10
        )

# FEATURE FREQUENCY HEATMAP
def plot_feature_frequency_heatmap(
    model_feature_freq,
    target_feature=None,
    save_path=None,
):
    
    """
    Plot the feature selection frequency heatmap.

    Parameters
    ----------
    model_feature_freq : dict
        Feature frequency information for every model.

    target_feature : str
        Feature to highlight in the heatmap.

    save_path : str
        Output filename (without extension).
    """

    # Prepare heatmap data
    plot_df = build_feature_frequency_matrix(
        model_feature_freq
    )

    annot_df = build_annotation_matrix(
        plot_df
    )

    plt.figure(
        figsize=(8, 6)
    )

    ax = sns.heatmap(
        plot_df,
        cmap="Blues",
        annot=annot_df,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=False,
        vmin=0,
        vmax=0.8,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10,
        fontstyle="italic",
    )

    if target_feature is not None:

        for label in ax.get_yticklabels():
            if label.get_text() == target_feature:
                label.set_color(
                    "darkred"
                )

                label.set_fontweight(
                    "bold"
                )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        fontsize=10,
        fontweight="bold",
    )

    plt.xlabel(
        "Models",
        fontsize=11,
        fontweight="bold",
    )

    plt.ylabel(
        "miRNAs",
        fontsize=11,
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.gcf().patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            f"{save_path}.tiff",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

    plt.close()

# PREPARE STABILITY HEATMAP
def prepare_stability_heatmap(
    stability_tables,
    best_model,
):
 
    """
    Prepare the stability table for heatmap visualization.

    Parameters
    ----------
    stability_tables : dict
        Stability tables for all models.

    best_model : str
        Model selected for visualization.

    Returns
    -------
    pd.DataFrame
        Stability table formatted for plotting.
    """

    heatmap_df = stability_tables[
        best_model
    ].copy()

    heatmap_df = heatmap_df[
        [
            "feature",
            "frequency",
            "importance_mean",
            "rank_mean",
            "stability_score",
        ]
    ]

    heatmap_df = heatmap_df.sort_values(
        "stability_score",
        ascending=False,
    )

    heatmap_df = heatmap_df.set_index(
        "feature"
    )
   
    return heatmap_df

# NORMALIZE STABILITY TABLE
def normalize_stability_heatmap(
    heatmap_df,
):
    
    """
    Normalize stability metrics for color visualization.

    Parameters
    ----------
    heatmap_df : pd.DataFrame
        Stability summary table.

    Returns
    -------
    pd.DataFrame
        Normalized values used for heatmap colors.
    """

    freq_vis = heatmap_df[
        "frequency"
    ]

    imp_vis = (
        heatmap_df["importance_mean"]
        /
        heatmap_df["importance_mean"].max()
    )

    rank_vis = (
        heatmap_df["rank_mean"].max()
        -
        heatmap_df["rank_mean"]
    )

    rank_vis = (
        rank_vis
        /
        rank_vis.max()
    )

    stab_vis = heatmap_df[
        "stability_score"
    ]

    plot_df = pd.DataFrame(
        {
            "Selection\nFrequency": freq_vis,
            "Mean\nImportance": imp_vis,
            "Mean Rank\n(↓ better)": rank_vis,
            "Consensus\nScore": stab_vis,
        }
    )

    return plot_df

# BUILD ORIGINAL VALUE LABELS
def build_stability_labels(
    heatmap_df,
):
    
    """
    Generate row labels for the stability heatmap.

    Features exceeding the predefined stability threshold
    are marked with a star.

    Parameters
    ----------
    heatmap_df : pd.DataFrame
        Stability table.

    Returns
    -------
    list
        Feature labels for plotting.
    """

    labels = []

    for feature in heatmap_df.index:

        if (
            heatmap_df.loc[
                feature,
                "frequency"
            ]
            >= 0.8
        ):

            labels.append(
                "★ " + feature
            )

        else:
            labels.append(
                feature
            )

    return labels

# STABILITY HEATMAP
def plot_stability_heatmap(
    stability_tables,
    best_model,
    save_path=None,
):
    
    """
    Plot the feature stability heatmap.

    Parameters
    ----------
    stability_tables : dict
        Stability tables for all evaluated models.

    best_model : str
        Model whose stability results will be visualized.

    save_path : str, optional
        Output filename (without extension).

    Returns
    -------
    None
    """

    heatmap_df = prepare_stability_heatmap(
        stability_tables,
        best_model,
    )

    heatmap_df = heatmap_df.head(20)

    plot_df = normalize_stability_heatmap(
        heatmap_df,
    )

    plt.style.use(
        "default"
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    sns.heatmap(
        plot_df,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=1.5,
        linecolor="white",
        cbar=False,
        annot=False,
        square=False,
        ax=ax,
    )

    for i, feature in enumerate(
        heatmap_df.index
    ):

        values = [
            heatmap_df.loc[
                feature,
                "frequency"
            ],

            heatmap_df.loc[
                feature,
                "importance_mean"
            ],

            heatmap_df.loc[
                feature,
                "rank_mean"
            ],

            heatmap_df.loc[
                feature,
                "stability_score"
            ],
        ]

        texts = [
            f"{values[0]:.2f}",
            f"{values[1]:.3f}",
            f"{values[2]:.1f}",
            f"{values[3]:.3f}",
        ]

        color_values = [
            plot_df.iloc[i, 0],
            plot_df.iloc[i, 1],
            plot_df.iloc[i, 2],
            plot_df.iloc[i, 3],
        ]

        for j in range(4):
            txt_color = (
                "white"
                if color_values[j] > 0.55
                else "black"
            )

            ax.text(
                j + 0.5,
                i + 0.5,
                texts[j],
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
            )

    labels = build_stability_labels(
        heatmap_df
    )

    ax.set_yticks(
        np.arange(len(labels)) + 0.5
    )
    
    ax.set_yticklabels(
        labels,
        rotation=0,
        fontsize=10,
        fontstyle="italic",
    )

    for label in ax.get_yticklabels():

        if "★" in label.get_text():
            label.set_color(
                "darkred"
            )

            label.set_fontweight(
                "bold"
            )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("")

    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            f"{save_path}.tiff",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

    plt.close()

# OOF ROC PR CURVES
def plot_oof_curves(
    all_model_curves,
    y_train,
    colors,
    save_path=None,
):
    
    """
    Plot out-of-fold ROC and Precision-Recall curves for all trained models.

    Parameters
    ----------
    all_model_curves : dict
        Dictionary containing ROC and PR curve data for each model.
    y_train : pd.Series
        Training labels used to compute the PR baseline.
    colors : dict
        Color mapping for model visualization.
    save_path : str, optional
        Output filename (without extension).

    Returns
    -------
    None
    """

    ordered_models = order_models_by_roc(
        all_model_curves
    )

    fig, ax1, ax2 = create_roc_pr_figure()

    # PLOT ROC CURVES
    for model in ordered_models:

        curve = all_model_curves[
            model
        ]

        ax1.plot(
            curve["fpr"],
            curve["tpr"],
            lw=2.5,
            color=colors.get(
                model
            ),

            label=(
                f"{model} "
                f"({curve['roc_auc']:.3f})"
            ),
        )

    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color="lightgray",
        lw=1.5,
    )

    ax1.set_xlabel(
        "False Positive Rate",
        fontsize=12,
    )

    ax1.set_ylabel(
        "True Positive Rate",
        fontsize=12,
    )

    ax1.set_title(
        "A. OOF ROC Curves",
        fontsize=13,
        fontweight="bold",
    )

    # ======================================================
    # PLOT PRECISION-RECALL CURVES
    # ======================================================

    for model in ordered_models:
        curve = all_model_curves[
            model
        ]

        ax2.plot(
            curve["recall"],
            curve["precision"],
            lw=2.5,
            color=colors.get(
                model
            ),

            label=(
                f"{model} "
                f"({curve['pr_auc']:.3f})"
            ),
        )

    baseline = y_train.mean()

    ax2.axhline(
        baseline,
        linestyle="--",
        color="lightgray",
        lw=1.5,
    )

    ax2.set_xlabel(
        "Recall",
        fontsize=12,
    )

    ax2.set_ylabel(
        "Precision",
        fontsize=12,
    )

    ax2.set_title(
        "B. OOF PR Curves",
        fontsize=13,
        fontweight="bold",
    )

    # FORMAT FIGURE
    format_roc_pr_axes(
        [
            ax1,
            ax2,
        ]
    )

    ax1.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        title="ROC-AUC",
        title_fontsize=10,
    )

    ax2.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        title="PR-AUC",
        title_fontsize=10,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            f"{save_path}.tiff",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

    plt.close()

# PREPARE RAINCLOUD DATA
def prepare_raincloud_data(
    plot_data,
    plot_data_pr,
):
    
    """
    Prepare ROC-AUC and PR-AUC results for RainCloud visualization.

    Parameters
    ----------
    plot_data : list
        ROC-AUC results collected across folds.

    plot_data_pr : list
        PR-AUC results collected across folds.

    Returns
    -------
    tuple
        Combined ROC dataframe, PR dataframe, and model order.
    """

    rain_df = pd.concat(
        plot_data,
        ignore_index=True,
    )

    rain_df_pr = pd.concat(
        plot_data_pr,
        ignore_index=True,
    )

    model_order = (
        rain_df["model"]
        .drop_duplicates()
        .tolist()
    )

    return (
        rain_df,
        rain_df_pr,
        model_order,
    )

# CREATE RAINCLOUD FIGURE
def create_raincloud_figure():

    """
    Create the figure layout used for RainCloud plots.

    Returns
    -------
    tuple
        Figure and four matplotlib axes.
    """

    fig = plt.figure(
        figsize=(16, 10),
        dpi=SAVE_DPI,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.5, 1.3],
        height_ratios=[1, 1],
        wspace=0.15,
        hspace=0.35,
    )

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    ax3 = fig.add_subplot(
        gs[1, 0]
    )

    ax4 = fig.add_subplot(
        gs[1, 1]
    )

    return (
        fig,
        ax1,
        ax2,
        ax3,
        ax4,
    )

# COMPUTE MEAN SD TABLE
def compute_summary_table(
    dataframe,
    metric,
    model_order,
):
    
    """
    Compute mean and standard deviation of a performance metric.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Performance results.

    metric : str
        Metric to summarize.

    model_order : list
        Display order of models.

    Returns
    -------
    pd.DataFrame
        Summary statistics for each model.
    """

    summary = (
        dataframe
        .groupby(
            "model"
        )[metric]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reindex(
            model_order
        )
        .reset_index()
    )

    return summary

# ROC RAINCLOUD
def plot_roc_raincloud(
    ax,
    rain_df,
    model_order,
):
    
    """
    Plot RainCloud visualization for ROC-AUC scores.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    rain_df : pd.DataFrame

    model_order : list

    Returns
    -------
    None
    """

    pt.RainCloud(
        x="model",
        y="roc_auc",
        data=rain_df,
        order=model_order,
        palette="colorblind",
        bw=0.2,
        width_viol=0.7,
        width_box=0.25,
        move=0.2,
        pointplot=True,
        box_showfliers=False,
        alpha=0.85,
        ax=ax,
    )

    ax.set_title(
        "A. Model Performance Distribution (Nested CV ROC-AUC)",
        fontweight="bold",
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        "ROC-AUC"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    sns.despine(
        ax=ax
    )

    ax.grid(False)

# PR RAINCLOUD
def plot_pr_raincloud(
    ax,
    rain_df_pr,
    model_order,
):
    
    """
    Plot RainCloud visualization for PR-AUC scores.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    rain_df_pr : pd.DataFrame

    model_order : list

    Returns
    -------
    None
    """

    pt.RainCloud(
        x="model",
        y="pr_auc",
        data=rain_df_pr,
        order=model_order,
        palette="colorblind",
        bw=0.2,
        width_viol=0.7,
        width_box=0.25,
        move=0.2,
        pointplot=True,
        box_showfliers=False,
        alpha=0.85,
        ax=ax,
    )

    ax.set_title(
        "B. Model Performance Distribution (Nested CV PR-AUC)",
        fontweight="bold",
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        "PR-AUC"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    sns.despine(
        ax=ax
    )

    ax.grid(False)

# PLOT SUMMARY TABLE
def plot_summary_table(
    ax,
    summary,
):
    
    """
    Display a summary table containing the mean and standard deviation of
    model performance metrics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis used to display the table.

    summary : pd.DataFrame
        DataFrame containing model names with mean and standard deviation.

    Returns
    -------
    None
    """

    table_text = [
        [
            row["model"],
            f"{row['mean']:.3f} ± {row['std']:.3f}"
        ]

        for _, row in summary.iterrows()
    ]
    ax.axis(
        "off"
    )

    table = ax.table(
        cellText=table_text,
        colLabels=[
            "Model",
            "Mean ± SD",
        ],

        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)

    table.set_fontsize(9)

    table.scale(
        1.15,
        1.6,
    )

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(
            0.5
        )

        if r == 0:
            cell.set_facecolor(
                "#EFEFEF"
            )

            cell.set_text_props(
                weight="bold"
            )

# RAINCLOUD PLOT
def plot_raincloud(
    plot_data,
    plot_data_pr,
    save_path=None,
):
    
    """
    Generate RainCloud plots and summary tables for ROC-AUC and PR-AUC
    obtained during nested cross-validation.

    Parameters
    ----------
    plot_data : list
        ROC-AUC values collected across folds.

    plot_data_pr : list
        PR-AUC values collected across folds.

    save_path : str, optional
        Output filename (without extension).

    Returns
    -------
    None
    """

    rain_df, rain_df_pr, model_order = (
        prepare_raincloud_data(
            plot_data,
            plot_data_pr,
        )
    )

    fig, ax1, ax2, ax3, ax4 = (
        create_raincloud_figure()
    )

    # PLOT ROC-AUC RAINCLOUD
    plot_roc_raincloud(
        ax=ax1,
        rain_df=rain_df,
        model_order=model_order,
    )

    # PLOT PR-AUC RAINCLOUD
    plot_pr_raincloud(
        ax=ax3,
        rain_df_pr=rain_df_pr,
        model_order=model_order,
    )

    # ROC-AUC SUMMARY TABLE
    roc_summary = compute_summary_table(
        dataframe=rain_df,
        metric="roc_auc",
        model_order=model_order,
    )

    plot_summary_table(
        ax=ax2,
        summary=roc_summary,
    )

    # PR-AUC SUMMARY TABLE
    pr_summary = compute_summary_table(
        dataframe=rain_df_pr,
        metric="pr_auc",
        model_order=model_order,
    )

    plot_summary_table(
        ax=ax4,
        summary=pr_summary,
    )

    # SAVE FIGURE
    plt.tight_layout()

    if save_path is not None:
    
        plt.savefig(
            f"{save_path}.tiff",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

    plt.close()
   
# HOLDOUT ROC & PR CURVES
def plot_holdout_curves(
    holdout_curves,
    y_hold,
    colors,
):
    
    """
    Plot ROC and Precision-Recall curves on the independent holdout set.

    Parameters
    ----------
    holdout_curves : dict
        ROC and PR curve information for each trained model.

    y_hold : pd.Series
        True labels of the holdout dataset.

    colors : dict
        Color mapping for each model.

    Returns
    -------
    None
    """

    ordered_models = list(
        holdout_curves.keys()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5)
    )

    ax1, ax2 = axes

    # PLOT HOLDOUT ROC CURVES 
    for model in ordered_models:
        curve = holdout_curves[model]
        ax1.plot(
            curve["fpr"],
            curve["tpr"],
            lw=2.5,
            color=colors.get(model),
            label=f"{model} ({curve['roc_auc']:.3f})",
        )

    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color="lightgray",
        lw=1.5,
    )

    ax1.set_xlabel(
        "False Positive Rate"
    )

    ax1.set_ylabel(
        "True Positive Rate"
    )

    ax1.set_title(
        "A. Holdout ROC Curves",
        fontweight="bold",
    )

    ax1.legend(
        frameon=False,
        fontsize=12,
        title="ROC-AUC",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    # PLOT HOLDOUT PR CURVES
    for model in ordered_models:

        curve = holdout_curves[model]

        ax2.plot(
            curve["recall"],
            curve["precision"],
            lw=2.5,
            color=colors.get(model),
            label=f"{model} ({curve['pr_auc']:.3f})",
        )

    baseline = y_hold.mean()

    ax2.axhline(
        baseline,
        linestyle="--",
        color="lightgray",
        lw=1.5,
    )

    ax2.set_xlabel(
        "Recall"
    )

    ax2.set_ylabel(
        "Precision"
    )

    ax2.set_title(
        "B. Holdout PR Curves",
        fontweight="bold",
    )

    ax2.legend(
        frameon=False,
        fontsize=12,
        title="PR-AUC",
        loc="upper right",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    for ax in axes:
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(
        "Holdout_ROC_PR_curves.tiff",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close()

# RUN ALL VISUALIZATIONS
def run_all_visualizations(
    global_storage,
    holdout_curves,
    y_train,
    y_hold,
    colors,
):
    
    """
    Generate all figures used in the manuscript.

    Parameters
    ----------
    global_storage : dict
        Results generated during nested cross-validation.

    holdout_curves : dict
        Holdout ROC and PR curve information.

    y_train : pd.Series
        Training labels.

    y_hold : pd.Series
        Holdout labels.

    colors : dict
        Color mapping for all models.

    Returns
    -------
    None
    """

    plot_feature_frequency_heatmap(
        global_storage["model_feature_freq"]
    )

    plot_stability_heatmap(
        global_storage["stability_tables"],
        best_model="EEC",
    )

    plot_oof_curves(
        global_storage["all_model_curves"],
        y_train,
        colors,
    )

    plot_raincloud(
        global_storage["plot_data"],
        global_storage["plot_data_pr"],
    )

    plot_holdout_curves(
        holdout_curves,
        y_hold,
        colors,
    )

__all__ = [
    "plot_feature_frequency_heatmap",
    "plot_stability_heatmap",
    "plot_oof_curves",
    "plot_raincloud",
    "plot_holdout_curves",
    "run_all_visualizations",
]