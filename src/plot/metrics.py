from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_prc_curve(
    prc_df: pd.DataFrame,
    no_skill: float,
    best_ix: int,
    model_name: str,
    save_path: Path = None,
) -> plt.Axes | None:
    """Plot and save the precision-recall curve.

    Parameters:
        prc_df (pd.DataFrame): DataFrame containing precision, recall, and thresholds.
        no_skill (float): No skill line value (proportion of positive samples).
        best_ix (int): Index of the best F1 score.
        model_name (str): Name of the model (used for saving the file).
        save_path (Path): Path to save the plot.
    Returns:
        plt.Axes | None: The Axes object of the plot if save_path is None
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")

    # plot curve
    ax.plot(prc_df["recall"], prc_df["precision"], marker=".", label=model_name.title())

    # plot best cut
    ax.scatter(
        prc_df["recall"][best_ix],
        prc_df["precision"][best_ix],
        marker="o",
        color="black",
        label="Best",
    )

    # axis labels
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{model_name}_prc_curve.png")
        plt.close(fig)
    else:
        return ax


def plot_roc_curve(
    roc_df: pd.DataFrame,
    model_name: str,
    save_path: Path = None,
) -> plt.Axes | None:
    """Plot and save the ROC curve.

    Parameters:
        roc_df (pd.DataFrame): DataFrame containing FPR, TPR, and thresholds
        model_name (str): Name of the model (used for saving the file).
        save_path (Path): Path to save the plot.

    Returns:
        plt.Axes | None: The Axes object of the plot if save_path is None
    """
    J = roc_df["tpr"] - roc_df["fpr"]
    ix = np.argmax(J)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    ax.plot(roc_df["fpr"], roc_df["tpr"], marker=".", label=model_name.title())
    ax.scatter(
        roc_df["fpr"][ix], roc_df["tpr"][ix], marker="o", color="black", label="Best"
    )
    # axis labels
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    # show the plot
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{model_name}_roc_curve.png")
        plt.close(fig)
    else:
        return ax


def plot_conf_mat(
    conf_mat: pd.DataFrame,
    model_name: str,
    save_path: Path = None,
) -> plt.Axes | None:
    """Plot and save the confusion matrix as a heatmap.

    Parameters:
        conf_mat (pd.DataFrame): Confusion matrix to plot.
        model_name (str): Name of the model (used for saving the file).
        save_path (Path): Path to save the plot.

    Returns:
        plt.Axes | None: The Axes object of the plot if save_path is None
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax = sns.heatmap(conf_mat.astype(int), annot=True, ax=ax, fmt="d")

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{model_name}_conf_mat.png")
        plt.close(fig)
    else:
        return ax


def plot_summary_roc_curve(
    feat_theme: str,
    fpr_tpr_dfs_all: dict[str, list[pd.DataFrame]],
    model_names_all: dict[str, list[str]],
    metrics_all: dict[str, pd.DataFrame],
    plot_all: bool = True,
    save_path: Path | None = None,
) -> plt.Axes:
    """Plot the average ROC curve for the given feature theme.

    Parameters:
        feat_theme (str): The feature theme to plot.
        fpr_tpr_dfs_all (dict): Dictionary of FPR/TPR Data
        model_names_all (dict): Dictionary of model names for each feature theme.
        metrics_all (dict): Dictionary of metrics DataFrames for each feature theme.
        plot_all (bool): Whether to plot all individual ROC curves. Default is True.
        save_path (Path | None): Path to save the plot. If None, the plot is returned.

    Returns:
        plt.Axes: The Axes object of the plot if save_path is None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    best_models = model_names_all[feat_theme]
    fpr_tpr_dfs = fpr_tpr_dfs_all[feat_theme]

    # get mean and std auc
    mean_auc = metrics_all[feat_theme]["roc_auc"].mean()
    std_auc = metrics_all[feat_theme]["roc_auc"].std()

    tprs = []
    colors = sns.color_palette("tab20", n_colors=len(best_models)).as_hex()
    mean_fpr = np.linspace(0, 1, 100)

    for i, (fpr_tpr, bm) in enumerate(zip(fpr_tpr_dfs, best_models)):
        fpr = fpr_tpr["fpr"]
        tpr = fpr_tpr["tpr"]

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)

        if plot_all:
            ax.plot(fpr, tpr, label=bm, color=colors[i], linewidth=0.3)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {round(mean_auc, 3)} $\pm$ {round(std_auc, 3)})",
        lw=2.5,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.plot([0, 1], [0, 1], linestyle="--")
    # axis labels
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC Curve Combination Features \n(Positive Label 'reg')",
    )
    ax.legend(prop={"size": 12})
    plt.margins(x=0.01, y=0.01, tight=True)

    if save_path is not None:
        fig.savefig(save_path / "average_roc_auc.png")
        plt.close(fig)
    else:
        return ax
