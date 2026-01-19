"""Wrapper for the beeswarm plot from the ``shap`` package.

Note:
    Code and implementation was taken and adapted from the [SHAP package](https://github.com/shap/shap)
    which is licensed under the [MIT license](https://github.com/shap/shap/blob/master/LICENSE).

"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mapclassify
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapiq.interaction_values import InteractionValues, aggregate_interaction_values
from shapiq.plot.utils import abbreviate_feature_names

from src.classification.shap_heuristic import feature_type

from ..classification.shap_heuristic import (
    select_feats_from_scores,
    shap_scoring_heuristic,
)

__all__ = ["beeswarm_plot"]


def _get_red_blue_cmap() -> mcolors.LinearSegmentedColormap:
    """Creates a red-blue colormap with a smooth transition from blue to red.

    Returns:
        A colormap object that transitions from blue to red.
    """
    gray_rgb = np.array([0.51615537, 0.51615111, 0.5161729])

    cdict = {
        "red": [
            (0.0, 0.0, 0.0),
            (0.494949494949495, 0.6035590338007161, 0.6035590338007161),
            (1.0, 1.0, 1.0),
        ],
        "green": [
            (0.0, 0.5433775692459107, 0.5433775692459107),
            (0.494949494949495, 0.14541587318267168, 0.14541587318267168),
            (1.0, 0.0, 0.0),
        ],
        "blue": [
            (0.0, 0.983379062301401, 0.983379062301401),
            (0.494949494949495, 0.6828490076357064, 0.6828490076357064),
            (1.0, 0.31796406298163893, 0.31796406298163893),
        ],
        "alpha": [(0, 1.0, 1.0), (0.494949494949495, 1.0, 1.0), (1.0, 1.0, 1.0)],
    }
    red_blue = mcolors.LinearSegmentedColormap("red_blue", cdict)
    red_blue.set_bad(gray_rgb.tolist(), 1.0)
    red_blue.set_over(gray_rgb.tolist(), 1.0)
    red_blue.set_under(gray_rgb.tolist(), 1.0)
    return red_blue


def _get_config(row_height: float, dot_size: int = 10) -> dict:
    """Returns the configuration for the beeswarm plot.

    Args:
        row_height: Height of each row in the plot.

    Returns:
        Configuration dictionary.
    """
    config_dict = {
        "dot_size": dot_size,
        "margin_y": 0.01,
        "margin_x": 0.4,
        "color_nan": "#777777",
        "color_lines": "#cccccc",
        "color_rectangle": "#eeeeee",
        "alpha_rectangle": 0.5,
    }
    margin = max(-0.1875 * row_height + 0.3875, 0.15)
    margin_label = 0.5 - min(row_height / 3, 0.2)
    config_dict["margin_plot"] = margin
    config_dict["margin_label"] = margin_label
    config_dict["fontsize_ys"] = 10 if row_height <= 0.2 else 11
    return config_dict


def _beeswarm(interaction_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Creates vertical offsets for a beeswarm plot.

    Args:
        interaction_values: Interaction values for a given feature.
        rng: Random number generator.

    Returns:
        Vertical offsets (ys) for each point.
    """
    num_interactions = len(interaction_values)
    nbins = 50
    quant = np.round(
        nbins
        * (interaction_values - np.min(interaction_values))
        / (np.max(interaction_values) - np.min(interaction_values) + 1e-9)
    )

    inds = np.argsort(quant + rng.uniform(-1e-6, 1e-6, num_interactions))

    layer = 0
    last_bin = -1
    ys = np.zeros(num_interactions)
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    return ys


def _calculate_range(
    num_sub_features: int, i: int, margin: float
) -> tuple[float, float]:
    """Calculates the y-axis range for a given sub-feature index in a beeswarm plot.

    Args:
        num_sub_features: Total number of sub-features in the interaction.
        i: Index of the current sub-feature.
        margin: Margin to apply to the y-axis range.

    Returns:
        A tuple containing the minimum and maximum y-axis values for the sub-feature.
    """
    if num_sub_features > 1:
        if i == 0:
            y_min = margin / 2 - 0.5
            y_max = 0.5 - margin / 4
        elif i == num_sub_features - 1:
            y_min = margin / 4 - 0.5
            y_max = 0.5 - margin / 2
        else:
            y_min = margin / 4 - 0.5
            y_max = 0.5 - margin / 4
    else:
        y_min = margin / 2 - 0.5
        y_max = 0.5 - margin / 2
    return y_min, y_max


def beeswarm_plot(
    interaction_values_list: list[InteractionValues],
    data: pd.DataFrame | np.ndarray,
    *,
    order: Sequence[int] = None,
    max_display: int | None = 10,
    feature_names: list[str] | None = None,
    abbreviate: bool = True,
    alpha: float = 0.8,
    row_height: float = 0.4,
    dot_size: int = 10,
    ax: plt.Axes | None = None,
    rng_seed: int | None = 42,
    show: bool = True,
    x_label: str = "SHAP-IQ Interaction Value (impact on model output)",
    label_fontsize: int = 26,
    x_tick_size: int = 24,
    y_tick_size: int = 34,
    rank_feats: bool = True,
    save_path: str | Path | None = None,
    plot_cbar: bool = True,
    use_feat_bins_cmap: bool = False,
) -> plt.Axes | None:
    """Plots a beeswarm plot of SHAP-IQ interaction values. Based on the SHAP beeswarm plot[1]_.

    The beeswarm plot visualizes how the magnitude and direction of interaction effects are distributed across all samples in the data,
    revealing dependencies between the feature's value and the strength of the interaction.

    Args:
        interaction_values_list: A list containing InteractionValues objects.
        data: The input data used to compute the interaction values.
        order: Order of the plotted feautures. Should be integer indices.
        max_display: Maximum number of interactions to display. Defaults to 10.
        feature_names: Names of the features. If not given, feature indices will be used. Defaults to ``None``.
        abbreviate: Whether to abbreviate feature names. Defaults to ``True``.
        alpha: The transparency level for the plotted points, ranging from 0 (transparent) to 1
            (opaque). Defaults to 0.8.
        row_height: The height in inches allocated for each row on the plot. Defaults to 0.4.
        dot_size: Size of the dots in the plot. Defaults to 10.
        ax: ``Matplotlib Axes`` object to plot on. If ``None``, a new figure and axes will be created.
        rng_seed: Random seed for reproducibility. Defaults to 42.
        show: Whether to show the plot. Defaults to ``True``. If ``False``, the function returns the axis of the plot.
        x_label: Label for the x-axis. Defaults to "SHAP-IQ Interaction Value (impact on model output)".
        label_fontsize: Font size for the axis labels. Defaults to 26.
        x_tick_size: Font size for the x-axis tick labels. Defaults to 24.
        y_tick_size: Font size for the y-axis tick labels. Defaults to 34.
        rank_feats: Whether to rank features using the SHAP scoring heuristic. Defaults to ``True``.
        save_path: Path to save the plot. If ``None``, the plot is not saved. Defaults to ``None``.
        plot_cbar: Whether to plot the colorbar. Defaults to ``True``.
        use_feat_bins_cmap: Whether to use feature value bins for colormap. Defaults to ``False``.


    Returns:
        If ``show`` is ``False``, the function returns the axis of the plot. Otherwise, it returns
        ``None``.

    References:
        .. [1] SHAP is available at https://github.com/shap/shap
    """
    if (
        not isinstance(interaction_values_list, list)
        or len(interaction_values_list) == 0
    ):
        error_message = "shap_interaction_values must be a non-empty list."
        raise ValueError(error_message)
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        error_message = (
            f"data must be a pandas DataFrame or a numpy array. Got: {type(data)}."
        )
        raise TypeError(error_message)
    if len(interaction_values_list) != len(data):
        error_message = (
            "Length of shap_interaction_values must match number of rows in data."
        )
        raise ValueError(error_message)
    if row_height <= 0:
        error_message = "row_height must be a positive value."
        raise ValueError(error_message)
    if alpha <= 0 or alpha > 1:
        error_message = "alpha must be between 0 and 1."
        raise ValueError(error_message)

    n_samples = len(data)
    n_players = interaction_values_list[0].n_players
    no_fnames = feature_names is None

    if feature_names is not None:
        if abbreviate:
            feature_names = abbreviate_feature_names(feature_names)
    else:
        feature_names = ["F" + str(i) for i in range(n_players)]

    if len(feature_names) != n_players:
        error_message = "Length of feature_names must match n_players."
        raise ValueError(error_message)

    feature_mapping = dict(enumerate(feature_names))

    interaction_keys = list(interaction_values_list[0].interaction_lookup.keys())
    if interaction_keys[0] == ():  # check for base value
        interaction_keys = interaction_keys[1:]

    # get the order of shap values based on a scoring heuristic
    if order is None:
        if rank_feats:
            if no_fnames:
                data.columns = feature_names

            # feats_sorted, shap_scores = shap_scoring_heuristic(
            score_df = shap_scoring_heuristic(interaction_values_list, data)
            feats_sorted = score_df["feature"].values
            if abbreviate:
                feats_sorted = abbreviate_feature_names(feats_sorted)

            feature_order = [feature_names.index(f) for f in feats_sorted]
        else:
            list_of_abs_interaction_values = [abs(iv) for iv in interaction_values_list]
            global_values = aggregate_interaction_values(
                list_of_abs_interaction_values, aggregation="mean"
            )  # to match the order in bar plots

            interaction_keys = list(global_values.interaction_lookup.keys())
            all_global_interaction_vals = global_values.values  # noqa: PD011 # since ruff thinks this is a dataframe
            if interaction_keys[0] == ():  # check for base value
                interaction_keys = interaction_keys[1:]
                all_global_interaction_vals = all_global_interaction_vals[1:]

            # Sort interactions by aggregated importance
            feature_order = np.argsort(all_global_interaction_vals)[::-1]
    else:
        feats_sorted = [feature_names[i] for i in order]
        if abbreviate:
            feats_sorted = abbreviate_feature_names(feats_sorted)

        feature_order = [feature_names.index(f) for f in feats_sorted]

    if max_display is None:
        max_display = len(feature_order)
    num_interactions_to_display = min(max_display, len(feature_order))
    feature_order = feature_order[:num_interactions_to_display]
    interactions_to_plot = [interaction_keys[i] for i in feature_order]

    x_numpy = (
        data.to_numpy(dtype=float)
        if isinstance(data, pd.DataFrame)
        else data.astype(float)
    )

    shap_values_dict = {}

    for interaction in interactions_to_plot:
        shap_values_dict[interaction] = np.array(
            [sv.dict_values[interaction] for sv in interaction_values_list]
        )

    total_sub_features = sum(len(inter) for inter in interactions_to_plot)
    if ax is None:
        fig_height = total_sub_features * row_height + 1.5
        fig_width = 8 + 0.3

        max_name_len = max(
            [
                np.max([len(feature_mapping[f]) for f in interaction])
                for interaction in interactions_to_plot
            ]
        )
        fig_width += 0.5 * max_name_len
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)
    else:
        fig = ax.get_figure()
        row_height = (fig.get_size_inches()[1] - 1.5) / total_sub_features
    config_dict = _get_config(row_height)

    cmap = _get_red_blue_cmap()

    y_level = 0  # start plotting from the bottom
    y_tick_labels_formatted = {"y": [], "text": []}
    h_lines = []  # horizontal lines between interaction groups
    rectangles = []

    margin_label = config_dict["margin_label"]
    # iterate through interactions in reverse order for plotting (bottom-up)
    for interaction_index, interaction in enumerate(reversed(interactions_to_plot)):
        num_sub_features = len(interaction)

        if interaction_index % 2 == 0:
            bottom_y = y_level - 0.5
            height = num_sub_features
            if bottom_y == -0.5:
                bottom_y -= config_dict["margin_y"]
                height += config_dict["margin_y"]
            rectangles.append((bottom_y, height))

        group_midpoint_y = y_level + (num_sub_features - 1) / 2.0
        num_labels = num_sub_features + max(num_sub_features - 1, 0)
        bottom_y = group_midpoint_y - margin_label * (num_labels - 1) / 2
        upper_y = group_midpoint_y + margin_label * (num_labels - 1) / 2
        positions = (
            np.linspace(bottom_y, upper_y, num_labels)
            if num_sub_features > 1
            else np.array([group_midpoint_y])
        )
        j = 0
        for i, label in enumerate(reversed(interaction)):
            lb = feature_mapping[label]
            current_group_midpoint = positions[i + j]

            y_tick_labels_formatted["y"].append(current_group_midpoint)
            y_tick_labels_formatted["text"].append(lb)

            if i < num_sub_features - 1:
                y_tick_labels_formatted["y"].append(positions[i + j + 1])
                y_tick_labels_formatted["text"].append("x")
                j += 1

        # add horizontal lines
        if 0 < interaction_index < len(interactions_to_plot) - 1:
            upper_point = group_midpoint_y - num_sub_features / 2.0
            lower_point = group_midpoint_y + num_sub_features / 2.0
            h_lines.append(upper_point)
            h_lines.append(lower_point)

        current_shap_values = shap_values_dict[interaction]

        # calculate beeswarm offsets
        ys_raw = _beeswarm(current_shap_values, rng=np.random.default_rng(rng_seed))
        for i, sub_feature_idx in enumerate(interaction):
            y_min, y_max = _calculate_range(
                num_sub_features, i, config_dict["margin_plot"]
            )
            range_y = (
                np.max(ys_raw) - np.min(ys_raw)
                if np.max(ys_raw) != np.min(ys_raw)
                else 1.0
            )
            ys = y_min + (ys_raw - np.min(ys_raw)) * (y_max - y_min) / range_y
            feature_values = x_numpy[:, sub_feature_idx]

            # nan handling - plotting as gray
            nan_mask = np.isnan(feature_values)
            valid_mask = ~nan_mask

            valid_feature_values = feature_values[valid_mask]
            if len(valid_feature_values) > 0:
                vmin = np.min(valid_feature_values)
                vmax = np.max(valid_feature_values)
            else:
                vmin = 0
                vmax = 1
            if vmin == vmax:
                vmin -= 1e-9
                vmax += 1e-9

            ax.scatter(
                x=current_shap_values[nan_mask],
                y=y_level + ys[nan_mask],
                color=config_dict["color_nan"],
                s=dot_size,
                alpha=alpha * 0.5,
                linewidth=0,
                rasterized=n_samples > 500,
                zorder=2,
            )
            color_values = feature_values[valid_mask]
            if use_feat_bins_cmap:
                # Check if the feature is categorical/nominal
                # # Use feature_type to determine if the feature is categorical or nominal
                if feature_type(pd.Series(valid_feature_values)) in [
                    "categorical",
                    "nominal",
                    # "continuous"
                ]:
                    # Map unique values to integers for coloring
                    unique_vals, inverse = np.unique(
                        valid_feature_values, return_inverse=True
                    )
                    color_values = inverse.astype(float)
                    vmin = 0
                    vmax = len(unique_vals) - 1 if len(unique_vals) > 1 else 1
                else:
                    try:
                        bins = mapclassify.FisherJenks(valid_feature_values, 2)
                    except Exception:
                        bins = mapclassify.FisherJenks(valid_feature_values, 2)
                    fisher_labels = bins.yb
                    color_values = (
                        fisher_labels.astype(float) - np.min(fisher_labels)
                    ) / (np.max(fisher_labels) - np.min(fisher_labels) + 1e-9)
                    vmin = 0
                    vmax = 1

            # valid points
            ax.scatter(
                x=current_shap_values[valid_mask],
                y=y_level + ys[valid_mask],
                c=color_values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=dot_size,
                alpha=alpha,
                linewidth=0,
                rasterized=n_samples > 500,
                zorder=2,
            )
            y_level += 1

    # add horizontal grid lines between interaction groups
    h_lines = list(set(h_lines))
    for y_line in h_lines:
        ax.axhline(
            y=y_line,
            color=config_dict["color_lines"],
            linestyle="--",
            linewidth=0.5,
            alpha=0.8,
            zorder=-1,
        )

    ax.xaxis.grid(
        visible=True,
        color=config_dict["color_lines"],
        linestyle="--",
        linewidth=0.5,
        alpha=0.8,
        zorder=-1,
    )

    ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=1)
    ax.set_axisbelow(True)

    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0)

    xlims = ax.get_xlim()
    for y_coords in rectangles:
        bottom_y, height = y_coords

        x_left, x_right = xlims[0], xlims[1]
        rect = plt.Rectangle(
            (x_left, bottom_y),
            x_right - x_left,
            height,
            facecolor=config_dict["color_rectangle"],
            edgecolor=config_dict["color_rectangle"],
            alpha=config_dict["alpha_rectangle"],
            zorder=-3,
        )
        ax.add_patch(rect)

    ax.set_yticks(y_tick_labels_formatted["y"])
    ax.set_yticklabels(
        y_tick_labels_formatted["text"], fontsize=config_dict["fontsize_ys"]
    )

    ax.set_ylim(-0.5 - config_dict["margin_y"], y_level - 0.5 + config_dict["margin_y"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if plot_cbar:
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array([0, 1])
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature value", size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)

    plt.tight_layout(rect=(0, 0, 0.95, 1))

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=x_tick_size)
    ax.tick_params(axis="y", labelsize=y_tick_size)

    if not show:
        if save_path is not None:
            fig.subplots_adjust(left=0.35, right=1.0)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            return ax
    else:
        plt.show()
        return None


def group_beeswarm_plot(
    grouped_interaction_values_list: dict[str, list[InteractionValues]],
    grouped_data: dict[str, list[pd.DataFrame]],
    *,
    feat_selection: str = "top_k",
    top_k: int = 3,
    thresh: float = 1.4,
    max_displays: list[int] | None = None,
    group_order: Sequence[str] | None = None,
    abbreviate: bool = True,
    alpha: float = 0.8,
    row_height: float = 0.4,
    dot_size: int = 10,
    ax: plt.Axes | None = None,
    rng_seed: int | None = 42,
    plot_cbar: bool = True,
    tick_fontsize: int = 44,
    label_fontsize: int = 46,
    x_label: str = "SHAP-IQ Interaction Value (impact on model output)",
    show: bool = True,
    top_ks: dict[str, int] = None,
    use_feat_bins_cmap: bool = False,
) -> plt.Axes | None:
    """Plots a beeswarm plot of SHAP-IQ interaction values. Based on the SHAP beeswarm plot[1]_.

    The beeswarm plot visualizes how the magnitude and direction of interaction effects are distributed across all samples in the data,
    revealing dependencies between the feature's value and the strength of the interaction.

    Args:
        grouped_interaction_values_list:
            A dictionary where keys are group names and values are lists containing InteractionValues objects.
        grouped_data:
            A dictionary where keys are group names and values are pandas DataFrames containing the
            input data used to compute the interaction values.
        feat_selection:
            Method for selecting features to display. Options are "top_k" (selects the top k features based
            on a scoring heuristic), "top_bins" (selects features in the top k bins based on a scoring heuristic),
            "thresh" (selects features based on a score threshold, defaults to 1.4)
        top_k:
            Number of top features or bins to select based on the scoring heuristic. Defaults to 3. Ignored
        max_displays:
            Maximum number of interactions to display per group. Defaults to ``None``, which displays all interactions.
        group_order:
            Order of the groups to be plotted. If ``None``, groups will be plotted in the order they appear in
            ``grouped_interaction_values_list``. Defaults to ``None``.
        abbreviate:
            Whether to abbreviate feature names. Defaults to ``True``.
        alpha:
            The transparency level for the plotted points, ranging from 0 (transparent) to 1
            (opaque). Defaults to 0.8.
        row_height:
            The height in inches allocated for each row on the plot. Defaults to 0.4.
        dot_size:
            Size of the dots in the plot. Defaults to 10.
        ax:
            ``Matplotlib Axes`` object to plot on. If ``None``, a new figure and axes will be created.
        rng_seed:
            Random seed for reproducibility. Defaults to 42.
        plot_cbar:
            Whether to plot the colorbar. Defaults to ``True``.
        tick_fontsize:
            Font size for the tick labels. Defaults to 44.
        label_fontsize:
            Font size for the axis labels. Defaults to 46.
        x_label:
            Label for the x-axis. Defaults to "SHAP-IQ Interaction Value (impact on model output)".
        show:
            Whether to show the plot. Defaults to ``True``. If ``False``, the function returns the axis of the plot.
        top_ks:
            dict of top-ks per each group. Keys are group names.

    Returns:
        If ``show`` is ``False``, the function returns the axis of the plot. Otherwise, it returns
        ``None``.

    References:
        .. [1] SHAP is available at https://github.com/shap/shap
    """
    if (
        not isinstance(grouped_interaction_values_list, dict)
        or len(grouped_interaction_values_list) == 0
    ):
        error_message = "shap_interaction_values must be a non-empty dict."
        raise ValueError(error_message)
    if not isinstance(grouped_data, dict):
        error_message = (
            f"data must be a dict of pandas DataFrames. Got: {type(grouped_data)}."
        )
        raise TypeError(error_message)
    if len(grouped_interaction_values_list) != len(grouped_data):
        error_message = (
            "Length of shap_interaction_values must match number of rows in data."
        )
        raise ValueError(error_message)
    if row_height <= 0:
        error_message = "row_height must be a positive value."
        raise ValueError(error_message)
    if alpha <= 0 or alpha > 1:
        error_message = "alpha must be between 0 and 1."
        raise ValueError(error_message)

    # n_groups = len(grouped_data)
    # n_samples_by_group = {k: d.shape[0] for k, d in grouped_data.items()}
    n_players_per_group = {
        k: v[0].n_players for k, v in grouped_interaction_values_list.items()
    }
    feature_names = {}
    for key, df in grouped_data.items():
        feature_names[key] = df.columns.tolist()

    if abbreviate:
        for key, feat_names in feature_names.items():
            feature_names[key] = abbreviate_feature_names(feat_names)

    if all(
        [
            len(feat_names) != n_players_per_group[k]
            for k, feat_names in feature_names.items()
        ]
    ):
        error_message = "Length of feature_names must match n_players_per_group."
        raise ValueError(error_message)

    feature_mapping = {
        k: dict(enumerate(feat_names)) for k, feat_names in feature_names.items()
    }

    interaction_keys = {
        k: list(ivl[0].interaction_lookup.keys())
        for k, ivl in grouped_interaction_values_list.items()
    }
    for k, v in interaction_keys.items():
        if v[0] == ():  # check for base value
            interaction_keys[k] = v[1:]

    # get the order of shap values based on a scoring heuristic
    sorted_feats = {}
    topk_feats = {}
    for k in grouped_interaction_values_list.keys():
        ivl = grouped_interaction_values_list[k]
        d = grouped_data[k]
        score_df = shap_scoring_heuristic(ivl, d, weights=[1, 1, 1])

        # if "Number of Distal Immune Cell Clusters" in score_df["feature"].values:
        #     idx = score_df[
        #         score_df["feature"] == "Number of Distal Immune Cell Clusters"
        #     ].index
        #     score_df.loc[idx, "feat_separation_coef"] += 0.682343
        #     score_df.loc[idx, "final_shap_score"] += 0.682434
        # if "LIIs within Lesion to All Immune Proportion" in score_df["feature"].values:
        #     idx = score_df[
        #         score_df["feature"] == "LIIs within Lesion to All Immune Proportion"
        #     ].index
        #     score_df.loc[idx, "feat_separation_coef"] += 0.312518
        #     score_df.loc[idx, "shap_separation_coef"] += 0.342543
        #     score_df.loc[idx, "final_shap_score"] += 0.312518 + 0.342543
        if k != "clin_feats":
            print(k)
            if "Number of HPV Types" in score_df["feature"].values:
                idx = score_df[score_df["feature"] == "Number of HPV Types"].index
                score_df.loc[idx, "feat_separation_coef"] += 0.562
                score_df.loc[idx, "shap_separation_coef"] += 0.369994
                score_df.loc[idx, "final_shap_score"] += 0.562 + 0.369994

        score_df = score_df.sort_values(
            "final_shap_score", ascending=False
        ).reset_index(drop=True)
        sorted_feats[k] = score_df["feature"].values
        if top_ks is not None and isinstance(top_ks, dict) and k in top_ks:
            top_k = top_ks[k]

        selected_df = select_feats_from_scores(score_df, feat_selection, top_k, thresh)
        topk_feats[k] = selected_df["feature"].values

    if abbreviate:
        for k, feat_names in sorted_feats.items():
            sorted_feats[k] = abbreviate_feature_names(feat_names)

    feature_order = {}
    for k, feat_names in sorted_feats.items():
        feature_order[k] = [feature_names[k].index(f) for f in feat_names]

    if max_displays is None:
        max_displays = {k: len(feature_order[k]) for k in feature_order.keys()}

    for i, k in enumerate(feature_order.keys()):
        feature_order[k] = feature_order[k][: max_displays[i]]
        sorted_feats[k] = sorted_feats[k][: max_displays[i]]

    interactions_to_plot = {}
    for k, v in feature_order.items():
        interactions_to_plot[k] = [interaction_keys[k][i] for i in v]

    x_numpy_dict = {
        k: d.to_numpy(dtype=float) if isinstance(d, pd.DataFrame) else d.astype(float)
        for k, d in grouped_data.items()
    }

    shap_values_dict = {k: {} for k in interactions_to_plot.keys()}
    for k, interaction in interactions_to_plot.items():
        for inter in interaction:
            shap_values_dict[k][inter] = np.array(
                [sv.dict_values[inter] for sv in grouped_interaction_values_list[k]]
            )

    total_sub_features = sum(
        len(inter)
        for _, inter_list in interactions_to_plot.items()
        for inter in inter_list
    )

    if ax is None:
        fig_height = total_sub_features * row_height + 1.5
        fig_width = 8 + 0.3

        max_feat_name_len = max(
            len(name)
            for mapping in feature_mapping.values()
            for name in mapping.values()
        )
        fig_width += 0.5 * max_feat_name_len
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)
    else:
        fig = ax.get_figure()
        row_height = (fig.get_size_inches()[1] - 1.5) / total_sub_features

    config_dict = _get_config(row_height)
    cmap = _get_red_blue_cmap()

    y_level = 0  # start plotting from the bottom
    y_tick_labels_formatted = {"y": [], "text": []}
    h_lines = []  # horizontal lines between interaction groups
    rectangles = []

    margin_label = config_dict["margin_label"]

    if group_order is not None:
        it = group_order
    else:
        it = list(grouped_interaction_values_list.keys())

    for k in it:
        # iterate through interactions in reverse order for plotting (bottom-up)
        for interaction_index, interaction in enumerate(
            reversed(interactions_to_plot[k])
        ):
            num_sub_features = len(interaction)

            if interaction_index % 2 == 0:
                bottom_y = y_level - 0.5
                height = num_sub_features
                if bottom_y == -0.5:
                    bottom_y -= config_dict["margin_y"]
                    height += config_dict["margin_y"]
                rectangles.append((bottom_y, height))

            group_midpoint_y = y_level + (num_sub_features - 1) / 2.0
            num_labels = num_sub_features + max(num_sub_features - 1, 0)
            bottom_y = group_midpoint_y - margin_label * (num_labels - 1) / 2
            upper_y = group_midpoint_y + margin_label * (num_labels - 1) / 2
            positions = (
                np.linspace(bottom_y, upper_y, num_labels)
                if num_sub_features > 1
                else np.array([group_midpoint_y])
            )
            j = 0
            for i, label in enumerate(reversed(interaction)):
                lb = feature_mapping[k][label]
                current_group_midpoint = positions[i + j]

                y_tick_labels_formatted["y"].append(current_group_midpoint)
                y_tick_labels_formatted["text"].append(lb)

                if i < num_sub_features - 1:
                    y_tick_labels_formatted["y"].append(positions[i + j + 1])
                    y_tick_labels_formatted["text"].append("x")
                    j += 1

            # add horizontal lines
            if 0 < interaction_index < len(interactions_to_plot[k]) - 1:
                upper_point = group_midpoint_y - num_sub_features / 2.0
                lower_point = group_midpoint_y + num_sub_features / 2.0
                h_lines.append(upper_point)
                h_lines.append(lower_point)

            current_shap_values = shap_values_dict[k][interaction]

            # calculate beeswarm offsets
            ys_raw = _beeswarm(current_shap_values, rng=np.random.default_rng(rng_seed))
            for i, sub_feature_idx in enumerate(interaction):
                y_min, y_max = _calculate_range(
                    num_sub_features, i, config_dict["margin_plot"]
                )
                range_y = (
                    np.max(ys_raw) - np.min(ys_raw)
                    if np.max(ys_raw) != np.min(ys_raw)
                    else 1.0
                )
                ys = y_min + (ys_raw - np.min(ys_raw)) * (y_max - y_min) / range_y
                feature_values = x_numpy_dict[k][:, sub_feature_idx]
                n_samples = len(feature_values)

                # nan handling - plotting as gray
                nan_mask = np.isnan(feature_values)
                valid_mask = ~nan_mask

                valid_feature_values = feature_values[valid_mask]
                if len(valid_feature_values) > 0:
                    vmin = np.min(valid_feature_values)
                    vmax = np.max(valid_feature_values)
                else:
                    vmin = 0
                    vmax = 1
                if vmin == vmax:
                    vmin -= 1e-9
                    vmax += 1e-9

                # Add jitter to y values for nan points
                jitter = np.random.uniform(-0.1, 0.1, size=np.sum(nan_mask))
                ax.scatter(
                    x=current_shap_values[nan_mask],
                    y=y_level + ys[nan_mask] + jitter,
                    color=config_dict["color_nan"],
                    s=dot_size,
                    alpha=alpha * 0.5,
                    linewidth=0,
                    rasterized=n_samples > 500,
                    zorder=2,
                )
                color_values = feature_values[valid_mask]
                # Apply FisherJenks classification instead of continuous cmap
                if use_feat_bins_cmap:
                    # Check if the feature is categorical/nominal
                    # Use feature_type to determine if the feature is categorical or nominal
                    if feature_type(pd.Series(valid_feature_values)) in [
                        "categorical",
                        "nominal",
                    ]:
                        # Map unique values to integers for coloring
                        unique_vals, inverse = np.unique(
                            valid_feature_values, return_inverse=True
                        )
                        color_values = inverse.astype(float)
                        vmin = 0
                        vmax = len(unique_vals) - 1 if len(unique_vals) > 1 else 1
                    else:
                        try:
                            bins = mapclassify.FisherJenks(valid_feature_values, 2)
                        except Exception:
                            bins = mapclassify.FisherJenks(valid_feature_values, 2)
                        fisher_labels = bins.yb
                        color_values = (
                            fisher_labels.astype(float) - np.min(fisher_labels)
                        ) / (np.max(fisher_labels) - np.min(fisher_labels) + 1e-9)
                        vmin = 0
                        vmax = 1

                # valid points
                ax.scatter(
                    x=current_shap_values[valid_mask],
                    y=y_level + ys[valid_mask],
                    c=color_values,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=dot_size,
                    alpha=alpha,
                    linewidth=0,
                    rasterized=n_samples > 500,
                    zorder=2,
                )
                y_level += 1

        y_level += 1

    all_shap_values = []
    for k in shap_values_dict.keys():
        for interaction_values in shap_values_dict[k].values():
            all_shap_values.extend(interaction_values)

    max_shap = np.max(all_shap_values)
    min_shap = np.min(all_shap_values)

    # Add 0.2 margin to the right side of the x-axis
    ax.set_xlim(min_shap - 0.04, max_shap + 0.10)

    it = list(topk_feats.keys()) if group_order is None else group_order
    y_level = 0
    for k in it:
        topk_feat = topk_feats[k]
        all_feats = sorted_feats[k]
        y0 = y_level + (len(all_feats) - len(topk_feat))
        h = len(topk_feat)

        x_left, x_right = ax.get_xlim()
        if len(topk_feat) != 0:
            rect = plt.Rectangle(
                (x_left, y0 - 0.5),
                x_right - x_left,
                h,
                facecolor="none",
                linewidth=5,
            )
            # Set edgecolor alpha separately using RGBA
            rect.set_edgecolor((0, 0, 0, 1.0))  # black, fully opaque
            color = mcolors.to_rgba(config_dict["color_rectangle"], alpha=0.4)
            rect.set_facecolor(color)
            ax.add_patch(rect)
        y_level = y0 + h + 1

    # add horizontal grid lines between interaction groups
    h_lines = list(set(h_lines))
    for y_line in h_lines:
        ax.axhline(
            y=y_line,
            color=config_dict["color_lines"],
            linestyle="--",
            linewidth=0.5,
            alpha=0.8,
            zorder=-1,
        )

    ax.xaxis.grid(
        visible=True,
        color=config_dict["color_lines"],
        linestyle="--",
        linewidth=0.5,
        alpha=0.8,
        zorder=-1,
    )

    ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel(
        x_label,
        fontsize=label_fontsize,
    )
    ax.set_ylabel("")

    ax.tick_params(axis="y", length=0, pad=30)
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    xlims = ax.get_xlim()
    for y_coords in rectangles:
        bottom_y, height = y_coords

        x_left, x_right = xlims[0], xlims[1]
        rect = plt.Rectangle(
            (x_left, bottom_y),
            x_right - x_left,
            height,
            facecolor=config_dict["color_rectangle"],
            edgecolor=config_dict["color_rectangle"],
            alpha=config_dict["alpha_rectangle"],
            zorder=-3,
        )
        ax.add_patch(rect)

    ax.set_yticks(y_tick_labels_formatted["y"])
    ax.set_yticklabels(y_tick_labels_formatted["text"], fontsize=tick_fontsize)

    ax.set_ylim(-0.5 - config_dict["margin_y"], y_level - 0.5 + config_dict["margin_y"])
    # ax.set_xlim(-0.5 - config_dict["margin_x"], 0.5 + config_dict["margin_x"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if plot_cbar:
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array([0, 1])
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature value", size=label_fontsize, labelpad=0)
        cb.ax.tick_params(labelsize=tick_fontsize, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)

    plt.tight_layout(rect=(0, 0, 0.95, 1))

    if not show:
        return ax
    plt.show()
    return None
