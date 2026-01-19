from typing import List

import mapclassify
import numpy as np
import pandas as pd
from shapiq import InteractionValues

from src.classification.utils import shap2df


def _is_binary(series: pd.Series) -> bool:
    """Returns True if the column contains only two unique values (0/1, True/False, etc)."""
    unique = series.dropna().unique()
    return len(unique) == 2


def _is_nominal(series: pd.Series, max_unique=20) -> bool:
    """Returns True if the column is categorical with a small number of unique values."""
    unique = series.dropna().unique()
    return (series.dtype.kind in "biufc") and (len(unique) <= max_unique)


def _is_categorical(series: pd.Series, max_unique=5) -> bool:
    """Returns True if the column is object dtype or has few unique values."""
    return series.dtype == "object" or _is_nominal(series, max_unique=max_unique)


def _is_continuous(series: pd.Series, min_unique=20) -> bool:
    """Returns True if the column is numeric and has many unique values."""
    unique = series.dropna().unique()
    return (series.dtype.kind in "fi") and (len(unique) >= min_unique)


def feature_type(series: pd.Series) -> str:
    """Classifies the feature type."""
    if _is_binary(series):
        return "binary"
    elif _is_nominal(series):
        return "nominal"
    elif _is_continuous(series):
        return "continuous"
    elif _is_categorical(series):
        return "categorical"
    else:
        return "unknown"


def scale_column(values: np.ndarray, norm: bool = True) -> np.ndarray:
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)
    if vmin == vmax:
        vmin = np.nanpercentile(values, 1)
        vmax = np.nanpercentile(values, 99)
        if vmin == vmax:
            vmin = np.min(values)
            vmax = np.max(values)
    if vmin > vmax:  # fixes rare numerical precision issues
        vmin = vmax

    # plot the non-nan values colored by the trimmed feature value
    nan_mask = np.isnan(values)
    cvals = values[np.invert(nan_mask)].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
    cvals[cvals_imp > vmax] = vmax
    cvals[cvals_imp < vmin] = vmin

    if norm:
        cvals = (cvals - np.min(cvals)) / (np.max(cvals) - np.min(cvals) + 1e-8)

    return cvals


def _scale_column_with_nans(values: np.ndarray, norm: bool = True) -> np.ndarray:
    """Scale values to [0, 1] with percentile trimming while preserving NaNs."""
    values = values.astype(float)
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)
    if vmin == vmax:
        vmin = np.nanpercentile(values, 1)
        vmax = np.nanpercentile(values, 99)
        if vmin == vmax:
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
    if vmin > vmax:  # fixes rare numerical precision issues
        vmin = vmax

    scaled = values.copy()
    scaled = np.where(np.isnan(scaled), np.nan, np.clip(scaled, vmin, vmax))

    if norm:
        smin = np.nanmin(scaled)
        smax = np.nanmax(scaled)
        if smin == smax:
            return scaled
        scaled = (scaled - smin) / (smax - smin + 1e-8)

    return scaled


def drop_min_max(df: pd.DataFrame) -> pd.DataFrame:
    maxidx = df.loc[:, np.log(df.max().values) > 0].idxmax()  # .to_list()
    minidx = df.loc[:, np.log(np.abs(df.min()).values) > 0].idxmin()

    return df.drop(index=pd.concat((maxidx, minidx)).unique())


def compute_feat_mixing_coef(
    feat_vals: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray
) -> float:
    """Compute the feature mixing coefficient.

    Computes the feature mixing coefficient as the average proportion of the most common
    feature value on the positive and negative side of the shap origin. If the dominant
    value is different for positive and negative sides, the score is high; if the same,
    the score is zero.

    For continuous/nominal features, values are binned into two groups before computing proportions.

    Returns:
        float: Mixing score, high if dominant values differ, zero if they are the same.
    """
    feat_type = feature_type(pd.Series(feat_vals))

    # convert continuous/nominal features to binary using equal interval binning ['low', 'high']
    if feat_type in ["nominal", "continuous"]:
        # bins = mapclassify.EqualInterval(scale_column(feat_vals, norm=True), 2)
        # bins = mapclassify.EqualInterval(feat_vals, 2)
        # bins = mapclassify.FisherJenks(scale_column(feat_vals, norm=True), 2)
        bins = mapclassify.FisherJenks(feat_vals, 2)
        feat_vals = bins.yb

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0

    pos_cls, pos_cnt = np.unique(feat_vals[pos_mask], return_counts=True)
    pos_prop = pos_cnt / pos_cnt.sum()
    neg_cls, neg_cnt = np.unique(feat_vals[neg_mask], return_counts=True)
    neg_prop = neg_cnt / neg_cnt.sum()

    if pos_prop.size == 0 or neg_prop.size == 0:
        return 0.0

    pos_major_cls = pos_cls[np.argmax(pos_prop)]
    pos_major_prop = np.max(pos_prop)
    neg_major_cls = neg_cls[np.argmax(neg_prop)]
    neg_major_prop = np.max(neg_prop)

    if neg_major_cls != pos_major_cls:
        score = (pos_major_prop + neg_major_prop) / 2
    else:
        score = 0

    return score


def compute_shap_abs_coef(shap_vals: np.ndarray, global_max: float) -> float:
    """Compute the absolute 95th percentile shap value.

    Computes the normalized 95th percentile absolute shap value. More robust to outliers
    than max and preserves more magnitude information than mean.

    Note:
        Normalizes to global min and max of mean abs shap values across all features.
    """
    # compute the mean absolute shap value and scale to global min and max
    abs_coef = np.sum(np.abs(shap_vals))
    abs_coef = abs_coef / (global_max + 1e-8)
    return abs_coef


def compute_shap_separation_coef(pos_mask: np.ndarray, neg_mask: np.ndarray) -> float:
    """Compute the SHAP value separation coefficient.

    This metric measures the balance of samples on the positive and negative side of the SHAP origin.
    It is calculated as the harmonic mean of the proportions of samples with positive and negative SHAP values.
    The coefficient is high when both sides have similar and substantial proportions, and low when one side dominates.

    Returns:
        float: Separation coefficient, ranging from 0 (one side dominates) to 1 (both sides balanced).
    """
    pos_prop = np.sum(pos_mask) / (np.sum(pos_mask) + np.sum(neg_mask))
    neg_prop = np.sum(neg_mask) / (np.sum(pos_mask) + np.sum(neg_mask))
    shap_separation_coef = (
        2 * pos_prop * neg_prop / (pos_prop + neg_prop)
        if pos_prop > 0 and neg_prop > 0
        else 0
    )

    return shap_separation_coef


def _compute_shap_side_stats(
    shap_vals: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray
) -> dict[str, float]:
    """Compute SHAP summary statistics for positive and negative sides."""
    pos_vals = shap_vals[pos_mask]
    neg_vals = shap_vals[neg_mask]

    pos_abs_sum = np.abs(pos_vals).sum() if pos_vals.size else 0.0
    neg_abs_sum = np.abs(neg_vals).sum() if neg_vals.size else 0.0
    pos_mean_abs = np.abs(pos_vals).mean() if pos_vals.size else 0.0
    neg_mean_abs = np.abs(neg_vals).mean() if neg_vals.size else 0.0

    return {
        "pos_abs_sum": pos_abs_sum,
        "neg_abs_sum": neg_abs_sum,
        "pos_mean_abs": pos_mean_abs,
        "neg_mean_abs": neg_mean_abs,
    }


def _bin_feature_values(values: np.ndarray, k: int = 2) -> np.ndarray:
    """Bin feature values into k clusters while preserving NaN positions."""
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        return np.full(values.shape, np.nan)

    bins = mapclassify.FisherJenks(values[valid_mask], k=k)
    binned = np.full(values.shape, np.nan)
    binned[valid_mask] = bins.yb
    return binned


def _mixing_coef_for_side(binned_vals: np.ndarray, side_mask: np.ndarray) -> float:
    """Compute mixing coefficient for a given side of SHAP origin."""
    valid = side_mask & ~np.isnan(binned_vals)
    side_vals = binned_vals[valid]
    if side_vals.size == 0:
        return 0.0

    counts = pd.Series(side_vals).value_counts(normalize=True)
    if len(counts) == 1:
        return 2.0

    max_prop = counts.max()
    if max_prop > 0.85:
        max_prop *= 2
    return float(max_prop)


def select_feats_from_scores(
    score_df: pd.DataFrame,
    feat_selection_type: str,
    top_k: int = 3,
    thresh: float = 1.4,
) -> pd.DataFrame:
    """Select the top-k or thresholded features from score_df.

    Parameters:
        score_df (pd.DataFrame):
            Output from the `shap_scoring_heuristic` function.
        feat_selection_type (str):
            One of "thresh", "top_bins", "top_k".
        top_k (int):
            Number of top k bins or features to be selected. Ignored if 'thresh'.
        thresh (float):
            Threshold for the shap score. Features above the threshold are selected.

    Returns:
        pd.DataFrame:
            The input score dataframe with only selected features.
    """
    if feat_selection_type == "top_bins":
        score_bins = mapclassify.FisherJenks(score_df["final_shap_score"].values, 5)
        upper_bin_mask = score_bins.yb >= (score_bins.k - top_k)
        res_feats = score_df.loc[upper_bin_mask]
    elif feat_selection_type == "top_k":
        res_feats = score_df.sort_values("final_shap_score", ascending=False).head(
            top_k
        )
    elif feat_selection_type == "thresh":
        res_feats = score_df.loc[score_df["final_shap_score"] > thresh]
    else:
        raise ValueError("feat_selection must be 'thresh', 'top_bins' or 'top_k'")

    return res_feats


def shap_scoring_heuristic(
    interaction_values_list: list[InteractionValues],
    feat_df: pd.DataFrame,
    weights: list[float] = np.array([1.0, 1.0, 1.0]),
    global_max: float = None,
) -> pd.DataFrame:
    """Score features using SHAP values and feature value distributions.

    This heuristic ranks features by combining three metrics:
    - Normalized sum of absolute SHAP values (feature impact magnitude).
    - SHAP separation coefficient: harmonic mean of the proportions of samples with positive
      and negative SHAP values (balance across SHAP origin).
    - Feature mixing coefficient: average proportion of the most common feature value on
      the positive and negative side of the SHAP origin; high if dominant values differ,
      zero if the same.

    Each metric is computed for every feature and combined using the provided weights.

    Parameters:
        interaction_values_list (list[InteractionValues]):
            List of interaction values for each sample.
        feat_df (pd.DataFrame):
            DataFrame containing the feature values. Every row corresponds to an element
            (InteractionValues object) in the `interaction_values_list`
        weights (list[float], optional):
            Weights for the three score components. Order: [abs_shap, shap_sep, feat_mix].
            Defaults to [1.0, 1.0, 1.0].

    Returns:
        pd.DataFrame:
            Sorted df containing the feature scores and score coefficients.
    """
    shap_iq_df = pd.concat(
        [shap2df(shap, feat_df.columns.to_list()) for shap in interaction_values_list]
    )
    shap_iq_df = drop_min_max(shap_iq_df)

    # get global min and max of mean abs shap values across all features for scaling
    if global_max is None:
        global_max = (
            shap_iq_df.loc[:, shap_iq_df.columns != "baseline"].abs().sum().max()
        )

    feat_scores = []
    for col in shap_iq_df.columns:
        scores = {}
        if col == "baseline":
            continue

        shap_vals = shap_iq_df[col].values
        feat_vals = feat_df[col].values

        # samples on positive and negative side of the shap origin
        neg_mask = shap_vals < 0
        pos_mask = shap_vals > 0

        # compute the coefficients
        feat_separation_coef = compute_feat_mixing_coef(feat_vals, pos_mask, neg_mask)
        shap_abs_coef = compute_shap_abs_coef(shap_vals, global_max)
        shap_separation_coef = compute_shap_separation_coef(pos_mask, neg_mask)

        # compute the final score as the weighted sum of the three coefficients
        score_values = np.array(
            [shap_abs_coef, shap_separation_coef, feat_separation_coef]
        )
        score = np.sum(score_values * weights)
        scores["feature"] = col
        scores["feat_separation_coef"] = feat_separation_coef
        scores["shap_separation_coef"] = shap_separation_coef
        scores["shap_abs_coef"] = shap_abs_coef
        scores["final_shap_score"] = score
        feat_scores.append(scores)

    score_df = pd.DataFrame(feat_scores)
    score_df = score_df.sort_values("final_shap_score", ascending=False)
    return score_df


def shap_scoring_heuristic_(
    interaction_values_list: list[InteractionValues],
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank features using the legacy SHAP/feature mixing heuristic.

    This implementation mirrors the input containers used by
    `shap_scoring_heuristic()` and produces a score DataFrame.

    The heuristic is based on:
    1. Features with high absolute SHAP values are important.
    2. Features with high mean SHAP values are important.
    3. Features that have little mixing of feature values around the SHAP origin
       get a boost.
    """
    shap_iq_df = pd.concat(
        [shap2df(shap, feat_df.columns.to_list()) for shap in interaction_values_list]
    )
    shap_iq_df = drop_min_max(shap_iq_df)

    feat_scores = []
    for col in shap_iq_df.columns:
        if col == "baseline":
            continue

        shap_vals = shap_iq_df[col].values
        feat_vals = feat_df[col].values

        pos_mask = shap_vals > 0
        neg_mask = shap_vals < 0

        shap_stats = _compute_shap_side_stats(shap_vals, pos_mask, neg_mask)
        fvalues = _scale_column_with_nans(feat_vals, norm=True)
        binned = _bin_feature_values(fvalues, k=2)

        mixing_coef_pos = _mixing_coef_for_side(binned, pos_mask)
        mixing_coef_neg = _mixing_coef_for_side(binned, neg_mask)

        score = (
            mixing_coef_neg
            * mixing_coef_pos
            * (
                shap_stats["pos_mean_abs"] * shap_stats["pos_abs_sum"]
                + shap_stats["neg_mean_abs"] * shap_stats["neg_abs_sum"]
            )
        )

        feat_scores.append(
            {
                "feature": col,
                "mixing_coef_pos": mixing_coef_pos,
                "mixing_coef_neg": mixing_coef_neg,
                "pos_abs_shap": shap_stats["pos_abs_sum"],
                "neg_abs_shap": shap_stats["neg_abs_sum"],
                "pos_mean_shap": shap_stats["pos_mean_abs"],
                "neg_mean_shap": shap_stats["neg_mean_abs"],
                "final_shap_score": score,
            }
        )

    score_df = pd.DataFrame(feat_scores)
    score_df = score_df.sort_values("final_shap_score", ascending=False)
    return score_df


def select_features(feature_scores: pd.Series) -> List[str]:
    """Select the top k features based on their scores.

    Do a 1-D clustering to the scores with FisherJenks algorithm and select the features
    belonging to the top-k clusters.
    """
    k = 5 if len(feature_scores) > 5 else 3
    top_k = 3 if len(feature_scores) > 5 else 2

    score_bins = mapclassify.FisherJenks(feature_scores, k=k)
    score_cls = score_bins.yb
    top_bins = list(range(score_bins.k))[::-1][:top_k]
    top_scores = feature_scores[np.isin(score_cls, top_bins)].sort_values(
        ascending=False
    )

    return top_scores
