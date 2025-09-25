import numpy as np
import pandas as pd

from src.classification.utils import shap2df


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


def drop_min_max(df: pd.DataFrame) -> pd.DataFrame:
    maxidx = df.loc[:, np.log(df.max().values) > 0].idxmax()  # .to_list()
    minidx = df.loc[:, np.log(np.abs(df.min()).values) > 0].idxmin()

    return df.drop(index=pd.concat((maxidx, minidx)).unique())


def compute_mixing_coef(
    feat_vals: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray
) -> float:
    """Compute the feature separation coefficient.

    Computes the feature separation coefficient as the absolute difference between
    the median feature value on the positive and negative side of the shap origin.
    """
    # truncate the feature values between 5th and 95th percentile like in shap beeswarm
    feat_vals_scaled = scale_column(feat_vals, norm=True)

    # get the feature values on positive and negative side of the shap origin
    feats_on_pos_side = feat_vals_scaled[pos_mask]
    feats_on_neg_side = feat_vals_scaled[neg_mask]

    # compute the median feature value on positive and negative side of the shap origin
    # the feature separation coef is the absolute difference between the two medians
    pos_feat_median = np.median(feats_on_pos_side) if np.any(pos_mask) else 0
    neg_feat_median = np.median(feats_on_neg_side) if np.any(neg_mask) else 0

    return np.abs(pos_feat_median - neg_feat_median)


def compute_95th_abs_shap_coef(
    shap_vals: np.ndarray, global_min: float, global_max: float
) -> float:
    """Compute the absolute 95th percentile shap value.

    Computes the normalized 95th percentile absolute shap value. More robust to outliers
    than max and preserves more magnitude information than mean.

    Note:
        Normalizes to global min and max of mean abs shap values across all features.
    """
    # compute the mean absolute shap value and scale to global min and max
    mean_abs_coef = np.nanpercentile(np.abs(shap_vals), 95)
    mean_abs_coef = (mean_abs_coef - global_min) / (global_max - global_min + 1e-8)
    return mean_abs_coef


def compute_separation_coef(
    shap_vals: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray
) -> float:
    """Compute the shap value separation coefficient.

    The shap value separation coef is the harmonic mean of the proportion of samples
    on the positive and negative side of the shap origin this is high when both proportions
    are high and low when one of the proportions is low.
    """
    pos_prop = np.sum(pos_mask) / len(shap_vals)
    neg_prop = np.sum(neg_mask) / len(shap_vals)
    shap_separation_coef = (
        2 * pos_prop * neg_prop / (pos_prop + neg_prop)
        if pos_prop > 0 and neg_prop > 0
        else 0
    )

    return shap_separation_coef


def shap_scoring_heuristic(
    interaction_values_list: pd.DataFrame,
    feat_df: pd.DataFrame,
    weights: list[float] = np.array([1.0, 1.0, 1.0]),
) -> np.ndarray | np.ndarray:
    """Score the features based on a heuristic of shap values and feature values.

    This heuristic favors features that have high absolute shap values and a good
    separation between the positive and negative shap values. A good separation means
    that there are many samples on both sides of the shap origin and that the feature
    values on the positive and negative side of the shap origin are different.

    The score is computed as the sum of:
    - 95th percentile of the absolute shap values (normalized to global min and max) [0-1].
    - Shap value separation coef: harmonic mean of the proportion of samples on the positive
      and negative side of the shap origin [0-1].
    - Feature separation coef: absolute difference between the median feature value on the
      positive and negative side of the shap origin [0-1].

    Parameters:
        interaction_values_list (list[InteractionValues]):
            List of interaction values for each sample.
        feat_df (pd.DataFrame):
            Dataframe containing the feature values.
        weights (list[float], optional):
            Weights for the three score components. Order: [mean_abs, shap_sep, feat_sep].
            Defaults to [1.0, 1.0, 1.0].

    Returns:
        np.ndarray | np.ndarray:
            List of the top-k features in order. Scores of the top-k features.
    """
    shap_iq_df = pd.concat(
        [shap2df(shap, feat_df.columns.to_list()) for shap in interaction_values_list]
    )
    shap_iq_df = drop_min_max(shap_iq_df)

    # get global min and max of mean abs shap values across all features for scaling
    global_min = shap_iq_df.loc[:, shap_iq_df.columns != "baseline"].min().min()
    global_max = shap_iq_df.loc[:, shap_iq_df.columns != "baseline"].max().max()

    scores = {}
    for col in shap_iq_df.columns:
        if col == "baseline":
            continue

        shap_vals = shap_iq_df[col].values
        feat_vals = feat_df[col].values

        # samples on positive and negative side of the shap origin
        neg_mask = shap_vals < 0
        pos_mask = shap_vals > 0

        # compute the coefficients
        feat_separation_coef = compute_mixing_coef(feat_vals, pos_mask, neg_mask)
        mean_abs_coef = compute_95th_abs_shap_coef(shap_vals, global_min, global_max)
        shap_separation_coef = compute_separation_coef(shap_vals, pos_mask, neg_mask)

        # compute the final score as the weighted sum of the three coefficients
        score_values = np.array(
            [mean_abs_coef, shap_separation_coef, feat_separation_coef]
        )
        score = np.sum(score_values * weights)
        scores[col] = score

    sorted_scores = np.array(sorted(scores.values(), reverse=True))
    sorted_feats = np.array(sorted(scores, key=scores.get, reverse=True))
    return sorted_feats, sorted_scores
