"""Run CIN2 classification and SHAP-based feature ranking."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

import shap
import shapiq
import tabpfn
from shapiq.interaction_values import InteractionValues

from src.classification.autogluon_wrapper import AutoGluonSklearnWrapper
from src.classification.feature_maps import (
    FEATURE_LABELS,
    FEATURE_THEMES,
    available_feature_themes,
)
from src.classification.metrics import (
    compute_metrics_table,
    get_conf_mat,
    get_prc_curve,
    get_roc_curve,
)
from src.classification.shap_heuristic import shap_scoring_heuristic_
from src.plot.beeswarm import beeswarm_plot
from src.plot.metrics import (
    plot_conf_mat,
    plot_prc_curve,
    plot_roc_curve,
    plot_summary_roc_curve,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run CIN2 classification feature ranking from a single input dataset.",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        type=Path,
        help="Path to the input dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./classification_results"),
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--csv-sep",
        type=str,
        default=",",
        help="Separator for CSV files (ignored for parquet).",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default="outcome",
        help="Name of the outcome column.",
    )
    parser.add_argument(
        "--patient-id-col",
        type=str,
        default="dg_sample_number",
        help="Name of the patient/sample identifier column.",
    )
    parser.add_argument(
        "--positive-class",
        type=str,
        default="reg",
        help="Label name for the positive class.",
    )
    parser.add_argument(
        "--negative-class",
        type=str,
        default="non-reg",
        help="Label name for the negative class.",
    )
    parser.add_argument(
        "--merge-part-reg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map 'part-reg' to the negative class.",
    )
    parser.add_argument(
        "--dropna-outcome",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with missing outcomes.",
    )
    parser.add_argument(
        "--fillna-value",
        type=float,
        default=0.0,
        help="Value used to fill missing feature values.",
    )
    parser.add_argument(
        "--apply-feature-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rename raw feature columns to manuscript labels.",
    )
    parser.add_argument(
        "--feature-themes",
        nargs="+",
        default=["clin_feats", "neo_nuc_feats", "tissue_feats", "immune_feats"],
        help=(
            "Feature theme(s) to run. Use 'all' to run all themes. "
            f"Available: {', '.join(available_feature_themes())}."
        ),
    )
    parser.add_argument(
        "--allow-missing-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow missing feature columns (they will be dropped).",
    )
    parser.add_argument(
        "--pap-smear-col",
        type=str,
        default="Entry Pap smear result",
        help="Column name for Pap smear result (if present).",
    )
    parser.add_argument(
        "--recode-pap-smear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recode Pap smear result into low/high grade bins.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of CV splits per repeat.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=4,
        help="Number of CV repeats.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30,
        help="Autogluon training time limit (seconds).",
    )
    parser.add_argument(
        "--infer-limit",
        type=float,
        default=0.005,
        help="Autogluon inference time limit (seconds).",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="f1_weighted",
        help="Autogluon evaluation metric.",
    )
    parser.add_argument(
        "--num-gpus",
        type=str,
        default="auto",
        help="Number of GPUs for Autogluon ('auto', '0', '1', etc.).",
    )
    parser.add_argument(
        "--compute-shap-for-all-folds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute SHAP for all folds (not only non-random classifiers).",
    )
    parser.add_argument(
        "--shap-index",
        type=str,
        default="SV",
        help="SHAP index type (e.g., 'SV', 'FSII').",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=1,
        help="Maximum interaction order for SHAP-IQ values.",
    )
    parser.add_argument(
        "--shap-budget",
        type=int,
        default=256,
        help="SHAP budget.",
    )
    parser.add_argument(
        "--verbose-shap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print verbose output during SHAP computation.",
    )
    parser.add_argument(
        "--compute-iq",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute SHAP-IQ values in addition to SHAP values.",
    )
    parser.add_argument(
        "--use-kernel-shap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KernelSHAP instead of SHAP-IQ Explainer.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="autogluon_run",
        help="Name for the model run (used in output paths).",
    )
    parser.add_argument(
        "--results-pkl",
        type=str,
        default="results.pkl",
        help="Filename for saving serialized results within output-dir.",
    )
    return parser.parse_args()


def load_dataset(path: Path, csv_sep: str) -> pd.DataFrame:
    """Load a dataset from CSV or Parquet.

    Args:
        path (Path): Path to the dataset file.
        csv_sep (str): CSV delimiter used for .csv/.tsv files.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep=csv_sep)
    raise ValueError("Input data must be a .csv/.tsv or .parquet file.")


def apply_feature_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw feature names to manuscript labels.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with renamed columns.
    """
    rename_map = {k: v for k, v in FEATURE_LABELS.items() if k in df.columns}
    if not rename_map:
        return df
    mapped = df.rename(columns=rename_map)
    duplicated = mapped.columns[mapped.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Duplicate columns detected after feature mapping. "
            "Either drop the raw columns or run with --no-apply-feature-mapping. "
            f"Duplicates: {duplicated}"
        )
    return mapped


def select_feature_themes(requested: Iterable[str]) -> dict[str, list[str]]:
    """Select feature themes from CLI input.

    Args:
        requested (Iterable[str]): Theme names or "all".

    Returns:
        dict[str, list[str]]: Mapping of theme name to feature list.
    """
    themes = [t.strip() for t in requested]
    if "all" in themes:
        return FEATURE_THEMES
    unknown = [t for t in themes if t not in FEATURE_THEMES]
    if unknown:
        raise ValueError(
            f"Unknown feature theme(s): {unknown}. Available: {available_feature_themes()}"
        )
    return {name: FEATURE_THEMES[name] for name in themes}


def recode_pap_smear(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Recode Pap smear results into low/high grades.

    Args:
        df (pd.DataFrame): Input dataset.
        col (str): Column name for Pap smear result.

    Returns:
        pd.DataFrame: Updated dataset.
    """
    if col not in df.columns:
        return df
    df = df.copy()
    df.loc[df[col].isin([1.0, 2.0, 5.0]), col] = 1.0  # low grade
    df.loc[df[col].isin([3.0, 4.0, 6.0]), col] = 2.0  # high grade
    df.loc[df[col] == 99.0, col] = np.nan
    return df


def coerce_class_labels(
    labels: pd.Series,
    positive: str,
    negative: str,
) -> tuple[Any, Any]:
    """Coerce class labels to match dataset types.

    Args:
        labels (pd.Series): Outcome labels.
        positive (str): Positive class label.
        negative (str): Negative class label.

    Returns:
        tuple[Any, Any]: Coerced (positive, negative) labels.
    """
    unique_vals = labels.dropna().unique().tolist()

    def _coerce_label(label: str) -> Any:
        if label in unique_vals:
            return label
        for cast in (int, float):
            try:
                casted = cast(label)
            except (TypeError, ValueError):
                continue
            if casted in unique_vals:
                return casted
        return label

    positive = _coerce_label(positive)
    negative = _coerce_label(negative)

    if positive not in unique_vals or negative not in unique_vals:
        raise ValueError(
            "Outcome labels must include both positive and negative classes. "
            f"Found: {sorted(unique_vals)}"
        )
    return positive, negative


def encode_labels(labels: pd.Series, positive: Any, negative: Any) -> np.ndarray:
    """Encode labels to 0/1 based on positive and negative classes.

    Args:
        labels (pd.Series): Outcome labels.
        positive (Any): Positive class label.
        negative (Any): Negative class label.

    Returns:
        np.ndarray: Encoded labels as 0/1.
    """
    return labels.map({negative: 0, positive: 1}).to_numpy()


def compute_shap(
    X: pd.DataFrame | np.ndarray,
    background_data: pd.DataFrame | np.ndarray,
    clf: Any,
    index: str = "SII",
    max_order: int = 2,
    imputer: str = "marginal",
    budget: int = 256,
    verbose: bool = True,
    random_state: int = 42,
    use_kernel_shap: bool = False,
    **kwargs: Any,
) -> list[InteractionValues]:
    """Compute SHAP or SHAP-IQ values for a model.

    Args:
        X (pd.DataFrame | np.ndarray): Samples to explain.
        background_data (pd.DataFrame | np.ndarray): Background data for explainer.
        clf (Any): Trained classifier.
        index (str): SHAP index type.
        max_order (int): Maximum interaction order.
        imputer (str): Imputer strategy (unused for TabPFN).
        budget (int): SHAP computation budget.
        verbose (bool): Verbosity flag.
        random_state (int): Random seed.
        use_kernel_shap (bool): Whether to use KernelSHAP.
        **kwargs (Any): Additional parameters.

    Returns:
        list[InteractionValues]: SHAP interaction values per sample.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(background_data, pd.DataFrame):
        background_data = background_data.values

    if isinstance(clf, tabpfn.TabPFNClassifier):
        try:
            kwargs = {"labels": kwargs["labels"]}
        except KeyError as exc:
            raise ValueError(
                "For TabPFNClassifier, 'labels' must be provided in kwargs."
            ) from exc
        explainer = shapiq.TabularExplainer(
            clf,
            data=background_data,
            index=index,
            max_order=max_order,
            imputer="baseline",
        )
    elif use_kernel_shap:
        explainer = shap.KernelExplainer(
            clf.predict_proba,
            data=background_data,
        )
        shap_values = explainer.shap_values(X, nsamples=500)
        baseline_value = explainer.expected_value[1]

        interaction_values = []
        interaction_lookup = {tuple(): 0}
        for i in range(X.shape[1]):
            interaction_lookup[(i,)] = i + 1

        for sv in shap_values[..., 1]:
            sv = np.insert(sv, 0, baseline_value)
            interaction_values.append(
                InteractionValues(
                    values=sv,
                    baseline_value=baseline_value,
                    interaction_lookup=interaction_lookup,
                    index="SV",
                    max_order=1,
                    min_order=0,
                    n_players=X.shape[1],
                )
            )
        return interaction_values
    else:
        kwargs = {"imputer": "baseline"}
        explainer = shapiq.Explainer(
            model=clf,
            data=background_data,
            index=index,
            max_order=max_order,
            **kwargs,
        )

    shap_vals = explainer.explain_X(
        X, budget=budget, verbose=verbose, random_state=random_state, n_jobs=None
    )
    return shap_vals


def _positive_class_index(clf: AutoGluonSklearnWrapper, positive_class: str) -> int:
    """Get index of the positive class in AutoGluon class labels.

    Args:
        clf (AutoGluonSklearnWrapper): Trained classifier.
        positive_class (str): Positive class label.

    Returns:
        int: Index of the positive class.
    """
    try:
        class_labels = list(clf.predictor.class_labels)
    except Exception:
        return 1
    if positive_class in class_labels:
        return class_labels.index(positive_class)
    return 1


def _align_probabilities(
    probs: np.ndarray,
    pos_index: int,
) -> np.ndarray:
    """Align class probabilities so column 1 is positive class.

    Args:
        probs (np.ndarray): Raw probability array.
        pos_index (int): Index of positive class in probs.

    Returns:
        np.ndarray: Reordered probabilities with [neg, pos] columns.
    """
    if probs.shape[1] != 2:
        raise ValueError("Expected binary classification probabilities with 2 columns.")
    neg_index = 1 - pos_index
    return np.column_stack([probs[:, neg_index], probs[:, pos_index]])


def rank_features(
    data: pd.DataFrame,
    feature_themes: dict[str, list[str]],
    model_name: str,
    save_path: Path,
    outcome_col: str,
    patient_id_col: str,
    positive_class: str,
    negative_class: str,
    n_repeats: int = 4,
    n_splits: int = 4,
    random_state: int = 42,
    compute_shap_for_all_folds: bool = False,
    shap_index: str = "SV",
    max_order: int = 1,
    verbose_shap: bool = True,
    compute_iq: bool = False,
    shap_budget: int = 256,
    use_kernel_shap: bool = True,
    eval_metric: str = "f1_weighted",
    time_limit: float = 30,
    infer_limit: float = 0.005,
    num_gpus: str = "auto",
) -> dict[str, Any]:
    """Train models and rank features using SHAP scores.

    Args:
        data (pd.DataFrame): Input dataset with features and outcome.
        feature_themes (dict[str, list[str]]): Feature sets to evaluate.
        model_name (str): Model run name.
        save_path (Path): Output directory.
        outcome_col (str): Outcome column name.
        patient_id_col (str): Patient/sample identifier column name.
        positive_class (str): Positive class label.
        negative_class (str): Negative class label.
        n_repeats (int): CV repeats.
        n_splits (int): CV splits.
        random_state (int): Random seed.
        compute_shap_for_all_folds (bool): Compute SHAP for all folds.
        shap_index (str): SHAP index type.
        max_order (int): Maximum interaction order.
        verbose_shap (bool): Verbosity flag for SHAP.
        compute_iq (bool): Whether to compute SHAP-IQ.
        shap_budget (int): SHAP budget.
        use_kernel_shap (bool): Use KernelSHAP.
        eval_metric (str): AutoGluon evaluation metric.
        time_limit (float): Training time limit (seconds).
        infer_limit (float): Inference time limit (seconds).
        num_gpus (str): GPU count for training.

    Returns:
        dict[str, Any]: Results including SHAP scores, metrics, and plots.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    in_data = data.copy()
    test_sets_all: dict[str, list[pd.DataFrame]] = {}
    shap_scores: dict[str, pd.DataFrame] = {}
    shap_vals_all: dict[str, list] = {}
    shap_iq_vals_all: dict[str, list] = {}
    metrics_all: dict[str, pd.DataFrame] = {}
    fpr_tpr_dfs_all: dict[str, list[pd.DataFrame]] = {}
    model_names_all: dict[str, list[str]] = {}
    test_labels_all: dict[str, list[np.ndarray]] = {}
    conf_mats: dict[str, pd.DataFrame] = {}
    conf_mats_all: dict[str, list[pd.DataFrame]] = {}
    samples_test_all: dict[str, list[list]] = {}
    feat_importances_all: dict[str, list] = {}

    for feat_theme, feats in feature_themes.items():
        print(f"Fitting model for feature set: {feat_theme} ...")

        conf_mat_sum = pd.DataFrame(
            np.zeros((2, 2)), index=["non-reg", "reg"], columns=["non-reg", "reg"]
        )
        shap_values: list[InteractionValues] = []
        shap_iq_values: list[InteractionValues] = []
        test_sets: list[pd.DataFrame] = []
        test_labels: list[np.ndarray] = []
        test_samples: list[list] = []
        metrics_dfs: list[pd.Series] = []
        fpr_tpr_dfs: list[pd.DataFrame] = []
        model_names: list[str] = []
        conf_matrices: list[pd.DataFrame] = []
        feat_importances: list = []

        skf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        for i, (train_ix, test_ix) in enumerate(
            skf.split(in_data[feats], in_data[outcome_col])
        ):
            (save_path / model_name / feat_theme).mkdir(exist_ok=True, parents=True)
            X_train = in_data[feats].iloc[train_ix]
            y_train = in_data[outcome_col].iloc[train_ix]
            X_test = in_data[feats].iloc[test_ix]
            y_test = in_data[outcome_col].iloc[test_ix]

            y_test_enc = encode_labels(y_test, positive_class, negative_class)
            test_labels.append(y_test_enc)
            samples_test = in_data[patient_id_col].iloc[test_ix].to_list()

            clf = AutoGluonSklearnWrapper(
                feats,
                save_path / model_name / feat_theme / f"fold_{i}",
                target_class=positive_class,
                label=outcome_col,
                eval_metric=eval_metric,
                y_test=y_test,
                X_test=X_test,
            )
            clf.fit(
                X_train,
                y_train,
                time_limit=time_limit,
                infer_limit=infer_limit,
                num_gpus=num_gpus,
            )

            no_skill = len(y_test_enc[y_test_enc == 1]) / len(y_test_enc)
            pred_labels = clf.predict(X_test)
            pred = np.array([1 if v == positive_class else 0 for v in pred_labels])
            probs = clf.predict_proba(X_test)
            pos_idx = _positive_class_index(clf, positive_class)
            probs_aligned = _align_probabilities(probs, pos_idx)
            pos_probs = probs_aligned[:, 1]

            metrics_df = compute_metrics_table(
                y_true=y_test_enc,
                predictions=pred,
                prediction_probabilities=probs_aligned,
            )

            auprc, best_ix, best_thresh, prc_df = get_prc_curve(
                y=y_test_enc,
                yhat=pos_probs,
            )
            auroc, best_ix, best_thresh, roc_df = get_roc_curve(
                y=y_test_enc,
                yhat=pos_probs,
            )
            conf_mat = get_conf_mat(
                y=y_test_enc,
                yhat=pred,
                best_f1_thresh=best_thresh,
            )

            gg = "autogluon"
            plot_prc_curve(
                prc_df,
                no_skill,
                best_ix,
                model_name + f"_{gg}" + f"_fold{i}",
                save_path=save_path / model_name / feat_theme / "prc_curves",
            )
            plot_roc_curve(
                roc_df,
                model_name + f"_{gg}" + f"_fold{i}",
                save_path=save_path / model_name / feat_theme / "roc_curves",
            )
            plot_conf_mat(
                conf_mat,
                model_name + f"_{gg}" + f"_fold{i}",
                save_path=save_path / model_name / feat_theme / "conf_mats",
            )

            conf_mat_sum += conf_mat

            is_not_random_classifier = (
                metrics_df["balanced_accuracy"] > 0.5
                and metrics_df["f1_weighted"] > 0.5
                and metrics_df["auprc_above_baseline"] > 0
                and metrics_df["mcc"] > 0
            )

            if is_not_random_classifier and not compute_shap_for_all_folds:
                shap_vals = compute_shap(
                    X=X_test,
                    background_data=X_train[y_train == negative_class],
                    clf=clf,
                    feature_names=feats,
                    index=shap_index,
                    max_order=max_order,
                    budget=shap_budget,
                    verbose=verbose_shap,
                    labels=y_test_enc,
                    use_kernel_shap=use_kernel_shap,
                )
                if compute_iq:
                    shap_iq_vals = compute_shap(
                        X=X_test,
                        background_data=X_train[y_train == negative_class],
                        clf=clf,
                        feature_names=feats,
                        index="SII",
                        max_order=2,
                        budget=shap_budget,
                        verbose=verbose_shap,
                        labels=y_test_enc,
                        use_kernel_shap=False,
                    )
                    shap_iq_values.extend(shap_iq_vals)

                shap_values.extend(shap_vals)
                test_sets.append(X_test)
                test_samples.append(samples_test)
                metrics_dfs.append(metrics_df)
                fpr_tpr_dfs.append(roc_df)
                model_names.append(model_name + f"_{gg}" + f"_fold{i}")
                test_samples.append(samples_test)
                conf_matrices.append(conf_mat)
            elif compute_shap_for_all_folds:
                shap_vals = compute_shap(
                    X=X_test,
                    background_data=X_train[y_train == negative_class],
                    clf=clf,
                    feature_names=feats,
                    index=shap_index,
                    max_order=max_order,
                    budget=shap_budget,
                    verbose=verbose_shap,
                    labels=y_test_enc,
                    use_kernel_shap=use_kernel_shap,
                )
                shap_values.extend(shap_vals)
                test_sets.append(X_test)
                test_samples.append(samples_test)
                metrics_dfs.append(metrics_df)
                fpr_tpr_dfs.append(roc_df)
                model_names.append(model_name + f"_{gg}" + f"_fold{i}")
                test_samples.append(samples_test)
                conf_matrices.append(conf_mat)

        if len(shap_values) == 0:
            continue

        shap_scores[feat_theme] = {}
        score_df = shap_scoring_heuristic_(shap_values, feat_df=pd.concat(test_sets))
        shap_scores[feat_theme] = score_df

        metrics = pd.DataFrame(metrics_dfs)
        metrics.to_csv(
            save_path / model_name / feat_theme / "perf_metrics.csv", index=False
        )
        shap_vals_all[feat_theme] = shap_values
        shap_iq_vals_all[feat_theme] = shap_iq_values
        metrics_all[feat_theme] = metrics
        test_sets_all[feat_theme] = test_sets
        samples_test_all[feat_theme] = test_samples
        fpr_tpr_dfs_all[feat_theme] = fpr_tpr_dfs
        model_names_all[feat_theme] = model_names
        test_labels_all[feat_theme] = test_labels
        conf_mats[feat_theme] = conf_mat_sum
        conf_mats_all[feat_theme] = conf_matrices
        feat_importances_all[feat_theme] = feat_importances

        if feat_theme not in fpr_tpr_dfs_all.keys():
            continue

        plot_summary_roc_curve(
            fpr_tpr_dfs=fpr_tpr_dfs_all[feat_theme],
            model_names=model_names_all[feat_theme],
            metrics_df=metrics_all[feat_theme],
            plot_all=False,
            save_path=save_path / model_name / feat_theme,
        )
        plot_conf_mat(
            conf_mats[feat_theme],
            model_name + f"_{feat_theme}_all_folds",
            save_path=save_path / model_name / feat_theme,
        )

        beeswarm_plot(
            interaction_values_list=shap_vals_all[feat_theme],
            data=pd.concat(test_sets_all[feat_theme]),
            feature_names=test_sets_all[feat_theme][0].columns.to_list(),
            max_display=30,
            abbreviate=False,
            dot_size=30,
            row_height=1.0,
            show=False,
            x_label="SHAP Value (impact on model output)",
            label_fontsize=26,
            x_tick_size=24,
            y_tick_size=34,
            rank_feats=True,
            save_path=save_path / model_name / feat_theme / "shap_beeswarm.png",
        )

    return {
        "shap_scores": shap_scores,
        "shap_vals": shap_vals_all,
        "shap_iq_vals": shap_iq_vals_all,
        "metrics": metrics_all,
        "test_sets": test_sets_all,
        "fpr_tpr_dfs": fpr_tpr_dfs_all,
        "model_names": model_names_all,
        "test_labels": test_labels_all,
        "samples_test": samples_test_all,
        "conf_mats": conf_mats_all,
    }


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    allow_missing: bool,
) -> list[str]:
    """Validate required columns in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        required_cols (Iterable[str]): Required column names.
        allow_missing (bool): Whether to allow missing columns.

    Returns:
        list[str]: Missing column names.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing and not allow_missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Provide the columns or use --allow-missing-features."
        )
    return missing


def main() -> None:
    """Run classification and feature ranking from the CLI.

    Returns:
        None: Writes outputs to disk.
    """
    args = parse_args()

    df = load_dataset(args.input_data, args.csv_sep)

    if args.apply_feature_mapping:
        df = apply_feature_mapping(df)

    if args.merge_part_reg and args.outcome_col in df.columns:
        df.loc[df[args.outcome_col] == "part-reg", args.outcome_col] = (
            args.negative_class
        )

    if args.dropna_outcome and args.outcome_col in df.columns:
        df = df.loc[~df[args.outcome_col].isna()].copy()

    if args.recode_pap_smear:
        df = recode_pap_smear(df, args.pap_smear_col)

    feature_themes = select_feature_themes(args.feature_themes)

    required_columns = [args.patient_id_col, args.outcome_col]
    for theme_features in feature_themes.values():
        required_columns.extend(theme_features)

    missing = validate_required_columns(
        df, required_columns, args.allow_missing_features
    )
    if missing:
        print("Warning: missing features will be dropped:")
        for col in missing:
            print(f"  - {col}")

    # Drop missing features if allowed
    for theme, feats in feature_themes.items():
        feature_themes[theme] = [f for f in feats if f in df.columns]

    empty_themes = [t for t, feats in feature_themes.items() if len(feats) == 0]
    if empty_themes:
        for theme in empty_themes:
            print(f"Warning: no available features for theme '{theme}', skipping.")
            feature_themes.pop(theme, None)

    if not feature_themes:
        raise ValueError("No feature themes remain after filtering missing columns.")

    df = df.fillna(args.fillna_value)

    if args.outcome_col not in df.columns:
        raise ValueError(f"Outcome column not found: {args.outcome_col}")
    if args.patient_id_col not in df.columns:
        raise ValueError(f"Patient ID column not found: {args.patient_id_col}")

    save_path = args.output_dir
    save_path.mkdir(parents=True, exist_ok=True)

    positive_class, negative_class = coerce_class_labels(
        df[args.outcome_col], args.positive_class, args.negative_class
    )

    print("🚀 Starting Autogluon Feature Ranking...")
    results = rank_features(
        data=df,
        feature_themes=feature_themes,
        model_name=args.model_name,
        save_path=save_path,
        outcome_col=args.outcome_col,
        patient_id_col=args.patient_id_col,
        positive_class=positive_class,
        negative_class=negative_class,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        random_state=args.random_state,
        compute_shap_for_all_folds=args.compute_shap_for_all_folds,
        shap_index=args.shap_index,
        max_order=args.max_order,
        verbose_shap=args.verbose_shap,
        compute_iq=args.compute_iq,
        shap_budget=args.shap_budget,
        use_kernel_shap=args.use_kernel_shap,
        eval_metric=args.eval_metric,
        time_limit=args.time_limit,
        infer_limit=args.infer_limit,
        num_gpus=args.num_gpus,
    )

    results_path = save_path / args.results_pkl
    with results_path.open("wb") as f:
        pickle.dump(results, f)

    print(f"✅ Completed. Results saved to {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
