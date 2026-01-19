"""Compute feature interaction matrices using FACET for a user-defined feature list."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.feature_maps import FEATURE_LABELS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Quantify feature interactions from a user-provided feature list using FACET."
        )
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to input dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="List of feature names to analyze.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=None,
        help="Path to a file with one feature name per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./interaction_results"),
        help="Directory to save interaction outputs.",
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
        help="Outcome column name.",
    )
    parser.add_argument(
        "--apply-feature-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rename raw feature columns to manuscript labels.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of CV splits (FACET model selection).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=4,
        help="Number of CV repeats (FACET model selection).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=8,
        help="Number of random seeds to sample.",
    )
    parser.add_argument(
        "--min-synergy",
        type=float,
        default=5.0,
        help="Minimum synergy percentage to report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top synergy pairs to save if none exceed the threshold.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for model selection.",
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


def load_features(features: list[str] | None, features_file: Path | None) -> list[str]:
    """Load features from CLI list and/or file.

    Args:
        features (list[str] | None): Features provided via --features.
        features_file (Path | None): File path with one feature per line.

    Returns:
        list[str]: Deduplicated list of feature names in order.
    """
    if features_file is not None:
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        file_features = [
            line.strip()
            for line in features_file.read_text().splitlines()
            if line.strip()
        ]
    else:
        file_features = []

    list_features = features or []
    all_features = [*file_features, *list_features]
    if not all_features:
        raise ValueError("Provide features via --features or --features-file.")

    # preserve order while removing duplicates
    seen = set()
    deduped = []
    for feat in all_features:
        if feat not in seen:
            seen.add(feat)
            deduped.append(feat)

    return deduped


def compute_interactions_facet(
    df: pd.DataFrame,
    selected_features: list[str],
    outcome_col: str,
    n_splits: int,
    n_repeats: int,
    n_seeds: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute redundancy and synergy matrices using FACET.

    Args:
        df (pd.DataFrame): Input dataset.
        selected_features (list[str]): Feature names to analyze.
        outcome_col (str): Outcome column name.
        n_splits (int): Number of CV splits for model selection.
        n_repeats (int): Number of CV repeats for model selection.
        n_seeds (int): Number of random seeds to sample.
        n_jobs (int): Number of parallel jobs.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Redundancy matrix, synergy matrix.
    """
    try:
        from facet.data import Sample
        from facet.inspection import LearnerInspector
        from facet.selection import LearnerSelector, ParameterSpace
        from scipy import stats
        from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
        from sklearndf.classification import (
            ExtraTreesClassifierDF,
            RandomForestClassifierDF,
        )
        from sklearndf.pipeline import ClassifierPipelineDF
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "FACET dependencies not installed. Install: pip install gamma-facet sklearndf"
        ) from exc

    cin2 = Sample(
        observations=df,
        feature_names=selected_features,
        target_name=outcome_col,
    )

    rforest_clf = ClassifierPipelineDF(
        classifier=RandomForestClassifierDF(random_state=42)
    )
    etrees_clf = ClassifierPipelineDF(
        classifier=ExtraTreesClassifierDF(random_state=42)
    )

    rforest_ps = ParameterSpace(rforest_clf)
    rforest_ps.classifier.max_depth = stats.randint(4, 10)
    rforest_ps.classifier.min_samples_leaf = stats.zipfian(a=1, n=12, loc=7)
    rforest_ps.classifier.n_estimators = stats.zipfian(a=1 / 2, n=380, loc=20)

    etrees_ps = ParameterSpace(etrees_clf)
    etrees_ps.classifier.max_depth = stats.randint(4, 10)
    etrees_ps.classifier.min_samples_leaf = stats.zipfian(a=1, n=12, loc=7)
    etrees_ps.classifier.n_estimators = stats.zipfian(a=1 / 2, n=380, loc=20)

    models = []
    seeds = list(range(42, 42 + n_seeds))
    for seed in seeds:
        selector = LearnerSelector(
            searcher_type=RandomizedSearchCV,
            parameter_space=[rforest_ps, etrees_ps],
            cv=RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=seed
            ),
            n_jobs=n_jobs,
            scoring="roc_auc",
            random_state=seed,
        ).fit(cin2.keep(feature_names=selected_features))
        models.append(selector.best_estimator_)

    redundancies = []
    synergies = []
    for model in models:
        inspector = LearnerInspector(
            model=model,
            n_jobs=n_jobs,
            verbose=False,
        ).fit(cin2.keep(feature_names=selected_features))
        redundancies.append(inspector.feature_redundancy_matrix(clustered=False).values)
        synergies.append(inspector.feature_synergy_matrix(clustered=False).values)

    redundancies = np.array(redundancies)
    synergies = np.array(synergies)

    red_mean = redundancies.mean(axis=0)
    syn_mean = synergies.mean(axis=0)

    red_df = pd.DataFrame(red_mean, index=selected_features, columns=selected_features)
    syn_df = pd.DataFrame(syn_mean, index=selected_features, columns=selected_features)

    return red_df, syn_df


def main() -> None:
    """Run feature interaction analysis and save outputs.

    Returns:
        None: Writes CSV outputs to the output directory.
    """
    args = parse_args()

    df = load_dataset(args.input_data, args.csv_sep)
    selected_features = load_features(args.features, args.features_file)
    if args.apply_feature_mapping:
        df = apply_feature_mapping(df)
        selected_features = [FEATURE_LABELS.get(f, f) for f in selected_features]

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(
            "Missing selected features in dataset: " + ", ".join(sorted(missing))
        )

    if args.outcome_col not in df.columns:
        raise ValueError(f"Outcome column not found: {args.outcome_col}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    red_df, syn_df = compute_interactions_facet(
        df=df,
        selected_features=selected_features,
        outcome_col=args.outcome_col,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_seeds=args.n_seeds,
        n_jobs=args.n_jobs,
    )

    red_df.to_csv(args.output_dir / "redundancy_matrix.csv")
    syn_df.to_csv(args.output_dir / "synergy_matrix.csv")

    # summarize strong synergies
    all_pairs = []
    for i, feat_a in enumerate(syn_df.index):
        for j, feat_b in enumerate(syn_df.columns):
            if j <= i:
                continue
            score = syn_df.iloc[i, j] * 100
            all_pairs.append(
                {"feature_a": feat_a, "feature_b": feat_b, "synergy_pct": score}
            )

    top_df = pd.DataFrame(all_pairs).sort_values("synergy_pct", ascending=False)
    filtered = top_df[top_df["synergy_pct"] >= args.min_synergy]
    if filtered.empty:
        filtered = top_df.head(max(args.top_k, 1))

    filtered.to_csv(args.output_dir / "top_synergies.csv", index=False)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
