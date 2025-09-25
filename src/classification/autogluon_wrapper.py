import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


class AutoGluonSklearnWrapper:
    def __init__(
        self,
        feature_names: Sequence[str],
        save_path: str = None,
        target_class: str = "reg",
        label: str = "Outcome",
        eval_metric: str = "f1_weighted",
        warnings: bool = True,
    ) -> None:
        """Simple wrapper to use autogluon autoML with sklearn api.

        Parameters:
            feature_names (Sequence[str]):
                Names of the features in correct order.
            save_path (str):
                Path to save the autogluon models.
            target_class (str):
                Name of the positive class for classification tasks.
            label (str):
                Name of the outcome label column.
            eval_metric (str):
                Metric to optimize during training.
            warnings (bool):
                Whether to show warnings.
        """
        self.feature_names = feature_names
        self.save_path = Path(save_path)
        self.target_class = target_class
        self.label = label
        self.warnings = warnings
        self.predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric,
            path=save_path,
            positive_class=target_class,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series | np.ndarray,
        verbosity: int = 2,
        time_limit: float = 300.0,
        infer_limit: float = 0.005,
        hyperparameters: dict = None,
        num_gpus: int = "auto",
    ) -> None:
        """Run autogluon predictor.fit."""
        self.X_train = TabularDataset(pd.concat([X, y], axis=1))
        self.predictor.fit(
            self.X_train,
            presets=["best_quality"],
            auto_stack=True,
            calibrate_decision_threshold=True,
            time_limit=time_limit,
            infer_limit=infer_limit,
            verbosity=verbosity,
            num_gpus=num_gpus,
            hyperparameters=hyperparameters,
        )

    def _get_best_model(
        self, test_data: pd.DataFrame, crit: str = "score_test"
    ) -> None:
        leaders = self.predictor.leaderboard(test_data, silent=True)
        best_model = leaders.loc[leaders[crit] == leaders[crit].max()]["model"].iloc[0]
        return best_model

    def _check_shape(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == len(self.feature_names):
            if self.warnings:
                warnings.warn(
                    "The input X has the same number of features as the training data. "
                    "Assuming the label column is missing. Adding a zero col as dummy label."
                )

            if isinstance(X, np.ndarray):
                X = np.column_stack([X, np.zeros((X.shape[0], 1))])
                X = pd.DataFrame(X, columns=self.feature_names + [self.label])

            if isinstance(X, pd.DataFrame) and self.label not in X.columns:
                X[self.label] = 0

        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # The X needs to have the outcome label column
        X = self._check_shape(X)

        best_model = self._get_best_model(X, "score_test")
        preds = self.predictor.predict_proba(X, model=best_model, as_pandas=False)

        return preds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # The X needs to have the outcome label column
        X = self._check_shape(X)

        best_model = self._get_best_model(X, "score_test")
        preds = self.predictor.predict(X, model=best_model, as_pandas=False)

        return preds
