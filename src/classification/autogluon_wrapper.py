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
        y_test: pd.DataFrame | pd.Series | np.ndarray = None,
        X_test: pd.DataFrame = None,
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
            y_test (pd.DataFrame | pd.Series | np.ndarray):
                The outcome labels for the test set. Used if the input X to predict
                does not contain the label column.
            X_test (pd.DataFrame):
                The test set features. Used to get the best model after training.
        """
        self.feature_names = feature_names
        self.save_path = Path(save_path)
        self.target_class = target_class
        self.label = label
        self.warnings = warnings
        self.y_test = y_test
        self.X_test = X_test
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
        # random_seed: int = 42,
        hyperparameters: dict = None,
        num_gpus: int = "auto",
    ) -> None:
        """Run autogluon predictor.fit."""
        self.X_train = TabularDataset(pd.concat([X, y], axis=1))
        self.y_train = y
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
            raise_on_no_models_fitted=True,
        )

    @property
    def feature_importances(self) -> pd.DataFrame:
        return self.predictor.feature_importance(
            pd.concat([self.X_test, self.y_test], axis=1), self.best_model
        )

    @property
    def best_model(self) -> str:
        crit = "score_test"
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        leaders = self.predictor.leaderboard(test_data, silent=True)
        return leaders.loc[leaders[crit] == leaders[crit].max()]["model"].iloc[0]

    def _check_shape(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        res = X.copy()

        if res.shape[1] == len(self.feature_names):
            if isinstance(X, np.ndarray):
                res = np.column_stack([res, np.zeros((res.shape[0], 1))])
                if isinstance(self.feature_names, np.ndarray):
                    cols = self.feature_names.tolist() + [self.label]
                else:
                    cols = self.feature_names + [self.label]
                res = pd.DataFrame(res, columns=cols)

            if isinstance(res, pd.DataFrame) and self.label not in res.columns:
                res[self.label] = 0

        return res

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # The X needs to have the outcome label column
        X = self._check_shape(X)
        preds = self.predictor.predict_proba(X, model=self.best_model, as_pandas=False)

        return preds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # The X needs to have the outcome label column
        X = self._check_shape(X)
        preds = self.predictor.predict(X, model=self.best_model, as_pandas=False)

        return preds
