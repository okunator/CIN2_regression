import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from shapiq.interaction_values import InteractionValues
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def shap2df(shap_vals: InteractionValues, feature_names: Sequence[str]) -> pd.DataFrame:
    """Convert shapiq interaction value objs into a pd.DataFrame.

    Parameters:
        shap_vals (InteractionValues):
            result of `explainer.explain_X`.
        feature_name (Sequence[str]):
            Names of the features in correct order.

    Returns:
        pd.DataFrame:
            shap values in a df.
    """
    features_to_names = {i: str(name) for i, name in enumerate(feature_names)}

    columns = []
    for k in shap_vals.dict_values.keys():
        if len(k) == 0:
            columns.append("baseline")
        else:
            columns.append(" x ".join(features_to_names[i] for i in k))

    values = [shap_vals.dict_values[k] for k in shap_vals.dict_values.keys()]
    shap_df = pd.DataFrame([values], columns=columns)
    return shap_df


def to_numpy(res: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(res, pd.DataFrame) or isinstance(res, pd.Series):
        res = res.to_numpy()

    if isinstance(res, np.ndarray) and res.dtype.kind in {"U", "S", "O"}:
        le = LabelEncoder()
        res = le.fit_transform(res)

    return res
