import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_conf_mat(
    y: np.ndarray,
    yhat: np.ndarray,
    best_f1_thresh: float,
) -> pd.DataFrame:
    """Get the confusion matrix based on the best F1 threshold.

    Parameters:
        y (np.ndarray): True binary labels.
        yhat (np.ndarray): Predicted probabilities.
        best_f1_thresh (float): Threshold that gives the best F1 score.

    Returns:
        pd.DataFrame: Confusion matrix with labels 0 (non-reg) and 1 (reg).
    """
    conf_mat = pd.DataFrame(
        confusion_matrix(y, (yhat > best_f1_thresh).astype(int), labels=[0, 1]),
        columns=["non-reg", "reg"],
        index=["non-reg", "reg"],
    )

    return conf_mat


def get_prc_curve(
    y: np.ndarray, yhat: np.ndarray
) -> tuple[float, int, float, pd.DataFrame]:
    """Get the precision-recall curve data and best threshold based on F1 score.

    Parameters:
        y (np.ndarray): True binary labels.
        yhat (np.ndarray): Predicted probabilities.

    Returns:
        tuple: A tuple containing:
            - auprc (float): Area under the precision-recall curve.
            - best_ix (int): index for the best f1 score.
            - best_thresh (float): Threshold that gives the best F1 score.
            - prc_df (pd.DataFrame): df with precision, recall, F1 score, & thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y, yhat)
    auprc = auc(recall, precision)

    # save roc and prc values
    prc_df = pd.DataFrame([precision, recall]).T
    prc_df.columns = ["precision", "recall"]

    # calculate f1 score for each threshold
    prc_df["f1"] = (2 * prc_df["precision"] * prc_df["recall"]) / (
        prc_df["precision"] + prc_df["recall"]
    )

    # add thresholds to df
    prc_df = pd.concat([prc_df, pd.Series(thresholds)], axis=1)
    prc_df.rename(columns={0: "threshold"}, inplace=True)

    # if we have more balanced precision and recall over 0.7, choose the best f1 from those
    over_70_precision = prc_df.loc[
        (prc_df["precision"] >= 0.7) & (prc_df["recall"] >= 0.7)
    ]

    if over_70_precision.empty:
        best_ix = prc_df["f1"].idxmax()
    else:
        best_ix = over_70_precision["f1"].idxmax()

    best_thresh = prc_df.iloc[best_ix]["threshold"]

    return auprc, best_ix, best_thresh, prc_df


def get_roc_curve(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    """Get the ROC curve data and best threshold based on TPR and FPR.

    Parameters:
        y (np.ndarray): True binary labels.
        yhat (np.ndarray): Predicted probabilities.

    Returns:
        tuple: A tuple containing:
            - roc_auc (float): Area under the ROC curve.
            - best_ix (int): index for the best tpr score.
            - best_thresh (float): Threshold that gives the best TPR with FPR >= 0.7 if possible.
            - fpr_tpr_df (pd.DataFrame): df with FPR, TPR, & thresholds.
    """
    roc_auc = roc_auc_score(y, yhat)
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, yhat, drop_intermediate=False)
    fpr_tpr_df = pd.DataFrame([fpr, tpr]).T
    fpr_tpr_df.columns = ["fpr", "tpr"]
    fpr_tpr_df = pd.concat([fpr_tpr_df, pd.Series(thresholds)], axis=1)
    fpr_tpr_df.rename(columns={0: "threshold"}, inplace=True)

    # if we have more balanced tpr and fpr over 0.7, choose the best tpr from those
    over_70_fpr = fpr_tpr_df.loc[
        (fpr_tpr_df["fpr"] >= 0.7) & (fpr_tpr_df["tpr"] >= 0.7)
    ]

    if over_70_fpr.empty:
        best_ix = fpr_tpr_df["tpr"].idxmax()
    else:
        best_ix = over_70_fpr["tpr"].idxmax()

    best_thresh = fpr_tpr_df.iloc[best_ix]["threshold"]

    return roc_auc, best_ix, best_thresh, fpr_tpr_df


def compute_metrics_table(
    y_true: np.ndarray,
    predictions: np.ndarray,
    prediction_probabilities: np.ndarray,
    target_class: int = 1,
) -> pd.Series:
    """Compute various classification metrics for the binary classification task.

    Parameters:
        y_true: np.ndarray
            True labels.
        predictions: np.ndarray
            Predicted labels.
        prediction_probabilities: np.ndarray
            Predicted probabilities for each class.

    Returns:
        pd.Series:
            Series containing computed metrics.
    """
    metrics = {}
    metrics["f1_weighted"] = f1_score(y_true, predictions, average="weighted")
    metrics["accuracy"] = accuracy_score(y_true, predictions)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, predictions)
    metrics["mcc"] = matthews_corrcoef(y_true, predictions)
    metrics["roc_auc"] = roc_auc_score(y_true, prediction_probabilities[:, 1])
    metrics["f1"] = f1_score(y_true, predictions, pos_label=target_class)
    metrics["precision"] = precision_score(y_true, predictions, pos_label=target_class)
    metrics["recall"] = recall_score(y_true, predictions, pos_label=target_class)
    metrics["auprc"] = average_precision_score(y_true, prediction_probabilities[:, 1])

    # Baseline AUPRC (random classifier)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    metrics["auprc_baseline"] = baseline
    metrics["auprc_above_baseline"] = metrics["auprc"] - baseline

    # Best F1 threshold
    thresholds = np.linspace(0, 1, 101)
    f1s = [
        f1_score(
            y_true,
            (prediction_probabilities[:, 1] > t).astype(int),
            pos_label=1,
        )
        for t in thresholds
    ]
    metrics["best_f1_thresh"] = thresholds[np.argmax(f1s)]

    return pd.Series(metrics)
