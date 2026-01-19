from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.metrics import compute_metrics_table, get_conf_mat
from tabpfn import TabPFNClassifier


def setup_model_and_optimizer(config: dict) -> tuple[TabPFNClassifier, Optimizer, dict]:
    """Initializes the TabPFN classifier, optimizer, and training configs."""
    print("--- 2. Model and Optimizer Setup ---")
    classifier_config = {
        "ignore_pretraining_limits": True,
        "device": "cuda",
        "n_estimators": 8,
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    classifier = TabPFNClassifier(
        **classifier_config, fit_mode="batched", differentiable_input=False
    )
    classifier._initialize_model_variables()
    # Optimizer uses finetuning-specific learning rate
    optimizer = Adam(
        classifier.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )

    print(f"Using device: {classifier_config['device']}")
    print(f"Optimizer: Adam, Finetuning LR: {config['finetuning']['learning_rate']}")
    print("----------------------------------\n")
    return classifier, optimizer, classifier_config


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluates the model's performance on the test set."""
    eval_classifier = clone_model_for_evaluation(
        classifier, eval_config, TabPFNClassifier
    )
    eval_classifier.fit(X_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(X_test)
        predictions = eval_classifier.predict(X_test)
        metrics_df = compute_metrics_table(
            y_true=y_test,
            predictions=predictions,
            prediction_probabilities=probabilities,
        )
        conf_mat = get_conf_mat(
            y=y_test,
            yhat=predictions,
            best_f1_thresh=0,
        )

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        metrics_df, conf_mat = pd.Series(), pd.DataFrame()

    return metrics_df, conf_mat


def finetune_tabpfn(
    X_train, X_test, y_train, y_test, save_path: str = None, best_metric: str = "mcc"
) -> TabPFNClassifier:
    """Main function to configure and run the finetuning workflow with checkpointing."""
    # --- Master Configuration ---
    config = {
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": len(y_train),
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.2,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": len(y_train),
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 50,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": 1e-4,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": 16,
    }

    # --- Setup Data, Model, and Dataloader ---
    classifier, optimizer, classifier_config = setup_model_and_optimizer(config)

    splitter = partial(train_test_split, test_size=0.2)
    training_datasets = classifier.get_preprocessed_datasets(
        X_train, y_train, splitter, config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )
    loss_function = torch.nn.CrossEntropyLoss()

    eval_config = {
        **classifier_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Checkpointing setup ---
    best_mcc = 0.0
    best_recall = 0.0
    best_balanced_accuracy = 0.0
    best_classifier = None
    best_metrics = None
    checkpoint_epoch = -1

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")

    # Create a single progress bar for all epochs
    total_epochs = config["finetuning"]["epochs"] + 1  # +1 for initial evaluation
    main_progress = tqdm(total=total_epochs, desc="TabPFN Finetuning", unit="epoch")

    for epoch in range(total_epochs):
        if epoch > 0:
            # Finetuning Step - no nested progress bar, just process batches
            for (
                X_train_batch,
                X_test_batch,
                y_train_batch,
                y_test_batch,
                cat_ixs,
                confs,
            ) in finetuning_dataloader:
                if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                    continue  # Skip batch if splits don't have all classes

                optimizer.zero_grad()
                classifier.fit_from_preprocessed(
                    X_train_batch, y_train_batch, cat_ixs, confs
                )
                predictions = classifier.forward(X_test_batch, return_logits=True)
                loss = loss_function(predictions, y_test_batch.to(config["device"]))
                loss.backward()
                optimizer.step()

        # Evaluation Step (runs before finetuning and after each epoch)
        metrics_df, conf_mat = evaluate_model(
            classifier, eval_config, X_train, y_train, X_test, y_test
        )

        # Extract metrics for progress bar display
        mcc = metrics_df.get("mcc", 0.0) if not metrics_df.empty else 0.0
        precision = metrics_df.get("precision", 0.0) if not metrics_df.empty else 0.0
        recall = metrics_df.get("recall", 0.0) if not metrics_df.empty else 0.0
        balanced_accuracy = (
            metrics_df.get("balanced_accuracy", 0.0) if not metrics_df.empty else 0.0
        )

        # Extract confusion matrix components (TP, FP, TN, FN)
        if not conf_mat.empty and conf_mat.shape == (2, 2):
            # conf_mat structure: rows=actual, cols=predicted
            # [TN, FP]
            # [FN, TP]
            tn = conf_mat.iloc[0, 0]  # True Negative
            fp = conf_mat.iloc[0, 1]  # False Positive
            fn = conf_mat.iloc[1, 0]  # False Negative
            tp = conf_mat.iloc[1, 1]  # True Positive
        else:
            tn = fp = fn = tp = 0

        # Update progress bar with comprehensive metrics
        status = "Initial" if epoch == 0 else f"E{epoch}"
        main_progress.set_postfix(
            {
                "Status": status,
                "MCC": f"{mcc:.3f}",
                "Bal Acc": f"{balanced_accuracy:.3f}",
                "Prec": f"{precision:.3f}",
                "Rec": f"{recall:.3f}",
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            }
        )
        main_progress.update(1)

        # Checkpoint model if it's non-random (MCC > 0), has precision > 0, and has better MCC
        is_better_recall = recall > best_recall
        is_better_mcc = mcc > best_mcc
        is_better_balanced_accuracy = balanced_accuracy > best_balanced_accuracy
        if best_metric == "recall":
            cond = is_better_recall
        elif best_metric == "balanced_accuracy":
            cond = is_better_balanced_accuracy
        else:
            cond = is_better_mcc

        if mcc > 0 and precision > 0 and cond:
            best_mcc = mcc
            best_recall = recall
            best_balanced_accuracy = balanced_accuracy
            best_metrics = metrics_df.copy()
            checkpoint_epoch = epoch

            # Brief checkpoint notification (don't interrupt progress bar too much)
            main_progress.write(
                f"🎯 New best model at epoch {epoch}! MCC: {mcc:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall}, Bal Acc: {balanced_accuracy}, TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}"
            )

            # Save checkpoint if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "state_dict": classifier.model_.state_dict(),
                    "config": classifier.config_,
                    "classifier_config": classifier_config,
                    "eval_config": eval_config,
                    "metrics": best_metrics,
                    "epoch": checkpoint_epoch,
                    "mcc": best_mcc,
                }

                torch.save(checkpoint_data, save_path / "best_tabpfn_checkpoint.pt")
                # Also store a copy of the model state for returning
                best_classifier = clone_model_for_evaluation(
                    classifier, eval_config, TabPFNClassifier
                )

    main_progress.close()

    print("--- ✅ Finetuning Finished ---")

    # Return best model if we found one
    if best_classifier is not None:
        print(
            f"Returning best model from epoch {checkpoint_epoch} with MCC {best_mcc:.4f}, recall: {best_recall:.4f}, Bal Acc: {best_balanced_accuracy:.4f}"
        )
        return best_classifier
    else:
        print("No non-random model found (MCC > 0). Returning last model.")
        return classifier


def load_finetuned_tabpfn(checkpoint_path: str) -> tuple[TabPFNClassifier, dict]:
    """Load a finetuned TabPFN model from checkpoint."""
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cuda", weights_only=False
    )

    # Recreate classifier with saved config
    classifier_config = checkpoint_data["classifier_config"]
    classifier = TabPFNClassifier(
        **classifier_config, fit_mode="batched", differentiable_input=False
    )
    classifier._initialize_model_variables()

    # Load the model state dict
    classifier.model_.load_state_dict(checkpoint_data["state_dict"])

    # Important: Keep the model in batched mode to match the finetuned state
    # This prevents the automatic mode switching that causes prediction differences

    return classifier, checkpoint_data


def create_inference_ready_tabpfn(
    checkpoint_path: str, X_train: np.ndarray, y_train: np.ndarray
) -> TabPFNClassifier:
    """
    Load a finetuned TabPFN model and fit it for immediate inference use.

    Args:
        checkpoint_path: Path to the saved checkpoint
        X_train: Training features to fit the model
        y_train: Training labels to fit the model

    Returns:
        Ready-to-use TabPFNClassifier for predictions
    """
    classifier, _ = load_finetuned_tabpfn(checkpoint_path)
    classifier.fit(X_train, y_train)
    return classifier
