"""
setfit_experiment.py
────────────────────
Experiment 3: SetFit few-shot fine-tuning for binary email classification.

SetFit trains a sentence-transformer via contrastive learning on a small
number of labelled examples, then fits a lightweight sklearn head.

Base model : sentence-transformers/all-MiniLM-L6-v2
Few-shot   : SAMPLES_PER_CLASS examples drawn from the training set
             (full val/test used for evaluation)

MLflow tracking
───────────────
  Experiment : Smart_Email_Classifier
  Run name   : SetFit_<model_short>_<timestamp>

Logged
  params  : model_id, num_epochs, num_iterations, samples_per_class,
            head_type
  metrics : val_* and test_* – accuracy, precision_weighted,
            recall_weighted, f1_macro, f1_weighted, roc_auc, avg_precision
  artifacts: classification_report.txt, confusion_matrix.png,
             roc_curve.png, pr_curve.png
  model   : mlflow.pyfunc wrapper (SetFit model saved locally)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer, TrainingArguments as SetFitTrainingArguments
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from Model_Experimentation.data_utils import ID2LABEL, LABEL2ID, load_data

# ── Offline-safe model loading ────────────────────────────────────────────────
# SetFit calls the HF Hub on every from_pretrained() by default.
# This wrapper tries local_files_only first to avoid network calls.

def _load_setfit_pretrained(model_id: str, **kwargs) -> SetFitModel:
    """
    Load a SetFit model from local HF cache if available,
    otherwise download from the Hub.
    Respects HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 env vars.
    """
    force_offline = (
        os.getenv("HF_HUB_OFFLINE", "0") == "1"
        or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    )
    try:
        return SetFitModel.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        if force_offline:
            raise RuntimeError(
                f"SetFit model '{model_id}' not in local cache and "
                "offline mode is enabled.\n"
                "Run:  uv run python Model_Experimentation/download_models.py"
            )
        print(f"  Cache miss for '{model_id}' — downloading from HuggingFace Hub …")
        return SetFitModel.from_pretrained(model_id, **kwargs)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID          = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLES_PER_CLASS = 64          # few-shot sample budget per class
NUM_EPOCHS        = 1           # contrastive training epochs
NUM_ITERATIONS    = 20          # number of sentence pairs per epoch
SEED              = 42


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=[ID2LABEL[i] for i in range(2)],
        yticklabels=[ID2LABEL[i] for i in range(2)], ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _plot_roc(y_true, y_prob, save_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="grey")
    ax.set(xlabel="FPR", ylabel="TPR", title=title); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def _plot_pr(y_true, y_prob, save_path: Path, title: str) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ── MLflow pyfunc wrapper ─────────────────────────────────────────────────────

class SetFitPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    Wraps a SetFit model as an mlflow.pyfunc model so it can be logged
    and loaded uniformly.
    """

    def load_context(self, context):
        self.model = SetFitModel.from_pretrained(
            context.artifacts["setfit_model"]
        )

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        texts = model_input.iloc[:, 0].tolist()
        return self.model.predict(texts).numpy()


# ── Few-shot sampling ─────────────────────────────────────────────────────────

def _sample_few_shot(
    train_df: pd.DataFrame,
    samples_per_class: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Stratified sample of `samples_per_class` rows per label."""
    frames = []
    for label in sorted(train_df["label"].unique()):
        subset = train_df[train_df["label"] == label]
        n = min(samples_per_class, len(subset))
        frames.append(subset.sample(n=n, random_state=seed))
    return pd.concat(frames).reset_index(drop=True)


# ── Main experiment ────────────────────────────────────────────────────────────

def run_setfit_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_id: str = MODEL_ID,
    samples_per_class: int = SAMPLES_PER_CLASS,
    run_name: str | None = None,
) -> dict:
    """
    Train a SetFit model and log everything to MLflow.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames from data_utils.load_data()
    model_id          : sentence-transformers model ID
    samples_per_class : few-shot budget per class
    run_name          : optional MLflow run name override

    Returns
    -------
    dict of test metrics
    """
    short_name = model_id.split("/")[-1]
    if run_name is None:
        run_name = f"SetFit_{short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  Experiment 3: SetFit  [{run_name}]")
    print(f"{'='*60}")

    # ── Few-shot subset ───────────────────────────────────────────────────────
    few_shot_df = _sample_few_shot(train_df, samples_per_class)
    print(f"  Few-shot training set: {len(few_shot_df)} examples "
          f"({samples_per_class} per class)")

    # Convert to HuggingFace Dataset
    train_ds = Dataset.from_dict({
        "text":  few_shot_df["text"].tolist(),
        "label": few_shot_df["label"].tolist(),
    })
    val_ds = Dataset.from_dict({
        "text":  val_df["text"].tolist(),
        "label": val_df["label"].tolist(),
    })

    # ── SetFit model & trainer (offline-safe) ───────────────────────────
    # Tries local HF cache first; only hits the network on a true cache miss.
    print(f"  Loading SetFit model: {model_id} …")
    setfit_model = _load_setfit_pretrained(
        model_id,
        labels=[ID2LABEL[0], ID2LABEL[1]],
    )

    training_args = SetFitTrainingArguments(
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        seed=SEED,
    )

    trainer = SetFitTrainer(
        model=setfit_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric="f1",
        metric_kwargs={"average": "weighted"},
    )

    print("  Training (contrastive phase) …")
    trainer.train()

    # ── Predict ───────────────────────────────────────────────────────────────
    def _predict(df: pd.DataFrame):
        texts  = df["text"].tolist()
        labels = df["label"].values

        # SetFit predict_proba returns class probabilities
        probs_tensor = setfit_model.predict_proba(texts)
        # Returns shape (N, num_classes) as torch tensor
        probs = probs_tensor.numpy() if hasattr(probs_tensor, "numpy") else np.array(probs_tensor)
        y_prob = probs[:, 1]  # probability of positive class
        y_pred = (y_prob >= 0.5).astype(int)
        return labels, y_pred, y_prob

    y_val,  y_val_pred,  y_val_prob  = _predict(val_df)
    y_test, y_test_pred, y_test_prob = _predict(test_df)

    def _metrics(y_true, y_pred, y_prob, prefix):
        return {
            f"{prefix}_accuracy":           accuracy_score(y_true, y_pred),
            f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
            f"{prefix}_f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_roc_auc":            roc_auc_score(y_true, y_prob),
            f"{prefix}_avg_precision":      average_precision_score(y_true, y_prob),
        }

    val_metrics  = _metrics(y_val,  y_val_pred,  y_val_prob,  "val")
    test_metrics = _metrics(y_test, y_test_pred, y_test_prob, "test")

    print(f"  Val   F1-weighted : {val_metrics['val_f1_weighted']:.4f}  |  ROC-AUC : {val_metrics['val_roc_auc']:.4f}")
    print(f"  Test  F1-weighted : {test_metrics['test_f1_weighted']:.4f}  |  ROC-AUC : {test_metrics['test_roc_auc']:.4f}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name, tags={"model_type": "setfit"}):
        mlflow.log_params({
            "model_id":           model_id,
            "num_epochs":         NUM_EPOCHS,
            "num_iterations":     NUM_ITERATIONS,
            "samples_per_class":  samples_per_class,
            "total_few_shot":     len(few_shot_df),
            "head_type":          "logistic_regression",
            "train_size":         len(train_df),
            "val_size":           len(val_df),
            "test_size":          len(test_df),
        })
        mlflow.log_metrics({**val_metrics, **test_metrics})

        with tempfile.TemporaryDirectory() as artdir:
            artdir = Path(artdir)

            # Save SetFit model locally so pyfunc can reference it
            model_save_dir = artdir / "setfit_model"
            setfit_model.save_pretrained(str(model_save_dir))

            # Classification report
            rp = classification_report(
                y_test, y_test_pred,
                target_names=[ID2LABEL[i] for i in range(2)],
            )
            rp_path = artdir / "classification_report.txt"
            rp_path.write_text(rp, encoding="utf-8")
            mlflow.log_artifact(str(rp_path), artifact_path="reports")

            # Plots
            _plot_confusion_matrix(y_test, y_test_pred, artdir / "confusion_matrix.png",
                                   f"Confusion Matrix – SetFit ({short_name})")
            _plot_roc(y_test, y_test_prob, artdir / "roc_curve.png",
                      f"ROC Curve – SetFit ({short_name})")
            _plot_pr(y_test, y_test_prob, artdir / "pr_curve.png",
                     f"PR Curve – SetFit ({short_name})")

            mlflow.log_artifact(str(artdir / "confusion_matrix.png"), "plots")
            mlflow.log_artifact(str(artdir / "roc_curve.png"),        "plots")
            mlflow.log_artifact(str(artdir / "pr_curve.png"),         "plots")

            # Log as pyfunc
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SetFitPyfuncModel(),
                artifacts={"setfit_model": str(model_save_dir)},
                registered_model_name="SetFit_Email_Classifier",
            )

    print("  ✓ MLflow run complete.")
    return test_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    mlflow.set_experiment("Smart_Email_Classifier")
    train_df, val_df, test_df = load_data()
    run_setfit_experiment(train_df, val_df, test_df)
