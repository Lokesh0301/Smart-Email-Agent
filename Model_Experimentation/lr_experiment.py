"""
lr_experiment.py
────────────────
Experiment 1: TF-IDF + Logistic Regression baseline.

MLflow tracking
───────────────
  Experiment : Smart_Email_Classifier
  Run name   : LR_TFIDF_<timestamp>

Logged
  params  : tfidf_max_features, tfidf_ngram_range, tfidf_sublinear_tf,
            lr_C, lr_solver, lr_max_iter, lr_class_weight
  metrics : accuracy, precision_weighted, recall_weighted,
            f1_macro, f1_weighted, roc_auc, avg_precision
  artifacts: classification_report.txt, confusion_matrix.png,
             roc_curve.png, pr_curve.png
  model   : sklearn pipeline (mlflow.sklearn)
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

from Model_Experimentation.data_utils import load_data, ID2LABEL

# ── Hyper-parameters (easy to modify) ────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{2,}",
)

LR_PARAMS = dict(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(2)],
        yticklabels=[ID2LABEL[i] for i in range(2)],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Logistic Regression")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_roc_curve(y_true, y_prob, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Logistic Regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_pr_curve(y_true, y_prob, save_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve – Logistic Regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ── Main experiment function ──────────────────────────────────────────────────

def run_lr_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str | None = None,
) -> dict:
    """
    Train TF-IDF + Logistic Regression, log everything to MLflow.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames from data_utils.load_data()
    run_name : optional override for the MLflow run name

    Returns
    -------
    dict of test metrics
    """
    if run_name is None:
        run_name = f"LR_TFIDF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  Experiment 1: Logistic Regression  [{run_name}]")
    print(f"{'='*60}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(**LR_PARAMS)),
    ])

    X_train, y_train = train_df["text"].tolist(), train_df["label"].values
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].values
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].values

    # ── Train ────────────────────────────────────────────────────────────────
    print("  Training …")
    pipeline.fit(X_train, y_train)

    # ── Evaluate on validation set ────────────────────────────────────────────
    y_val_pred  = pipeline.predict(X_val)
    y_val_prob  = pipeline.predict_proba(X_val)[:, 1]

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    def _compute_metrics(y_true, y_pred, y_prob, prefix="test") -> dict:
        return {
            f"{prefix}_accuracy":           accuracy_score(y_true, y_pred),
            f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
            f"{prefix}_f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
            f"{prefix}_roc_auc":            roc_auc_score(y_true, y_prob),
            f"{prefix}_avg_precision":      average_precision_score(y_true, y_prob),
        }

    val_metrics  = _compute_metrics(y_val,  y_val_pred,  y_val_prob,  prefix="val")
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_prob, prefix="test")

    print(f"  Val   F1-weighted : {val_metrics['val_f1_weighted']:.4f}  |  ROC-AUC : {val_metrics['val_roc_auc']:.4f}")
    print(f"  Test  F1-weighted : {test_metrics['test_f1_weighted']:.4f}  |  ROC-AUC : {test_metrics['test_roc_auc']:.4f}")

    # ── MLflow logging ────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name, tags={"model_type": "logistic_regression"}):

        # Params
        mlflow.log_params({
            "tfidf_max_features":  TFIDF_PARAMS["max_features"],
            "tfidf_ngram_range":   str(TFIDF_PARAMS["ngram_range"]),
            "tfidf_sublinear_tf":  TFIDF_PARAMS["sublinear_tf"],
            "lr_C":                LR_PARAMS["C"],
            "lr_solver":           LR_PARAMS["solver"],
            "lr_max_iter":         LR_PARAMS["max_iter"],
            "lr_class_weight":     str(LR_PARAMS["class_weight"]),
            "train_size":          len(train_df),
            "val_size":            len(val_df),
            "test_size":           len(test_df),
        })

        # Metrics
        mlflow.log_metrics({**val_metrics, **test_metrics})

        # Artifacts – use a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Classification report (test set)
            report = classification_report(
                y_test, y_test_pred,
                target_names=[ID2LABEL[i] for i in range(2)],
            )
            report_path = tmpdir / "classification_report.txt"
            report_path.write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            # Plots
            cm_path  = tmpdir / "confusion_matrix.png"
            roc_path = tmpdir / "roc_curve.png"
            pr_path  = tmpdir / "pr_curve.png"

            _plot_confusion_matrix(y_test, y_test_pred, cm_path)
            _plot_roc_curve(y_test, y_test_prob, roc_path)
            _plot_pr_curve(y_test, y_test_prob, pr_path)

            mlflow.log_artifact(str(cm_path),  artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(pr_path),  artifact_path="plots")

        # Model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="LR_Email_Classifier",
        )

    print("  ✓ MLflow run complete.")
    return test_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    mlflow.set_experiment("Smart_Email_Classifier")
    train_df, val_df, test_df = load_data()
    run_lr_experiment(train_df, val_df, test_df)
