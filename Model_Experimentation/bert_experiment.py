"""
bert_experiment.py
──────────────────
Experiment 2: DistilBERT fine-tuning for binary email classification.

Model : distilbert-base-uncased  (swap MODEL_NAME to change variant)
Input : subject [SEP] body  (truncated to max_length=256 tokens)

MLflow tracking
───────────────
  Experiment : Smart_Email_Classifier
  Run name   : BERT_<model_short>_<timestamp>

Logged
  params  : model_name, max_length, num_epochs, batch_size,
            lr, warmup_ratio, weight_decay
  metrics : val_* and test_* – accuracy, precision_weighted,
            recall_weighted, f1_macro, f1_weighted, roc_auc, avg_precision
  artifacts: classification_report.txt, confusion_matrix.png,
             roc_curve.png, pr_curve.png
  model   : transformers pipeline (mlflow.transformers)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from Model_Experimentation.data_utils import ID2LABEL, LABEL2ID, get_class_weights, load_data

# ── Offline-safe model loading ────────────────────────────────────────────────
# HuggingFace always pings the Hub by default, even when the model is cached.
# _load_pretrained() tries local cache first, falls back to online download.
# Set env var  HF_HUB_OFFLINE=1  to force fully offline mode.

def _load_pretrained(loader_fn, model_id: str, **kwargs):
    """
    Try loading from local cache first (local_files_only=True).
    Falls back to an online download only if the cache misses.
    Respects HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 env vars.
    """
    import os
    force_offline = (
        os.getenv("HF_HUB_OFFLINE", "0") == "1"
        or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    )
    try:
        # Always attempt local cache first — no network call
        return loader_fn(model_id, local_files_only=True, **kwargs)
    except Exception:
        if force_offline:
            raise RuntimeError(
                f"Model '{model_id}' not found in local cache and "
                "offline mode is enabled (HF_HUB_OFFLINE=1).\n"
                "Run:  uv run python Model_Experimentation/download_models.py"
            )
        print(f"  Cache miss for '{model_id}' — downloading from HuggingFace Hub …")
        return loader_fn(model_id, **kwargs)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 256
NUM_EPOCHS   = 3
BATCH_SIZE   = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
SEED          = 42


# ── Custom Trainer with class-weighted loss ────────────────────────────────────

class WeightedTrainer(Trainer):
    """Overrides compute_loss to use class-weighted cross-entropy."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics for Trainer ────────────────────────────────────────────────────────

def make_compute_metrics(threshold: float = 0.5):
    """Return a compute_metrics function for the HuggingFace Trainer."""
    import evaluate
    f1_metric = evaluate.load("f1")

    def _compute(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = (probs >= threshold).astype(int)
        f1 = f1_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]
        acc = accuracy_score(labels, preds)
        return {"f1_weighted": f1, "accuracy": acc}

    return _compute


# ── Plotting helpers (re-use same style as LR) ────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
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


# ── Main experiment ────────────────────────────────────────────────────────────

def run_bert_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    run_name: str | None = None,
) -> dict:
    """
    Fine-tune a BERT-family model and log everything to MLflow.

    Parameters
    ----------
    train_df, val_df, test_df : DataFrames from data_utils.load_data()
    model_name : HuggingFace model hub ID
    run_name   : optional MLflow run name override

    Returns
    -------
    dict of test metrics
    """
    short_name = model_name.split("/")[-1]
    if run_name is None:
        run_name = f"BERT_{short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  Experiment 2: BERT Fine-tuning  [{run_name}]")
    print(f"{'='*60}")

    # ── Tokenizer & model (offline-safe) ────────────────────────────────────
    # Tries local HF cache first; only hits the network on a true cache miss.
    print(f"  Loading tokenizer and model: {model_name} …")
    tokenizer = _load_pretrained(AutoTokenizer.from_pretrained, model_name)
    model = _load_pretrained(
        AutoModelForSequenceClassification.from_pretrained,
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Tokenise splits ───────────────────────────────────────────────────────
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # dynamic padding via DataCollator
        )

    def df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label"]])
        ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch")
        return ds

    print("  Tokenising datasets …")
    train_ds = df_to_hf_dataset(train_df)
    val_ds   = df_to_hf_dataset(val_df)
    test_ds  = df_to_hf_dataset(test_df)

    # ── Class weights ─────────────────────────────────────────────────────────
    cw = get_class_weights(train_df)
    class_weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32)
    print(f"  Class weights: {cw}")

    # ── Training arguments ────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            logging_steps=50,
            seed=SEED,
            report_to="none",  # we handle MLflow manually
            fp16=torch.cuda.is_available(),
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("  Training …")
        trainer.train()

        # ── Collect predictions ───────────────────────────────────────────────
        def _predict(dataset, df):
            preds_out = trainer.predict(dataset)
            logits = preds_out.predictions
            probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            preds  = (probs >= 0.5).astype(int)
            labels = df["label"].values
            return labels, preds, probs

        y_val,  y_val_pred,  y_val_prob  = _predict(val_ds,  val_df)
        y_test, y_test_pred, y_test_prob = _predict(test_ds, test_df)

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

        # ── MLflow ───────────────────────────────────────────────────────────
        with mlflow.start_run(run_name=run_name, tags={"model_type": "distilbert"}):
            mlflow.log_params({
                "model_name":   model_name,
                "max_length":   MAX_LENGTH,
                "num_epochs":   NUM_EPOCHS,
                "batch_size":   BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "class_weight": "balanced (weighted loss)",
                "train_size":   len(train_df),
                "val_size":     len(val_df),
                "test_size":    len(test_df),
            })
            mlflow.log_metrics({**val_metrics, **test_metrics})

            with tempfile.TemporaryDirectory() as artdir:
                artdir = Path(artdir)

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
                                       f"Confusion Matrix – {short_name}")
                _plot_roc(y_test, y_test_prob, artdir / "roc_curve.png",
                          f"ROC Curve – {short_name}")
                _plot_pr(y_test, y_test_prob, artdir / "pr_curve.png",
                         f"PR Curve – {short_name}")

                mlflow.log_artifact(str(artdir / "confusion_matrix.png"), "plots")
                mlflow.log_artifact(str(artdir / "roc_curve.png"),       "plots")
                mlflow.log_artifact(str(artdir / "pr_curve.png"),        "plots")

            # Log the best model as a HuggingFace transformers pipeline
            from transformers import pipeline as hf_pipeline
            clf_pipeline = hf_pipeline(
                "text-classification",
                model=trainer.model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            mlflow.transformers.log_model(
                transformers_model=clf_pipeline,
                artifact_path="model",
                registered_model_name=f"BERT_Email_Classifier_{short_name}",
            )

    print("  ✓ MLflow run complete.")
    return test_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    mlflow.set_experiment("Smart_Email_Classifier")
    train_df, val_df, test_df = load_data()
    run_bert_experiment(train_df, val_df, test_df)
