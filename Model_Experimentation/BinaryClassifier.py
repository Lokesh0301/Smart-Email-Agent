"""
BinaryClassifier.py
───────────────────
Main orchestrator for all email binary classification experiments.

Runs three experiments sequentially, all tracked under the
"Smart_Email_Classifier" MLflow experiment:

  1. Logistic Regression  (TF-IDF features, sklearn pipeline)
  2. DistilBERT           (fine-tuned transformer via HuggingFace Trainer)
  3. SetFit               (few-shot sentence-transformer + LR head)

Usage (from project root):
    uv run python -m Model_Experimentation.BinaryClassifier

MLflow UI (after run):
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    → open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on sys.path so all imports resolve ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd

from Model_Experimentation.data_utils import load_data
from Model_Experimentation.lr_experiment import run_lr_experiment
from Model_Experimentation.bert_experiment import run_bert_experiment
from Model_Experimentation.setfit_experiment import run_setfit_experiment

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_DB_URI   = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "Smart_Email_Classifier"

# ── Which experiments to run (set False to skip) ──────────────────────────────
RUN_LR     = True
RUN_BERT   = True
RUN_SETFIT = True


# ── Result display ─────────────────────────────────────────────────────────────

METRIC_DISPLAY_ORDER = [
    "test_accuracy",
    "test_f1_weighted",
    "test_f1_macro",
    "test_precision_weighted",
    "test_recall_weighted",
    "test_roc_auc",
    "test_avg_precision",
]

METRIC_LABELS = {
    "test_accuracy":           "Accuracy",
    "test_f1_weighted":        "F1 (weighted)",
    "test_f1_macro":           "F1 (macro)",
    "test_precision_weighted": "Precision (weighted)",
    "test_recall_weighted":    "Recall (weighted)",
    "test_roc_auc":            "ROC-AUC",
    "test_avg_precision":      "Avg. Precision",
}


def _print_banner(text: str) -> None:
    width = 64
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  {text:<{width-2}}║")
    print(f"╚{border}╝")


def _print_results_table(results: dict[str, dict]) -> None:
    """Pretty-print a comparison table of all experiment results."""
    if not results:
        print("  (no results to display)")
        return

    models = list(results.keys())
    col_w  = max(24, *(len(m) + 2 for m in models))
    label_w = 24

    # Header
    header = f"  {'Metric':<{label_w}}" + "".join(f"{m:^{col_w}}" for m in models)
    sep    = "  " + "─" * (label_w + col_w * len(models))
    print(sep)
    print(header)
    print(sep)

    for metric_key in METRIC_DISPLAY_ORDER:
        label = METRIC_LABELS.get(metric_key, metric_key)
        row = f"  {label:<{label_w}}"
        for model in models:
            val = results[model].get(metric_key)
            if val is None:
                row += f"{'N/A':^{col_w}}"
            else:
                row += f"{val:^{col_w}.4f}"
        print(row)

    print(sep)


def _fetch_mlflow_results(experiment_name: str, run_ids: list[str]) -> None:
    """Query MLflow and print a final comparison table from stored runs."""
    print("\n  Fetching results from MLflow …")
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print("  Could not find experiment in MLflow.")
        return

    rows = []
    for run_id in run_ids:
        try:
            run  = client.get_run(run_id)
            tags = run.data.tags
            name = run.info.run_name or run_id[:8]
            mtype = tags.get("model_type", "unknown")
            metrics = {k: v for k, v in run.data.metrics.items()
                       if k.startswith("test_")}
            rows.append({"Run": name, "Type": mtype, **metrics})
        except Exception:
            continue

    if not rows:
        print("  No runs found.")
        return

    df = pd.DataFrame(rows).set_index("Run")

    # Select and rename metric columns
    display_cols = [c for c in METRIC_DISPLAY_ORDER if c in df.columns]
    df = df[["Type"] + display_cols].rename(columns=METRIC_LABELS)

    # Format floats
    float_cols = [METRIC_LABELS[c] for c in display_cols]
    df[float_cols] = df[float_cols].applymap(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print("\n" + df.to_string())
    print()


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    _print_banner(f"Smart Email Classifier  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  MLflow backend : {MLFLOW_DB_URI}")
    print(f"  Experiment     : {EXPERIMENT_NAME}")

    # ── Load & split data once (shared across all experiments) ────────────────
    _print_banner("Step 0 · Loading & splitting data")
    train_df, val_df, test_df = load_data()
    print(f"\n  Data loaded  →  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ── Track run IDs and local results ──────────────────────────────────────
    run_ids: list[str] = []
    local_results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    # ────────────────────────────────────────────────────────────────────────
    # Experiment 1 · Logistic Regression
    # ────────────────────────────────────────────────────────────────────────
    if RUN_LR:
        _print_banner("Experiment 1 · Logistic Regression (TF-IDF)")
        t0 = time.time()

        # Capture the active run ID right after start
        with mlflow.start_run(nested=False) as _probe:
            pass  # just to warm up — actual run is created inside lr_experiment

        metrics = run_lr_experiment(train_df, val_df, test_df)
        timings["Logistic Regression"] = time.time() - t0
        local_results["Logistic Regression"] = metrics

        # Retrieve the last created run for this experiment
        client  = mlflow.tracking.MlflowClient()
        exp_obj = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs    = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string='tags.model_type = "logistic_regression"',
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            run_ids.append(runs[0].info.run_id)

        print(f"\n  ⏱  Completed in {timings['Logistic Regression']:.1f}s")

    # ────────────────────────────────────────────────────────────────────────
    # Experiment 2 · DistilBERT
    # ────────────────────────────────────────────────────────────────────────
    if RUN_BERT:
        _print_banner("Experiment 2 · DistilBERT Fine-Tuning")
        t0 = time.time()
        metrics = run_bert_experiment(train_df, val_df, test_df)
        timings["DistilBERT"] = time.time() - t0
        local_results["DistilBERT"] = metrics

        client  = mlflow.tracking.MlflowClient()
        exp_obj = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs    = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string='tags.model_type = "distilbert"',
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            run_ids.append(runs[0].info.run_id)

        print(f"\n  ⏱  Completed in {timings['DistilBERT']:.1f}s")

    # ────────────────────────────────────────────────────────────────────────
    # Experiment 3 · SetFit
    # ────────────────────────────────────────────────────────────────────────
    if RUN_SETFIT:
        _print_banner("Experiment 3 · SetFit (Few-Shot)")
        t0 = time.time()
        metrics = run_setfit_experiment(train_df, val_df, test_df)
        timings["SetFit"] = time.time() - t0
        local_results["SetFit"] = metrics

        client  = mlflow.tracking.MlflowClient()
        exp_obj = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs    = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string='tags.model_type = "setfit"',
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            run_ids.append(runs[0].info.run_id)

        print(f"\n  ⏱  Completed in {timings['SetFit']:.1f}s")

    # ────────────────────────────────────────────────────────────────────────
    # Final Results Summary
    # ────────────────────────────────────────────────────────────────────────
    _print_banner("Results Summary · All Experiments (Test Set)")

    if local_results:
        _print_results_table(local_results)

    # Timing breakdown
    if timings:
        print("  Training time breakdown:")
        for name, secs in timings.items():
            mins, sec = divmod(int(secs), 60)
            print(f"    {name:<24} {mins:>2}m {sec:02d}s")

    # Best model
    if local_results:
        best = max(local_results, key=lambda m: local_results[m].get("test_f1_weighted", 0))
        best_f1 = local_results[best].get("test_f1_weighted", 0)
        print(f"\n  🏆  Best model  →  {best}  (F1-weighted = {best_f1:.4f})")

    # ── MLflow cross-run results (from DB) ────────────────────────────────────
    if run_ids:
        _print_banner("MLflow Stored Results (from DB)")
        _fetch_mlflow_results(EXPERIMENT_NAME, run_ids)

    _print_banner("Done · Launch MLflow UI to explore runs interactively")
    print(f"\n  Run:  mlflow ui --backend-store-uri {MLFLOW_DB_URI}")
    print(f"  Open: http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()
