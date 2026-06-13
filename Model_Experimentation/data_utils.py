"""
data_utils.py
─────────────
Shared data preparation utilities for all binary classifier experiments.

Dataset: Data_Extraction_Setup/gmail_emails.csv
  columns : subject (str), body (str), labels (str)
  classes : "Ignore" → 0 | "Need to check" → 1
  size    : ~13,668 rows  (imbalanced ~85/15)

Split strategy: stratified 70 / 15 / 15  (train / val / test)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data_Extraction_Setup" / "gmail_emails.csv"

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL2ID = {"Ignore": 0, "Need to check": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── Text helpers ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Minimal cleaning: collapse whitespace, strip leading/trailing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_input_text(row: pd.Series) -> str:
    """Concatenate subject and body with a [SEP] separator.
    Handles NaN in either field gracefully.
    """
    subject = _clean_text(row.get("subject", ""))
    body = _clean_text(row.get("body", ""))

    if subject and body:
        return f"{subject} [SEP] {body}"
    return subject or body


# ── Main loader ───────────────────────────────────────────────────────────────

def load_data(
    data_path: str | Path = DATA_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    max_body_chars: int = 2000,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # raw splits (text, label_int)
]:
    """
    Load, clean, and split the email dataset.

    Returns
    -------
    train_df, val_df, test_df
        Each DataFrame has columns: ['text', 'label', 'label_name']
        where `label` is 0/1 (int) and `label_name` is the original string.
    """
    df = pd.read_csv(data_path, encoding="utf-8")

    # ── Drop rows where both subject and body are missing ────────────────────
    both_null = df["subject"].isna() & df["body"].isna()
    df = df[~both_null].copy()

    # Truncate body to avoid memory issues with very long emails
    df["body"] = df["body"].fillna("").str[:max_body_chars]
    df["subject"] = df["subject"].fillna("")

    # ── Build combined text feature ──────────────────────────────────────────
    df["text"] = df.apply(build_input_text, axis=1)

    # ── Encode labels ────────────────────────────────────────────────────────
    df["label"] = df["labels"].map(LABEL2ID)
    df["label_name"] = df["labels"]

    # Keep only needed columns
    df = df[["text", "label", "label_name"]].dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ── Stratified split: train / (val + test) ───────────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state,
    )

    # Split temp into val / test  (equal halves when val_size == test_size)
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    _log_split_info(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print split statistics."""
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = split["label"].sum()
        neg = len(split) - pos
        print(
            f"  {name:5s}: {len(split):5d} rows | "
            f"pos={pos} ({100*pos/len(split):.1f}%) | "
            f"neg={neg} ({100*neg/len(split):.1f}%)"
        )


def get_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    """Return {class_id: weight} for handling imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = sorted(train_df["label"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values,
    )
    return dict(zip(classes, weights))


if __name__ == "__main__":
    print("Loading data …")
    train, val, test = load_data()
    print(f"\nTotal usable rows: {len(train) + len(val) + len(test)}")
    print("\nSample text:")
    print(train["text"].iloc[0][:200])
