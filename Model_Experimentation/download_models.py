"""
download_models.py
──────────────────
One-time script to download all models needed for the experiments
into the local HuggingFace cache.

Run this ONCE while connected to the internet:
    uv run python Model_Experimentation/download_models.py

After this, all experiments will run fully offline using
the cached files at: C:/Users/<user>/.cache/huggingface/hub/
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Models to download ────────────────────────────────────────────────────────
MODELS = [
    "distilbert-base-uncased",                      # BERT experiment
    "sentence-transformers/all-MiniLM-L6-v2",       # SetFit experiment
]


def download_all() -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer

    for model_id in MODELS:
        print(f"\n{'─'*60}")
        print(f"  Downloading: {model_id}")
        print(f"{'─'*60}")

        try:
            if "MiniLM" in model_id or "sentence-transformers" in model_id:
                # sentence-transformers model (used by SetFit)
                _ = SentenceTransformer(model_id)
                print(f"  ✓ {model_id} cached via sentence-transformers")
            else:
                # Transformer model (BERT/DistilBERT/RoBERTa)
                _ = AutoTokenizer.from_pretrained(model_id)
                _ = AutoModelForSequenceClassification.from_pretrained(
                    model_id, num_labels=2
                )
                print(f"  ✓ {model_id} tokenizer + model cached")
        except Exception as e:
            print(f"  ✗ Failed to download {model_id}: {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  All models downloaded. You can now run offline.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    download_all()
