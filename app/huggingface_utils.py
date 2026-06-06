from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMS_TEST_DATASET_PATH = PROJECT_ROOT / "sms+spam+collection" / "SMSSpamCollection"
HUGGINGFACE_MODEL_ID = "Goodmotion/spam-mail-classifier"
SECONDARY_HUGGINGFACE_MODEL_ID = "niru-nny/SMS_Spam_Detection"
HUGGINGFACE_MODEL_SPECS = {
    "goodmotion": {
        "label": "Hugging Face: Goodmotion spam-mail-classifier",
        "model_id": HUGGINGFACE_MODEL_ID,
    },
    "niru_nny": {
        "label": "Hugging Face: niru-nny SMS Spam Detection",
        "model_id": SECONDARY_HUGGINGFACE_MODEL_ID,
    },
}
HUGGINGFACE_LABELS = ("HAM", "NOSPAM", "NOT_SPAM", "LABEL_0", "SPAM", "LABEL_1")
HUGGINGFACE_MAX_LENGTH = 128


@dataclass(frozen=True)
class HuggingFaceSpamClassifier:
    model_id: str
    pipeline: Any


def load_sms_test_dataset(data_path: Path = SMS_TEST_DATASET_PATH) -> pd.DataFrame:
    """Load the external SMS collection for validation only."""
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        usecols=[0, 1],
    )
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].copy()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df


def load_huggingface_model(
    model_id: str = HUGGINGFACE_MODEL_ID,
) -> HuggingFaceSpamClassifier:
    """Load the external Hugging Face classifier for inference."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    local_files_only = _is_huggingface_model_cached(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )
    return HuggingFaceSpamClassifier(model_id=model_id, pipeline=classifier)


def predict_message_huggingface(
    classifier: HuggingFaceSpamClassifier,
    message: str,
) -> dict[str, float | str]:
    """Return spam probability and predicted label using the Hugging Face model."""
    text = message.strip()
    if not text:
        return {
            "label": "empty",
            "spam_probability": 0.0,
            "confidence": 0.0,
            "source_label": "EMPTY",
        }

    raw_scores = classifier.pipeline(
        text,
        top_k=None,
        truncation=True,
        max_length=HUGGINGFACE_MAX_LENGTH,
    )
    scores = raw_scores[0] if raw_scores and isinstance(raw_scores[0], list) else raw_scores
    spam_score = 0.0
    ham_score = 0.0
    best_label = ""
    best_score = 0.0

    for item in scores:
        source_label = str(item["label"]).upper()
        score = float(item["score"])
        if score > best_score:
            best_label = source_label
            best_score = score
        if "SPAM" in source_label and not source_label.startswith(("NO", "NOT")):
            spam_score = max(spam_score, score)
        elif source_label in {"LABEL_1"}:
            spam_score = max(spam_score, score)
        else:
            ham_score = max(ham_score, score)

    if spam_score == 0.0 and ham_score == 0.0:
        spam_score = best_score if "SPAM" in best_label else 1.0 - best_score
        ham_score = 1.0 - spam_score

    label = "spam" if spam_score >= ham_score else "ham"

    return {
        "label": label,
        "spam_probability": spam_score,
        "confidence": max(spam_score, ham_score),
        "source_label": best_label,
    }


def _is_huggingface_model_cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    required_files = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    return all(
        isinstance(try_to_load_from_cache(model_id, filename), str)
        for filename in required_files
    )
