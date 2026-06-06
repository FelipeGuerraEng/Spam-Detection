from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMS_TEST_DATASET_PATH = PROJECT_ROOT / "sms+spam+collection" / "SMSSpamCollection"
HUGGINGFACE_MODEL_ID = "Goodmotion/spam-mail-classifier"
HUGGINGFACE_LABELS = ("NOSPAM", "SPAM")
HUGGINGFACE_MAX_LENGTH = 128


@dataclass(frozen=True)
class HuggingFaceSpamClassifier:
    tokenizer: Any
    model: Any


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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_files_only = _is_huggingface_model_cached(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model.eval()
    return HuggingFaceSpamClassifier(tokenizer=tokenizer, model=model)


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

    import torch

    inputs = classifier.tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=HUGGINGFACE_MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = classifier.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    nospam_probability = float(probabilities[0].item())
    spam_probability = float(probabilities[1].item())
    label = "spam" if spam_probability >= nospam_probability else "ham"
    source_label = HUGGINGFACE_LABELS[1] if label == "spam" else HUGGINGFACE_LABELS[0]

    return {
        "label": label,
        "spam_probability": spam_probability,
        "confidence": max(nospam_probability, spam_probability),
        "source_label": source_label,
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
