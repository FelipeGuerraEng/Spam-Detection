from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_utils import METRICS_PATH, MODEL_PATH, load_model, predict_message


st.set_page_config(page_title="SMS Spam Detector", layout="centered")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


st.title("SMS Spam Detector")
st.write(
    "Simple spam classifier trained with TF-IDF features and Logistic Regression."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Model training is required before application startup."
    )
    st.stop()

message = st.text_area(
    "SMS message",
    height=140,
    placeholder="Example: WINNER!! Free prize available. Claim code active now...",
)

if st.button("Detect spam", type="primary"):
    result = predict_message(get_model(), message)
    probability = result["spam_probability"]

    if result["label"] == "empty":
        st.warning("A message is required for classification.")
    elif result["label"] == "spam":
        st.error(f"Prediction: SPAM ({probability:.1%} spam probability)")
    else:
        st.success(f"Prediction: HAM ({probability:.1%} spam probability)")

metrics = get_metrics()
if metrics:
    st.divider()
    st.subheader("Model metrics")
    test_metrics = metrics["test_metrics"]
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{test_metrics['accuracy']:.3f}")
    cols[1].metric("Spam precision", f"{test_metrics['precision_spam']:.3f}")
    cols[2].metric("Spam recall", f"{test_metrics['recall_spam']:.3f}")
    cols[3].metric("Spam F1", f"{test_metrics['f1_spam']:.3f}")

    dataset = metrics["dataset"]
    st.caption(
        "Dataset balance: "
        f"{dataset['label_distribution_percent']['ham']:.2f}% ham, "
        f"{dataset['label_distribution_percent']['spam']:.2f}% spam."
    )
