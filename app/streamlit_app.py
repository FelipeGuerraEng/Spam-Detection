from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.huggingface_utils import (
    HUGGINGFACE_MODEL_ID,
    load_huggingface_model,
    predict_message_huggingface,
)
from app.model_utils import METRICS_PATH, MODEL_PATH, load_model, predict_message


st.set_page_config(page_title="SMS Spam Detector", layout="centered")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_huggingface_model():
    return load_huggingface_model(HUGGINGFACE_MODEL_ID)


@st.cache_data
def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


st.title("SMS Spam Detector")
st.write(
    "Simple spam classifier with local and Hugging Face inference options."
)

model_options = {
    "Local TF-IDF": "local",
    "Hugging Face": "huggingface",
}
selected_model = st.radio(
    "Detection model",
    options=list(model_options.keys()),
    horizontal=True,
)
selected_model_key = model_options[selected_model]

message = st.text_area(
    "SMS message",
    height=140,
    placeholder="Example: WINNER!! Free prize available. Claim code active now...",
)

if st.button("Detect spam", type="primary"):
    if selected_model_key == "local":
        if not MODEL_PATH.exists():
            st.error(
                "The trained local model was not found. Model training is required before local inference."
            )
            st.stop()
        result = predict_message(get_model(), message)
    else:
        try:
            with st.spinner("Loading Hugging Face model..."):
                result = predict_message_huggingface(get_huggingface_model(), message)
        except Exception as exc:
            st.error("The Hugging Face model could not be loaded or executed.")
            st.caption(str(exc))
            st.stop()

    probability = result["spam_probability"]

    if result["label"] == "empty":
        st.warning("A message is required for classification.")
    elif result["label"] == "spam":
        st.error(f"Prediction: SPAM ({probability:.1%} spam probability)")
    else:
        st.success(f"Prediction: HAM ({probability:.1%} spam probability)")

    if selected_model_key == "huggingface" and result["label"] != "empty":
        st.caption(
            f"Hugging Face label: {result['source_label']} "
            f"({result['confidence']:.1%} confidence)."
        )

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
