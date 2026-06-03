from app.model_utils import MODEL_PATH, load_model, predict_message


def test_trained_model_artifact_exists():
    assert MODEL_PATH.exists(), "Model training is required before functional tests."


def test_spam_message_is_detected():
    model = load_model()
    result = predict_message(
        model,
        "WINNER! Free cash prize available. Call now to claim the reward.",
    )

    assert result["label"] == "spam"
    assert 0.5 <= result["spam_probability"] <= 1.0


def test_ham_message_is_detected():
    model = load_model()
    result = predict_message(model, "Hi, the meeting starts soon. See everyone later.")

    assert result["label"] == "ham"
    assert 0.0 <= result["spam_probability"] < 0.5


def test_empty_message_is_handled():
    model = load_model()
    result = predict_message(model, "   ")

    assert result == {"label": "empty", "spam_probability": 0.0}
