import pytest

from app.huggingface_utils import (
    HUGGINGFACE_MODEL_ID,
    HUGGINGFACE_LABELS,
    load_huggingface_model,
    load_sms_test_dataset,
    predict_message_huggingface,
)


SMS_COLLECTION_CASES = [
    (
        "spam",
        "WINNER!! As a valued network customer you have been selected to receivea "
        "\xa3900 prize reward! To claim call 09061701461. Claim code KL341. "
        "Valid 12 hours only.",
    ),
    (
        "spam",
        "URGENT! You have won a 1 week FREE membership in our \xa3100,000 Prize Jackpot! "
        "Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD "
        "POBOX 4403LDNW1A7RW18",
    ),
    ("ham", "I HAVE A DATE ON SUNDAY WITH WILL!!"),
    ("ham", "Ok lar... Joking wif u oni..."),
]


@pytest.fixture(scope="session")
def huggingface_classifier():
    return load_huggingface_model(HUGGINGFACE_MODEL_ID)


def test_sms_collection_dataset_loads_for_external_validation():
    df = load_sms_test_dataset()

    assert len(df) == 5572
    assert set(df["label"].unique()) == {"ham", "spam"}
    assert df["label"].value_counts().to_dict() == {"ham": 4825, "spam": 747}


@pytest.mark.huggingface
def test_huggingface_model_detects_representative_sms_collection_cases(
    huggingface_classifier,
):
    df = load_sms_test_dataset()

    for expected_label, message in SMS_COLLECTION_CASES:
        is_dataset_case = (df["label"] == expected_label) & (df["message"] == message)
        assert is_dataset_case.any()

        result = predict_message_huggingface(huggingface_classifier, message)

        assert result["label"] == expected_label
        assert result["source_label"] in HUGGINGFACE_LABELS
        assert 0.0 <= result["spam_probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0


def test_huggingface_empty_message_is_handled_without_inference():
    result = predict_message_huggingface(classifier=None, message="   ")

    assert result == {
        "label": "empty",
        "spam_probability": 0.0,
        "confidence": 0.0,
        "source_label": "EMPTY",
    }
