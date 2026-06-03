from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "spam.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spam_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
RANDOM_STATE = 42


def load_sms_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the SMS Spam Collection dataset."""
    df = pd.read_csv(data_path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""].copy()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Duplicate SMS messages can leak almost identical samples across train/test.
    df = df.drop_duplicates(subset=["message", "label"]).reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    """Build the text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, Any]:
    """Train, evaluate, and persist the spam detector."""
    df = load_sms_dataset(data_path)
    x = df["message"]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "tfidf__min_df": [1, 2],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.5, 1.0, 2.0],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]

    label_counts = df["label"].value_counts().to_dict()
    label_distribution = (df["label"].value_counts(normalize=True) * 100).round(2).to_dict()
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics: dict[str, Any] = {
        "dataset": {
            "rows_after_cleaning": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "label_distribution_percent": {
                str(k): float(v) for k, v in label_distribution.items()
            },
            "is_imbalanced": bool(label_distribution.get("spam", 0) < 35),
        },
        "split": {
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "test_size": 0.2,
            "stratified": True,
            "random_state": RANDOM_STATE,
        },
        "best_params": search.best_params_,
        "cv_best_f1": float(search.best_score_),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_spam": float(precision_score(y_test, y_pred, pos_label=1)),
            "recall_spam": float(recall_score(y_test, y_pred, pos_label=1)),
            "f1_spam": float(f1_score(y_test, y_pred, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
        },
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": cm.tolist(),
            "tn_ham_as_ham": int(cm[0, 0]),
            "fp_ham_as_spam": int(cm[0, 1]),
            "fn_spam_as_ham": int(cm[1, 0]),
            "tp_spam_as_spam": int(cm[1, 1]),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load a trained spam detector."""
    return joblib.load(model_path)


def predict_message(model: Pipeline, message: str) -> dict[str, float | str]:
    """Return spam probability and predicted label for one message."""
    text = message.strip()
    if not text:
        return {"label": "empty", "spam_probability": 0.0}

    spam_probability = float(model.predict_proba([text])[0, 1])
    label = "spam" if spam_probability >= 0.5 else "ham"
    return {"label": label, "spam_probability": spam_probability}
