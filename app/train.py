from __future__ import annotations

import json

from app.model_utils import METRICS_PATH, MODEL_PATH, train_model


def main() -> None:
    metrics = train_model()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps(metrics["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
