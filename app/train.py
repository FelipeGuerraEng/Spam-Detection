from __future__ import annotations

import json
import argparse

from app.model_utils import LOCAL_MODEL_SPECS, train_all_local_models, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local SMS spam classifiers.")
    parser.add_argument(
        "--model",
        choices=sorted(LOCAL_MODEL_SPECS),
        default="logistic_regression",
        help="Local model to train.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train every configured local model.",
    )
    args = parser.parse_args()

    if args.all:
        all_metrics = train_all_local_models()
        for model_key, metrics in all_metrics.items():
            spec = LOCAL_MODEL_SPECS[model_key]
            print(f"{spec['label']}")
            print(f"Model saved to: {spec['model_path']}")
            print(f"Metrics saved to: {spec['metrics_path']}")
            print(json.dumps(metrics["test_metrics"], indent=2))
        return

    spec = LOCAL_MODEL_SPECS[args.model]
    metrics = train_model(
        model_path=spec["model_path"],
        metrics_path=spec["metrics_path"],
        model_key=args.model,
    )
    print(f"Model saved to: {spec['model_path']}")
    print(f"Metrics saved to: {spec['metrics_path']}")
    print(json.dumps(metrics["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
