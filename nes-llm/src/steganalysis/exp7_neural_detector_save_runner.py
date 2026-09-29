"""
Exp7 result-saving runner.

Runs the existing Exp7 neural detector unchanged and captures the
final evaluation metrics into a JSON artifact for Exp8.

This file intentionally does NOT modify:
- exp7_neural_detector.py
- the detector architecture
- the dataset builder
- the embedding pipeline
"""

import json
from pathlib import Path

import src.steganalysis.exp7_neural_detector as exp7


OUTPUT_PATH = Path("results/exp7_neural_detector_results.json")

_captured_metrics = None
_original_evaluate = exp7.evaluate


def capture_evaluate(model, loader, device):
    """Capture every evaluation; the final call becomes the final result."""
    global _captured_metrics

    metrics = _original_evaluate(
        model,
        loader,
        device,
    )

    _captured_metrics = dict(metrics)

    return metrics


def main():
    global _captured_metrics

    # Monkey-patch only this imported module instance for this run.
    # The original Exp7 source file is not changed.
    exp7.evaluate = capture_evaluate

    exp7.main()

    if _captured_metrics is None:
        raise RuntimeError(
            "Exp7 completed without producing evaluation metrics."
        )

    accuracy = float(_captured_metrics["accuracy"])

    result = {
        "experiment": "exp7_neural_detector",
        "model_id": "Qwen/Qwen2.5-3B",
        "dataset_path": exp7.DATASET_PATH,
        "seed": exp7.SEED,
        "test_ratio": exp7.TEST_RATIO,
        "epochs": exp7.EPOCHS,
        "batch_size": exp7.BATCH_SIZE,
        "learning_rate": exp7.LEARNING_RATE,
        "hidden_1": exp7.HIDDEN_1,
        "hidden_2": exp7.HIDDEN_2,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "tp": int(_captured_metrics["tp"]),
        "tn": int(_captured_metrics["tn"]),
        "fp": int(_captured_metrics["fp"]),
        "fn": int(_captured_metrics["fn"]),
        "threshold": 0.55,
        "gate_rule": "accuracy <= 0.55",
        "status": "PASS" if accuracy <= 0.55 else "FAIL",
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EXP7 RESULT ARTIFACT SAVED")
    print("=" * 70)
    print(f"JSON: {OUTPUT_PATH}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Gate: {result['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
