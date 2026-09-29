from pathlib import Path
import json


def find_neural_detector_result(model_id: str):
    """
    Exp8-only adapter.

    Search both:
    1. the current working directory's results/
    2. the actual nes-llm/results/ directory

    Exp8 only consumes an already-produced detector result.
    It never trains or modifies the detector.
    """
    here = Path(__file__).resolve()

    candidates = [
        Path("results/exp7_neural_detector_results.json"),
        here.parents[2] / "results" / "exp7_neural_detector_results.json",
        here.parents[3] / "nes-llm" / "results" / "exp7_neural_detector_results.json",
    ]

    seen = set()

    for path in candidates:
        path = path.resolve()

        if path in seen or not path.exists():
            continue

        seen.add(path)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        # Current Exp7 artifact is a single-model result.
        if data.get("model_id") == model_id and "accuracy" in data:
            return data

        # Also support a future multi-model artifact.
        item = data.get(model_id)
        if isinstance(item, dict) and "accuracy" in item:
            return item

    return None