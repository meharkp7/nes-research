from pathlib import Path
import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Exp5 results
# ---------------------------------------------------------------------------
#
# These are the already-completed real Exp5 results for Qwen2.5-3B.
# Exp8 consumes them; it does NOT rerun the expensive PPL experiment.
#
EXP5_RESULTS = {
    "Qwen/Qwen2.5-3B": {
        "baseline_ppl": 12.4707,
        "reconstruction_control_ppl": 11.3494,
        "embedded_ppl": 11.3488,
        "ppl_delta_pct": -0.005286623081384053,
        "ppl_degradation_pct": 0.005286623081384053,
        "threshold_pct": 2.0,
        "source": "prior_completed_exp5_run",
    }
}


def get_exp5_result(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the already-completed real Exp5 result for a model.

    Exp8 uses this adapter so that it does not rerun the expensive
    WikiText-2 PPL evaluation.
    """
    result = EXP5_RESULTS.get(model_id)

    if result is None:
        return None

    return dict(result)


# ---------------------------------------------------------------------------
# Exp7 neural detector results
# ---------------------------------------------------------------------------

def find_neural_detector_result(
    model_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Read an already-produced Exp7 neural-detector result.

    Exp8 never trains a detector implicitly.

    The Exp7 result may be located in:
      1. results/ relative to the current working directory
      2. nes-llm/results/ relative to this project
      3. the parent nes-research/results/ directory

    This is intentionally an Exp8-only adapter. It does not modify or
    rerun Exp7.
    """

    adapter_file = Path(__file__).resolve()

    candidates = [
        # If Exp8 is run from nes-llm/
        Path("results/exp7_neural_detector_results.json"),

        # If this adapter is inside:
        # nes-research/nes-llm/src/model/
        adapter_file.parents[2]
        / "results"
        / "exp7_neural_detector_results.json",

        # If Exp8 is run from the outer:
        # nes-research/
        adapter_file.parents[3]
        / "results"
        / "exp7_neural_detector_results.json",

        # Backward-compatible older artifact name
        Path("results/exp7_detector_results.json"),

        adapter_file.parents[2]
        / "results"
        / "exp7_detector_results.json",

        adapter_file.parents[3]
        / "results"
        / "exp7_detector_results.json",
    ]

    seen = set()

    for path in candidates:
        path = path.resolve()

        if path in seen:
            continue

        seen.add(path)

        if not path.exists():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(data, dict):
            continue

        # Current Exp7 artifact format:
        #
        # {
        #   "experiment": "...",
        #   "model_id": "Qwen/Qwen2.5-3B",
        #   "accuracy": 0.705,
        #   ...
        # }
        if (
            data.get("model_id") == model_id
            and "accuracy" in data
        ):
            return data

        # Future / multi-model format:
        #
        # {
        #   "Qwen/Qwen2.5-3B": {
        #       "accuracy": ...
        #   }
        # }
        item = data.get(model_id)

        if isinstance(item, dict) and "accuracy" in item:
            return item

    return None