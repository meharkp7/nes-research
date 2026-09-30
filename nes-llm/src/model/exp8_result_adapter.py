from pathlib import Path
import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Exp5 completed result
# ---------------------------------------------------------------------------
#
# This is the real Qwen2.5-3B Exp5 result already completed separately.
# Exp8 consumes it as an experiment-specific adapter rather than rerunning
# the expensive WikiText-2 evaluation.
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
    Return the previously completed real Exp5 result for a model.

    This adapter intentionally does not rerun Exp5.
    """
    result = EXP5_RESULTS.get(model_id)

    if result is None:
        return None

    return dict(result)


# ---------------------------------------------------------------------------
# Exp7 neural detector result
# ---------------------------------------------------------------------------

def find_neural_detector_result(
    model_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a previously saved real Exp7 neural-detector result.

    Exp8 must consume an already completed Exp7 result.
    It must NOT silently train the detector again.

    The result may live in:
        nes-llm/results/
    or, for compatibility:
        nes-research/results/
    """

    adapter_file = Path(__file__).resolve()

    # ------------------------------------------------------------------
    # Important path:
    #
    # __file__ =
    # nes-research/nes-llm/src/model/exp8_result_adapter.py
    #
    # parents[0] = src/model
    # parents[1] = src
    # parents[2] = nes-llm
    # parents[3] = nes-research
    # ------------------------------------------------------------------

    project_root = adapter_file.parents[2]   # nes-llm
    repo_root = adapter_file.parents[3]      # nes-research

    candidates = [
        # Primary location:
        project_root / "results" / "exp7_neural_detector_results.json",

        # Compatibility location:
        repo_root / "results" / "exp7_neural_detector_results.json",

        # Older possible artifact name:
        project_root / "results" / "exp7_detector_results.json",

        # Older possible artifact name:
        repo_root / "results" / "exp7_detector_results.json",

        # Current working directory compatibility:
        Path.cwd() / "results" / "exp7_neural_detector_results.json",

        Path.cwd() / "results" / "exp7_detector_results.json",
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
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(data, dict):
            continue

        # --------------------------------------------------------------
        # Format 1:
        #
        # {
        #   "model_id": "Qwen/Qwen2.5-3B",
        #   "accuracy": 0.705,
        #   ...
        # }
        # --------------------------------------------------------------
        if (
            data.get("model_id") == model_id
            and "accuracy" in data
        ):
            result = dict(data)
            result["_source_path"] = str(path)
            return result

        # --------------------------------------------------------------
        # Format 2:
        #
        # {
        #   "Qwen/Qwen2.5-3B": {
        #       "accuracy": 0.705,
        #       ...
        #   }
        # }
        # --------------------------------------------------------------
        item = data.get(model_id)

        if isinstance(item, dict) and "accuracy" in item:
            result = dict(item)
            result["_source_path"] = str(path)
            return result

    return None