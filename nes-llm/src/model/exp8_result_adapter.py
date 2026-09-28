"""Exp8-only adapters for previously completed experiment artifacts.

This module intentionally does not modify Exp5/Exp6/Exp7 or shared model code.
It only normalizes already-measured results for the Exp8 cross-model table.
"""

from typing import Any, Dict, Optional


# Historical Exp5 Qwen2.5-3B run completed before Exp8 aggregation.
# These are recorded here only because the original Exp5 run did not emit the
# structured artifact that Exp8 expects. Do not treat this as a new measurement.
PRIOR_EXP5_RESULTS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen2.5-3B": {
        "baseline_ppl": 12.4707,
        "reconstruction_control_ppl": 11.3494,
        "embedded_ppl": 11.3488,
        "source": "prior_completed_exp5_run",
    },
}


def get_exp5_result(model_id: str) -> Optional[Dict[str, Any]]:
    """Return a normalized Exp5 result, if a prior result is known."""
    item = PRIOR_EXP5_RESULTS.get(model_id)
    if item is None:
        return None

    control = float(item["reconstruction_control_ppl"])
    embedded = float(item["embedded_ppl"])

    # G4 measures the embedding-induced change relative to the reconstruction
    # control, not the unrelated NF4 -> reconstruction change.
    delta_pct = ((embedded - control) / control) * 100.0

    return {
        **item,
        "ppl_delta_pct": delta_pct,
        "ppl_degradation_pct": abs(delta_pct),
        "source": item.get("source", "prior_completed_exp5_run"),
    }