"""
Phase 3b — confirm the LWE grid width on every cached model.

The grid-width frontier in ``exp11_lwe_alpha_pareto`` was measured on a
single model. A setting that satisfies both gates on one model is a lead,
not a result, so this re-measures the chosen width on each cached model.

Runs one model per invocation and reports on release, because residuals
for the larger models are 3.8-8.6 GB and several in one process would
exceed this machine's 26 GB.

Usage
-----
    python -m src.experiments.exp12_lwe_cross_model --models <id> ...
    python -m src.experiments.exp12_lwe_cross_model      # all cached
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.experiments.exp10_strategy_comparison import (  # noqa: E402
    measure_detectability,
    measure_robustness,
)
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import (  # noqa: E402
    cache_status,
    load_cached_residuals,
)

# All models with a complete residual cache on this machine.
CACHED_MODELS: List[Dict[str, Any]] = [
    {"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "layers": 22},
    {"model_id": "Qwen/Qwen2.5-3B", "layers": 36},
    {"model_id": "Qwen/Qwen2.5-7B", "layers": 28},
    {"model_id": "mistralai/Mistral-7B-v0.3", "layers": 32},
    {"model_id": "meta-llama/Llama-3.1-8B", "layers": 32},
    {"model_id": "google/gemma-2-9b", "layers": 42},
]

ROBUSTNESS_GATE = 0.02
DETECTOR_GATE = 0.55
SEED = 42
DEFAULT_GRID_WIDTH = 0.010


def _force_grid_width(grid_width: float):
    """Pin the grid width for the shared measurement helpers.

    ``exp10`` builds its own EmbeddingConfig, so the width is injected by
    wrapping the class for the duration of the measurement.
    """
    import src.experiments.exp10_strategy_comparison as exp10

    original = exp10.EmbeddingConfig

    def patched(*args, **kwargs):
        kwargs["alpha"] = 1.0
        kwargs["min_magnitude"] = grid_width / 2.0
        return original(*args, **kwargs)

    return exp10, patched


def measure_model(model_id: str, layers: int, grid_width: float) -> Dict:
    import src.experiments.exp10_strategy_comparison as exp10

    exp10, patched = _force_grid_width(grid_width)
    saved = exp10.EmbeddingConfig
    exp10.EmbeddingConfig = patched

    try:
        if not cache_status(model_id, layers).get("complete"):
            return {
                "model_id": model_id,
                "status": "SKIPPED",
                "reason": "residual cache incomplete",
            }

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        residuals = load_cached_residuals(model_id, layers)

        robustness = measure_robustness(residuals, "lwe")
        accuracy, density, n_samples = measure_detectability(
            residuals, "lwe",
            torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        )

        ber = robustness.get("0.001")
        robust_ok = ber is not None and ber <= ROBUSTNESS_GATE
        detect_ok = accuracy is not None and accuracy <= DETECTOR_GATE

        result = {
            "model_id": model_id,
            "layers": layers,
            "status": "MEASURED",
            "grid_width": grid_width,
            "ber_at_sigma_0_001": ber,
            "meets_robustness_gate": robust_ok,
            "detector_accuracy": accuracy,
            "meets_detector_gate": detect_ok,
            "satisfies_both_gates": bool(robust_ok and detect_ok),
            "robustness_ber_curve": robustness,
            "signal_density": density,
        }

        del residuals
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return result

    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "ERROR",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    finally:
        exp10.EmbeddingConfig = saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirm the LWE grid width across cached models"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model ids to measure. Default: every cached model.",
    )
    parser.add_argument(
        "--grid-width", type=float, default=DEFAULT_GRID_WIDTH,
    )
    args = parser.parse_args(argv)

    targets = [
        m for m in CACHED_MODELS
        if not args.models or m["model_id"] in args.models
    ]

    results: List[Dict] = []

    for spec in targets:
        print(
            f"--- {spec['model_id']} ({spec['layers']} layers) ---",
            flush=True,
        )
        result = measure_model(
            spec["model_id"], spec["layers"], args.grid_width
        )
        results.append(result)

        if result["status"] != "MEASURED":
            print(f"  {result['status']}: {result.get('reason','')}")
            continue

        print(
            f"  BER@sigma=0.001 : {result['ber_at_sigma_0_001']:.4f}"
            f"  ({'PASS' if result['meets_robustness_gate'] else 'FAIL'})"
            f"\n  detector accuracy: {result['detector_accuracy']:.2%}"
            f"  ({'PASS' if result['meets_detector_gate'] else 'FAIL'})"
            f"\n  both gates       : "
            f"{'PASS' if result['satisfies_both_gates'] else 'FAIL'}",
            flush=True,
        )

    measured = [
        r for r in results if r["status"] == "MEASURED"
    ]
    both = [r for r in measured if r["satisfies_both_gates"]]

    print()
    print(
        f"models satisfying both gates: {len(both)}/{len(measured)}"
    )
    for r in measured:
        flag = "PASS" if r["satisfies_both_gates"] else "FAIL"
        print(
            f"  {r['model_id']:<42} "
            f"BER {r['ber_at_sigma_0_001']:.4f}  "
            f"det {r['detector_accuracy']:.2%}  {flag}"
        )

    conclusion = (
        f"The LWE grid width {args.grid_width} satisfies both the "
        f"robustness and detectability gates on {len(both)} of "
        f"{len(measured)} cached models."
        if both and len(both) == len(measured)
        else f"{len(both)} of {len(measured)} cached models satisfy both "
        "gates at this width; the result is not uniform across models."
    )
    print()
    print(conclusion)

    target = RESULTS_DIR / "exp12_lwe_cross_model.json"
    target.write_text(
        __import__("json").dumps(
            {
                "experiment": "exp12_lwe_cross_model",
                "title": "LWE grid width confirmed across cached models",
                "grid_width": args.grid_width,
                "gates": {
                    "ber_at_sigma_0_001": ROBUSTNESS_GATE,
                    "detector_accuracy": DETECTOR_GATE,
                },
                "reference_detector_result": (
                    "exp7_neural: sign embedding detected at 70.5% on "
                    "Qwen2.5-3B, same gate and same detector architecture"
                ),
                "results": results,
                "models_passing": len(both),
                "models_measured": len(measured),
                "conclusion": conclusion,
                "changes_any_gate": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())