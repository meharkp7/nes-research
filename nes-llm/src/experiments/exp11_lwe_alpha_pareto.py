"""
Phase 3b — can any LWE setting satisfy both gates?

Phase 3 fixed LWE's extractability and produced a clean trade-off:

    small perturbation  -> undetectable (detector at chance)
    large perturbation  -> survives noise

The grid width is ``residual_std * alpha * scale``, so ``alpha`` moves
along exactly that trade-off. This sweeps it and records both gates per
setting.

The question is whether the two gates are satisfiable at once. If no
alpha gives BER <= 0.02 at sigma=0.001 together with detector accuracy
<= 55%, the LWE-inspired family cannot replace sign and that is a clean
negative result rather than a tuning failure.

Does not change the 55% gate or the 2% robustness gate.
"""

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import (  # noqa: E402
    IntelligentEmbedder,
)
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.exp10_strategy_comparison import (  # noqa: E402
    measure_detectability,
    measure_robustness,
)
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import (  # noqa: E402
    load_cached_residuals,
)

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36

PAYLOAD_BITS = 10_000
MESSAGE = "A" * 1_250

DETECTOR_GATE = 0.55
ROBUSTNESS_GATE = 0.02
ROBUSTNESS_SIGMA = 0.001

SEED = 42

# Which knob actually moves the grid width
# ---------------------------------------
# LWEStrategy computes
#     interval_width = max(std * alpha * scale, min_magnitude * 2)
# With std ~ 0.002 and alpha = 0.001 the first term is ~2e-6 while
# min_magnitude * 2 is 2e-3, so the floor dominates by ~1000x and the
# shipped defaults are pinned to the floor. Sweeping alpha alone would be
# a no-op, so this sweeps grid_width directly by pairing alpha with a
# min_magnitude floor that never dominates it.
#
# Values span just above the floor up to a tenth of the residual spread.
GRID_WIDTHS = [2e-3, 5e-3, 1e-2, 2e-2, 5e-2]


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading residuals from cache...", flush=True)
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}\n", flush=True)

    results: List[Dict[str, Any]] = []

    print(
        f"{'alpha':>8}{'BER@.001':>11}{'robust':>8}"
        f"{'accuracy':>10}{'detect':>9}  verdict"
    )
    print("-" * 58)

    for grid_width in GRID_WIDTHS:
        # Measure both gates at this grid width by overriding the config
        # the shared measurement helpers build.
        import src.experiments.exp10_strategy_comparison as exp10

        original_config = _patch_grid_width(exp10, grid_width)
        try:
            robustness = measure_robustness(residuals, "lwe")
            accuracy, density, n_samples = measure_detectability(
                residuals, "lwe", device
            )
        finally:
            _unpatch(exp10, original_config)

        ber = robustness.get(str(ROBUSTNESS_SIGMA))
        robust_ok = ber is not None and ber <= ROBUSTNESS_GATE
        detect_ok = accuracy is not None and accuracy <= DETECTOR_GATE
        wins = bool(robust_ok and detect_ok)

        results.append(
            {
                "grid_width": grid_width,
                "robustness_ber_curve": robustness,
                "ber_at_gate_sigma": ber,
                "meets_robustness_gate": robust_ok,
                "detector_accuracy": accuracy,
                "meets_detector_gate": detect_ok,
                "signal_density": density,
                "satisfies_both_gates": wins,
            }
        )

        print(
            f"{grid_width:>9.4f}"
            f"{ber if ber is not None else float('nan'):>11.4f}"
            f"{'PASS' if robust_ok else 'FAIL':>8}"
            f"{accuracy if accuracy is not None else float('nan'):>10.2%}"
            f"{'PASS' if detect_ok else 'FAIL':>9}"
            f"  {'BOTH' if wins else ''}",
            flush=True,
        )

    winners = [r for r in results if r["satisfies_both_gates"]]

    print()

    if winners:
        conclusion = (
            f"{len(winners)} alpha setting(s) satisfy both gates: "
            + ", ".join(str(r["grid_width"]) for r in winners)
            + ". Promote to a second model to confirm before replacing "
            "sign."
        )
    else:
        conclusion = (
            "No grid width satisfies both gates. The two gates are in direct "
            "tension for this scheme: the perturbation must be small "
            "enough to defeat the detector and large enough to survive "
            "noise. The LWE-inspired family cannot replace sign "
            "embedding as implemented, which is a negative result about "
            "this scheme rather than a tuning failure."
        )

    print(conclusion)

    target = RESULTS_DIR / "exp11_lwe_alpha_pareto.json"
    target.write_text(
        __import__("json").dumps(
            {
                "experiment": "exp11_lwe_alpha_pareto",
                "title": "LWE grid width vs both gates",
                "model_id": MODEL_ID,
                "gates": {
                    "detector_accuracy": DETECTOR_GATE,
                    "ber_at_sigma_0_001": ROBUSTNESS_GATE,
                },
                "grid_widths_swept": GRID_WIDTHS,
                "results": results,
                "winners": [r["grid_width"] for r in winners],
                "conclusion": conclusion,
                "changes_any_gate": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {target}")
    return 0


def _patch_grid_width(exp10_module, grid_width: float) -> None:
    """Force the measurement helpers to embed with this grid width.

    alpha is set to 1.0 so the width term always wins, and
    min_magnitude is set to half the width so the floor matches it
    exactly rather than clamping to the 0.002 default.
    """
    original = exp10_module.EmbeddingConfig

    def patched(*args, **kwargs):
        kwargs["alpha"] = 1.0
        kwargs["min_magnitude"] = grid_width / 2.0
        return original(*args, **kwargs)

    exp10_module.EmbeddingConfig = patched
    return original


def _unpatch(exp10_module, original) -> None:
    exp10_module.EmbeddingConfig = original


if __name__ == "__main__":
    raise SystemExit(main())