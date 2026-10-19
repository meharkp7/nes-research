"""
W1.3 — strategy x model matrix, first pass (3 models).

RESEARCH_PLAN §2 W1.3: "All viable strategies x 7 models, one table,
four axes: extractability, BER, robustness (BER @ sigma=0.001/0.002),
detectability." Cost control from the plan: first pass on 3 models,
one per size class, one model per process.

    small  google/gemma-2-2b     26 layers
    mid    Qwen/Qwen2.5-3B       36 layers
    large  meta-llama/Llama-3.1-8B  32 layers

The plan's small-class pick was TinyLlama 1.1B; its cache is
incomplete and the standing rule is SKIPPED, never defaulted —
gemma-2-2b (complete cache, the exp15 second model) takes the slot.

Four axes per (strategy, model) cell are exp10's own three
measurement functions — the protocol is unchanged, only parametrised
by the model under test:

    extractability   round trip, strategy's own extractor, no cover
                     (= BER axis; DecryptPipeline consumability
                     recorded alongside as exp10 records it)
    robustness       BER curve at sigma 0.0 / 0.001 / 0.002 / 0.005,
                     3 trials per nonzero sigma (exp6's grid)
    detectability    carrier-centered neural detector trained on the
                     strategy's own embedding: 400 pairs / 20
                     embeds / 30 epochs (exp10's, 800 samples >= the
                     suite's 400-pair floor)

Strategies: the four READY ones. The two not run are excluded BY
NAME, never silently dropped:

    neural    NEEDS_TRAINING — W1.2, never trained; not in §7 order
    nf4_qae   BLOCKED — exp17: its reference residual needs the
              fp16/nf4 weight tensors the embed contract does not
              carry

Gates: THRESHOLDS['exp18'] — four reused numbers, none new:
round trip exp3's 0.0, robustness exp6's 0.02 @ 0.001 and 0.10 @
0.002, detector exp7's 0.55. A cell "wins" (exp10's word) only if it
passes all three; cells that fail any axis are reported as failing,
per cell, with no experiment-level PASS/FAIL — this is a measurement
table, and sign's detectability failing on one model is data, not an
error.

Usage:
    python -m src.experiments.exp18_strategy_model_matrix --model <id>
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.embedding.strategy_registry import spec  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp10_strategy_comparison import (  # noqa: E402
    EMBEDDINGS_PER_VARIANT,
    EPOCHS,
    PAIRS,
    PAYLOAD_BITS,
    PATCH_SIZE,
    SEED,
    measure_detectability,
    measure_extractability,
    measure_robustness,
)
# _context_for/_slug: registry family/layers with exp15's gemma-2-2b
# fallback, so family/layers have one source across standalone runs.
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp18"

# First pass, one model per size class (§2 W1.3 cost control).
DEFAULT_MODELS = [
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.1-8B",
]

STRATEGIES = ["sign", "magnitude_aware", "lwe", "qae"]

EXCLUDED: Dict[str, str] = {
    "neural": (
        "NEEDS_TRAINING — train_sampled() never run (W1.2, not in "
        "§7's suggested order); registering it here untrained would "
        "measure a load error, not a strategy."
    ),
    "nf4_qae": (
        "BLOCKED — exp17: reference residual needs the fp16/nf4 "
        "weight tensors neither strategy.embed nor EmbeddingConfig "
        "carries."
    ),
}


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def measure_cell(residuals, strategy: str, family: str, layers: int,
                 device) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    log(f"  [{strategy}] extractability (round trip, no cover) ...")
    ext = measure_extractability(
        residuals, strategy, family=family, num_layers=layers
    )

    log(f"  [{strategy}] robustness (sigma grid, 3 trials each) ...")
    curve = measure_robustness(
        residuals, strategy, family=family, num_layers=layers
    )

    log(f"  [{strategy}] detectability (400-pair detector) ...")
    accuracy, density, samples = measure_detectability(
        residuals, strategy, device, family=family, num_layers=layers
    )

    round_trip = bool(
        ext.get("structurally_usable")
        and ext.get("ber") == gate["max_ber"]
    )
    b1 = curve.get("0.001")
    b2 = curve.get("0.002")
    robust = bool(
        b1 is not None
        and b1 <= gate["max_ber_at_sigma_0_001"]
        and b2 is not None
        and b2 <= gate["max_ber_at_sigma_0_002"]
    )
    detectable = accuracy is not None
    under_gate = bool(
        detectable and accuracy <= gate["max_detector_accuracy"]
    )

    s = spec(strategy)
    cell = {
        "strategy": strategy,
        "status": s.status,
        "forces_sign_flip": s.forces_sign_flip,
        "extract_needs_cover": s.extract_needs_cover,
        "extractability": ext,
        "robustness_ber_curve": curve,
        "meets_round_trip_gate": round_trip,
        "meets_robustness_gate": robust,
        "meets_detector_gate": under_gate,
        "detector_accuracy": accuracy,
        "detector_samples": samples,
        "signal_density": density,
        "wins": round_trip and robust and under_gate,
    }
    log(
        f"  [{strategy}] ber={ext.get('ber')} "
        f"sigma.001={b1} sigma.002={b2} "
        f"detector={accuracy} ({samples} samples) -> "
        f"round_trip={round_trip} robust={robust} "
        f"under_gate={under_gate} wins={cell['wins']}"
    )
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        required=True,
        help="model id whose residual cache is complete",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="comma-separated strategy names (default: the four READY)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
    strategies = [s for s in args.strategies.split(",") if s]

    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")
    log("Loading residuals from cache ...")
    residuals = load_cached_residuals(
        args.model, context.expected_layers
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    cells: List[Dict[str, Any]] = []
    for strategy in strategies:
        log(f"[exp18] {args.model} x {strategy} ...")
        cells.append(
            measure_cell(
                residuals, strategy,
                context.family, context.expected_layers, device,
            )
        )

    gate = gate_for(EXPERIMENT)
    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W1.3 — strategy x model matrix (one model per run)",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp18']",
        },
        "cells": cells,
        "excluded_strategies": EXCLUDED,
        "method": {
            "axes_source": (
                "exp10_strategy_comparison measurement functions, "
                "parametrised by model (family/num_layers); protocol "
                "identical to exp10/exp12"
            ),
            "payload_bits": PAYLOAD_BITS,
            "detector_pairs": PAIRS,
            "detector_embeddings": EMBEDDINGS_PER_VARIANT,
            "detector_epochs": EPOCHS,
            "patch_size": PATCH_SIZE,
            "seed": SEED,
            "robustness_sigmas": ["0.0", "0.001", "0.002", "0.005"],
            "robustness_trials": "3 per nonzero sigma, 1 at 0.0",
            "one_model_per_process": True,
            "first_pass_models": DEFAULT_MODELS,
            "strategy_notes": {
                s: spec(s).notes for s in strategies
            },
        },
        "notes": [
            "Per-cell verdicts only: a cell that fails an axis is "
            "reported failing for that axis, never as an experiment "
            "error. 'wins' = round trip + robustness + under detector "
            "gate, exp10's definition.",
            "Detector accuracy is strategy-specific (each strategy's "
            "own dataset), comparable as 'this detector against this "
            "embedding' — exp10's comparability note applies.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_matrix_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    for cell in cells:
        log(f"{cell['strategy']:16s} "
            f"ber={cell['extractability'].get('ber')} "
            f"det={cell['detector_accuracy']} "
            f"wins={cell['wins']}")
    log(f"wrote {target}")

    del residuals
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
