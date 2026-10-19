"""
W5.3 — the sign/parity split dial (exp20).

RESEARCH_PLAN §4 W5.3: *"Parity on a fraction of carriers, sign on the
rest. Makes the stealth-vs-robustness trade-off an explicit dial rather
than a per-scheme guess. Most interesting scientifically."*

``SplitStrategy`` delegates: parity carriers go through LWEStrategy,
sign carriers through production sign, on disjoint positions. The dial
is ``split_fraction`` (EmbeddingConfig), swept here:

    f = 0.00   pure sign     (sign's robustness, sign's detectability)
    f = 0.25 / 0.50 / 0.75   the dial between them
    f = 1.00   pure parity   (LWE's stealth, LWE's fragility)

Per fraction, three axes — exp10's own measurement functions, protocol
unchanged, only the config varies (config_overrides):

    extractability   round trip, the scheme's own extractor, no cover
    robustness       BER curve, exp6's sigma grid, 3 trials
    detectability    400-pair carrier-centred neural detector (exp10's)

Anchors: f=0.0 (pure sign) must land near exp18's direct `sign` cell
on the same model and f=1.0 (pure parity) near its `lwe` cell —
exp18's artifact is read for
the comparison and recorded as a delta (anchors are checked, not
gated; a missing exp18 artifact is recorded missing, never defaulted).

Gates: THRESHOLDS['exp20'] — the same four numbers exp18 reused
(exp3's 0.0, exp6's 0.02/0.10, exp7's 0.55), per cell. Verdicts are
per cell; the TRADE-OFF SHAPE across fractions is the measurement,
not a gate. First pass: one model (Qwen2.5-3B, exp10's anchor, so the
f=0/f=1 endpoints sit beside exp10/exp18's own numbers).

Usage:
    python -m src.experiments.exp20_split_dial --model <id>
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp20"

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# split_fraction is the PARITY share (plan: "parity on a fraction of
# carriers"), so f=0.0 is pure sign and f=1.0 pure parity: f=0.0 must
# behave like exp18's `sign` cell, f=1.0 like its `lwe` cell.
ANCHOR_FRACTIONS = {0.0: "sign", 1.0: "lwe"}


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def measure_cell(residuals, fraction: float, family: str, layers: int,
                 device) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    overrides = {"split_fraction": fraction}

    log(f"  [f={fraction:.2f}] extractability (round trip, no cover) ...")
    ext = measure_extractability(
        residuals, "split", family=family, num_layers=layers,
        config_overrides=overrides,
    )

    log(f"  [f={fraction:.2f}] robustness (sigma grid, 3 trials each) ...")
    curve = measure_robustness(
        residuals, "split", family=family, num_layers=layers,
        config_overrides=overrides,
    )

    log(f"  [f={fraction:.2f}] detectability (400-pair detector) ...")
    accuracy, density, samples = measure_detectability(
        residuals, "split", device, family=family, num_layers=layers,
        config_overrides=overrides,
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
    under_gate = bool(
        accuracy is not None and accuracy <= gate["max_detector_accuracy"]
    )

    cell = {
        "split_fraction": fraction,
        "extractability": ext,
        "robustness_ber_curve": curve,
        "detector_accuracy": accuracy,
        "detector_samples": samples,
        "signal_density": density,
        "meets_round_trip_gate": round_trip,
        "meets_robustness_gate": robust,
        "meets_detector_gate": under_gate,
        "wins": round_trip and robust and under_gate,
    }
    log(
        f"  [f={fraction:.2f}] ber={ext.get('ber')} "
        f"sigma.001={b1} sigma.002={b2} "
        f"detector={accuracy} ({samples} samples) -> "
        f"round_trip={round_trip} robust={robust} "
        f"under_gate={under_gate} wins={cell['wins']}"
    )
    return cell


def exp18_anchors(model_slug: str) -> Dict[str, Any]:
    """exp18's same-model detector numbers for the two endpoints.

    Recorded as comparison, never gated; a missing artifact or cell is
    reported as such — never defaulted to a number.
    """
    source = f"exp18_matrix_{model_slug}.json"
    out: Dict[str, Any] = {"source": source, "available": False,
                           "anchors": {}}
    path = RESULTS_DIR / source
    if not path.exists():
        out["reason"] = "exp18 artifact missing on disk"
        return out
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    out["available"] = True
    cells = {
        c.get("strategy"): c.get("detector_accuracy")
        for c in artifact.get("cells", [])
    }
    for fraction, strategy in ANCHOR_FRACTIONS.items():
        out["anchors"][str(fraction)] = {
            "exp18_strategy": strategy,
            "exp18_detector_accuracy": cells.get(strategy),
            "present": strategy in cells,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp10's anchor)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
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
    for fraction in FRACTIONS:
        log(f"[exp20] split_fraction={fraction} ...")
        cells.append(
            measure_cell(
                residuals, fraction,
                context.family, context.expected_layers, device,
            )
        )

    anchors = exp18_anchors(_slug(args.model))
    # Endpoint deltas, computed only from numbers actually present.
    for fraction, anchor in anchors.get("anchors", {}).items():
        f = float(fraction)
        match = next(
            (c for c in cells if c["split_fraction"] == f), None
        )
        anchor["exp20_detector_accuracy"] = (
            match.get("detector_accuracy") if match else None
        )
        a = anchor.get("exp18_detector_accuracy")
        b = anchor.get("exp20_detector_accuracy")
        anchor["delta"] = (b - a) if (a is not None and b is not None) else None

    gate = gate_for(EXPERIMENT)
    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W5.3 — sign/parity split dial (first pass, one model)",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp20']",
        },
        "cells": cells,
        "fractions": list(FRACTIONS),
        "endpoint_anchors": anchors,
        "excluded_strategies": {},
        "method": {
            "axes_source": (
                "exp10_strategy_comparison measurement functions, "
                "same protocol, only EmbeddingConfig.split_fraction "
                "varies per cell (config_overrides)"
            ),
            "split_spec": spec("split").notes,
            "partition_rule": (
                "keyless position predicate: blake2b('nes-split-v1:"
                "{layer}:{position}') below fraction -> parity; "
                "public by design (exp13: grid width is public too)"
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
        },
        "notes": [
            "Per-cell verdicts only, exp18's rule: a cell failing an "
            "axis is reported failing for that axis, never as an "
            "experiment error. The dial's SHAPE across fractions is "
            "the measurement.",
            "Detector accuracy is trained on each cell's own "
            "embedding (exp10's comparability note): comparable as "
            "'this detector against this fraction'.",
            "Endpoint anchors vs exp18 are recorded deltas, not "
            "gates; a missing anchor stays recorded as missing.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_split_dial_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    for cell in cells:
        log(f"f={cell['split_fraction']:<5} "
            f"ber={cell['extractability'].get('ber')} "
            f"det={cell['detector_accuracy']} "
            f"sigma.002={cell['robustness_ber_curve'].get('0.002')} "
            f"wins={cell['wins']}")
    log(f"wrote {target}")

    del residuals
    if device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
