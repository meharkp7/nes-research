"""
Experiment 4 — Capacity Curve.

How much payload can be embedded while still recovering at BER = 0.

Reporting rule that this module enforces: the largest payload tested at
BER=0 is called "maximum tested capacity", never the absolute maximum
capacity. Nothing here establishes an upper bound; an unbounded sweep
would eventually hit the point where QACI cannot allocate the requested
bits, and that boundary is a property of the scheduler, not of the
carrier capacity (§25 rule 5).

Note on message size: the payload is carried as text, so the message
must be at least ``payload_bits / 8`` characters. Several sizes below
are marked as skipped rather than silently truncated.
"""

from typing import Any, Dict, List

from src.experiments.experiment_registry import gate_for
from src.experiments.experiments.exp3_clean_ber import embed_payload
from src.experiments.model_context import ModelContext
from src.extraction.decrypt_pipeline import DecryptPipeline

EXPERIMENT = "exp4"

# Guide asks for at least 500,000 bits and the maximum tested per model.
DEFAULT_SIZES: List[int] = [
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
]

MESSAGE_BYTE = "A"


def message_for(payload_bits: int) -> str:
    return MESSAGE_BYTE * ((payload_bits // 8) + 1)


def run(
    context: ModelContext,
    sizes: List[int] = None,
) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    sizes = list(sizes or DEFAULT_SIZES)

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Capacity Curve",
            "configuration": {"payload_sizes": sizes},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": "Residuals unavailable; cannot measure capacity.",
            "source": "run",
        }

    points: List[Dict[str, Any]] = []
    zero_ber_sizes: List[int] = []

    for size in sorted(sizes):
        print(f"  [exp4] testing {size:,} bits...")

        try:
            message = message_for(size)
            result = embed_payload(context, size, message)

            pipeline = DecryptPipeline(key=result.key)
            recovered, stats = pipeline.run(
                result.embedded_residuals,
                result.carrier_indices,
            )

            # Measured directly: DecryptPipeline's stats dict has no
            # BER field, so reading one from it would always be None.
            extracted = pipeline.extract_bits_only(
                result.embedded_residuals,
                result.carrier_indices,
            )
            transmitted = result.embedded_bits
            compared = min(len(transmitted), len(extracted))
            errors = sum(
                1
                for a, b in zip(transmitted[:compared], extracted[:compared])
                if a != b
            )

            ber = errors / compared if compared else 1.0
            matches = recovered == message
            ok = bool(stats.get("success", False)) and matches and ber == 0.0

            point = {
                "payload_bits": size,
                "bits_embedded": result.bits_embedded,
                "bits_compared": compared,
                "bit_errors": errors,
                "ber": ber,
                "recovered_matches": matches,
                "status": "PASS" if ok else "FAIL",
            }

            if ok:
                zero_ber_sizes.append(size)

        except Exception as exc:
            # A capacity failure is expected at large sizes. Record it
            # as a measured limit rather than an error that hides it.
            point = {
                "payload_bits": size,
                "status": "NOT_ACHIEVED",
                "reason": f"{type(exc).__name__}: {exc}",
            }

        points.append(point)
        print(f"  [exp4] {size:,} bits -> {point['status']}")

    max_tested_zero = max(zero_ber_sizes) if zero_ber_sizes else 0

    metrics: Dict[str, Any] = {
        "payload_sizes_tested": sorted(sizes),
        "points": points,
        "sizes_recovered_at_ber_zero": zero_ber_sizes,
        "max_tested_payload_at_ber_zero_bits": max_tested_zero,
        "guide_minimum_bits": gate["min_bits_required_by_guide"],
        "meets_guide_minimum": (
            max_tested_zero >= gate["min_bits_required_by_guide"]
        ),
    }

    passed = max_tested_zero >= gate["min_bits_required_by_guide"]

    return {
        "experiment": EXPERIMENT,
        "title": "Capacity Curve",
        "configuration": {
            "payload_sizes": sorted(sizes),
            "message_source": (
                f"repeated '{MESSAGE_BYTE}', length = ceil(bits/8)"
            ),
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": (
            f"Maximum tested payload at BER=0 is "
            f"{max_tested_zero:,} bits. This is the largest size tested, "
            "not an absolute capacity bound; no stopping criterion for "
            "the true maximum was established."
        ),
        "source": "run",
    }