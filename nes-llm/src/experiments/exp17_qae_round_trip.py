"""
W1.1 — QAE round trip.

RESEARCH_PLAN §2 W1.1: two quantization-aware strategies exist on disk
and have never been run. `QuantizationStrategy` implements the
per-tensor `EmbeddingStrategy` ABC, not production's dict contract —
the registry now carries `QaeDictAdapter`, which delegates each layer
to the ABC's own embed() so its margin logic runs unmodified, and
wraps the result in an EmbeddingResult exactly as BaseEmbedder does.

This experiment measures what the plan asks for: a clean
embed -> extract -> decrypt round trip through the production path
(exp3's flow, byte for byte — same DecryptPipeline, same honest BER
against the transmitted bit sequence rather than a stats field), plus
the registry's own no-cover probe for both strategies.

Two records, honestly separated:

    qae      READY — measured here.
    nf4_qae  BLOCKED — its reference residual needs the fp16/nf4
             weight tensors, which neither strategy.embed nor
             EmbeddingConfig carries and no caller supplies; the
             probe's recorded error IS the measurement of its
             status. Registered, not faked, not silently embedded.

Trap note (plan §2 W1.1): QAE is quantization-*aware*. Any future
comparison where QAE wins must say "QAE is well-matched to NF4", not
"LWE is stealthy". This experiment compares nothing — it only
establishes the round trip.

Gate: THRESHOLDS['exp17'].max_ber = 0.0 — exp3's own number, reused,
plus exp3's decrypt-and-match conditions.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    probe_extraction,
    spec,
    structural_report,
)
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp17"

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36

PAYLOAD_BITS = 50_000
MESSAGE = "A" * 6_000

PROBED_STRATEGIES = ("qae", "nf4_qae")


def main() -> int:
    torch.manual_seed(42)

    print("Loading residuals from cache ...", flush=True)
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    # --- 1. Production round trip through the new adapter -----------
    print("Embedding via strategy='qae' ...", flush=True)
    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy="qae",
        model_family=FAMILY,
        num_hidden_layers=NUM_LAYERS,
    )
    result = IntelligentEmbedder(config).embed(MESSAGE, residuals)
    print(
        f"  {result.total_bits} bits, "
        f"{result.bits_embedded} embedded, "
        f"{sum(1 for v in result.layer_allocation.values() if v > 0)} "
        f"layers",
        flush=True,
    )

    from src.extraction.decrypt_pipeline import DecryptPipeline

    pipeline = DecryptPipeline(key=result.key)
    recovered, stats = pipeline.run(
        result.embedded_residuals,
        result.carrier_indices,
    )
    matches = recovered == MESSAGE
    decrypt_ok = bool(stats.get("success", False))

    # exp3's rule: BER against the transmitted bits, never a stats
    # field that could yield None and report a clean trip silently.
    extracted_bits = pipeline.extract_bits_only(
        result.embedded_residuals,
        result.carrier_indices,
    )
    transmitted_bits = result.embedded_bits
    compared = min(len(transmitted_bits), len(extracted_bits))
    errors = sum(
        1
        for a, b in zip(transmitted_bits[:compared], extracted_bits[:compared])
        if a != b
    )
    ber = errors / compared if compared else 1.0

    # --- 2. No-cover probes for both registered strategies ----------
    probes: Dict[str, Any] = {}
    for name in PROBED_STRATEGIES:
        report = probe_extraction(
            name,
            residuals,
            transmitted_bits,
            result.carrier_indices,
        )
        probes[name] = report
        print(
            f"  probe {name}: embed_ok={report['embed_ok']} "
            f"no_cover_ok={report['extract_without_cover_ok']} "
            f"ber={report['ber']} usable={report['structurally_usable']} "
            f"error={report['error'] or '-'}",
            flush=True,
        )

    # --- 3. Distortion actually written ------------------------------
    changed = 0
    total = 0
    max_delta = 0.0
    sum_delta = 0.0
    for lid, orig in residuals.items():
        delta = (result.embedded_residuals[lid] - orig).abs()
        total += delta.numel()
        n = int((delta > 0).sum().item())
        changed += n
        if n:
            max_delta = max(max_delta, float(delta.max().item()))
            sum_delta += float(delta.sum().item())

    gate = gate_for(EXPERIMENT)
    passed = decrypt_ok and matches and ber <= gate["max_ber"]

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W1.1 — QAE round trip (adapter -> production path)",
        "model_id": MODEL_ID,
        "family": FAMILY,
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "max_ber": gate["max_ber"],
            "measured_ber": ber,
            "decrypt_ok": decrypt_ok,
            "recovered_matches_original": matches,
            "status": "PASS" if passed else "FAIL",
            "gate_source": "experiment_registry.THRESHOLDS['exp17']",
        },
        "metrics": {
            "payload_bits_requested": PAYLOAD_BITS,
            "bits_embedded": result.bits_embedded,
            "bits_transmitted": len(transmitted_bits),
            "bits_extracted": len(extracted_bits),
            "bits_compared": compared,
            "bit_errors": errors,
            "ber": ber,
            "decrypt_success": decrypt_ok,
            "recovered_matches_original": matches,
            "layers_used": sum(
                1 for v in result.layer_allocation.values() if v > 0
            ),
            "total_values": total,
            "changed_values": changed,
            "changed_ratio": changed / total if total else None,
            "max_abs_delta": max_delta,
            "mean_abs_delta_over_changed": (
                sum_delta / changed if changed else None
            ),
        },
        "probes": probes,
        "structural": [
            r
            for r in structural_report()
            if r["strategy"] in PROBED_STRATEGIES
        ],
        "method": {
            "adapter": (
                "QaeDictAdapter in strategy_registry — per-layer "
                "delegation to QuantizationStrategy.embed, "
                "BaseEmbedder-shaped result wrapping"
            ),
            "round_trip": (
                "exp3's exact flow: DecryptPipeline, BER compared "
                "against the transmitted bit sequence"
            ),
            "probe": (
                "probe_extraction without the cover (residuals_ref "
                "deliberately withheld)"
            ),
            "strategy_status": {
                name: spec(name).status for name in PROBED_STRATEGIES
            },
        },
        "notes": (
            "qae is measured through the production path; nf4_qae is "
            "registered BLOCKED and its probe error records why "
            "(reference residual needs weight tensors the contract "
            "does not carry). Per plan §2 W1.1: QAE is "
            "quantization-aware — any comparison it wins must be "
            "stated as 'well-matched to NF4', not as a stealth claim "
            "about other schemes."
        ),
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_qae_round_trip.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"qae round trip: BER = {ber} over {compared} bits "
          f"(errors {errors})")
    print(f"decrypt_ok={decrypt_ok} matches={matches}")
    print(f"changed {changed}/{total} values, "
          f"mean |delta| over changed = "
          f"{artifact['metrics']['mean_abs_delta_over_changed']:.6f}")
    print(f"gate (<= {gate['max_ber']}): {artifact['status']}")
    print("=" * 70)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
