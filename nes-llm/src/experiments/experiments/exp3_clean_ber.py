"""
Experiment 3 — Clean BER.

The fundamental correctness check: embed a payload, extract it, decrypt
it, and confirm bit-exact recovery.

Gate: BER = 0 and recovered message == original message.
"""

from typing import Any, Dict

from src.core.types import EmbeddingConfig
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.experiments.experiment_registry import gate_for
from src.experiments.model_context import ModelContext

EXPERIMENT = "exp3"

PAYLOAD_BITS = 50_000
MESSAGE = "A" * 6_000


def embed_payload(
    context: ModelContext,
    payload_bits: int,
    message: str,
):
    """Run the production embedding path for one payload.

    Shared by Exp3/Exp4/Exp6/Exp7 so every experiment embeds through the
    exact same validated code path, and no experiment holds a private
    copy of the embedding logic that could drift.
    """
    config = EmbeddingConfig(
        total_payload_bits=payload_bits,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )

    embedder = IntelligentEmbedder(config)
    return embedder.embed(message, context.residuals)


def run(context: ModelContext) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Clean BER",
            "configuration": {"payload_bits": PAYLOAD_BITS},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": "Residuals unavailable; cannot embed.",
            "source": "run",
        }

    from src.extraction.decrypt_pipeline import DecryptPipeline

    result = embed_payload(context, PAYLOAD_BITS, MESSAGE)

    pipeline = DecryptPipeline(key=result.key)
    recovered, stats = pipeline.run(
        result.embedded_residuals,
        result.carrier_indices,
    )

    matches = recovered == MESSAGE
    decrypt_ok = bool(stats.get("success", False))

    # BER is measured against the transmitted bit sequence, not taken
    # from the decrypt stats: DecryptPipeline's stats dict reports
    # success/bits_extracted but carries no BER field, so reading one
    # from there would silently yield None and report a clean round trip
    # without ever comparing a single bit.
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

    metrics: Dict[str, Any] = {
        "payload_bits_requested": PAYLOAD_BITS,
        "bits_embedded": result.bits_embedded,
        "bits_transmitted": len(transmitted_bits),
        "bits_extracted": len(extracted_bits),
        "bits_compared": compared,
        "bit_errors": errors,
        "ber": ber,
        "decrypt_success": decrypt_ok,
        "recovered_matches_original": matches,
        "key_id": result.key_id,
        "layers_used": sum(
            1 for b in result.layer_allocation.values() if b > 0
        ),
        "recovery_stats": {
            k: v
            for k, v in stats.items()
            if isinstance(v, (int, float, str, bool))
        },
    }

    passed = (
        decrypt_ok
        and matches
        and ber <= gate["max_ber"]
    )

    return {
        "experiment": EXPERIMENT,
        "title": "Clean BER",
        "configuration": {
            "payload_bits": PAYLOAD_BITS,
            "message_length_chars": len(MESSAGE),
            "embedding_strategy": "sign",
            "carrier_selection": "qaci",
            "message_source": (
                "repeated 'A'; AES-256-GCM nonce is fresh per run"
            ),
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": (
            "Clean round trip recovered exactly."
            if passed
            else "Round trip did not recover the original message."
        ),
        "source": "run",
    }