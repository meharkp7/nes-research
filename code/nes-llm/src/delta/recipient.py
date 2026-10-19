"""
Recipient recovery — W8.1's "base model + delta + key → payload".

The recipient holds three things and never speaks to the sender:

    base model   the same (NF4, FP16) pair the sender used —
                 R_clean = W_FP16 - dequant(W_NF4)
    delta        the file ``format.py`` describes, integrity-verified
                 on load
    key          the 32-byte AES-256 key, out of band

Recovery has exactly one arithmetic path:

    R_embed = R_clean + delta          (= W_stego - W_nf4_dequant)
    bit     = 1 if R_embed >= 0 else 0 (production sign rule)
    message = AES-GCM decrypt of the bit stream

so the recipient reads the payload through the same extraction code
production used (``DecryptPipeline``), not a private reimplementation.

The delta records the strategy it was embedded with. This tool decodes
the production ``sign`` scheme — what exp3/6/7 embed and exp13 reads —
and refuses any other scheme by name rather than reading a parity/grid
stream with a sign rule and returning noise.
"""

from typing import Dict, List, Tuple

import torch

from src.delta.format import (
    DeltaIntegrityError,
    verify_delta,
)
from src.extraction.decrypt_pipeline import DecryptPipeline


class UnsupportedStrategyError(Exception):
    """The delta was embedded with a scheme this reader cannot decode."""


def reconstruct_residuals(
    clean_residuals: Dict[int, torch.Tensor],
    payload: dict,
) -> Dict[int, torch.Tensor]:
    """R_embed = R_clean + delta at the recorded carriers.

    Layers without carriers are unchanged by construction and are not
    needed to read the payload (the extractor only visits recorded
    positions), so only recorded layers are returned. Call
    ``verify_delta`` first — this function trusts the arrays it is
    given.
    """

    embedded: Dict[int, torch.Tensor] = {}

    for layer_id, pos in payload["positions"].items():
        clean_full = clean_residuals[layer_id]
        clean = clean_full.flatten().detach()
        reconstructed = clean.clone()
        reconstructed[pos] = clean[pos] + payload["values"][layer_id]
        embedded[layer_id] = reconstructed.view(clean_full.shape)

    return embedded


def recover_payload(
    clean_residuals: Dict[int, torch.Tensor],
    payload: dict,
    key: bytes,
) -> Tuple[str, dict]:
    """Base residuals + delta + key → (message, report).

    Verifies integrity first (W8.2), reconstructs the embedded
    residuals, reads them through ``DecryptPipeline``, then cross-checks
    the decoded length header against the metadata's payload length —
    a delta whose numbers were rewritten and re-signed still cannot
    claim a payload length the stream does not carry.
    """

    report = verify_delta(payload)
    meta = payload["metadata"]

    strategy = meta.get("strategy") or "sign"
    if strategy != "sign":
        raise UnsupportedStrategyError(
            f"delta was embedded with strategy {strategy!r}; this reader "
            "decodes the production 'sign' scheme only (a parity/grid "
            "scheme stores no sign information — reading it back as "
            "signs returns noise). Decode it with that scheme's own "
            "extractor via DecryptPipeline(strategy_name=..., "
            "strategy_instance=...)."
        )

    if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
        raise ValueError("key must be the 32-byte AES-256 key")

    embedded = reconstruct_residuals(clean_residuals, payload)
    carriers: Dict[int, List[int]] = {
        layer_id: pos.tolist()
        for layer_id, pos in payload["positions"].items()
    }

    pipeline = DecryptPipeline(key=bytes(key), strategy_name="sign")
    message, stats = pipeline.run(embedded, carriers)

    if not stats.get("success"):
        raise DeltaIntegrityError(
            "decryption failed: "
            f"{stats.get('error')} — wrong key, or a delta whose payload "
            "does not decrypt"
        )

    decoded_bits = stats.get("payload_bits")
    if decoded_bits != meta["payload_bits"]:
        raise DeltaIntegrityError(
            f"payload length mismatch: header decodes {decoded_bits} bits, "
            f"metadata claims {meta['payload_bits']}"
        )

    report["payload_bits_from_header"] = decoded_bits
    report["message_length"] = stats.get("message_length", len(message))
    return message, report
