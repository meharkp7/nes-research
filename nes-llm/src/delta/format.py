"""
Delta file format — W8.2's integrity metadata (delta-only distribution).

The embedding changes 10,256 of ~811M values (0.0013%), so what ships
is not a modified checkpoint but a delta, shipped like a LoRA adapter:

    delta = W_stego - W_clean           (weight space)
          = R_embed - R_clean           (residual space — the same
          numbers, because W = W_nf4_dequant + R by WeightPatcher's
          definition, so the dequant baseline cancels)

nonzero only at carrier positions. The file carries exactly what an
auditor and a recipient need:

    positions {layer: [flat index]}   every carrier, in read order
    values     {layer: [float32]}      the delta at those carriers
    metadata   W8.2's own list — hash, carrier count, payload length —
               plus the identifiers needed to interpret the arrays

Integrity is verified on load (``verify_delta``):

    sha256         over canonical metadata JSON + canonical tensor
                   bytes — catches silent corruption of either side
    carrier_count  must equal what the position arrays contain
    changed_count  must equal the nonzero value count
    payload length must be internally consistent (total_bits =
                   payload_bits + the 32-bit outer header) and is
                   re-checked against the header the recipient
                   decodes at extraction time

Nothing in this module loads a model: inspection must be cheap, so a
recipient can audit a delta before paying for the base model.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.embedding.payload_encoder import PayloadEncoder


DELTA_FORMAT = "nes-delta"
DELTA_VERSION = 1

# The definition string is part of the hashed metadata: an auditor can
# see, inside the file itself, what the numbers are supposed to mean.
DELTA_DEFINITION = (
    "delta = W_stego - W_clean = R_embed - R_clean, "
    "nonzero only at carrier positions"
)


class DeltaIntegrityError(Exception):
    """The delta failed its integrity checks — refuse to use it."""


# ======================================================================
# Canonical hashing
# ======================================================================

def canonical_sha256(
    metadata: Dict[str, Any],
    positions: Dict[int, torch.Tensor],
    values: Dict[int, torch.Tensor],
) -> str:
    """sha256 over canonical metadata JSON + canonical tensor bytes.

    The metadata hash excludes the ``sha256`` field itself. Tensors are
    hashed layer by layer in sorted layer order, as contiguous little
    raw bytes of their declared dtypes — the hash describes the
    numbers, not the container torch happened to serialize them in.
    """

    meta_body = {
        k: v for k, v in metadata.items() if k != "sha256"
    }
    blob = json.dumps(
        meta_body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(blob)

    for layer_id in sorted(positions.keys()):
        digest.update(f"|{layer_id}:".encode("utf-8"))
        digest.update(
            positions[layer_id]
            .detach()
            .to(torch.int64)
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        digest.update(
            values[layer_id]
            .detach()
            .to(torch.float32)
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


# ======================================================================
# Build (sender side)
# ======================================================================

def build_delta(
    clean_residuals: Dict[int, torch.Tensor],
    embed_result,
    *,
    model_id: str,
    family: str,
) -> dict:
    """Build a delta payload from one embedding run.

    ``embed_result`` is ``IntelligentEmbedder.embed``'s return value.
    ``clean_residuals`` are the residuals the embed read from (the
    embed clones, so they are still clean).

    Refuses to build when the embed changed anything the delta format
    cannot represent: a change outside the recorded carriers would be
    silently dropped, and a payload that did not fit its carrier budget
    would extract as noise. Both raise instead.
    """

    if not embed_result.success:
        raise ValueError("embedding reported success=False; nothing to export")

    if embed_result.bits_embedded != embed_result.total_bits:
        raise ValueError(
            "payload did not fit the carrier budget: "
            f"{embed_result.bits_embedded}/{embed_result.total_bits} bits "
            "embedded — a delta built from it would extract as noise"
        )

    positions: Dict[int, torch.Tensor] = {}
    values: Dict[int, torch.Tensor] = {}
    shapes: Dict[str, List[int]] = {}
    layer_carriers: Dict[str, int] = {}
    changed_count = 0

    embedded = embed_result.embedded_residuals
    carriers = embed_result.carrier_indices

    for layer_id in sorted(embedded.keys()):
        # Model-derived residuals carry requires_grad; the delta is a
        # distribution artifact, not a graph — detach at the boundary.
        clean = clean_residuals[layer_id].flatten().detach()
        emb = embedded[layer_id].flatten().detach()
        carrier_idx = carriers.get(layer_id, [])

        if not carrier_idx:
            # No carriers recorded: the embed must not have touched
            # this layer at all.
            if not torch.equal(emb, clean):
                raise ValueError(
                    f"layer {layer_id} changed with no recorded carriers"
                )
            continue

        delta = emb - clean
        pos = torch.tensor(carrier_idx, dtype=torch.int64)

        # Completeness: the delta is zero outside the recorded
        # carriers, so positions+values over carriers IS the whole
        # change to the model.
        mask = torch.zeros(clean.numel(), dtype=torch.bool)
        mask[pos] = True
        if bool((delta[~mask] != 0).any()):
            raise ValueError(
                f"layer {layer_id}: values changed outside the recorded "
                "carriers — this format would drop them"
            )

        values_at_carriers = delta[pos].to(torch.float32)

        positions[layer_id] = pos
        values[layer_id] = values_at_carriers
        shapes[str(layer_id)] = list(clean_residuals[layer_id].shape)
        layer_carriers[str(layer_id)] = len(carrier_idx)
        changed_count += int((values_at_carriers != 0).sum().item())

    carrier_count = sum(len(p) for p in positions.values())
    total_bits = int(embed_result.total_bits)
    header = int(PayloadEncoder.HEADER_BITS)

    metadata: Dict[str, Any] = {
        "format": DELTA_FORMAT,
        "version": DELTA_VERSION,
        "definition": DELTA_DEFINITION,
        "model_id": model_id,
        "family": family,
        "strategy": embed_result.strategy or "sign",
        "key_id": embed_result.key_id,
        "payload_bits": total_bits - header,
        "total_bits": total_bits,
        "header_bits": header,
        "carrier_count": carrier_count,
        "changed_count": changed_count,
        "layer_count": len(positions),
        "shapes": shapes,
        "layer_carriers": layer_carriers,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    metadata["sha256"] = canonical_sha256(metadata, positions, values)

    return {
        "metadata": metadata,
        "positions": positions,
        "values": values,
    }


# ======================================================================
# Verify (both sides)
# ======================================================================

def verify_delta(payload: dict) -> dict:
    """Run every W8.2 integrity check. Returns a report; raises
    ``DeltaIntegrityError`` on the first failure."""

    if not isinstance(payload, dict) or "metadata" not in payload:
        raise DeltaIntegrityError("not a delta file: missing metadata")

    meta = payload["metadata"]
    positions = payload.get("positions")
    values = payload.get("values")

    if not isinstance(positions, dict) or not isinstance(values, dict):
        raise DeltaIntegrityError("missing position/value arrays")

    def fail(what: str):
        raise DeltaIntegrityError(f"delta integrity: {what}")

    if meta.get("format") != DELTA_FORMAT:
        fail(f"unknown format {meta.get('format')!r}")
    if meta.get("version") != DELTA_VERSION:
        fail(f"unsupported version {meta.get('version')!r}")

    # --- arrays: presence, dtype, alignment, bounds ------------------
    if set(positions.keys()) != set(values.keys()):
        fail("positions and values cover different layers")

    carrier_count = 0
    changed_count = 0

    for layer_id in sorted(positions.keys()):
        pos = positions[layer_id]
        val = values[layer_id]

        if pos.dtype != torch.int64 or pos.dim() != 1:
            fail(f"layer {layer_id}: positions must be a 1-D int64 tensor")
        if val.dtype != torch.float32 or val.dim() != 1:
            fail(f"layer {layer_id}: values must be a 1-D float32 tensor")
        if pos.numel() != val.numel():
            fail(
                f"layer {layer_id}: {pos.numel()} positions but "
                f"{val.numel()} values"
            )

        shape = meta.get("shapes", {}).get(str(layer_id))
        if shape is None:
            fail(f"layer {layer_id}: no shape recorded")
        numel = 1
        for dim in shape:
            numel *= dim
        if int(pos.numel()) and int(pos.max()) >= numel:
            fail(
                f"layer {layer_id}: index {int(pos.max())} out of bounds "
                f"for shape {shape}"
            )
        if len(set(pos.tolist())) != pos.numel():
            fail(f"layer {layer_id}: duplicate carrier positions")

        carrier_count += int(pos.numel())
        changed_count += int((val != 0).sum().item())

    # --- W8.2's three fields, against the arrays ---------------------
    if meta.get("carrier_count") != carrier_count:
        fail(
            f"carrier count {meta.get('carrier_count')} != "
            f"{carrier_count} positions on disk"
        )
    if meta.get("changed_count") != changed_count:
        fail(
            f"changed count {meta.get('changed_count')} != "
            f"{changed_count} nonzero values on disk"
        )
    if meta.get("layer_count") != len(positions):
        fail("layer count does not match the arrays")

    payload_bits = meta.get("payload_bits")
    total_bits = meta.get("total_bits")
    header_bits = meta.get("header_bits", PayloadEncoder.HEADER_BITS)
    if not isinstance(payload_bits, int) or payload_bits < 0:
        fail("payload length missing or negative")
    if total_bits != payload_bits + header_bits:
        fail(
            f"payload length inconsistent: total_bits {total_bits} != "
            f"payload_bits {payload_bits} + header {header_bits}"
        )
    if carrier_count < total_bits:
        fail(
            f"{carrier_count} carriers cannot hold {total_bits} bits"
        )

    # --- the hash itself ---------------------------------------------
    expected = meta.get("sha256")
    actual = canonical_sha256(meta, positions, values)
    if expected != actual:
        fail(f"sha256 mismatch (recorded {expected}, computed {actual})")

    return {
        "ok": True,
        "model_id": meta["model_id"],
        "family": meta["family"],
        "strategy": meta["strategy"],
        "key_id": meta["key_id"],
        "carrier_count": carrier_count,
        "changed_count": changed_count,
        "layer_count": len(positions),
        "payload_bits": payload_bits,
        "total_bits": total_bits,
        "sha256": actual,
        "created_utc": meta["created_utc"],
    }


# ======================================================================
# Save / load
# ======================================================================

def save_delta(payload: dict, path) -> Path:
    """Serialize a payload (hash must already be embedded)."""

    verify_delta(payload)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    return out


def load_delta(path) -> dict:
    """Load and verify a delta. Raises on corruption — never returns a
    payload that failed its integrity checks."""

    payload = torch.load(Path(path), weights_only=True)
    verify_delta(payload)
    return payload
