"""Experimental NF4 interior-code channel; no novelty or security claim.

The candidate leaves codes 0 and 15 (block extrema) untouched, and maps
interior codes only to interior codes. Sender and receiver reconstruct the
eligible set from their own artifact. This avoids transmitting a carrier map.
The baseline envelope and HMAC ordering are reused from frozen B1.4 so that
the unrestricted pair arm is byte-for-byte comparable with that baseline.
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np

from scripts.contract_b_nf4_b14 import DOMAIN, NONCE, MAGIC, envelope

METHODS = ("pair_parity", "nearest_parity", "interior_nearest", "interior_pair")


def unpack(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8)
    codes = np.empty(len(data) * 2, dtype=np.uint8)
    codes[::2], codes[1::2] = data >> 4, data & 15
    return codes


def pack(codes: np.ndarray) -> bytes:
    codes = np.asarray(codes)
    if codes.ndim != 1 or len(codes) % 2 or codes.dtype.kind not in "iu":
        raise ValueError("Expected an even one-dimensional integer code array")
    if np.any(codes < 0) or np.any(codes > 15):
        raise ValueError("NF4 code out of range")
    return ((codes[::2] << 4) | codes[1::2]).astype(np.uint8).tobytes()


def eligible(codes: np.ndarray, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == "interior_nearest":
        return (codes > 0) & (codes < 15)
    if method == "interior_pair":
        return (codes >= 2) & (codes <= 13)
    return np.ones(len(codes), dtype=bool)


def carrier_positions(codes: np.ndarray, count: int, key: bytes,
                      tensor_key: str, method: str) -> np.ndarray:
    mask = eligible(codes, method)
    if len(key) < 16:
        raise ValueError("Research key must contain at least 16 bytes")
    if not 0 <= count <= int(mask.sum()) or not len(codes):
        raise ValueError("Payload exceeds eligible carrier capacity")
    if len(codes) > 2**32:
        raise ValueError("B1.4-compatible ordering supports at most 2**32 codes")
    out, seen, counter = [], set(), 0
    while len(out) < count:
        digest = hmac.new(key, DOMAIN + NONCE + tensor_key.encode()
                          + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
        for off in range(0, 32, 4):
            pos = int.from_bytes(digest[off:off + 4], "big") % len(codes)
            if mask[pos] and pos not in seen:
                seen.add(pos)
                out.append(pos)
                if len(out) == count:
                    break
    return np.asarray(out, dtype=np.int64)


def embed(raw: bytes, payload: bytes, key: bytes, tensor_key: str,
          method: str, codebook) -> tuple[bytes, dict]:
    if not payload:
        raise ValueError("Empty application payload is unsupported by B1.4 framing")
    codes = unpack(raw)
    cb = np.asarray(codebook, dtype=np.float64)
    if cb.shape != (16,) or not np.isfinite(cb).all() or not np.all(np.diff(cb) > 0):
        raise ValueError("Expected a finite, strictly increasing 16-entry NF4 codebook")
    if cb[0] != -1 or cb[-1] != 1:
        raise ValueError("Expected NF4 endpoint anchors -1 and +1")
    bits = np.unpackbits(np.frombuffer(envelope(payload), dtype=np.uint8))
    pos = carrier_positions(codes, len(bits), key, tensor_key, method)
    updated = codes.copy()
    if method in ("pair_parity", "interior_pair"):
        updated[pos] = (codes[pos] & 14) | bits
    else:
        alphabet = np.arange(1, 15) if method == "interior_nearest" else np.arange(16)
        for bit in (0, 1):
            candidates = alphabet[alphabet % 2 == bit]
            lookup = candidates[np.argmin((cb[:, None] - cb[candidates])**2, axis=1)]
            selected = pos[bits == bit]
            updated[selected] = lookup[codes[selected]]
    if not np.array_equal(eligible(codes, method), eligible(updated, method)):
        raise AssertionError("Embedding changed receiver eligibility")
    if not np.array_equal(updated[pos] & 1, bits):
        raise AssertionError("Embedding did not reproduce the envelope bits")
    return pack(updated), {
        "net_payload_bits": len(payload) * 8,
        "framed_bits": len(bits),
        "framing_bytes": 16,
        "eligible_codes": int(eligible(codes, method).sum()),
        "total_codes": len(codes),
        "changed_codes": int(np.count_nonzero(codes != updated)),
        "carrier_map_bytes": 0,
        "reference_weight_bytes": 0,
        "delivered_file_growth_bytes": 0,
        "receiver_inputs": ["packed artifact", "key", "tensor name", "fixed method/protocol"],
    }


def extract(raw: bytes, key: bytes, tensor_key: str, method: str) -> bytes:
    codes = unpack(raw)
    pos = carrier_positions(codes, 64, key, tensor_key, method)
    header = np.packbits(codes[pos] & 1).tobytes()
    if header[:4] != MAGIC:
        raise ValueError("Envelope magic mismatch")
    size = int.from_bytes(header[4:8], "big")
    if size <= 0 or (size + 16) * 8 > int(eligible(codes, method).sum()):
        raise ValueError("Envelope length exceeds receiver capacity")
    pos = carrier_positions(codes, (size + 16) * 8, key, tensor_key, method)
    framed = np.packbits(codes[pos] & 1).tobytes()
    payload = framed[8:8 + size]
    if framed[-8:] != hashlib.sha256(payload).digest()[:8]:
        raise ValueError("Envelope checksum mismatch")
    return payload


def recovery_diagnostic(raw: bytes, expected: bytes, key: bytes,
                        tensor_key: str, method: str) -> dict:
    """Expected bytes are used only for scoring; extract() never receives them."""
    result = {"exact_recovery": False, "decode_error": None,
              "payload_bit_errors": None, "payload_ber": None}
    try:
        recovered = extract(raw, key, tensor_key, method)
        result["exact_recovery"] = recovered == expected
        result["recovered_sha256"] = hashlib.sha256(recovered).hexdigest()
    except ValueError as exc:
        result["decode_error"] = str(exc)
    try:
        codes = unpack(raw)
        pos = carrier_positions(codes, (len(expected) + 16) * 8, key, tensor_key, method)
        observed = codes[pos[64:64 + len(expected) * 8]] & 1
        wanted = np.unpackbits(np.frombuffer(expected, dtype=np.uint8))
        result["payload_bit_errors"] = int(np.count_nonzero(observed != wanted))
        result["payload_ber"] = result["payload_bit_errors"] / len(wanted)
    except ValueError:
        pass
    result["ber_uses_evaluator_known_length"] = True
    return result
