"""Pure-Python helpers for packed BitsAndBytes NF4 artifact experiments.

Nibble convention: even value index is the high nibble; odd value index is
the low nibble. This module intentionally has no PyTorch dependency so CI can
test byte-level serialization without downloading model runtime dependencies.
"""
from __future__ import annotations


def _packed_bytes(packed) -> bytes:
    if isinstance(packed, (bytes, bytearray, memoryview)):
        return bytes(packed)
    # Support a CPU/GPU tensor without importing torch at module import time.
    raw = packed.detach().cpu().contiguous().view(__import__("torch").uint8).flatten().tolist()
    return bytes(raw)


def unpack_codes(packed, n_values: int) -> list[int]:
    raw = _packed_bytes(packed)
    if n_values < 0 or n_values > len(raw) * 2:
        raise ValueError("n_values must fit the packed tensor")
    return [((raw[i // 2] >> 4) & 15) if i % 2 == 0 else (raw[i // 2] & 15)
            for i in range(n_values)]


def pack_codes(codes: list[int]) -> bytes:
    if len(codes) % 2:
        raise ValueError("packed NF4 code count must be even")
    if any(not isinstance(code, int) or code < 0 or code > 15 for code in codes):
        raise ValueError("NF4 codes must be integers in [0, 15]")
    raw = bytearray(len(codes) // 2)
    for i in range(0, len(codes), 2):
        raw[i // 2] = ((codes[i] & 15) << 4) | (codes[i + 1] & 15)
    return bytes(raw)


def tensor_dequant(codes: list[int], codebook, absmax, blocksize: int) -> list[float]:
    if blocksize < 1:
        raise ValueError("blocksize must be positive")
    cb = _to_float_list(codebook)
    scales = _to_float_list(absmax)
    if len(cb) != 16:
        raise ValueError("NF4 codebook must contain 16 values")
    if codes and (max(codes) > 15 or min(codes) < 0):
        raise ValueError("NF4 codes must be in [0, 15]")
    required_scales = (len(codes) + blocksize - 1) // blocksize
    if required_scales > len(scales):
        raise ValueError("absmax scales do not cover the code sequence")
    return [cb[c] * scales[i // blocksize] for i, c in enumerate(codes)]


def _to_float_list(values) -> list[float]:
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values.detach().float().cpu().flatten().tolist()]
