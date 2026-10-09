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


def allocate_payload_segments(capacities: dict[str, int], payload_bits: int) -> list[tuple[str, int, int]]:
    """Spread a bitstream across selected tensors, respecting each capacity.

    Each tensor gets an approximately equal share when capacity permits. This
    prevents a nominal multi-tensor run from silently using only its first
    (usually very large) tensor.
    """
    if payload_bits < 0:
        raise ValueError("payload_bits must be non-negative")
    if any(not isinstance(cap, int) or cap < 0 for cap in capacities.values()):
        raise ValueError("tensor capacities must be non-negative integers")
    if payload_bits > sum(capacities.values()):
        raise ValueError(f"payload needs {payload_bits} bits but capacity is {sum(capacities.values())}")
    names = list(capacities)
    segments: list[tuple[str, int, int]] = []
    offset = 0
    for index, name in enumerate(names):
        remaining = payload_bits - offset
        if remaining <= 0:
            break
        tensors_left = len(names) - index
        fair_share = (remaining + tensors_left - 1) // tensors_left
        count = min(capacities[name], fair_share)
        if count:
            segments.append((name, offset, offset + count))
            offset += count
    # If an early tensor has low capacity, any unallocated bits are assigned
    # to remaining tensors in order, still without exceeding capacity.
    if offset < payload_bits:
        for name in names:
            already = sum(end - start for n, start, end in segments if n == name)
            available = capacities[name] - already
            count = min(available, payload_bits - offset)
            if count:
                segments.append((name, offset, offset + count))
                offset += count
            if offset == payload_bits:
                break
    if offset != payload_bits:
        raise ValueError("could not allocate entire payload")
    # Keep one contiguous segment per tensor, merging any overflow assignment.
    merged: dict[str, tuple[int, int]] = {}
    for name, start, end in segments:
        if name in merged:
            merged[name] = (min(merged[name][0], start), max(merged[name][1], end))
        else:
            merged[name] = (start, end)
    return [(name, *merged[name]) for name in names if name in merged]
