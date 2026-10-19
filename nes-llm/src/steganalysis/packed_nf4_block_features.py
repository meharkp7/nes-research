"""Artifact-only, per-block features for packed NF4 code sequences.

Feature values depend only on packed NF4 codes and the declared logical value
count/block size. Carrier positions, payload/reference values, corpus contents,
artifact paths, labels, and source/run identifiers are never feature inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.nf4_artifact_codec import unpack_codes


def packed_to_codes(packed: Any, n_values: int) -> np.ndarray:
    """Decode packed bytes/tensor to uint8 logical NF4 codes."""
    if not isinstance(n_values, int) or isinstance(n_values, bool) or n_values < 0:
        raise ValueError("n_values must be a non-negative integer")
    codes = np.asarray(unpack_codes(packed, n_values), dtype=np.uint8)
    if codes.size != n_values:
        raise ValueError(f"decoded {codes.size} codes; expected {n_values}")
    if codes.size and int(codes.max()) > 15:
        raise ValueError("NF4 codes must be in [0, 15]")
    return codes


def _features_for_block(block: np.ndarray) -> dict[str, float]:
    n = int(block.size)
    if n == 0:
        raise ValueError("cannot extract features from an empty block")
    counts = np.bincount(block, minlength=16).astype(np.float64)
    probs = counts / n
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    transitions = (
        float(np.mean(block[1:] != block[:-1])) if n > 1 else 0.0
    )
    return {
        "block_value_count": float(n),
        "entropy_bits": entropy,
        "unique_code_count": float(np.count_nonzero(counts)),
        "max_code_frequency": float(counts.max() / n),
        "lsb_one_fraction": float(np.mean(block & 1)),
        "adjacent_change_rate": transitions,
        **{
            f"code_frequency_{code:02d}": float(counts[code] / n)
            for code in range(16)
        },
    }


def packed_block_feature_rows(
    packed: Any, n_values: int, blocksize: int = 64
) -> list[dict[str, float]]:
    """Return one feature-only row per contiguous logical-code block.

    Block index is intentionally not included: callers may store it as row
    metadata, but must not feed it to a classifier as a feature.
    The final partial block is retained and reports its actual value count.
    """
    if not isinstance(blocksize, int) or isinstance(blocksize, bool) or blocksize < 2:
        raise ValueError("blocksize must be an integer >= 2")
    codes = packed_to_codes(packed, n_values)
    if codes.size == 0:
        raise ValueError("packed code sequence must not be empty")
    return [
        _features_for_block(codes[start : start + blocksize])
        for start in range(0, codes.size, blocksize)
    ]
