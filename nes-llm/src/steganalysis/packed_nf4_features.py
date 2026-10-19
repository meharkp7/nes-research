"""Blind, artifact-only features for packed NF4 code sequences.

No carrier positions, reference weights, payload bits, or run/path labels
are accepted by this module. Inputs are packed bytes/codes only.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from src.experiments.nf4_artifact_codec import unpack_codes


def packed_to_codes(packed: Any, n_values: int) -> np.ndarray:
    """Decode the repository's high-nibble-first packed NF4 representation."""
    codes = np.asarray(unpack_codes(packed, n_values), dtype=np.uint8)
    if codes.size != n_values:
        raise ValueError(f"Expected {n_values} codes, got {codes.size}")
    if codes.size and int(codes.max()) > 15:
        raise ValueError("NF4 codes must be in [0, 15]")
    return codes


def block_features(codes: Any, blocksize: int = 64) -> dict[str, float]:
    """Compute blind summary features over fixed, non-overlapping code blocks.

    A final partial block is allowed and identified by its length fraction.
    Features describe the code sequence; they do not establish detectability.
    """
    arr = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if blocksize < 2:
        raise ValueError("blocksize must be at least 2")
    if arr.size == 0:
        raise ValueError("codes must not be empty")
    if int(arr.max()) > 15:
        raise ValueError("NF4 codes must be in [0, 15]")

    n = int(arr.size)
    counts = np.bincount(arr, minlength=16).astype(np.float64)
    probs = counts / n
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    transitions = arr[:-1].astype(np.int16) * 16 + arr[1:].astype(np.int16)
    trans_counts = np.bincount(transitions, minlength=256).astype(np.float64)
    trans_probs = trans_counts[trans_counts > 0] / max(1, n - 1)
    transition_entropy = float(-(trans_probs * np.log2(trans_probs)).sum())

    blocks = [arr[i:i + blocksize] for i in range(0, n, blocksize)]
    changed_adjacent_rates = []
    block_entropies = []
    block_unique_counts = []
    block_max_frequencies = []
    block_lsb_fractions = []
    for block in blocks:
        if block.size == 0:
            continue
        changed_adjacent_rates.append(
            float(np.mean(block[1:] != block[:-1])) if block.size > 1 else 0.0
        )
        bc = np.bincount(block, minlength=16).astype(np.float64)
        bp = bc[bc > 0] / block.size
        block_entropies.append(float(-(bp * np.log2(bp)).sum()))
        block_unique_counts.append(float(np.count_nonzero(bc)))
        block_max_frequencies.append(float(bc.max() / block.size))
        block_lsb_fractions.append(float(np.mean(block & 1)))

    features: dict[str, float] = {
        "log_code_count": float(math.log1p(n)),
        "global_entropy_bits": entropy,
        "transition_entropy_bits": transition_entropy,
        "adjacent_change_rate": float(np.mean(arr[1:] != arr[:-1])),
        "block_count": float(len(blocks)),
        "partial_block_fraction": float((n % blocksize) / blocksize),
        "block_entropy_mean": float(np.mean(block_entropies)),
        "block_entropy_std": float(np.std(block_entropies)),
        "block_unique_mean": float(np.mean(block_unique_counts)),
        "block_unique_std": float(np.std(block_unique_counts)),
        "block_max_frequency_mean": float(np.mean(block_max_frequencies)),
        "block_lsb_fraction_mean": float(np.mean(block_lsb_fractions)),
        "block_lsb_fraction_std": float(np.std(block_lsb_fractions)),
        "block_adjacent_change_rate_mean": float(np.mean(changed_adjacent_rates)),
        "block_adjacent_change_rate_std": float(np.std(changed_adjacent_rates)),
    }
    for code, count in enumerate(counts):
        features[f"code_frequency_{code:02d}"] = float(count / n)

    return features
