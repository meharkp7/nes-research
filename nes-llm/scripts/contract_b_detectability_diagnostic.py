#!/usr/bin/env python3
"""Descriptive packed-code distribution comparison for Contract B.

Compares an original NF4 artifact and a B1.4 stego artifact. It is a descriptive
diagnostic, not a trained detector and not a stealth/security claim.
Run from nes-llm:
../.venv/bin/python scripts/contract_b_detectability_diagnostic.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY, TEST_KEY_HEX, find_shard, positions, read_tensor, code_at,
)


def summarize(original: bytes, stego: bytes, tensor_key: str, key: bytes) -> dict:
    if len(original) != len(stego):
        raise ValueError(f"Packed tensor lengths differ: {len(original)} vs {len(stego)}")
    n_codes = len(original) * 2
    orig_hist = [0] * 16
    stego_hist = [0] * 16
    changed_positions = []
    pair_id_changes = 0
    orig_lsb_ones = 0
    stego_lsb_ones = 0
    for i in range(n_codes):
        a = code_at(original, i)
        b = code_at(stego, i)
        orig_hist[a] += 1
        stego_hist[b] += 1
        orig_lsb_ones += a & 1
        stego_lsb_ones += b & 1
        if a != b:
            changed_positions.append(i)
            pair_id_changes += (a // 2) != (b // 2)

    p = [x / n_codes for x in orig_hist]
    q = [x / n_codes for x in stego_hist]
    total_variation = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    epsilon = 1e-15
    kl_pq = sum(a * math.log(a / max(b, epsilon)) for a, b in zip(p, q) if a > 0)
    expected_envelope_carriers = 1266 * 8
    expected_positions = set(positions(key, n_codes, expected_envelope_carriers, tensor_key))
    changed_set = set(changed_positions)
    return {
        "logical_code_count": n_codes,
        "changed_code_count": len(changed_positions),
        "changed_code_fraction": len(changed_positions) / n_codes,
        "pair_id_changes": pair_id_changes,
        "original_code_histogram": orig_hist,
        "stego_code_histogram": stego_hist,
        "histogram_total_variation_distance": total_variation,
        "histogram_kl_original_to_stego_nats": kl_pq,
        "original_lsb_one_fraction": orig_lsb_ones / n_codes,
        "stego_lsb_one_fraction": stego_lsb_ones / n_codes,
        "expected_carrier_count_for_b14_10k": expected_envelope_carriers,
        "changed_codes_within_expected_carriers": changed_set.issubset(expected_positions),
        "changed_codes_outside_expected_carriers": len(changed_set - expected_positions),
        "payload_carriers_that_changed": len(changed_set & expected_positions),
        "original_tensor_sha256": hashlib.sha256(original).hexdigest(),
        "stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--stego", required=True)
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--test-key-hex", default=TEST_KEY_HEX)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    original_dir = Path(args.original).expanduser().resolve()
    stego_dir = Path(args.stego).expanduser().resolve()
    if not original_dir.is_dir() or not stego_dir.is_dir():
        raise FileNotFoundError("Both --original and --stego must be existing checkpoint directories")
    original_shard = find_shard(original_dir, args.tensor_key)
    stego_shard = find_shard(stego_dir, args.tensor_key)
    _, _, _, _, original = read_tensor(original_shard, args.tensor_key)
    _, _, _, _, stego = read_tensor(stego_shard, args.tensor_key)
    key = bytes.fromhex(args.test_key_hex)
    metrics = summarize(original, stego, args.tensor_key, key)
    report = {
        "stage": "Contract-B-descriptive-detectability",
        "status": "MEASURED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_checkpoint": str(original_dir),
        "stego_checkpoint": str(stego_dir),
        "tensor_key": args.tensor_key,
        "metrics": metrics,
        "interpretation": (
            "Descriptive histogram/LSB comparison only. It is not a trained or held-out detector, "
            "does not estimate an attacker's success probability, and cannot establish stealth. "
            "The original artifact is used only for offline evaluation, not by the receiver."
        ),
    }
    output = Path(args.output).expanduser().resolve() if args.output else stego_dir.parent / "contract_b_detectability_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Contract B detectability] Changed codes: {metrics['changed_code_count']}/{metrics['logical_code_count']}")
    print(f"[Contract B detectability] Histogram TV distance: {metrics['histogram_total_variation_distance']:.8g}")
    print(f"[Contract B detectability] LSB one fraction: {metrics['original_lsb_one_fraction']:.6f} -> {metrics['stego_lsb_one_fraction']:.6f}")
    print(f"[Contract B detectability] Changed codes outside expected carriers: {metrics['changed_codes_outside_expected_carriers']}")
    print(f"[Contract B detectability] Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
