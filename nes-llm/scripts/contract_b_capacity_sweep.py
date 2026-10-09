#!/usr/bin/env python3
"""Contract B capacity sweep on the packed NF4 tensor, without writing checkpoints.

Run from nes-llm:
../.venv/bin/python scripts/contract_b_capacity_sweep.py \
  --checkpoint ../cache/contract_b_nf4_probe_retry \
  --sizes-bits 1000 10000 25000 50000 --repeats 3

Mechanism-level capacity/recovery only; not utility, detectability, robustness, or security.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY, TEST_KEY_HEX, embed, envelope, extract,
    find_shard, read_tensor, to_bits,
)


def make_payload(bit_count: int, trial: int) -> bytes:
    if bit_count <= 0 or bit_count % 8:
        raise ValueError("Payload sizes must be positive multiples of 8 bits")
    target = bit_count // 8
    out = bytearray()
    counter = 0
    while len(out) < target:
        out.extend(hashlib.sha256(
            b"NES-B1.4-capacity-sweep-v1" + bit_count.to_bytes(8, "big")
            + trial.to_bytes(4, "big") + counter.to_bytes(8, "big")
        ).digest())
        counter += 1
    return bytes(out[:target])


def run_sweep(checkpoint: Path, tensor_key: str, sizes: list[int], repeats: int) -> dict:
    shard = find_shard(checkpoint, tensor_key)
    _, _, _, _, packed = read_tensor(shard, tensor_key)
    base_key = bytes.fromhex(TEST_KEY_HEX)
    runs = []
    for size_bits in sizes:
        if size_bits <= 0 or size_bits % 8:
            raise ValueError(f"Invalid payload size {size_bits}; sizes must be positive and byte-aligned")
        for trial in range(repeats):
            payload = make_payload(size_bits, trial)
            wrapped = envelope(payload)
            key = hashlib.sha256(base_key + b"B1.4-capacity-trial" + trial.to_bytes(4, "big")).digest()
            stego, carrier_positions, changed, checks = embed(packed, to_bits(wrapped), key, tensor_key)
            recovered, extraction = extract(stego, key, tensor_key)
            bit_errors = sum((a ^ b).bit_count() for a, b in zip(payload, recovered))
            ber = bit_errors / (len(payload) * 8)
            passed = recovered == payload and ber == 0.0 and all(checks.values())
            runs.append({
                "payload_bits": size_bits, "payload_bytes": len(payload), "trial": trial,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "recovered_sha256": hashlib.sha256(recovered).hexdigest(),
                "exact_recovery": recovered == payload, "bit_errors": bit_errors, "ber": ber,
                "carrier_count": len(carrier_positions),
                "expected_carrier_count": (len(payload) + 16) * 8,
                "changed_code_count": changed, "checks": checks,
                "extraction_carrier_count": extraction["carrier_count"],
                "status": "PASS" if passed else "FAIL",
            })
    summaries = []
    for size_bits in sizes:
        subset = [r for r in runs if r["payload_bits"] == size_bits]
        summaries.append({
            "payload_bits": size_bits, "runs": len(subset),
            "pass_count": sum(r["status"] == "PASS" for r in subset),
            "exact_recovery_rate": sum(r["exact_recovery"] for r in subset) / len(subset),
            "max_ber": max(r["ber"] for r in subset),
            "mean_changed_codes": statistics.mean(r["changed_code_count"] for r in subset),
            "carrier_count": subset[0]["carrier_count"],
        })
    return {
        "stage": "B1.4-capacity-sweep",
        "status": "PASS" if all(r["status"] == "PASS" for r in runs) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "tensor_key": tensor_key,
        "logical_code_count": len(packed) * 2, "repeats_per_size": repeats,
        "sizes_bits": sizes, "summaries": summaries, "runs": runs,
        "interpretation": (
            "Mechanism-level packed-code capacity/recovery only. Carrier counts include the "
            "16-byte envelope. No utility, detectability, robustness, confidentiality, stealth, "
            "or novelty conclusion follows from this sweep."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--sizes-bits", nargs="+", type=int, default=[1000, 10000, 25000, 50000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    report = run_sweep(checkpoint, args.tensor_key, args.sizes_bits, args.repeats)
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.parent / "contract_b_b14_capacity_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[B1.4 capacity] Status: {report['status']}")
    for row in report["summaries"]:
        print(f"[B1.4 capacity] {row['payload_bits']:>6} bits: {row['pass_count']}/{row['runs']} PASS; "
              f"max BER={row['max_ber']:.6f}; mean changed codes={row['mean_changed_codes']:.1f}")
    print(f"[B1.4 capacity] Report: {output}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
