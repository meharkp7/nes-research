#!/usr/bin/env python3
"""B1.4 packed-code perturbation matrix.

This is a channel-level integrity/locality diagnostic, NOT a substitute for
real fine-tuning, pruning, adapter merge, task-vector merge, or requantization.
Those operations must be performed on separate checkpoint copies and evaluated
with the artifact receiver.

Run from nes-llm:
../.venv/bin/python scripts/contract_b_robustness_matrix.py \
  --stego ../cache/contract_b_nf4_b14_10k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY,
    TEST_KEY_HEX,
    code_at,
    extract,
    find_shard,
    positions,
    read_tensor,
    set_code,
)


def flip_parity(packed: bytes, indices: list[int]) -> bytes:
    """Flip only the parity/LSB of selected packed NF4 codes."""
    out = bytearray(packed)
    for index in indices:
        set_code(out, index, code_at(out, index) ^ 1)
    return bytes(out)


def measure_recovery(packed: bytes, key: bytes, tensor_key: str,
                     expected_payload: bytes) -> dict:
    try:
        recovered, detail = extract(packed, key, tensor_key)
        errors = sum((a ^ b).bit_count() for a, b in zip(expected_payload, recovered))
        # Account for a length mismatch rather than silently truncating zip().
        errors += 8 * abs(len(expected_payload) - len(recovered))
        ber = errors / max(1, len(expected_payload) * 8)
        return {
            "extraction_completed": True,
            "exact_recovery": recovered == expected_payload,
            "recovered_bytes": len(recovered),
            "bit_errors": errors,
            "ber": ber,
            "recovered_sha256": hashlib.sha256(recovered).hexdigest(),
            "checksum_valid": detail.get("checksum_valid", False),
            "error": None,
        }
    except Exception as exc:  # A failed decode is an outcome to record.
        return {
            "extraction_completed": False,
            "exact_recovery": False,
            "recovered_bytes": None,
            "bit_errors": None,
            "ber": None,
            "recovered_sha256": None,
            "checksum_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_matrix(packed: bytes, tensor_key: str, key: bytes,
               noncarrier_counts: list[int], carrier_counts: list[int],
               seed: int = 20261009) -> dict:
    payload, _ = extract(packed, key, tensor_key)
    payload_hash = hashlib.sha256(payload).hexdigest()
    n_codes = len(packed) * 2
    carrier_count = (len(payload) + 16) * 8
    carrier_positions = positions(key, n_codes, carrier_count, tensor_key)
    carrier_set = set(carrier_positions)
    noncarriers = [i for i in range(n_codes) if i not in carrier_set]
    # Keep header bits intact in the carrier-corruption leg so these trials
    # test payload/checksum integrity rather than merely header rejection.
    carrier_data_positions = carrier_positions[64:]
    rng = random.Random(seed)
    rows = [{
        "case": "pristine",
        "mutation_domain": "none",
        "mutated_code_count": 0,
        **measure_recovery(packed, key, tensor_key, payload),
        "expected_outcome": "exact recovery",
    }]

    for count in noncarrier_counts:
        if count < 0 or count > len(noncarriers):
            raise ValueError(f"Invalid non-carrier mutation count: {count}")
        chosen = rng.sample(noncarriers, count)
        mutated = flip_parity(packed, chosen)
        measured = measure_recovery(mutated, key, tensor_key, payload)
        rows.append({
            "case": f"noncarrier_lsb_flip_{count}",
            "mutation_domain": "non-carrier code LSBs only",
            "mutated_code_count": count,
            **measured,
            "expected_outcome": "exact recovery; payload carriers untouched",
        })

    for count in carrier_counts:
        if count < 1 or count > len(carrier_data_positions):
            raise ValueError(f"Invalid carrier corruption count: {count}")
        chosen = rng.sample(carrier_data_positions, count)
        mutated = flip_parity(packed, chosen)
        measured = measure_recovery(mutated, key, tensor_key, payload)
        rows.append({
            "case": f"carrier_lsb_corruption_{count}",
            "mutation_domain": "payload/envelope carriers after 64-bit header",
            "mutated_code_count": count,
            **measured,
            "expected_outcome": "checksum rejection; not silent recovery",
        })

    # Non-carrier mutations must preserve exact payload; carrier corruption
    # should be rejected by envelope checksum or otherwise fail exact recovery.
    for row in rows:
        if row["case"] == "pristine" or row["case"].startswith("noncarrier_"):
            row["meets_expectation"] = bool(row["exact_recovery"])
        else:
            row["meets_expectation"] = (
                not row["exact_recovery"] and not row["checksum_valid"]
            )

    return {
        "stage": "Contract-B-B1.4-packed-code-robustness-matrix",
        "status": "PASS" if all(r["meets_expectation"] for r in rows) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensor_key": tensor_key,
        "logical_code_count": n_codes,
        "payload_bits": len(payload) * 8,
        "payload_bytes": len(payload),
        "payload_sha256": payload_hash,
        "carrier_count_including_envelope": carrier_count,
        "seed": seed,
        "cases": rows,
        "scope": (
            "Controlled packed-code parity mutations only. Non-carrier changes "
            "test carrier locality; carrier changes test envelope error detection. "
            "This is not evidence for survival under model fine-tuning, pruning, "
            "LoRA/task-vector merging, or quantization conversion."
        ),
        "lifecycle_axes_not_run_here": [
            "fresh-process model reload (run the B1.4 receiver; previously measured separately)",
            "real LoRA fine-tuning",
            "real magnitude pruning of model weights",
            "real adapter merge",
            "real task-vector merge",
            "NF4 requantization / NF4-to-GPTQ / NF4-to-AWQ conversion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stego", required=True, help="Saved B1.4 stego checkpoint")
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--test-key-hex", default=TEST_KEY_HEX)
    parser.add_argument("--noncarrier-counts", nargs="+", type=int, default=[1, 100, 1000])
    parser.add_argument("--carrier-counts", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    folder = Path(args.stego).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    shard = find_shard(folder, args.tensor_key)
    _, _, _, _, packed = read_tensor(shard, args.tensor_key)
    key = bytes.fromhex(args.test_key_hex)
    report = run_matrix(
        packed, args.tensor_key, key,
        args.noncarrier_counts, args.carrier_counts, args.seed,
    )
    report["stego_checkpoint"] = str(folder)
    report["tensor_shard"] = str(shard.relative_to(folder))
    report["packed_tensor_sha256"] = hashlib.sha256(packed).hexdigest()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else folder.parent / "contract_b_b14_robustness_matrix.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[B1.4 robustness] Status: {report['status']}")
    for row in report["cases"]:
        outcome = "exact" if row["exact_recovery"] else "rejected/failed"
        print(
            f"[B1.4 robustness] {row['case']}: {outcome}; "
            f"mutated={row['mutated_code_count']}; "
            f"BER={row['ber'] if row['ber'] is not None else 'n/a'}; "
            f"meets_expectation={row['meets_expectation']}"
        )
    print(f"[B1.4 robustness] Report: {output}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
