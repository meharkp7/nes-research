#!/usr/bin/env python3
"""Compare a pristine B1.4 artifact with a transformed artifact.

Reports exact application-payload bit error positions, counts in fixed-size blocks,
and contiguous error runs. It does not attempt correction and does not modify either
checkpoint. Run from nes-llm/:

python scripts/contract_b_error_pattern.py \
  --reference-stego ../cache/contract_b_nf4_b14_10k \
  --transformed-stego ../cache/contract_b_b14_nf4_requantized_retry3 \
  --output ../cache/contract_b_b14_nf4_requantized_retry3_error_pattern.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY,
    TEST_KEY_HEX,
    code_at,
    extract,
    find_shard,
    from_bits,
    positions,
    read_tensor,
)


def contiguous_runs(indices: list[int]) -> list[dict]:
    """Summarize sorted error indices as half-open contiguous runs."""
    if not indices:
        return []
    runs = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append({"start_bit": start, "end_bit_exclusive": previous + 1,
                         "length_bits": previous - start + 1})
            start = index
        previous = index
    runs.append({"start_bit": start, "end_bit_exclusive": previous + 1,
                 "length_bits": previous - start + 1})
    return runs


def analyze_packed_tensors(reference_packed: bytes, transformed_packed: bytes,
                           key: bytes, tensor_key: str,
                           block_bits: int = 128) -> dict:
    """Analyze errors relative to the payload recovered from a pristine artifact."""
    if block_bits <= 0:
        raise ValueError("block_bits must be a positive integer")
    if len(reference_packed) != len(transformed_packed):
        raise ValueError("Reference and transformed packed tensors differ in byte length")

    expected_payload, _ = extract(reference_packed, key, tensor_key)
    payload_bit_count = len(expected_payload) * 8
    envelope_bytes = len(expected_payload) + 16
    carrier_positions = positions(key, len(transformed_packed) * 2,
                                  envelope_bytes * 8, tensor_key)
    decoded_envelope = from_bits([
        code_at(transformed_packed, position) & 1
        for position in carrier_positions
    ])
    header_valid = (
        decoded_envelope[:4] == b"NB14"
        and len(decoded_envelope) >= 16
        and int.from_bytes(decoded_envelope[4:8], "big") == len(expected_payload)
    )
    base = {
        "payload_bits": payload_bit_count,
        "block_bits": block_bits,
        "reference_payload_sha256": hashlib.sha256(expected_payload).hexdigest(),
        "reference_packed_tensor_sha256": hashlib.sha256(reference_packed).hexdigest(),
        "transformed_packed_tensor_sha256": hashlib.sha256(transformed_packed).hexdigest(),
        "header_valid": header_valid,
    }
    if not header_valid:
        return {
            **base,
            "status": "HEADER_INVALID",
            "bit_errors": None,
            "ber": None,
            "error_positions": [],
            "blocks": [],
            "contiguous_error_runs": [],
            "max_contiguous_error_run_bits": None,
            "interpretation": (
                "The transformed artifact's envelope header could not be decoded. "
                "Payload bit positions cannot be reported reliably."
            ),
        }

    recovered_payload = decoded_envelope[8:8 + len(expected_payload)]
    error_positions = []
    for byte_index, (expected_byte, actual_byte) in enumerate(
        zip(expected_payload, recovered_payload)
    ):
        difference = expected_byte ^ actual_byte
        for bit_in_byte in range(8):
            if difference & (1 << (7 - bit_in_byte)):
                error_positions.append(byte_index * 8 + bit_in_byte)

    blocks = []
    for start in range(0, payload_bit_count, block_bits):
        end = min(start + block_bits, payload_bit_count)
        count = sum(start <= position < end for position in error_positions)
        blocks.append({
            "start_bit": start,
            "end_bit_exclusive": end,
            "bit_count": end - start,
            "error_count": count,
            "ber": count / (end - start),
        })

    runs = contiguous_runs(error_positions)
    bit_errors = len(error_positions)
    return {
        **base,
        "status": "ANALYZED",
        "recovered_payload_sha256": hashlib.sha256(recovered_payload).hexdigest(),
        "payload_checksum_valid": (
            decoded_envelope[8 + len(expected_payload):]
            == hashlib.sha256(recovered_payload).digest()[:8]
        ),
        "exact_payload_recovery": recovered_payload == expected_payload,
        "bit_errors": bit_errors,
        "ber": bit_errors / payload_bit_count if payload_bit_count else 0.0,
        "error_positions": error_positions,
        "blocks": blocks,
        "contiguous_error_runs": runs,
        "max_contiguous_error_run_bits": max(
            (run["length_bits"] for run in runs), default=0
        ),
        "interpretation": (
            "Descriptive analysis of this artifact pair only. Error clustering and "
            "repeatability require additional independent transformation trials; "
            "this report does not establish an error-channel model or ECC guarantee."
        ),
    }


def run(args: argparse.Namespace) -> int:
    reference = Path(args.reference_stego).expanduser().resolve()
    transformed = Path(args.transformed_stego).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"Reference artifact not found: {reference}")
    if not transformed.is_dir():
        raise FileNotFoundError(f"Transformed artifact not found: {transformed}")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    key = bytes.fromhex(args.test_key_hex)
    reference_shard = find_shard(reference, args.tensor_key)
    transformed_shard = find_shard(transformed, args.tensor_key)
    _, _, _, _, reference_packed = read_tensor(reference_shard, args.tensor_key)
    _, _, _, _, transformed_packed = read_tensor(transformed_shard, args.tensor_key)
    report = {
        "stage": "Contract-B-B1.4-error-pattern-diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_artifact": str(reference),
        "transformed_artifact": str(transformed),
        "tensor_key": args.tensor_key,
        "block_bits": args.block_bits,
        "analysis": analyze_packed_tensors(
            reference_packed, transformed_packed, key, args.tensor_key, args.block_bits
        ),
        "scope": (
            "Read-only payload error-pattern diagnostic. No checkpoint is modified; "
            "no error correction, utility claim, or stealth claim is made."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    analysis = report["analysis"]
    print(f"[B1.4 error pattern] Status: {analysis['status']}")
    print(f"[B1.4 error pattern] Errors: {analysis['bit_errors']}; BER: {analysis['ber']}")
    print(f"[B1.4 error pattern] Max contiguous run: {analysis['max_contiguous_error_run_bits']}")
    print(f"[B1.4 error pattern] Report: {output}")
    return 0 if analysis["status"] == "ANALYZED" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-stego", required=True,
                        help="Pristine B1.4 stego checkpoint")
    parser.add_argument("--transformed-stego", required=True,
                        help="Transformed checkpoint to diagnose")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--block-bits", type=int, default=128,
                        help="Application-payload block size in bits (default: 128)")
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--test-key-hex", default=TEST_KEY_HEX)
    args = parser.parse_args()
    if args.block_bits <= 0:
        parser.error("--block-bits must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
