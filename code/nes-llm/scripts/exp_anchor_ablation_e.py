#!/usr/bin/env python3
"""Arm E: interior-eligible carrier set with the BASELINE parity operation.

Eligibility = interior NF4 codes 1..14 (no anchorless-block restriction).
Operation   = baseline parity, `code & 14 | bit` (same op as pair_parity).

Endpoint codes 0 and 15 are never eligible for selection. The op may still
map an interior code onto an endpoint (1->0 when bit=0; 14->15 when bit=1);
that is the phenomenon this arm isolates and measures.

Reuses the frozen codec and the A-D helpers unchanged. New outputs only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from bitsandbytes.functional import dequantize_4bit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contract_b_nf4_b14 import MAGIC, envelope
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from scripts.exp_anchor_ablation_ad import carrier_positions_mask, _anchored_mask
from src.experiments.nf4_anchor_channel import pack, unpack

ARM = "interior_parity_baseline_op"
BLOCKSIZE = 64


def eligible_e(codes):
    """Interior codes only; endpoints 0 and 15 are never selectable."""
    return (codes >= 1) & (codes <= 14)


def embed_e(raw, payload, key, tensor_key, codebook):
    codes = unpack(raw)
    cb = np.asarray(codebook, dtype=np.float64)
    if cb.shape != (16,) or not np.all(np.diff(cb) > 0) or cb[0] != -1 or cb[-1] != 1:
        raise ValueError("Expected NF4 codebook with endpoints -1 and +1")
    mask = eligible_e(codes)
    bits = np.unpackbits(np.frombuffer(envelope(payload), dtype=np.uint8))
    pos = carrier_positions_mask(codes, len(bits), key, tensor_key, mask)
    updated = codes.copy()
    updated[pos] = (codes[pos] & 14) | bits          # baseline parity op
    if not np.array_equal(updated[pos] & 1, bits):
        raise AssertionError("Embedding did not reproduce envelope bits")
    endpoint_origin = int(np.count_nonzero(mask[pos] & ((codes[pos] == 0) | (codes[pos] == 15))))
    wrote_endpoint = int(np.count_nonzero((updated[pos] == 0) | (updated[pos] == 15)))
    return pack(updated), {
        "eligible_codes": int(mask.sum()),
        "changed_codes": int(np.count_nonzero(codes != updated)),
        "embed_eligibility_changed": bool(
            not np.array_equal(mask, eligible_e(updated))),
        "selected_endpoint_origin_codes": endpoint_origin,   # must be 0 by construction
        "carriers_written_to_endpoint": wrote_endpoint,      # op side effect
        "framed_bits": int(len(bits)),
        "net_payload_bits": int(len(payload) * 8),
    }


def extract_e(raw, key, tensor_key):
    codes = unpack(raw)
    mask = eligible_e(codes)
    pos = carrier_positions_mask(codes, 64, key, tensor_key, mask)
    header = np.packbits(codes[pos] & 1).tobytes()
    if header[:4] != MAGIC:
        raise ValueError("Envelope magic mismatch")
    size = int.from_bytes(header[4:8], "big")
    if size <= 0 or (size + 16) * 8 > int(mask.sum()):
        raise ValueError("Envelope length exceeds receiver capacity")
    pos = carrier_positions_mask(codes, (size + 16) * 8, key, tensor_key, mask)
    framed = np.packbits(codes[pos] & 1).tobytes()
    payload = framed[8:8 + size]
    if framed[-8:] != hashlib.sha256(payload).digest()[:8]:
        raise ValueError("Envelope checksum mismatch")
    return payload


def recovery_e(raw, expected, key, tensor_key):
    res = {"exact_recovery": False, "decode_error": None, "payload_bit_errors": None,
           "payload_ber": None, "checksum_valid": False}
    try:
        rec = extract_e(raw, key, tensor_key)
        res["exact_recovery"] = rec == expected
        res["checksum_valid"] = True
        res["recovered_sha256"] = hashlib.sha256(rec).hexdigest()
    except ValueError as exc:
        res["decode_error"] = str(exc)
    try:
        codes = unpack(raw)
        mask = eligible_e(codes)
        pos = carrier_positions_mask(codes, (len(expected) + 16) * 8, key, tensor_key, mask)
        observed = codes[pos[64:64 + len(expected) * 8]] & 1
        wanted = np.unpackbits(np.frombuffer(expected, dtype=np.uint8))
        res["payload_bit_errors"] = int(np.count_nonzero(observed != wanted))
        res["payload_ber"] = res["payload_bit_errors"] / len(wanted)
    except ValueError:
        pass
    return res


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def self_test():
    """Deterministic unit test on a small synthetic code array."""
    cb = np.array([-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
                   -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
                   0.44070983, 0.562617, 0.72295684, 1.0])
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, size=64 * 40).astype(np.uint8)
    codes[0] = 0
    codes[63] = 15
    raw = pack(codes)
    payload = bytes(range(32))
    key = hashlib.sha256(b"NES-development-20261016").digest()
    out1, info = embed_e(raw, payload, key, "model.layers.0.self_attn.q_proj.weight", cb)
    out2, _ = embed_e(raw, payload, key, "model.layers.0.self_attn.q_proj.weight", cb)
    assert out1 == out2, "embedding not deterministic"
    updated = unpack(out1)
    # 1. endpoints never selected as carriers
    assert info["selected_endpoint_origin_codes"] == 0, "endpoint code was selected"
    # 2. endpoint-origin positions are untouched
    assert updated[0] == 0 and updated[63] == 15, "endpoint-origin position was modified"
    # 3. only interior positions changed
    assert np.all(updated[codes == 0] == 0) and np.all(updated[codes == 15] == 15)
    # 4. bits reproduced at carriers
    bits = np.unpackbits(np.frombuffer(envelope(payload), dtype=np.uint8))
    pos = carrier_positions_mask(codes, len(bits), key,
                                "model.layers.0.self_attn.q_proj.weight", eligible_e(codes))
    assert np.array_equal(updated[pos] & 1, bits)
    # 5. the baseline op writes some carriers onto endpoints (the E mechanism)
    assert info["carriers_written_to_endpoint"] > 0, "expected op to write endpoints"
    # 6. same-artifact recovery is expected to FAIL because eligibility shifts
    try:
        rec = extract_e(out1, key, "model.layers.0.self_attn.q_proj.weight")
        recovered = (rec == payload)
        err = None
    except ValueError as exc:
        recovered = False
        err = str(exc)
    print("self-test PASS:",
          json.dumps({"selected_endpoint_origin_codes": info["selected_endpoint_origin_codes"],
                      "carriers_written_to_endpoint": info["carriers_written_to_endpoint"],
                      "changed_codes": info["changed_codes"],
                      "embed_eligibility_changed": info["embed_eligibility_changed"],
                      "same_artifact_recovered": recovered,
                      "decode_error": err}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    p.add_argument("--bits", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.checkpoint or not args.output:
        p.error("--checkpoint and --output are required unless --self-test")

    args.output.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py",
               Path(__file__).parent / "contract_b_nf4_b14.py",
               Path(__file__).parent / "run_nf4_anchor_pilot.py",
               Path(__file__).parent / "exp_anchor_ablation_ad.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_anchor_arm_e_20261016",
        "scope": "Arm E only: interior eligibility (codes 1-14) with baseline parity op; one tensor",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_hashes": {str(Path(args.checkpoint).resolve() / "model.safetensors"):
                              sha_file(Path(args.checkpoint) / "model.safetensors")},
        "tensor": args.tensor, "payload_bits": args.bits, "seed": args.seed, "cycles": args.cycles,
        "arm": ARM,
        "eligibility": "codes 1..14 (no anchorless-block restriction); endpoints never selectable",
        "operation": "baseline parity code & 14 | bit",
        "transformation": "bitsandbytes dequantize to stored dtype then fresh NF4 with same block size and nesting",
        "source_hashes": {str(q.resolve()): sha_file(q) for q in scripts},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["torch", "bitsandbytes", "numpy", "safetensors"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    source, state = load_tensor(args.checkpoint, args.tensor)
    raw = source.numpy().tobytes()
    original_codes = unpack(raw)
    cb = state.code.float().numpy()
    anchor_mask = (original_codes == 0) | (original_codes == 15)

    payload = np.random.default_rng(args.seed).bytes(args.bits // 8)
    key = hashlib.sha256(f"NES-development-{args.seed}".encode()).digest()
    print(json.dumps({"arm": ARM, "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                      "blocksize": state.blocksize, "nested": state.nested,
                      "total_codes": len(original_codes)}), flush=True)

    stego, info = embed_e(raw, payload, key, args.tensor, cb)
    stego_codes = unpack(stego)
    base_positions = carrier_positions_mask(
        stego_codes, (args.bits // 8 + 16) * 8, key, args.tensor, eligible_e(stego_codes))
    tensor = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
    current_state = state
    count = 0
    with (args.output / "observations.jsonl").open("x") as log:
        for cycle in range(args.cycles + 1):
            if cycle:
                tensor, current_state = requantize(tensor, current_state)
            current = tensor.numpy().tobytes()
            cur_codes = unpack(current)
            cur_mask = eligible_e(cur_codes)
            try:
                cur_positions = carrier_positions_mask(
                    cur_codes, (args.bits // 8 + 16) * 8, key, args.tensor, cur_mask)
                pos_agree = float(np.mean(cur_positions == base_positions))
            except ValueError:
                pos_agree = None
            blocks = cur_codes.reshape(-1, BLOCKSIZE)
            missing = int(np.sum(~np.any((blocks == 0) | (blocks == 15), axis=1)))
            row = {
                "run_id": manifest["run_id"], "tensor": args.tensor, "arm": ARM,
                "net_payload_bits": args.bits, "seed": args.seed, "cycle": cycle,
                "status": "MEASURED",
                "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
                "nested_statistics": state.nested,
                "eligible_codes": info["eligible_codes"],
                "changed_codes": info["changed_codes"],
                "embed_eligibility_changed": info["embed_eligibility_changed"],
                "selected_endpoint_origin_codes": info["selected_endpoint_origin_codes"],
                "carriers_written_to_endpoint": info["carriers_written_to_endpoint"],
                "changed_anchor_codes": int(np.count_nonzero(original_codes[anchor_mask] != stego_codes[anchor_mask])),
                "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != original_codes)),
                "blocks_without_endpoint_after_transform": missing,
                "carrier_position_agreement": pos_agree,
                "eligibility_changed_after_transform": int(np.count_nonzero(cur_mask != eligible_e(original_codes))),
                "output_packed_sha256": hashlib.sha256(current).hexdigest(),
                **recovery_e(current, payload, key, args.tensor),
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            count += 1
    write_json(args.output / "completion.json", {"status": "COMPLETED", "measured_rows": count})
    print(json.dumps({"status": "COMPLETED", "measured_rows": count, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
