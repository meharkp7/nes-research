#!/usr/bin/env python3
"""Minimal matched causal A-D experiment for NF4 anchor preservation.

Four arms from ONE immutable source tensor (same tensor, payload, key, seed,
transformation). Reuses the frozen codec helpers unchanged where possible:

  A. interior_nearest_anchored  -> existing codec method `interior_nearest`
  B. nearest_parity_anchored    -> parity-nearest op, anchored blocks, endpoints ELIGIBLE
  C. interior_nearest_unanchored-> interior-nearest op, NO anchor restriction
  D. pair_parity_unrestricted   -> existing codec method `pair_parity`

B and C are the two ablation arms that isolate (i) touching anchors and
(ii) the anchorless-block exclusion. Ordering, framing, HMAC key schedule and
checksum are copied verbatim from the frozen codec so results are comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from bitsandbytes.functional import dequantize_4bit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contract_b_nf4_b14 import DOMAIN, NONCE, MAGIC, envelope
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from src.experiments.nf4_anchor_channel import unpack, pack

BLOCKSIZE = 64
ARMS = ("interior_nearest_anchored", "nearest_parity_anchored",
        "interior_nearest_unanchored", "pair_parity_unrestricted")


def _anchored_mask(codes):
    if len(codes) % BLOCKSIZE:
        raise ValueError("Interior protocol requires complete 64-code blocks")
    blocks = codes.reshape(-1, BLOCKSIZE)
    return np.repeat(np.any((blocks == 0) | (blocks == 15), axis=1), BLOCKSIZE)


def eligible_arm(codes, arm):
    """Receiver-reconstructable eligibility mask for each arm."""
    if arm == "interior_nearest_anchored":
        return _anchored_mask(codes) & (codes > 0) & (codes < 15)
    if arm == "nearest_parity_anchored":
        return _anchored_mask(codes)                       # endpoints remain eligible
    if arm == "interior_nearest_unanchored":
        return (codes > 0) & (codes < 15)                  # no anchor restriction
    if arm == "pair_parity_unrestricted":
        return np.ones(len(codes), dtype=bool)
    raise ValueError(f"Unknown arm: {arm}")


def carrier_positions_mask(codes, count, key, tensor_key, mask):
    """Verbatim copy of the frozen codec ordering, parameterised by mask."""
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


def embed_arm(raw, payload, key, tensor_key, arm, codebook):
    codes = unpack(raw)
    cb = np.asarray(codebook, dtype=np.float64)
    if cb.shape != (16,) or not np.all(np.diff(cb) > 0) or cb[0] != -1 or cb[-1] != 1:
        raise ValueError("Expected NF4 codebook with endpoints -1 and +1")
    mask = eligible_arm(codes, arm)
    bits = np.unpackbits(np.frombuffer(envelope(payload), dtype=np.uint8))
    pos = carrier_positions_mask(codes, len(bits), key, tensor_key, mask)
    updated = codes.copy()
    if arm in ("pair_parity_unrestricted",):
        updated[pos] = (codes[pos] & 14) | bits
    else:
        alphabet = np.arange(1, 15) if arm == "interior_nearest_anchored" or arm == "interior_nearest_unanchored" else np.arange(16)
        for bit in (0, 1):
            candidates = alphabet[alphabet % 2 == bit]
            mapping = candidates[np.argmin((cb[:, None] - cb[candidates]) ** 2, axis=1)]
            selected = pos[bits == bit]
            updated[selected] = mapping[codes[selected]]
    eligibility_changed = not np.array_equal(eligible_arm(codes, arm), eligible_arm(updated, arm))
    if not np.array_equal(updated[pos] & 1, bits):
        raise AssertionError("Embedding did not reproduce envelope bits")
    return pack(updated), {
        "eligible_codes": int(mask.sum()),
        "changed_codes": int(np.count_nonzero(codes != updated)),
        "embed_eligibility_changed": bool(eligibility_changed),
        "carrier_map_bytes": 0,
        "reference_weight_bytes": 0,
        "framed_bits": int(len(bits)),
        "net_payload_bits": int(len(payload) * 8),
    }


def extract_arm(raw, key, tensor_key, arm):
    codes = unpack(raw)
    mask = eligible_arm(codes, arm)
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


def recovery_arm(raw, expected, key, tensor_key, arm):
    res = {"exact_recovery": False, "decode_error": None,
           "payload_bit_errors": None, "payload_ber": None, "checksum_valid": False}
    try:
        rec = extract_arm(raw, key, tensor_key, arm)
        res["exact_recovery"] = rec == expected
        res["checksum_valid"] = True
        res["recovered_sha256"] = hashlib.sha256(rec).hexdigest()
    except ValueError as exc:
        res["decode_error"] = str(exc)
    try:
        codes = unpack(raw)
        mask = eligible_arm(codes, arm)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    p.add_argument("--bits", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cycles", type=int, default=3)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py",
               Path(__file__).parent / "contract_b_nf4_b14.py",
               Path(__file__).parent / "run_nf4_anchor_pilot.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_anchor_ablation_ad_20261016",
        "scope": "Minimal matched causal A-D anchor-preservation experiment; one tensor; no steganalysis or task utility",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_hashes": {str(Path(args.checkpoint).resolve() / "model.safetensors"):
                              sha_file(Path(args.checkpoint) / "model.safetensors")},
        "tensor": args.tensor, "payload_bits": args.bits, "seed": args.seed, "cycles": args.cycles,
        "arms": list(ARMS),
        "transformation": "bitsandbytes dequantize to stored dtype then fresh NF4 with same block size and nesting",
        "payload": "Deterministic random bytes; B1.4 16-byte framing; no ECC or encryption",
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
    clean_weights = None

    payload = np.random.default_rng(args.seed).bytes(args.bits // 8)
    key = hashlib.sha256(f"NES-development-{args.seed}".encode()).digest()
    print(json.dumps({"source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                      "blocksize": state.blocksize, "nested": state.nested,
                      "total_codes": len(original_codes)}), flush=True)

    count = 0
    with (args.output / "observations.jsonl").open("x") as log:
        for arm in ARMS:
            stego, info = embed_arm(raw, payload, key, args.tensor, arm, cb)
            stego_codes = unpack(stego)
            base_positions = carrier_positions_mask(
                stego_codes, (args.bits // 8 + 16) * 8, key, args.tensor, eligible_arm(stego_codes, arm))
            tensor = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
            current_state = state
            for cycle in range(args.cycles + 1):
                if cycle:
                    tensor, current_state = requantize(tensor, current_state)
                current = tensor.numpy().tobytes()
                cur_codes = unpack(current)
                weights = dequantize_4bit(
                    tensor, quant_state=current_state).float()
                cur_mask = eligible_arm(cur_codes, arm)
                try:
                    cur_positions = carrier_positions_mask(
                        cur_codes, (args.bits // 8 + 16) * 8, key, args.tensor, cur_mask)
                    pos_agree = float(np.mean(cur_positions == base_positions))
                except ValueError:
                    cur_positions = None
                    pos_agree = None
                blocks = cur_codes.reshape(-1, BLOCKSIZE)
                missing = int(np.sum(~np.any((blocks == 0) | (blocks == 15), axis=1)))
                row = {
                    "run_id": manifest["run_id"], "tensor": args.tensor, "arm": arm,
                    "net_payload_bits": args.bits, "seed": args.seed, "cycle": cycle,
                    "status": "MEASURED",
                    "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
                    "nested_statistics": state.nested,
                    "eligible_codes": info["eligible_codes"],
                    "changed_codes": info["changed_codes"],
                    "embed_eligibility_changed": info["embed_eligibility_changed"],
                    "changed_anchor_codes": int(np.count_nonzero(original_codes[anchor_mask] != stego_codes[anchor_mask])),
                    "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != original_codes)),
                    "blocks_without_endpoint_after_transform": missing,
                    "carrier_position_agreement": pos_agree,
                    "eligibility_changed_after_transform": int(np.count_nonzero(cur_mask != eligible_arm(original_codes, arm))),
                    "output_packed_sha256": hashlib.sha256(current).hexdigest(),
                    **recovery_arm(current, payload, key, args.tensor, arm),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                count += 1
    write_json(args.output / "completion.json", {"status": "COMPLETED", "measured_rows": count})
    print(json.dumps({"status": "COMPLETED", "measured_rows": count, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
