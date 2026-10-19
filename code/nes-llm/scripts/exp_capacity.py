#!/usr/bin/env python3
"""Payload-capacity characterization for the frozen interior_nearest method.

Two checkpoints, one predeclared representative tensor each, predeclared
payload sizes {1000, 5000, 10000, 20000, 50000} bits x 3 seeds x cycles 0-3.
pair_parity is run only as a matched baseline at 10,000 bits.

Uses the frozen codec (`embed`, `extract`, `eligible`, `carrier_positions`,
`recovery_diagnostic`) unchanged. New outputs only.
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
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from scripts.exp_tensor_matrix import block_scales, list_packed_tensors
from src.experiments.nf4_anchor_channel import (
    eligible, embed, recovery_diagnostic, unpack,
)

METHOD = "interior_nearest"
BASELINE = "pair_parity"
BLOCKSIZE = 64
PREDECLARED_SIZES = (1000, 5000, 10000, 20000, 50000)
PREDECLARED_SEEDS = (20261016, 20261017, 20261018)
BASELINE_BITS = 10000


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def self_test():
    """Deterministic unit test of framing, capacity, and reproducibility."""
    from src.experiments.nf4_anchor_channel import pack
    cb = np.array([-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
                   -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
                   0.44070983, 0.562617, 0.72295684, 1.0])
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 16, size=64 * 400).astype(np.uint8)
    codes[0] = 0
    codes[63] = 15
    raw = pack(codes)
    payload = payload_for(1000, 20261016)
    key = key_for(20261016)
    out1, info = embed(raw, payload, key, "synthetic", METHOD, cb)
    out2, _ = embed(raw, payload, key, "synthetic", METHOD, cb)
    assert out1 == out2, "embedding not deterministic"
    cap = int(eligible(codes, METHOD).sum())
    framed = len(payload) * 8 + 128
    assert framed <= cap, "test payload must fit"
    assert info["eligible_codes"] == cap
    assert info["framed_bits"] == framed, (info["framed_bits"], framed)
    print("self-test PASS:", json.dumps({
        "eligible_capacity_bits": cap, "framed_bits_1000": framed,
        "utilization": round(framed / cap, 6), "deterministic": True}))


def capacity_of(checkpoint, tensor):
    source, state = load_tensor(checkpoint, tensor)
    raw = source.numpy().tobytes()
    codes = unpack(raw)
    cap = int(eligible(codes, METHOD).sum())
    return raw, state, codes, cap, tuple(source.shape)


def run_condition(checkpoint, tensor, raw, state, codes, raw_shape, method, bits, seed, cycles, log, meta):
    payload = payload_for(bits, seed)
    key = key_for(seed)
    cb = state.code.float().numpy()
    framed = len(payload) * 8 + 128
    cap = int(eligible(codes, method).sum())
    if framed > cap:
        row = {**meta, "tensor": tensor, "method": method, "net_payload_bits": bits,
               "seed": seed, "cycle": None, "status": "CAPACITY_EXCEEDED",
               "eligible_codes": cap, "framed_bits": framed,
               "max_feasible_payload_bits": cap - 128}
        log.write(json.dumps(row) + "\n"); log.flush()
        return 1, {"tensor": tensor, "method": method, "bits": bits, "seed": seed,
                   "status": "CAPACITY_EXCEEDED", "max_feasible_payload_bits": cap - 128}
    anchor_mask = (codes == 0) | (codes == 15)
    try:
        stego, info = embed(raw, payload, key, tensor, method, cb)
    except (ValueError, AssertionError) as exc:
        row = {**meta, "tensor": tensor, "method": method, "net_payload_bits": bits,
               "seed": seed, "cycle": None, "status": "EMBED_FAILED",
               "error": f"{type(exc).__name__}: {exc}", "eligible_codes": cap, "framed_bits": framed}
        log.write(json.dumps(row) + "\n"); log.flush()
        return 1, {"tensor": tensor, "method": method, "bits": bits, "seed": seed,
                   "status": "EMBED_FAILED", "error": str(exc)}
    stego_codes = unpack(stego)
    tensor_t = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(raw_shape)
    current_state = state
    ref_scales = block_scales(dequantize_4bit(tensor_t, quant_state=current_state).float())
    rows = 0
    for cycle in range(cycles + 1):
        if cycle:
            tensor_t, current_state = requantize(tensor_t, current_state)
        current = tensor_t.numpy().tobytes()
        cur_codes = unpack(current)
        scales = block_scales(dequantize_4bit(tensor_t, quant_state=current_state).float())
        blocks = cur_codes.reshape(-1, BLOCKSIZE)
        missing = int(np.sum(~np.any((blocks == 0) | (blocks == 15), axis=1)))
        try:
            from src.experiments.nf4_anchor_channel import carrier_positions
            cur_pos = carrier_positions(cur_codes, framed, key, tensor, method)
            base_pos = carrier_positions(stego_codes, framed, key, tensor, method)
            pos_agree = float(np.mean(cur_pos == base_pos))
        except ValueError:
            pos_agree = None
        row = {
            **meta, "tensor": tensor, "method": method, "net_payload_bits": bits,
            "seed": seed, "cycle": cycle, "status": "MEASURED",
            "eligible_codes": cap, "framed_bits": framed,
            "carrier_utilization": framed / cap,
            "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
            "nested_statistics": state.nested,
            "changed_codes": int(info["changed_codes"]),
            "changed_anchor_codes": int(np.count_nonzero(codes[anchor_mask] != stego_codes[anchor_mask])),
            "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != codes)),
            "blocks_without_endpoint_after_transform": missing,
            "blocks_scale_changed_vs_cycle0": int(np.count_nonzero(scales != ref_scales)),
            "carrier_position_agreement": pos_agree,
            "output_packed_sha256": hashlib.sha256(current).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            **recovery_diagnostic(current, payload, key, tensor, method),
        }
        log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
        rows += 1
    return rows, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-id", default="nes_capacity_20261016")
    p.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    p.add_argument("--sizes", nargs="+", type=int, default=list(PREDECLARED_SIZES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(PREDECLARED_SEEDS))
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.checkpoint or not args.output:
        p.error("--checkpoint and --output required unless --self-test")

    args.output.mkdir(parents=True, exist_ok=False)
    available = set(list_packed_tensors(args.checkpoint))
    if args.tensor not in available:
        raise SystemExit(f"Tensor not found: {args.tensor}")

    scripts = [Path(__file__), Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py",
               Path(__file__).parent / "run_nf4_anchor_pilot.py",
               Path(__file__).parent / "exp_tensor_matrix.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "scope": "Payload-capacity characterization of frozen interior_nearest; one tensor per checkpoint",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_hashes": {str(q.resolve()): sha_file(q)
                              for q in sorted(Path(args.checkpoint).glob("*.safetensors"))},
        "tensor": args.tensor,
        "method": METHOD, "baseline_method": BASELINE, "baseline_bits": BASELINE_BITS,
        "predeclared_sizes": list(args.sizes), "predeclared_seeds": list(args.seeds),
        "cycles": args.cycles,
        "transformation": "bitsandbytes dequantize to stored dtype then fresh NF4 with same block size and nesting",
        "source_hashes": {str(q.resolve()): sha_file(q) for q in scripts},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["torch", "bitsandbytes", "numpy", "safetensors"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    raw, state, codes, cap, raw_shape = capacity_of(args.checkpoint, args.tensor)
    meta = {"run_id": args.run_id, "source_tensor_sha256": hashlib.sha256(raw).hexdigest()}
    print(json.dumps({"tensor": args.tensor, "eligible_capacity_bits": cap,
                      "max_feasible_payload_bits": cap - 128,
                      "source_tensor_sha256": meta["source_tensor_sha256"]}), flush=True)

    count = 0
    failures = []
    with (args.output / "observations.jsonl").open("x") as log:
        for bits in args.sizes:
            for seed in args.seeds:
                n, fail = run_condition(args.checkpoint, args.tensor, raw, state, codes,
                                        raw_shape, METHOD, bits, seed, args.cycles, log, meta)
                count += n
                if fail: failures.append(fail)
        for seed in args.seeds:
            n, fail = run_condition(args.checkpoint, args.tensor, raw, state, codes,
                                    raw_shape, BASELINE, BASELINE_BITS, seed, args.cycles, log, meta)
            count += n
            if fail: failures.append(fail)
    write_json(args.output / "completion.json",
               {"status": "COMPLETED", "measured_rows": count, "failures": failures})
    print(json.dumps({"status": "COMPLETED", "measured_rows": count, "failures": failures}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
