#!/usr/bin/env python3
"""Tensor-generalization matrix for interior_nearest vs pair_parity.

Predeclared representative tensors across layer depths and tensor types.
Same 10,000-bit payload, seed 20261016, blocksize 64, nested statistics,
and the same requantization lifecycle (cycles 0-3) for every tensor.

Uses the frozen codec (`embed`, `recovery_diagnostic`, `eligible`) unchanged.
New outputs only; no historical artifact is read or written.
"""
from __future__ import annotations

import argparse
import hashlib
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
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from src.experiments.nf4_anchor_channel import METHODS, embed, eligible, recovery_diagnostic, unpack

METHODS_MATRIX = ("interior_nearest", "pair_parity")
BLOCKSIZE = 64

PREDECLARED_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.11.self_attn.k_proj.weight",
    "model.layers.21.self_attn.v_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
    "model.layers.21.mlp.gate_proj.weight",
)


def list_packed_tensors(checkpoint):
    """List packed U8 tensors, supporting both single-file and sharded checkpoints."""
    folder = Path(checkpoint)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        by_shard = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        packed = []
        for shard, names in by_shard.items():
            with safe_open(str(folder / shard), framework="pt", device="cpu") as f:
                packed.extend(n for n in names if f.get_slice(n).get_dtype() == "U8")
        return packed
    with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as f:
        return [k for k in f.keys() if f.get_slice(k).get_dtype() == "U8"]


def block_scales(weights):
    flat = weights.reshape(-1, BLOCKSIZE)
    return flat.abs().amax(dim=1).numpy()


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def preflight(checkpoint, tensors):
    """Verify every declared tensor exists and is a packed U8 NF4 tensor with blocksize 64."""
    available = set(list_packed_tensors(checkpoint))
    report = []
    for name in tensors:
        entry = {"tensor": name, "exists": name in available}
        if entry["exists"]:
            raw, state = load_tensor(checkpoint, name)
            codes = unpack(raw.numpy().tobytes())
            entry.update({
                "num_codes": int(len(codes)),
                "blocks": int(len(codes) // BLOCKSIZE),
                "blocksize": int(state.blocksize),
                "nested": bool(state.nested),
                "quant_type": str(state.quant_type),
                "source_tensor_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                "eligible_interior_nearest": int(eligible(codes, "interior_nearest").sum()),
                "eligible_pair_parity": int(eligible(codes, "pair_parity").sum()),
            })
        report.append(entry)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--tensors", nargs="+", default=list(PREDECLARED_TENSORS))
    p.add_argument("--run-id", default="nes_tensor_matrix_20261016")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)

    pre = preflight(args.checkpoint, args.tensors)
    missing = [e["tensor"] for e in pre if not e["exists"]]
    if missing:
        write_json(args.output / "preflight.json", {"preflight": pre, "missing": missing})
        raise SystemExit(f"Missing tensors: {missing}")

    scripts = [Path(__file__), Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py",
               Path(__file__).parent / "contract_b_nf4_b14.py",
               Path(__file__).parent / "run_nf4_anchor_pilot.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "scope": "Tensor generalization of interior_nearest vs pair_parity; one checkpoint; no steganalysis or utility",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_hashes": {str(p.resolve()): sha_file(p)
                              for p in sorted(Path(args.checkpoint).glob("*.safetensors"))},
        "predeclared_tensors": list(args.tensors),
        "methods": list(METHODS_MATRIX),
        "payload_bits": args.bits, "seed": args.seed, "cycles": args.cycles,
        "transformation": "bitsandbytes dequantize to stored dtype then fresh NF4 with same block size and nesting",
        "payload": "Deterministic random bytes; B1.4 16-byte framing; no ECC or encryption",
        "preflight": pre,
        "source_hashes": {str(q.resolve()): sha_file(q) for q in scripts},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["torch", "bitsandbytes", "numpy", "safetensors"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    payload = np.random.default_rng(args.seed).bytes(args.bits // 8)
    key = hashlib.sha256(f"NES-development-{args.seed}".encode()).digest()
    count = 0
    failures = []
    with (args.output / "observations.jsonl").open("x") as log:
        for name in args.tensors:
            source, state = load_tensor(args.checkpoint, name)
            raw = source.numpy().tobytes()
            codes = unpack(raw)
            anchor_mask = (codes == 0) | (codes == 15)
            cb = state.code.float().numpy()
            clean_weights = dequantize_4bit(source, quant_state=state).float()
            clean_scales = block_scales(clean_weights)
            for method in METHODS_MATRIX:
                base = {"run_id": manifest["run_id"], "tensor": name, "method": method,
                        "net_payload_bits": args.bits, "seed": args.seed,
                        "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                        "payload_sha256": hashlib.sha256(payload).hexdigest()}
                start = time.monotonic()
                try:
                    stego, info = embed(raw, payload, key, name, method, cb)
                except ValueError as exc:
                    row = {**base, "status": "EMBED_FAILED", "error": str(exc),
                           "cycle": None}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    failures.append({"tensor": name, "method": method, "stage": "embed", "error": str(exc)})
                    count += 1
                    continue
                stego_codes = unpack(stego)
                from src.experiments.nf4_anchor_channel import carrier_positions
                base_pos = carrier_positions(stego_codes, (args.bits // 8 + 16) * 8, key, name, method)
                tensor = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
                current_state = state
                ref_scales = block_scales(dequantize_4bit(tensor, quant_state=current_state).float())
                for cycle in range(args.cycles + 1):
                    if cycle:
                        tensor, current_state = requantize(tensor, current_state)
                    current = tensor.numpy().tobytes()
                    cur_codes = unpack(current)
                    weights = dequantize_4bit(tensor, quant_state=current_state).float()
                    scales = block_scales(weights)
                    scale_changed = int(np.count_nonzero(scales != ref_scales))
                    max_scale_change = float(np.max(np.abs(scales - ref_scales))) if len(scales) else 0.0
                    blocks = cur_codes.reshape(-1, BLOCKSIZE)
                    missing_endpoints = int(np.sum(~np.any((blocks == 0) | (blocks == 15), axis=1)))
                    try:
                        cur_pos = carrier_positions(cur_codes, (args.bits // 8 + 16) * 8, key, name, method)
                        pos_agree = float(np.mean(cur_pos == base_pos))
                    except ValueError:
                        pos_agree = None
                    row = {
                        **base, "status": "MEASURED", "cycle": cycle,
                        "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
                        "nested_statistics": state.nested,
                        "eligible_codes": int(info["eligible_codes"]),
                        "changed_codes": int(info["changed_codes"]),
                        "changed_anchor_codes": int(np.count_nonzero(codes[anchor_mask] != stego_codes[anchor_mask])),
                        "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != codes)),
                        "blocks_without_endpoint_after_transform": missing_endpoints,
                        "blocks_scale_changed_vs_cycle0": scale_changed,
                        "max_abs_block_scale_change_vs_cycle0": max_scale_change,
                        "carrier_position_agreement": pos_agree,
                        "output_packed_sha256": hashlib.sha256(current).hexdigest(),
                        **recovery_diagnostic(current, payload, key, name, method),
                    }
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    count += 1
                # capture cycle-0 carrier positions for the next method loop
                # (recomputed per cycle above against this method's own cycle-0)
    write_json(args.output / "completion.json",
               {"status": "COMPLETED", "measured_rows": count, "failures": failures})
    print(json.dumps({"status": "COMPLETED", "measured_rows": count, "failures": failures}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
