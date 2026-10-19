#!/usr/bin/env python3
"""Resumable third-model validation matrix for the frozen interior_nearest codec.

One durable JSON per (tensor, method, cycle). Written to a temp file, validated,
then atomically finalized. On restart, already-finalized conditions are skipped.
Uses the frozen codec unchanged (`embed`, `eligible`, `recovery_diagnostic`,
`unpack`, `carrier_positions`). New outputs only.
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

sys.path.insert(0, "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm")
sys.path.insert(0, "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, requantize, sha_file
from src.experiments.nf4_anchor_channel import (
    METHODS, carrier_positions, embed, eligible, recovery_diagnostic, unpack,
)
CODEC = Path("/workspace/nes_run_20261016_anchor_ablation/code/nes-llm/src/experiments/nf4_anchor_channel.py")

BLOCKSIZE = 64
METHODS_MATRIX = ("interior_nearest", "pair_parity")
DEFAULT_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.11.self_attn.k_proj.weight",
    "model.layers.23.self_attn.v_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
    "model.layers.23.mlp.gate_proj.weight",
)


def atomic_write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def block_scales(weights):
    return weights.reshape(-1, BLOCKSIZE).abs().amax(dim=1).numpy()


def cond_id(tensor, method, cycle):
    return f"{tensor}__{method}__c{cycle}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--tensors", nargs="+", default=list(DEFAULT_TENSORS))
    p.add_argument("--run-id", default="nes_third_model_stablelm_20261016")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    conds_dir = args.output / "conditions"
    conds_dir.mkdir(exist_ok=True)

    payload = np.random.default_rng(args.seed).bytes(args.bits // 8)
    key = hashlib.sha256(f"NES-development-{args.seed}".encode()).digest()

    manifest_path = args.output / "manifest.json"
    if not manifest_path.exists():
        pre = []
        for name in args.tensors:
            raw, state = load_tensor(args.checkpoint, name)
            codes = unpack(raw.numpy().tobytes())
            pre.append({"tensor": name, "num_codes": int(len(codes)),
                        "blocks": int(len(codes) // BLOCKSIZE), "blocksize": int(state.blocksize),
                        "nested": bool(state.nested), "quant_type": str(state.quant_type),
                        "source_tensor_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                        "eligible_interior_nearest": int(eligible(codes, "interior_nearest").sum()),
                        "eligible_pair_parity": int(eligible(codes, "pair_parity").sum())})
        atomic_write_json(manifest_path, {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id, "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_hashes": {pp.name: sha_file(pp)
                                  for pp in sorted(args.checkpoint.glob("*.safetensors"))},
            "tensors": list(args.tensors), "methods": list(METHODS_MATRIX),
            "payload_bits": args.bits, "seed": args.seed, "cycles": args.cycles,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "codec_sha256": sha_file(CODEC),
            "versions": {m: importlib.metadata.version(m)
                         for m in ["torch", "bitsandbytes", "numpy", "safetensors"]},
            "platform": platform.platform(), "preflight": pre})

    done, skipped = [], []
    total = len(args.tensors) * len(METHODS_MATRIX) * (args.cycles + 1)
    for name in args.tensors:
        source, state = load_tensor(args.checkpoint, name)
        raw = source.numpy().tobytes()
        codes = unpack(raw)
        anchor_mask = (codes == 0) | (codes == 15)
        cb = state.code.float().numpy()
        clean_weights = dequantize_4bit(source, quant_state=state).float()
        clean_scales = block_scales(clean_weights)
        for method in METHODS_MATRIX:
            base = {"run_id": args.run_id, "tensor": name, "method": method,
                    "net_payload_bits": args.bits, "seed": args.seed,
                    "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                    "payload_sha256": hashlib.sha256(payload).hexdigest()}
            stego, info = embed(raw, payload, key, name, method, cb)
            stego_codes = unpack(stego)
            base_pos = carrier_positions(stego_codes, (args.bits // 8 + 16) * 8, key, name, method)
            tensor = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
            current_state = state
            ref_scales = block_scales(dequantize_4bit(tensor, quant_state=current_state).float())
            for cycle in range(args.cycles + 1):
                cid = cond_id(name, method, cycle)
                cpath = conds_dir / f"{cid}.json"
                if cpath.exists():
                    skipped.append(cid)
                    continue
                t0 = time.monotonic()
                try:
                    if cycle:
                        tensor, current_state = requantize(tensor, current_state)
                    current = tensor.numpy().tobytes()
                    cur_codes = unpack(current)
                    weights = dequantize_4bit(tensor, quant_state=current_state).float()
                    scales = block_scales(weights)
                    blocks = cur_codes.reshape(-1, BLOCKSIZE)
                    missing_endpoints = int(np.sum(~np.any((blocks == 0) | (blocks == 15), axis=1)))
                    try:
                        cur_pos = carrier_positions(cur_codes, (args.bits // 8 + 16) * 8, key, name, method)
                        pos_agree = float(np.mean(cur_pos == base_pos))
                    except ValueError:
                        pos_agree = None
                    row = {**base, "status": "MEASURED", "cycle": cycle,
                           "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
                           "nested_statistics": state.nested,
                           "eligible_codes": int(info["eligible_codes"]),
                           "changed_codes": int(info["changed_codes"]),
                           "changed_anchor_codes": int(np.count_nonzero(codes[anchor_mask] != stego_codes[anchor_mask])),
                           "packed_code_hamming_vs_source": int(np.count_nonzero(cur_codes != codes)),
                           "blocks_without_endpoint_after_transform": missing_endpoints,
                           "blocks_scale_changed_vs_cycle0": int(np.count_nonzero(scales != ref_scales)),
                           "max_abs_block_scale_change_vs_cycle0": float(np.max(np.abs(scales - ref_scales))) if len(scales) else 0.0,
                           "carrier_position_agreement": pos_agree,
                           "output_packed_sha256": hashlib.sha256(current).hexdigest(),
                           "elapsed_s": round(time.monotonic() - t0, 2),
                           **recovery_diagnostic(current, payload, key, name, method)}
                except Exception as exc:  # record, never crash the whole matrix
                    row = {**base, "status": "ERROR", "cycle": cycle,
                           "error": f"{type(exc).__name__}: {exc}"}
                # validate before finalizing
                if row["status"] == "MEASURED" and "exact_recovery" not in row:
                    row["status"] = "INVALID"
                atomic_write_json(cpath, row)
                done.append(cid)
                print(json.dumps({"cond": cid, "status": row["status"],
                                  "exact": row.get("exact_recovery"),
                                  "ber": row.get("payload_ber")}), flush=True)

    atomic_write_json(args.output / "completion.json", {
        "status": "COMPLETE" if (len(done) + len(skipped)) == total else "PARTIAL",
        "total_conditions": total, "completed_now": len(done), "skipped_existing": len(skipped)})
    print(json.dumps({"status": "COMPLETE" if (len(done) + len(skipped)) == total else "PARTIAL",
                      "total": total, "done_now": len(done), "skipped": len(skipped)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
