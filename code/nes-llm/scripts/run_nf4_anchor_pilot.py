#!/usr/bin/env python3
"""Development tensor pilot on saved NF4 checkpoints; not a full-model utility test."""
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
from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contract_b_nf4_b14 import find_shard
from src.experiments.nf4_anchor_channel import METHODS, embed, eligible, recovery_diagnostic, unpack


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_tensor(checkpoint, name):
    path = find_shard(checkpoint, name)
    with safe_open(path, framework="pt", device="cpu") as f:
        raw = f.get_tensor(name)
        state = QuantState.from_dict(
            {k: f.get_tensor(k) for k in f.keys() if k.startswith(name + ".")},
            torch.device("cpu"),
        )
    if raw.dtype != torch.uint8 or state.quant_type != "nf4":
        raise ValueError("Expected packed U8 NF4 tensor")
    if state.blocksize != 64:
        raise ValueError("This development channel supports blocksize 64 only")
    if int(np.prod(state.shape)) != raw.numel() * 2:
        raise ValueError("Padded or odd-length tensors need an explicit layout adapter")
    return raw, state


def requantize(raw, state):
    weights = dequantize_4bit(raw, quant_state=state)
    return quantize_4bit(weights, blocksize=state.blocksize,
                         compress_statistics=state.nested, quant_type="nf4")


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensors", nargs="+", default=[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.16.self_attn.k_proj.weight",
        "model.layers.32.self_attn.v_proj.weight",
    ])
    parser.add_argument("--bits", nargs="+", type=int, default=[10000, 50000, 250000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261016, 20261017, 20261018])
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.cycles < 1 or any(n <= 0 or n % 8 for n in args.bits):
        parser.error("Positive byte-aligned payload sizes and at least one cycle required")
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py",
               Path(__file__).parent / "contract_b_nf4_b14.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Development tensor experiment; one checkpoint; no steganalysis or task utility",
        "checkpoint": str(args.checkpoint.resolve()), "methods": METHODS,
        "tensors": args.tensors, "payload_bits": args.bits, "seeds": args.seeds,
        "cycles": args.cycles,
        "transformation": "bitsandbytes dequantize to stored dtype then fresh NF4 with same block size and nesting",
        "payload": "Deterministic random bytes; B1.4 16-byte framing; no ECC or encryption",
        "source_hashes": {str(p.resolve()): sha_file(p) for p in scripts},
        "checkpoint_hashes": {str(p.resolve()): sha_file(p)
                              for p in sorted({find_shard(args.checkpoint, n) for n in args.tensors})},
        "versions": {p: importlib.metadata.version(p)
                     for p in ["torch", "bitsandbytes", "numpy", "safetensors"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)
    count = 0
    with (args.output / "observations.jsonl").open("x") as log:
        for name in args.tensors:
            source, state = load_tensor(args.checkpoint, name)
            raw = source.numpy().tobytes()
            original_codes = unpack(raw)
            clean_weights = dequantize_4bit(source, quant_state=state).float()
            clean_transforms = {0: clean_weights}
            clean_q, clean_state = source, state
            for cycle in range(1, args.cycles + 1):
                clean_q, clean_state = requantize(clean_q, clean_state)
                clean_transforms[cycle] = dequantize_4bit(clean_q, quant_state=clean_state).float()
            cb = state.code.float().numpy()
            for bits in args.bits:
                for seed in args.seeds:
                    payload = np.random.default_rng(seed).bytes(bits // 8)
                    key = hashlib.sha256(f"NES-development-{seed}".encode()).digest()
                    for method in METHODS:
                        base = {"tensor": name, "method": method, "net_payload_bits": bits,
                                "seed": seed, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                                "source_tensor_sha256": hashlib.sha256(raw).hexdigest()}
                        start = time.monotonic()
                        try:
                            stego, info = embed(raw, payload, key, name, method, cb)
                        except ValueError as exc:
                            log.write(json.dumps({**base, "status": "EMBED_FAILED", "error": str(exc)}) + "\n")
                            log.flush()
                            continue
                        updated = unpack(stego)
                        anchor_mask = (original_codes == 0) | (original_codes == 15)
                        blocks = updated.reshape(-1, state.blocksize)
                        missing_anchors = ~np.any((blocks == 0) | (blocks == 15), axis=1)
                        tensor = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(source.shape)
                        current_state = state
                        for cycle in range(args.cycles + 1):
                            if cycle:
                                tensor, current_state = requantize(tensor, current_state)
                            current = tensor.numpy().tobytes()
                            weights = dequantize_4bit(tensor, quant_state=current_state).float()
                            row = {
                                **base, **info, "status": "MEASURED", "cycle": cycle,
                                "embedding_seconds": time.monotonic() - start if cycle == 0 else None,
                                "stored_dtype": str(state.dtype), "blocksize": state.blocksize,
                                "nested_statistics": state.nested,
                                "changed_anchor_codes": int(np.count_nonzero(original_codes[anchor_mask] != updated[anchor_mask])),
                                "blocks_without_endpoint_after_embedding": int(missing_anchors.sum()),
                                "eligibility_changed_after_transform": int(np.count_nonzero(
                                    eligible(updated, method) != eligible(unpack(current), method))),
                                "weight_mse_vs_transformed_clean": float(torch.mean((weights - clean_transforms[cycle])**2)),
                                "weight_mse_vs_original_clean": float(torch.mean((weights - clean_weights)**2)),
                                "output_packed_sha256": hashlib.sha256(current).hexdigest(),
                                **recovery_diagnostic(current, payload, key, name, method),
                            }
                            log.write(json.dumps(row, allow_nan=False) + "\n")
                            log.flush()
                            count += 1
                        print(json.dumps({**base, "final_exact_recovery": row["exact_recovery"],
                                          "final_ber": row["payload_ber"]}), flush=True)
    write_json(args.output / "completion.json", {"status": "COMPLETED", "measured_rows": count})


if __name__ == "__main__":
    main()
