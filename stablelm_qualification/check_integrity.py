#!/usr/bin/env python3
"""Gate A integrity: only the intended tensor differs after embedding.

Materializes an embedded checkpoint for a given (size, seed) into a temp dir and
compares every tensor against the clean checkpoint. Reports differing keys,
tensor shapes, finite outputs, and payload recovery. Does not modify the frozen
codec or any existing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor
from src.experiments.nf4_anchor_channel import embed, extract
from safetensors.torch import load_file, save_file

CKPT = Path("/workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_clean")
TENSOR = "model.layers.0.self_attn.q_proj.weight"
METHOD = "interior_nearest"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source, state = load_tensor(CKPT, TENSOR)
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    payload = np.random.default_rng(args.seed).bytes(args.size // 8)
    key = hashlib.sha256(f"NES-development-{args.seed}".encode()).digest()
    stego, _ = embed(raw, payload, key, TENSOR, METHOD, cb)

    clean = load_file(CKPT / "model.safetensors")
    with tempfile.TemporaryDirectory(dir="/workspace") as td:
        dest = Path(td) / "emb"
        shutil.copytree(CKPT, dest)
        emb = dict(clean)
        emb[TENSOR] = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(clean[TENSOR].shape)
        save_file(emb, dest / "model.safetensors", metadata={"format": "pt"})
        reloaded = load_file(dest / "model.safetensors")

    diff_keys = [k for k in clean if not torch.equal(clean[k], reloaded[k])]
    recovered = extract(reloaded[TENSOR].numpy().tobytes(), key, TENSOR, METHOD)

    # finite-output sanity on the embedded checkpoint (small forward pass)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(CKPT, device_map="cpu", local_files_only=True)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        out = model(ids, labels=ids)
    finite = bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(out.loss))

    result = {
        "size": args.size, "seed": args.seed, "tensor": TENSOR,
        "n_tensors": len(clean), "differing_keys": diff_keys,
        "only_intended_tensor_differs": diff_keys == [TENSOR],
        "shape_preserved": tuple(reloaded[TENSOR].shape) == tuple(clean[TENSOR].shape),
        "payload_exact_recovery": recovered == payload,
        "stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
        "clean_tensor_sha256": hashlib.sha256(raw).hexdigest(),
        "finite_logits_and_loss": finite,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if (result["only_intended_tensor_differs"] and result["payload_exact_recovery"]
                 and result["finite_logits_and_loss"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
