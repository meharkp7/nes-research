#!/usr/bin/env python3
"""Detectability evaluation for the frozen interior_nearest method.

Question: can clean vs embedded NF4 tensor artifacts be distinguished from
key-agnostic, predeclared block features (and a small neural detector)?

Design notes (leakage control):
- Examples are 64-code blocks. Blocks from the same source tensor are never
  split across train and test. Splits are grouped by source tensor and by
  checkpoint.
- Only two checkpoints exist, so any held-out neural result is exploratory.
- Controls: clean-vs-clean (two halves of the clean blocks) and
  embedded-seed-vs-seed, both expected at chance if there is no leakage.

The frozen codec is used unchanged; nothing is written outside --output.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_nf4_anchor_pilot import load_tensor, sha_file
from scripts.exp_tensor_matrix import list_packed_tensors
from src.experiments.nf4_anchor_channel import embed, unpack

METHOD = "interior_nearest"
BASELINE = "pair_parity"
BLOCKSIZE = 64
SIZES = (10000, 50000)
SEEDS = (20261016, 20261017, 20261018)
BLOCKS_PER_ARTIFACT = 1024
SAMPLE_SEED = 424242

MODELS = {
    "tinyllama": {
        "path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/tinyllama_clean",
        "tensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.11.self_attn.k_proj.weight",
            "model.layers.21.self_attn.v_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.11.mlp.down_proj.weight",
            "model.layers.21.mlp.gate_proj.weight",
        ],
    },
    "qwen": {
        "path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/qwen_clean",
        "tensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.17.self_attn.k_proj.weight",
            "model.layers.35.self_attn.v_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.17.mlp.down_proj.weight",
            "model.layers.35.mlp.gate_proj.weight",
        ],
    },
}

# --- Predeclared block features (computed on the 64 unpacked NF4 codes) -------
# Rationale: interior_nearest (i) never changes endpoint codes 0/15, (ii) forces
# each carrier's code parity to the payload bit, and (iii) remaps interior codes
# to the nearest same-parity interior code. Any of those can leave a signature in
# parity, interior-shape, or transition statistics. Scale features test the
# block-scale hypothesis. Endpoint features test whether anchor preservation is
# itself a tell.
FEATURES = (
    "endpoint_frac",          # anchor preservation: does keeping 0/15 leave a tell?
    "zero_count",             # endpoint -1 occupancy
    "fifteen_count",          # endpoint +1 occupancy
    "interior_frac",          # fraction of codes in 1..14 (embedding stays interior)
    "mean_code",              # interior remap shifts the code mean
    "std_code",               # interior remap changes spread
    "odd_frac",               # parity forcing flattens the odd/even ratio
    "parity_imbalance",       # |#odd - #even| / 64
    "odd_interior_frac",      # parity of interior codes specifically
    "distinct_codes",         # remap reduces the number of distinct codes
    "code_entropy",           # Shannon entropy of the 16-bin code histogram
    "mean_abs_value",         # dequantized magnitude shift
    "absmax_value",           # block scale proxy (max |codebook value|)
    "adjacent_equal_frac",    # transition statistics
    "adjacent_parity_change", # parity transition statistics
    "max_run",                # run-length statistic
)


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def features_for_blocks(blocks, cb):
    """blocks: (n, 64) uint8 codes. Returns (n, len(FEATURES)) float64."""
    n = blocks.shape[0]
    out = np.zeros((n, len(FEATURES)), dtype=np.float64)
    abs_cb = np.abs(cb)
    for i in range(n):
        c = blocks[i].astype(np.int64)
        odd = (c & 1).astype(bool)
        is_zero = c == 0
        is_fifteen = c == 15
        interior = (c >= 1) & (c <= 14)
        hist = np.bincount(c, minlength=16).astype(np.float64)
        p = hist / 64.0
        nz = p[p > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        eq = (c[1:] == c[:-1])
        pchg = ((c[1:] & 1) != (c[:-1] & 1))
        # longest run of equal codes
        max_run = run = 1
        for j in range(1, 64):
            run = run + 1 if c[j] == c[j - 1] else 1
            if run > max_run:
                max_run = run
        out[i] = (
            (is_zero | is_fifteen).mean(),
            is_zero.sum(),
            is_fifteen.sum(),
            interior.mean(),
            c.mean(),
            c.std(),
            odd.mean(),
            abs(int(odd.sum()) - int((~odd).sum())) / 64.0,
            (odd & interior).sum() / max(interior.sum(), 1),
            np.unique(c).size,
            entropy,
            abs_cb[c].mean(),
            abs_cb[c].max(),
            eq.mean(),
            pchg.mean(),
            float(max_run),
        )
    return out


def sample_blocks(codes, count, rng):
    nblocks = len(codes) // BLOCKSIZE
    blocks = codes[:nblocks * BLOCKSIZE].reshape(nblocks, BLOCKSIZE)
    if nblocks <= count:
        return blocks
    idx = np.sort(rng.choice(nblocks, size=count, replace=False))
    return blocks[idx]


def build_dataset(output):
    rows_feat = []
    rows_meta = []
    source_hashes = {}
    for model, spec in MODELS.items():
        ckpt = spec["path"]
        available = set(list_packed_tensors(ckpt))
        for tensor in spec["tensors"]:
            if tensor not in available:
                raise SystemExit(f"Tensor missing: {model} {tensor}")
            source, state = load_tensor(ckpt, tensor)
            raw = source.numpy().tobytes()
            codes = unpack(raw)
            cb = state.code.float().numpy()
            src_sha = hashlib.sha256(raw).hexdigest()
            source_hashes[f"{model}:{tensor}"] = src_sha
            rng = np.random.default_rng(SAMPLE_SEED)
            # clean control artifact
            blocks = sample_blocks(codes, BLOCKS_PER_ARTIFACT, rng)
            rows_feat.append(features_for_blocks(blocks, cb))
            rows_meta.append({"model": model, "tensor": tensor, "label": "clean",
                              "size": 0, "seed": 0, "source_tensor_sha256": src_sha,
                              "artifact_sha256": src_sha})
            # embedded artifacts
            for size in SIZES:
                for seed in SEEDS:
                    payload = payload_for(size, seed)
                    stego, info = embed(raw, payload, key_for(seed), tensor, METHOD, cb)
                    scodes = unpack(stego)
                    rng = np.random.default_rng(SAMPLE_SEED)
                    blocks = sample_blocks(scodes, BLOCKS_PER_ARTIFACT, rng)
                    rows_feat.append(features_for_blocks(blocks, cb))
                    rows_meta.append({"model": model, "tensor": tensor, "label": "embedded",
                                      "size": size, "seed": seed,
                                      "source_tensor_sha256": src_sha,
                                      "artifact_sha256": hashlib.sha256(stego).hexdigest(),
                                      "changed_codes": int(info["changed_codes"])})
    X = np.concatenate(rows_feat, axis=0)
    meta = []
    for j, m in enumerate(rows_meta):
        for _ in range(rows_feat[j].shape[0]):
            meta.append(m)
    np.savez_compressed(output / "features.npz", X=X,
                        feature_names=np.array(FEATURES),
                        meta_json=np.array([json.dumps(m) for m in meta]))
    write_json(output / "source_hashes.json", source_hashes)
    return X, meta


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def auc_score(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        p.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_detectability_20261016",
        "scope": "Detectability of frozen interior_nearest artifacts vs matched clean controls",
        "method": METHOD, "baseline": BASELINE,
        "sizes": list(SIZES), "seeds": list(SEEDS),
        "blocks_per_artifact": BLOCKS_PER_ARTIFACT, "sample_seed": SAMPLE_SEED,
        "features": list(FEATURES),
        "feature_rationale": {
            "endpoint_frac/zero_count/fifteen_count": "anchor preservation may itself be a tell",
            "interior_frac/mean_code/std_code": "interior remap shifts the code distribution",
            "odd_frac/parity_imbalance/odd_interior_frac": "embedding forces carrier parity",
            "distinct_codes/code_entropy": "remap reduces distinct codes / flattens entropy",
            "mean_abs_value/absmax_value": "dequantized magnitude and block-scale shift",
            "adjacent_equal_frac/adjacent_parity_change/max_run": "transition/run-length statistics",
        },
        "checkpoints": {m: s["path"] for m, s in MODELS.items()},
        "source_hashes": {str(Path(__file__).resolve()): sha_file(Path(__file__)),
                          "codec": sha_file(Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py")},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["torch", "bitsandbytes", "numpy", "safetensors", "scikit-learn"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    X, meta = build_dataset(args.output)
    write_json(args.output / "dataset.json", {
        "n_examples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_artifacts": len({(m["model"], m["tensor"], m["label"], m["size"], m["seed"]) for m in meta}),
    })
    print(json.dumps({"status": "DATASET_BUILT", "n_examples": int(X.shape[0])}), flush=True)


def self_test():
    cb = np.array([-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
                   -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
                   0.44070983, 0.562617, 0.72295684, 1.0])
    blocks = np.tile(np.arange(16, dtype=np.uint8).repeat(4), (2, 1))[:2]
    f = features_for_blocks(blocks, cb)
    assert f.shape == (2, len(FEATURES)), f.shape
    assert np.allclose(f[0], f[1]), "features must be deterministic"
    assert abs(f[0][FEATURES.index("endpoint_frac")] - 8 / 64) < 1e-9
    assert abs(f[0][FEATURES.index("odd_frac")] - 0.5) < 1e-9
    print("self-test PASS:", json.dumps({"features": len(FEATURES), "deterministic": True}))


if __name__ == "__main__":
    raise SystemExit(main())
