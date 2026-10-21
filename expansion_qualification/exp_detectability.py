#!/usr/bin/env python3
"""Per-model detectability evaluation (Track 3, V2 protocol).

Generalized from stablelm_qualification/exp_stablelm_detectability.py. Reuses the
16 predeclared block features and detector methodology unchanged. Each model is
evaluated SEPARATELY and never pooled with other models.

Frozen protocol (QUALIFICATION_PROTOCOL_V2.md Track A4):
  - frozen interior_nearest codec; matched clean controls;
  - payloads 10k and 50k; seeds 20261016/017/018;
  - grouped splits by model::tensor (GroupKFold);
  - detectors: logistic regression, gradient boosting, exploratory MLP;
  - report AUC + grouped-bootstrap CI, TPR@1%/5% FPR, Cohen's d, clean-vs-clean
    AUC, permutation test, group counts, runtime.
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

REPO = "/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, sha_file  # noqa: E402
from src.experiments.nf4_anchor_channel import embed, unpack  # noqa: E402
from scripts.exp_detectability import (  # noqa: E402
    FEATURES, BLOCKSIZE, payload_for, key_for, features_for_blocks, cohen_d,
)
from scripts.exp_detectability_analysis import tpr_at_fpr, lr_scores, gb_scores  # noqa: E402

METHOD = "interior_nearest"
SIZES = (10000, 50000)
SEEDS = (20261016, 20261017, 20261018)
BLOCKS_PER_ARTIFACT = 2048
SAMPLE_SEED = 424242


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_dataset(ckpt, tensors, output, group_prefix):
    X_list, meta = [], []
    source_hashes, sparsity = {}, []
    for tensor in tensors:
        source, state = load_tensor(ckpt, tensor)
        raw = source.numpy().tobytes()
        codes = unpack(raw)
        cb = state.code.float().numpy()
        nb = len(codes) // BLOCKSIZE
        clean_blocks = codes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE)
        src_sha = hashlib.sha256(raw).hexdigest()
        source_hashes[tensor] = src_sha
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = np.sort(rng.choice(nb, size=min(BLOCKS_PER_ARTIFACT, nb), replace=False))

        def add(blocks_all, m):
            f = features_for_blocks(blocks_all[idx], cb)
            X_list.append(f)
            meta.extend([{**m, "source_tensor_sha256": src_sha}] * len(f))

        add(clean_blocks, {"tensor": tensor, "label": "clean", "size": 0, "seed": 0})
        for size in SIZES:
            for seed in SEEDS:
                stego, _ = embed(raw, payload_for(size, seed), key_for(seed), tensor, METHOD, cb)
                scodes = unpack(stego)
                diff = scodes != codes
                touched = int(diff.reshape(nb, BLOCKSIZE).any(1).sum())
                sparsity.append({"tensor": tensor, "seed": seed, "size": size,
                                 "changed_codes": int(diff.sum()), "total_blocks": nb,
                                 "blocks_touched": touched, "blocks_touched_frac": touched / nb,
                                 "sampled_touched": int(diff.reshape(nb, BLOCKSIZE).any(1)[idx].sum()),
                                 "sampled_blocks": len(idx)})
                add(scodes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE),
                    {"tensor": tensor, "label": "embedded", "size": size, "seed": seed})
    X = np.concatenate(X_list)
    np.savez_compressed(output / "features.npz", X=X, feature_names=np.array(FEATURES),
                        meta_json=np.array([json.dumps(m) for m in meta]))
    write_json(output / "source_hashes.json", source_hashes)
    write_json(output / "block_sparsity.json", {"per_artifact": sparsity})
    return X, meta


def bootstrap_ci(scores, labels, groups, n_boot=2000, seed=20261016):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, scores[idx]))
    if not aucs:
        return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def permutation_test(scores, labels, groups, n_perm=200, seed=20261016):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(labels, scores)
    uniq = np.array(sorted(set(groups)))
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n_perm):
        y = labels.copy()
        for g in uniq:
            i = idx_by_group[g]
            y[i] = rng.permutation(y[i])
        if len(np.unique(y)) < 2:
            continue
        stats.append(roc_auc_score(y, scores))
    if not stats:
        return obs, None, 0
    stats = np.array(stats)
    p = float((np.sum(stats >= obs) + 1) / (len(stats) + 1))
    return obs, p, len(stats)


def grouped_oof(fn, Xa, ya, ga, n_splits=4):
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(n_splits, len(set(ga))))
    oof = np.zeros(len(ya))
    for tr, te in gkf.split(Xa, ya, ga):
        oof[te] = fn(Xa[tr], ya[tr], Xa[te])
    return oof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensors", nargs="+", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--blocks-per-artifact", type=int, default=BLOCKS_PER_ARTIFACT)
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    group_prefix = args.model_id.split("/")[-1].lower()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id, "model": args.model_id, "revision": args.revision,
        "method": METHOD, "sizes": list(args.sizes), "seeds": list(SEEDS),
        "tensors": args.tensors, "features": list(FEATURES),
        "blocks_per_artifact": args.blocks_per_artifact, "sample_seed": SAMPLE_SEED,
        "grouping": "GroupKFold by model::tensor",
        "screening_thresholds": {"auc_max": 0.55, "tpr_at_1pct_fpr_max": 0.02},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["numpy", "scikit-learn", "torch", "bitsandbytes", "safetensors"]},
        "codec_sha256": sha_file(Path(f"{REPO}/src/experiments/nf4_anchor_channel.py")),
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    t0 = time.time()
    X, meta = build_dataset(args.checkpoint, args.tensors, args.output, group_prefix)
    build_s = time.time() - t0
    label = np.array([1 if m["label"] == "embedded" else 0 for m in meta])
    tensor = np.array([m["tensor"] for m in meta])
    group = np.array([f"{group_prefix}::{m['tensor']}" for m in meta])
    n_groups = len(set(group))

    from sklearn.metrics import roc_auc_score
    results = {"n_examples": int(len(label)), "n_groups": n_groups,
               "n_embedded_artifacts": int(len({(m["tensor"], m["seed"], m["size"])
                                                for m in meta if m["label"] == "embedded"})),
               "group_labels": sorted(set(group.tolist())),
               "build_seconds": round(build_s, 1)}

    detectors = {"gradient_boosting": gb_scores, "logistic_regression": lr_scores}
    if not args.no_mlp:
        from scripts.exp_detectability_analysis import mlp_scores_factory
        detectors["mlp_exploratory"] = mlp_scores_factory()

    results["detectors"] = {}
    for name, fn in detectors.items():
        t = time.time()
        oof = grouped_oof(fn, X, label, group)
        auc = float(roc_auc_score(label, oof))
        lo, hi = bootstrap_ci(oof, label, group)
        t1, f1 = tpr_at_fpr(oof, label, 0.01)
        t5, f5 = tpr_at_fpr(oof, label, 0.05)
        obs, pval, nperm = permutation_test(oof, label, group)
        entry = {"auc": auc, "auc_ci95": [lo, hi],
                 "tpr_at_fpr_0.01": t1, "tpr_at_fpr_0.05": t5,
                 "permutation_p": pval, "permutation_n": nperm,
                 "runtime_s": round(time.time() - t, 1)}
        for tname in sorted(set(tensor)):
            sel = tensor == tname
            entry[f"auc_tensor_{tname}"] = float(roc_auc_score(label[sel], oof[sel]))
        results["detectors"][name] = entry
        print(json.dumps({"detector": name, "auc": auc, "tpr@1%": t1, "perm_p": pval}), flush=True)

    cl = np.where(label == 0)[0]
    cX, cy, cg = [], [], []
    for g in sorted(set(group[cl])):
        i = cl[group[cl] == g]
        h = len(i) // 2
        cX += [X[i[:h]], X[i[h:2 * h]]]
        cy += [np.zeros(h), np.ones(h)]
        cg += [np.array([g] * h), np.array([g] * h)]
    cX = np.concatenate(cX); cy = np.concatenate(cy); cg = np.concatenate(cg)
    coof = grouped_oof(gb_scores, cX, cy, cg)
    results["control_clean_vs_clean"] = {"auc": float(roc_auc_score(cy, coof)),
                                         "auc_ci95": list(bootstrap_ci(coof, cy, cg))}

    eff = []
    for tname in sorted(set(tensor)):
        sel = tensor == tname
        for j, fname in enumerate(FEATURES):
            eff.append({"tensor": tname, "feature": fname,
                        "cohen_d": cohen_d(X[sel & (label == 1)][:, j], X[sel & (label == 0)][:, j])})
    write_json(args.output / "effect_sizes.json", {"effects": eff})

    write_json(args.output / "detectability_results.json", results)
    print(json.dumps({"status": "COMPLETED", "n_examples": int(len(label)),
                      "n_groups": n_groups,
                      "aucs": {k: v["auc"] for k, v in results["detectors"].items()}}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
