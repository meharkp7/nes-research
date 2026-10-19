#!/usr/bin/env python3
"""Targeted detectability supplement: small-capacity tensors at 50,000 bits.

Removes the sparsity confound identified in RESULTS_DETECTABILITY.md by
evaluating only the small-capacity attention tensors (k_proj, v_proj), where
~95% of blocks are modified at 50,000 bits.

Reuses the 16 predeclared features and the existing detector pipeline. All
splits are grouped by model::tensor (artifact), so no tensor contributes to
both train and test. Results are labelled exploratory if the group count is
too small for a definitive conclusion. The frozen codec is unchanged.
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
from src.experiments.nf4_anchor_channel import embed, unpack
from scripts.exp_detectability import (
    FEATURES, BLOCKSIZE, METHOD, payload_for, key_for, features_for_blocks,
    write_json, cohen_d,
)
from scripts.exp_detectability_analysis import tpr_at_fpr, lr_scores, gb_scores

SIZE = 50000
SEEDS = (20261016, 20261017, 20261018)
BLOCKS_PER_ARTIFACT = 2048
SAMPLE_SEED = 424242

# Small-capacity tensors, one per model per type.
TARGETS = {
    "tinyllama": {
        "path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/tinyllama_clean",
        "tensors": [
            "model.layers.21.self_attn.k_proj.weight",
            "model.layers.21.self_attn.v_proj.weight",
        ],
    },
    "qwen": {
        "path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/qwen_clean",
        "tensors": [
            "model.layers.17.self_attn.k_proj.weight",
            "model.layers.35.self_attn.v_proj.weight",
        ],
    },
}


def build_dataset(output):
    X_list, meta = [], []
    source_hashes = {}
    sparsity = []
    for model, spec in TARGETS.items():
        for tensor in spec["tensors"]:
            source, state = load_tensor(spec["path"], tensor)
            raw = source.numpy().tobytes()
            codes = unpack(raw)
            cb = state.code.float().numpy()
            nb = len(codes) // BLOCKSIZE
            clean_blocks = codes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE)
            src_sha = hashlib.sha256(raw).hexdigest()
            source_hashes[f"{model}:{tensor}"] = src_sha
            rng = np.random.default_rng(SAMPLE_SEED)
            idx = np.sort(rng.choice(nb, size=min(BLOCKS_PER_ARTIFACT, nb), replace=False))

            def add(blocks_all, m):
                f = features_for_blocks(blocks_all[idx], cb)
                X_list.append(f)
                meta.extend([{**m, "source_tensor_sha256": src_sha}] * len(f))

            add(clean_blocks, {"model": model, "tensor": tensor, "label": "clean", "size": 0, "seed": 0})
            for seed in SEEDS:
                stego, info = embed(raw, payload_for(SIZE, seed), key_for(seed), tensor, METHOD, cb)
                scodes = unpack(stego)
                diff = scodes != codes
                touched = int(diff.reshape(nb, BLOCKSIZE).any(1).sum())
                sparsity.append({"model": model, "tensor": tensor, "seed": seed,
                                 "changed_codes": int(diff.sum()), "total_blocks": nb,
                                 "blocks_touched": touched, "blocks_touched_frac": touched / nb,
                                 "sampled_touched": int(diff.reshape(nb, BLOCKSIZE).any(1)[idx].sum())})
                add(scodes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE),
                    {"model": model, "tensor": tensor, "label": "embedded", "size": SIZE, "seed": seed})
    X = np.concatenate(X_list)
    np.savez_compressed(output / "features.npz", X=X, feature_names=np.array(FEATURES),
                        meta_json=np.array([json.dumps(m) for m in meta]))
    write_json(output / "source_hashes.json", source_hashes)
    write_json(output / "block_sparsity.json", {"per_artifact": sparsity})
    return X, meta


def bootstrap_ci(scores, labels, groups, n_boot=2000, seed=20261016):
    """Group-level bootstrap CI for AUC (resample groups with replacement)."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-mlp", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_detectability_smallcap_20261016",
        "scope": "Targeted supplement: small-capacity k_proj/v_proj at 50,000 bits",
        "method": METHOD, "size": SIZE, "seeds": list(SEEDS),
        "blocks_per_artifact": BLOCKS_PER_ARTIFACT, "sample_seed": SAMPLE_SEED,
        "features": list(FEATURES),
        "grouping": "GroupKFold / leave-one-tensor-out by model::tensor",
        "targets": {m: s["tensors"] for m, s in TARGETS.items()},
        "versions": {m: importlib.metadata.version(m)
                     for m in ["numpy", "scikit-learn", "torch", "bitsandbytes", "safetensors"]},
        "codec_sha256": sha_file(Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py"),
        "platform": platform.platform(),
    }
    write_json(args.output / "manifest.json", manifest)

    X, meta = build_dataset(args.output)
    label = np.array([1 if m["label"] == "embedded" else 0 for m in meta])
    model = np.array([m["model"] for m in meta])
    tensor = np.array([m["tensor"] for m in meta])
    group = np.array([f"{m['model']}::{m['tensor']}" for m in meta])
    n_groups = len(set(group))

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    results = {"n_examples": int(len(label)), "n_groups": n_groups,
               "n_embedded_artifacts": int(len({(m["model"], m["tensor"], m["seed"])
                                               for m in meta if m["label"] == "embedded"})),
               "group_labels": sorted(set(group.tolist()))}

    def grouped_oof(fn, Xa, ya, ga):
        gkf = GroupKFold(n_splits=min(4, len(set(ga))))
        oof = np.zeros(len(ya))
        for tr, te in gkf.split(Xa, ya, ga):
            oof[te] = fn(Xa[tr], ya[tr], Xa[te])
        return oof

    detectors = {"gradient_boosting": gb_scores, "logistic_regression": lr_scores}
    if not args.no_mlp:
        from scripts.exp_detectability_analysis import mlp_scores_factory
        detectors["mlp_exploratory"] = mlp_scores_factory()

    results["detectors"] = {}
    for name, fn in detectors.items():
        oof = grouped_oof(fn, X, label, group)
        auc = float(roc_auc_score(label, oof))
        lo, hi = bootstrap_ci(oof, label, group)
        t1, f1 = tpr_at_fpr(oof, label, 0.01)
        t5, f5 = tpr_at_fpr(oof, label, 0.05)
        entry = {"auc": auc, "auc_ci95": [lo, hi], "tpr_at_fpr_0.01": t1, "tpr_at_fpr_0.05": t5}
        for m in sorted(set(model)):
            sel = model == m
            entry[f"auc_model_{m}"] = float(roc_auc_score(label[sel], oof[sel]))
        for g in sorted(set(group)):
            sel = group == g
            if len(np.unique(label[sel])) > 1:
                entry[f"auc_group_{g}"] = float(roc_auc_score(label[sel], oof[sel]))
        results["detectors"][name] = entry

    # Clean-vs-clean control at the same granularity.
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

    # Effect sizes on the targeted tensors.
    eff = []
    for m in sorted(set(model)):
        for t in sorted(set(tensor[model == m])):
            sel = (model == m) & (tensor == t)
            for j, fname in enumerate(FEATURES):
                eff.append({"model": m, "tensor": t, "feature": fname,
                            "cohen_d": cohen_d(X[sel & (label == 1)][:, j], X[sel & (label == 0)][:, j])})
    write_json(args.output / "effect_sizes.json", {"effects": eff})

    write_json(args.output / "supplement_results.json", results)
    print(json.dumps({"status": "COMPLETED", "n_examples": int(len(label)),
                      "n_groups": n_groups,
                      "aucs": {k: v["auc"] for k, v in results["detectors"].items()}}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
