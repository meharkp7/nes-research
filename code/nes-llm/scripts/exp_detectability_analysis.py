#!/usr/bin/env python3
"""Analysis for the interior_nearest detectability evaluation.

Reads features.npz from exp_detectability.py and reports:
  * per-feature effect sizes (clean vs embedded), grouped by model and tensor
  * grouped-CV statistical detectors (logistic regression, gradient boosting)
  * a small neural detector (exploratory)
  * leakage controls: clean-vs-clean and embedded seed-vs-seed
  * leave-one-model-out transfer

Grouping is by source tensor for CV, so blocks from one tensor never appear in
both train and test. With only two checkpoints, model-transfer numbers are
exploratory and are labelled as such.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 20261016


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load(output):
    d = np.load(output / "features.npz", allow_pickle=True)
    X = d["X"]
    names = list(d["feature_names"])
    meta = [json.loads(s) for s in d["meta_json"]]
    return X, names, meta


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def tpr_at_fpr(scores, labels, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx]), float(fpr[idx])


def grouped_cv(scores_fn, X, y, groups, n_splits=6):
    """Out-of-fold scores via GroupKFold."""
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        oof[te] = scores_fn(X[tr], y[tr], X[te])
    return oof


def lr_scores(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict_proba(scaler.transform(Xte))[:, 1]


def gb_scores(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def mlp_scores_factory(epochs=60):
    import torch
    import torch.nn as nn

    def run(Xtr, ytr, Xte):
        torch.manual_seed(SEED)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd
        model = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
                              nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        xt = torch.tensor(Xtr_n, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(model(xt), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(Xte_n, dtype=torch.float32))).squeeze(1).numpy()
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-mlp", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    X, names, meta = load(args.features_dir)
    model = np.array([m["model"] for m in meta])
    tensor = np.array([m["tensor"] for m in meta])
    label = np.array([1 if m["label"] == "embedded" else 0 for m in meta])
    size = np.array([m["size"] for m in meta])
    seed = np.array([m["seed"] for m in meta])
    group = np.array([f"{m['model']}::{m['tensor']}" for m in meta])

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_detectability_analysis_20261016",
        "features_dir": str(args.features_dir.resolve()),
        "features_file_sha256": hashlib.sha256(
            (args.features_dir / "features.npz").read_bytes()).hexdigest(),
        "n_examples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_groups": int(len(set(group))),
        "grouping": "GroupKFold by model::tensor (blocks from one tensor never split)",
        "features": names,
        "versions": {m: importlib.metadata.version(m)
                     for m in ["numpy", "scikit-learn", "torch"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "analysis_manifest.json", manifest)

    # --- Effect sizes: clean vs embedded, per model and per tensor ------------
    effects = []
    for m in sorted(set(model)):
        for t in sorted(set(tensor[model == m])):
            sel = (model == m) & (tensor == t)
            cl = X[sel & (label == 0)]
            em = X[sel & (label == 1)]
            for j, fname in enumerate(names):
                d = cohen_d(em[:, j], cl[:, j])
                effects.append({"model": m, "tensor": t, "feature": fname,
                                "cohen_d": d,
                                "clean_mean": float(cl[:, j].mean()),
                                "embedded_mean": float(em[:, j].mean())})
    write_json(args.output / "effect_sizes.json", {"effects": effects})

    # pooled effect sizes per model
    pooled = {}
    for m in sorted(set(model)):
        sel = model == m
        pooled[m] = {fname: cohen_d(X[sel & (label == 1)][:, j], X[sel & (label == 0)][:, j])
                     for j, fname in enumerate(names)}
    write_json(args.output / "effect_sizes_pooled.json", pooled)

    # --- Detectors with grouped CV -------------------------------------------
    results = {"detectors": {}}
    for det_name, fn in [("logistic_regression", lr_scores),
                         ("gradient_boosting", gb_scores)] + (
                        [] if args.no_mlp else [("mlp_exploratory", mlp_scores_factory())]):
        oof = grouped_cv(fn, X, label, group)
        auc = float(roc_auc_score(label, oof))
        tpr1, fpr1 = tpr_at_fpr(oof, label, 0.01)
        tpr5, fpr5 = tpr_at_fpr(oof, label, 0.05)
        results["detectors"][det_name] = {
            "auc": auc, "tpr_at_fpr_0.01": tpr1, "tpr_at_fpr_0.05": tpr5,
        }
        # per-size AUC
        for s in sorted(set(size[label == 1])):
            sel = (label == 0) | (size == s)
            results["detectors"][det_name][f"auc_size_{s}"] = float(
                roc_auc_score(label[sel], oof[sel]))
        # per-model AUC (within-model, still grouped by tensor)
        for m in sorted(set(model)):
            sel = model == m
            results["detectors"][det_name][f"auc_model_{m}"] = float(
                roc_auc_score(label[sel], oof[sel]))

    # --- Control 1: clean vs clean (two disjoint block halves) ---------------
    # Relabel: even-index blocks of clean artifacts = class 0, odd = class 1.
    # We approximate by splitting each clean artifact's blocks in half.
    cl_idx = np.where(label == 0)[0]
    # Build a group-preserving split: within each (model,tensor) clean set, first
    # half vs second half.
    ctrl1_X, ctrl1_y, ctrl1_g = [], [], []
    for g in sorted(set(group[cl_idx])):
        idx = cl_idx[group[cl_idx] == g]
        half = len(idx) // 2
        ctrl1_X.append(X[idx[:half]]); ctrl1_y.append(np.zeros(half))
        ctrl1_X.append(X[idx[half:2 * half]]); ctrl1_y.append(np.ones(half))
        ctrl1_g.append(np.array([g] * half)); ctrl1_g.append(np.array([g] * half))
    Xc1 = np.concatenate(ctrl1_X); yc1 = np.concatenate(ctrl1_y); gc1 = np.concatenate(ctrl1_g)
    oof1 = grouped_cv(gb_scores, Xc1, yc1, gc1)
    results["control_clean_vs_clean"] = {"auc": float(roc_auc_score(yc1, oof1)),
                                         "expected": "~0.5"}

    # --- Control 2: embedded seed vs seed -----------------------------------
    em_idx = np.where(label == 1)[0]
    keep = np.isin(seed[em_idx], [20261016, 20261017])
    sel = em_idx[keep]
    Xc2 = X[sel]
    yc2 = (seed[sel] == 20261017).astype(int)
    gc2 = group[sel]
    oof2 = grouped_cv(gb_scores, Xc2, yc2, gc2)
    results["control_seed_vs_seed"] = {"auc": float(roc_auc_score(yc2, oof2)),
                                       "expected": "~0.5 (payload bits are the only difference)"}

    # --- Leave-one-model-out transfer (exploratory, 1 test checkpoint) ------
    transfer = {}
    for m in sorted(set(model)):
        tr = model != m
        te = model == m
        oof = gb_scores(X[tr], label[tr], X[te])
        transfer[m] = {"auc": float(roc_auc_score(label[te], oof)),
                       "note": "exploratory: single held-out checkpoint"}
    results["leave_one_model_out"] = transfer

    write_json(args.output / "detector_results.json", results)
    print(json.dumps({"status": "COMPLETED", "detectors": list(results["detectors"])}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
