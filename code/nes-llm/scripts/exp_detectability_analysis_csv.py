#!/usr/bin/env python3
"""Analysis for detectability evaluation from CSV output (stage1 CSV format)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 20261016
FEATURES = [
    "f_mean", "f_std", "f_min", "f_max", "f_median", "f_q25", "f_q75",
    "f_mean_square", "f_positive_fraction", "f_skew",
    "f_excess_kurtosis", "f_hist_entropy", "f_zero_fraction", "f_abs_mean",
]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_csv(output_dir: Path):
    """Load CSV output from stage1."""
    block_features = pd.read_csv(output_dir / "residual_paired_block_features.csv")
    layer_desc = pd.read_csv(output_dir / "residual_layer_descriptives.csv")
    cell_status = pd.read_csv(output_dir / "cell_status.csv")
    return block_features, layer_desc, cell_status


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


def bootstrap_ci(scores, labels, groups, n_boot=2000, seed=20261016):
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


def grouped_cv(scores_fn, X, y, groups, n_splits=6):
    """Out-of-fold scores via GroupKFold."""
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        oof[te] = scores_fn(X[tr], y[tr], X[te])
    return oof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-mlp", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # Load CSV data
    block_features = pd.read_csv(args.features_dir / "residual_paired_block_features.csv")
    layer_desc = pd.read_csv(args.features_dir / "residual_layer_descriptives.csv")
    cell_status = pd.read_csv(args.features_dir / "cell_status.csv")

    # Filter to PAIRED_DESCRIPTIVE_READY cells
    cell_status = cell_status[cell_status["status"] == "PAIRED_DESCRIPTIVE_READY"]
    
    # Merge block features with cell status to get model/method labels
    block_features = block_features.merge(
        cell_status[["model_id", "method", "artifact_sha256"]],
        on=["model_id", "method", "artifact_sha256"],
        how="inner"
    )

    # Prepare data for detectors
    feature_cols = [c for c in block_features.columns if c.startswith("f_")]
    X = block_features[feature_cols].values
    y = block_features["label"].values
    groups = (block_features["model_id"] + "::" + block_features["method"] + "::" + block_features["layer_id"].astype(str)).values
    model = block_features["model_id"].values
    tensor = block_features["layer_id"].astype(str).values  # using layer_id as tensor identifier
    group = block_features["model_id"] + "::" + block_features["method"] + "::" + block_features["layer_id"].astype(str)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": f"nes_detectability_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "features_dir": str(args.features_dir.resolve()),
        "features_file_sha256": hashlib.sha256(
            (args.features_dir / "residual_paired_block_features.csv").read_bytes()).hexdigest(),
        "n_examples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_groups": int(len(set(group))),
        "grouping": "GroupKFold by model::method::layer_id",
        "features": list(feature_cols),
        "versions": {m: importlib.metadata.version(m)
                     for m in ["numpy", "scikit-learn", "torch"]},
        "platform": platform.platform(),
    }
    write_json(args.output / "analysis_manifest.json", manifest)

    # --- Effect sizes: clean vs embedded, per model and per tensor ----------
    effects = []
    for m in sorted(set(model)):
        for t in sorted(set(tensor[model == m])):
            sel = (model == m) & (tensor == t)
            cl = X[sel & (y == 0)]
            em = X[sel & (y == 1)]
            for j, fname in enumerate(FEATURES):
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
        pooled[m] = {fname: cohen_d(X[sel & (y == 1)][:, j], X[sel & (y == 0)][:, j])
                     for j, fname in enumerate(FEATURES)}
    write_json(args.output / "effect_sizes_pooled.json", pooled)

    # --- Detectors with grouped CV -------------------------------------------
    results = {"detectors": {}}
    for det_name, fn in [("logistic_regression", lr_scores),
                          ("gradient_boosting", gb_scores)] + (
                         [] if args.no_mlp else [("mlp_exploratory", mlp_scores_factory())]):
        oof = grouped_cv(fn, X, y, group)
        auc = float(roc_auc_score(y, oof))
        lo, hi = bootstrap_ci(oof, y, group)
        t1, f1 = tpr_at_fpr(oof, y, 0.01)
        t5, f5 = tpr_at_fpr(oof, y, 0.05)
        obs, pval, nperm = permutation_test(oof, y, group)
        entry = {"auc": auc, "auc_ci95": [lo, hi],
                 "tpr_at_fpr_0.01": t1, "tpr_at_fpr_0.05": t5,
                 "permutation_p": pval, "permutation_n": nperm}
        for m in sorted(set(model)):
            sel = model == m
            entry[f"auc_model_{m}"] = float(roc_auc_score(y[sel], oof[sel]))
        results["detectors"][det_name] = entry

    # --- Control 1: clean vs clean (two disjoint block halves) ---------------
    cl_idx = np.where(y == 0)[0]
    ctrl1_X, ctrl1_y, ctrl1_g = [], [], []
    for g in sorted(set(group[cl_idx])):
        idx = cl_idx[group[cl_idx] == g]
        half = len(idx) // 2
        if half > 0:
            ctrl1_X.append(X[idx[:half]]); ctrl1_y.append(np.zeros(half))
            ctrl1_X.append(X[idx[half:2*half]]); ctrl1_y.append(np.ones(half))
            ctrl1_g.append(np.array([g] * half)); ctrl1_g.append(np.array([g] * half))
    Xc1 = np.concatenate(ctrl1_X); yc1 = np.concatenate(ctrl1_y); gc1 = np.concatenate(ctrl1_g)
    oof1 = grouped_cv(gb_scores, Xc1, yc1, gc1)
    results["control_clean_vs_clean"] = {"auc": float(roc_auc_score(yc1, oof1)),
                                         "expected": "~0.5"}

    # --- Control 2: embedded seed vs seed -----------------------------------
    em_idx = np.where(y == 1)[0]
    keep = np.isin(seed[em_idx], [20261016, 20261017])
    sel = em_idx[keep]
    Xc2 = X[sel]
    yc2 = (seed[sel] == 20261017).astype(int)
    gc2 = group[sel]
    oof2 = grouped_cv(gb_scores, Xc2, yc2, gc2)
    results["control_seed_vs_seed"] = {"auc": float(roc_auc_score(yc2, oof2)),
                                       "expected": "~0.5 (payload bits are the only difference)"}

    # --- Leave-one-model-out transfer (exploratory) ------
    transfer = {}
    for m in sorted(set(model)):
        tr = model != m
        te = model == m
        oof = gb_scores(X[tr], y[tr], X[te])
        transfer[m] = {"auc": float(roc_auc_score(y[te], oof)),
                       "note": "exploratory: single held-out checkpoint"}
    results["leave_one_model_out"] = transfer

    write_json(Path(args.output) / "detector_results.json", results)

    print(json.dumps({"status": "COMPLETED", "detectors": list(results["detectors"])}), flush=True)


if __name__ == "__main__":
    import argparse
    from datetime import timezone
    from pathlib import Path
    import hashlib
    import importlib.metadata

    # Add required imports
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())