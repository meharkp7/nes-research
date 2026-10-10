#!/usr/bin/env python3
"""Run a focused packed-NF4 detector ablation without modifying prior reports.

Conditions: Qwen-only Random Forest, Qwen entropy-only Random Forest,
Qwen full-feature Random Forest without entropy, plus paired-delta distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_packed_nf4_detector_benchmark import (  # noqa: E402
    GROUP_COLS, PAIR_COLS, SPLITS, make_mixed_split, metric_pack, sha256_file,
    validate_dataset,
)

QWEN = "Qwen/Qwen2.5-3B"
LABEL = "label"
ENTROPY = "entropy_bits"
QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


def rf(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
    )


def evaluate_condition(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                       features: list[str], seed: int) -> dict:
    """Fit one condition; choose threshold on validation and touch test once."""
    if not features:
        raise ValueError("A condition must have at least one feature")
    model = rf(seed)
    model.fit(train[features].to_numpy(float), train[LABEL].to_numpy(int))
    val_y = validation[LABEL].to_numpy(int)
    val_scores = model.predict_proba(validation[features].to_numpy(float))[:, 1]
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], val_scores)))
    threshold = max(
        thresholds,
        key=lambda t: (
            np.mean(val_scores[val_y == 1] >= t) - np.mean(val_scores[val_y == 0] >= t),
            -abs(float(t) - 0.5),
        ),
    )
    test_scores = model.predict_proba(test[features].to_numpy(float))[:, 1]
    scored = test[[LABEL, "model_id", *GROUP_COLS]].copy()
    scored["score"] = test_scores
    by_run = {}
    for (source_id, run_id), sub in scored.groupby(list(GROUP_COLS), sort=True):
        # Some individual runs may not contain both labels; AUC is then explicitly null.
        by_run[f"{source_id}::{run_id}"] = metric_pack(
            sub[LABEL].to_numpy(int), sub["score"].to_numpy(float), float(threshold)
        )
    return {
        "features": features,
        "n_features": len(features),
        "validation": metric_pack(val_y, val_scores, float(threshold)),
        "test_pooled": metric_pack(test[LABEL].to_numpy(int), test_scores, float(threshold)),
        "test_by_source_run": by_run,
        "validation_selected_threshold": float(threshold),
        "warning": "Qwen-only random pair split can still leak source/run-specific structure across partitions.",
    }


def qwen_grouped_sensitivity(qwen: pd.DataFrame, features: list[str], seed: int) -> dict:
    groups = qwen[list(GROUP_COLS)].astype(str).agg("::".join, axis=1).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        return {
            "status": "NOT_EVALUABLE",
            "group_count": n_groups,
            "reason": "At least 3 independent Qwen source/run groups are required for GroupKFold sensitivity.",
        }
    splitter = GroupKFold(n_splits=min(4, n_groups))
    y = qwen[LABEL].to_numpy(int)
    scores = np.full(len(qwen), np.nan)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(qwen, y, groups=groups)):
        train, test = qwen.iloc[train_idx], qwen.iloc[test_idx]
        if train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
            return {"status": "NOT_EVALUABLE", "group_count": n_groups,
                    "reason": f"Fold {fold} did not contain both labels."}
        model = rf(seed + fold)
        model.fit(train[features].to_numpy(float), train[LABEL].to_numpy(int))
        scores[test_idx] = model.predict_proba(test[features].to_numpy(float))[:, 1]
        folds.append({
            "fold": fold,
            "train_groups": sorted(set(groups[train_idx])),
            "held_out_groups": sorted(set(groups[test_idx])),
            "metrics": metric_pack(y[test_idx], scores[test_idx], 0.5),
        })
    return {
        "status": "COMPLETED",
        "method": "Qwen-only GroupKFold out-of-fold sensitivity",
        "group_count": n_groups,
        "fold_count": len(folds),
        "oof_pooled": metric_pack(y, scores, 0.5),
        "folds": folds,
        "warning": "Small group counts make these sensitivity results unstable; not a confirmatory estimate.",
    }


def paired_delta_distribution(df: pd.DataFrame, features: list[str]) -> dict:
    by_model = {}
    for model_id, subset in df.groupby("model_id", sort=True):
        feature_results = {}
        for feature in features:
            pairs = subset.pivot_table(
                index=list(PAIR_COLS), columns=LABEL, values=feature, aggfunc="first"
            )
            if 0 not in pairs.columns or 1 not in pairs.columns:
                continue
            delta = (pairs[1] - pairs[0]).dropna().to_numpy(float)
            qs = np.quantile(delta, QUANTILES) if len(delta) else np.array([])
            feature_results[feature] = {
                "matched_pairs": int(len(delta)),
                "nonzero_fraction": float(np.mean(delta != 0.0)) if len(delta) else None,
                "zero_fraction": float(np.mean(delta == 0.0)) if len(delta) else None,
                "mean": float(np.mean(delta)) if len(delta) else None,
                "std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, qs)},
            }
        by_model[str(model_id)] = feature_results
    return {
        "by_model": by_model,
        "interpretation": (
            "Descriptive matched-pair embedded-minus-clean distributions. Quantiles and nonzero "
            "fractions reveal sparse changes hidden by zero medians; no p-values or causal claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    df = pd.read_csv(dataset)
    features = validate_dataset(df)
    if ENTROPY not in features:
        raise ValueError(f"Required ablation feature {ENTROPY!r} is absent")
    qwen_all = df[df["model_id"].astype(str) == QWEN].copy()
    if qwen_all.empty:
        raise ValueError(f"Dataset contains no rows for required model {QWEN!r}")

    split = make_mixed_split(qwen_all, args.seed)
    parts = {name: split[split["split"] == name].copy() for name in SPLITS}
    for name, part in parts.items():
        if part.empty or set(part[LABEL].astype(int)) != {0, 1}:
            raise ValueError(f"Qwen-only {name} partition must contain both labels")
    conditions = {
        "qwen_only_random_forest_full_features": features,
        "qwen_only_entropy_only": [ENTROPY],
        "qwen_only_full_features_without_entropy": [f for f in features if f != ENTROPY],
    }
    results = {}
    for condition, columns in conditions.items():
        results[condition] = evaluate_condition(
            parts["train"], parts["validation"], parts["test"], columns, args.seed
        )

    # Grouped sensitivity on the full Qwen subset, plus distribution diagnostics over
    # matched pairs. These are diagnostic complements, not substitutes for the held-out test.
    grouped = qwen_grouped_sensitivity(qwen_all, features, args.seed)
    deltas = paired_delta_distribution(df, features)
    report = {
        "schema": "nes.packed_nf4_detector_ablation.v1",
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "seed": args.seed,
        "qwen_model_id": QWEN,
        "qwen_rows": int(len(qwen_all)),
        "qwen_matched_pairs": int(qwen_all[list(PAIR_COLS)].drop_duplicates().shape[0]),
        "features": features,
        "split": {
            "rows": {name: int(len(part)) for name, part in parts.items()},
            "pairs": {name: int(part[list(PAIR_COLS)].drop_duplicates().shape[0]) for name, part in parts.items()},
            "rule": "Matched clean/embedded pairs split together; 60/20/20 random pair split within Qwen.",
            "warning": "Source/run groups may cross partitions; use grouped sensitivity as a leakage diagnostic.",
        },
        "conditions": results,
        "qwen_grouped_sensitivity": grouped,
        "paired_delta_distributions": deltas,
        "interpretation_guardrails": [
            "Qwen-only random splitting tests within-model discrimination but does not guarantee source/run independence.",
            "Entropy-only and without-entropy conditions are ablations, not causal proofs.",
            "Permutation importance and feature ablation can be affected by correlated features.",
            "Do not interpret pooled block counts as independent sample sizes.",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "packed_nf4_detector_ablation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "qwen_rows": report["qwen_rows"],
        "qwen_matched_pairs": report["qwen_matched_pairs"],
        "split_rows": report["split"]["rows"],
        "test_roc_auc": {
            name: value["test_pooled"]["roc_auc"] for name, value in results.items()
        },
        "grouped_sensitivity_status": grouped["status"],
        "warning": "Exploratory; Qwen random split may retain source/run overlap.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
