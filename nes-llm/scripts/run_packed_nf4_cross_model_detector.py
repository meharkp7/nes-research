#!/usr/bin/env python3
"""Leave-one-model-family-out packed-NF4 detector evaluation.

Trains on two model families and evaluates on the third, rotating the held-out
family. This is a direct cross-model transfer test; it does not use held-out
model rows for fitting or threshold selection. Results remain exploratory when
the dataset has few independent source/run groups.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_packed_nf4_detector_benchmark import (
    GROUP_COLS, META, metric_pack, sha256_file, validate_dataset,
)


def cross_model_results(df: pd.DataFrame, features: list[str], seed: int) -> dict:
    models = sorted(df["model_id"].astype(str).unique())
    if len(models) < 3:
        return {"status": "NOT_EVALUABLE", "reason": f"Need at least 3 model families; found {len(models)}."}
    estimators = {
        "logistic_regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced", random_state=seed),
        ),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                   class_weight="balanced", n_jobs=-1, random_state=seed),
        ),
        "hist_gradient_boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08,
                                           max_leaf_nodes=15, l2_regularization=1.0,
                                           random_state=seed),
        ),
    }
    outputs = {}
    for held_out in models:
        train = df[df["model_id"].astype(str) != held_out]
        test = df[df["model_id"].astype(str) == held_out]
        if train["label"].nunique() != 2 or test["label"].nunique() != 2:
            outputs[held_out] = {"status": "NOT_EVALUABLE", "reason": "Training or held-out model lacks both labels."}
            continue
        # Matched pairs must remain intact; validation is not used to select a
        # threshold because the headline transfer metric is threshold-free AUC.
        pair_cols = ["source_id", "run_id", "block_index"]
        if train.duplicated(pair_cols + ["label"]).any() or test.duplicated(pair_cols + ["label"]).any():
            outputs[held_out] = {"status": "NOT_EVALUABLE", "reason": "Duplicate pair/label rows detected."}
            continue
        train_groups = train[list(GROUP_COLS)].astype(str).agg("::".join, axis=1).nunique()
        test_groups = test[list(GROUP_COLS)].astype(str).agg("::".join, axis=1).nunique()
        per_estimator = {}
        y_train = train["label"].to_numpy(int)
        y_test = test["label"].to_numpy(int)
        for name, factory in estimators.items():
            model = factory()
            model.fit(train[features].to_numpy(float), y_train)
            scores = model.predict_proba(test[features].to_numpy(float))[:, 1]
            per_estimator[name] = {
                "metrics_at_fixed_threshold_0_5": metric_pack(y_test, scores, 0.5),
                "roc_auc": float(roc_auc_score(y_test, scores)),
            }
        outputs[held_out] = {
            "status": "COMPLETED",
            "train_model_families": [m for m in models if m != held_out],
            "held_out_model_family": held_out,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_source_run_groups": int(train_groups),
            "test_source_run_groups": int(test_groups),
            "threshold_policy": "Fixed score threshold 0.5; no held-out-model threshold tuning. ROC-AUC is the primary transfer metric.",
            "estimators": per_estimator,
        }
    completed = [v for v in outputs.values() if v.get("status") == "COMPLETED"]
    return {
        "status": "COMPLETED" if len(completed) == len(models) else "PARTIAL",
        "method": "Leave-one-model-family-out: train on all rows from the other model families; evaluate only on the held-out family.",
        "model_count": len(models),
        "results_by_held_out_model": outputs,
        "warning": "Model transfer is tested, but source/run counts are small and the dataset's provenance determines whether clean/embedded pairs differ only by embedding.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    data_path = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    df = pd.read_csv(data_path)
    features = validate_dataset(df)
    result = {
        "schema": "nes.packed_nf4_cross_model_detector.v1",
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "rows": int(len(df)),
        "feature_columns": features,
        "metadata_excluded_from_features": sorted(META),
        "seed": args.seed,
        "evaluation": cross_model_results(df, features, args.seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["evaluation"]["status"],
        "report": str(output),
        "held_out_models": list(result["evaluation"].get("results_by_held_out_model", {})),
        "primary_metric": "ROC-AUC at the model-family level; threshold fixed at 0.5 for secondary metrics",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
