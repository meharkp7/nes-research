#!/usr/bin/env python3
"""Evaluate QSE and DCE detectors separately on held-out model families."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from packed_nf4_method_eval_utils import (
    infer_method, validate_chunk, validate_manifest,
)

META = {
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index", "split",
}
METHODS = ("qse", "dce")
SPLITS = ("train", "validation", "test")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_rows": int(len(y)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "threshold": float(threshold),
    }


def select_threshold(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    j = tpr - fpr
    best = np.flatnonzero(j == j.max())
    idx = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[idx])


def build_models(seed):
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1500, class_weight="balanced",
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed,
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--chunksize", type=int, default=100000)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    split_path = args.splits.expanduser().resolve()
    readiness_path = args.readiness.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    for path in (dataset, split_path, readiness_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive")

    manifest = json.loads(split_path.read_text())
    readiness = json.loads(readiness_path.read_text())
    if readiness.get("status") != "PASS":
        raise ValueError("Readiness report must have status PASS")
    group_counts = validate_manifest(manifest)
    features = manifest.get("feature_columns", [])
    if not features or set(features) & META:
        raise ValueError("Invalid feature_columns in frozen split manifest")

    usecols = sorted(META | set(features))
    dtype = {feature: "float32" for feature in features}
    dtype.update({"label": "int8", "block_index": "int32"})

    x_parts = {(m, s): [] for m in METHODS for s in SPLITS}
    y_parts = {(m, s): [] for m in METHODS for s in SPLITS}
    model_parts = {(m, s): [] for m in METHODS for s in SPLITS}
    row_counts = {
        (m, s, role): 0
        for m in METHODS for s in SPLITS for role in ("clean", "embedded")
    }
    observed_models = {s: set() for s in SPLITS}

    for chunk in pd.read_csv(
        dataset, usecols=usecols, dtype=dtype, chunksize=args.chunksize
    ):
        methods = validate_chunk(chunk, manifest)
        for split in SPLITS:
            observed_models[split].update(
                chunk.loc[chunk["split"] == split, "model_id"].unique()
            )
        for method in METHODS:
            method_mask = methods.eq(method)
            for split in SPLITS:
                mask = method_mask & chunk["split"].eq(split)
                if not mask.any():
                    continue
                part = chunk.loc[mask]
                x_parts[(method, split)].append(
                    part[features].to_numpy(dtype=np.float32, copy=True)
                )
                y_parts[(method, split)].append(
                    part["label"].to_numpy(dtype=np.int8, copy=True)
                )
                model_parts[(method, split)].append(
                    part["model_id"].to_numpy(dtype=str, copy=True)
                )
                for role in ("clean", "embedded"):
                    row_counts[(method, split, role)] += int(
                        part["role"].eq(role).sum()
                    )

    expected_models = manifest["group_to_split"]
    for split in SPLITS:
        actual = observed_models[split]
        expected = {m for m, s in expected_models.items() if s == split}
        if actual != expected:
            raise ValueError(f"Observed model groups mismatch for {split}")
    if any(v == 0 for v in row_counts.values()):
        raise ValueError("A method/split/role partition is empty")
    for method in METHODS:
        for split in SPLITS:
            if row_counts[(method, split, "clean")] != row_counts[
                (method, split, "embedded")
            ]:
                raise ValueError(f"Unbalanced labels for {method}/{split}")

    report = {
        "schema": "nes.packed_nf4_method_specific_detector.v1",
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "readiness_report": str(readiness_path),
        "readiness_status": readiness["status"],
        "seed": args.seed,
        "feature_columns": features,
        "metadata_excluded_from_features": sorted(META),
        "model_groups_per_split": group_counts,
        "models_by_split": {
            s: sorted(observed_models[s]) for s in SPLITS
        },
        "row_counts": {
            f"{m}/{s}/{r}": n for (m, s, r), n in row_counts.items()
        },
        "methods": {},
        "interpretation": {
            "design": "QSE and DCE are trained and evaluated independently.",
            "threshold": "Youden's J selected on validation only; test evaluated once.",
            "primary_metrics": ["roc_auc", "pr_auc"],
            "dependence": (
                "Block-level metrics are descriptive; blocks within an artifact "
                "are correlated. No independent-block confidence intervals."
            ),
            "generalization": (
                "Only two model families are held out; broad model-family "
                "generalization is not established."
            ),
            "stealth": "Detector metrics do not establish stealth or undetectability.",
            "controls": (
                "Clean controls have historical_qse_revision_match UNVERIFIED; "
                "this evaluator does not resolve that provenance limitation."
            ),
        },
    }

    for method in METHODS:
        arrays = {}
        for split in SPLITS:
            key = (method, split)
            arrays[split] = {
                "X": np.concatenate(x_parts[key]),
                "y": np.concatenate(y_parts[key]).astype(int),
                "models": np.concatenate(model_parts[key]),
            }
        train, val, test = (arrays[s] for s in SPLITS)
        if any(set(a["y"].tolist()) != {0, 1} for a in (train, val, test)):
            raise ValueError(f"Both classes required in every {method} split")

        report["methods"][method] = {"models": {}}
        for name, estimator in build_models(args.seed).items():
            estimator.fit(train["X"], train["y"])
            val_scores = estimator.predict_proba(val["X"])[:, 1]
            threshold = select_threshold(val["y"], val_scores)
            test_scores = estimator.predict_proba(test["X"])[:, 1]

            per_model = {}
            for model_id in sorted(np.unique(test["models"])):
                mask = test["models"] == model_id
                per_model[model_id] = metrics(
                    test["y"][mask], test_scores[mask], threshold
                )
            report["methods"][method]["models"][name] = {
                "validation": metrics(val["y"], val_scores, threshold),
                "test_pooled": metrics(test["y"], test_scores, threshold),
                "test_by_model": per_model,
                "selected_threshold": threshold,
            }
            print(json.dumps({
                "method": method,
                "model": name,
                "test_roc_auc": report["methods"][method]["models"][name][
                    "test_pooled"
                ]["roc_auc"],
                "test_pr_auc": report["methods"][method]["models"][name][
                    "test_pooled"
                ]["pr_auc"],
                "threshold": threshold,
            }), flush=True)

        del arrays

    out.mkdir(parents=True, exist_ok=False)
    destination = out / "method_specific_detector_report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPLETED",
        "report": str(destination),
        "methods": list(report["methods"]),
        "models_per_method": 2,
        "warning": report["interpretation"]["generalization"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
