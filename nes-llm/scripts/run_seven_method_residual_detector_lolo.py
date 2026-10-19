#!/usr/bin/env python3
"""Exploratory residual detectability benchmark with leave-one-layer-out evaluation.

This evaluates five residual methods on one TinyLlama run. It is not a
cross-model generalization benchmark. Existing input artifacts are read-only.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT.parent / "cache" / "detectability_tinyllama_residual_20261011" / "residual_paired_block_features.csv"
OUTPUT = ROOT.parent / "cache" / "detectability_tinyllama_detector_20261011"

METHODS = [
    "sign",
    "magnitude_aware",
    "qae",
    "lwe_grid_parity",
    "split_sign_parity",
]
FEATURES = [
    "f_mean", "f_std", "f_min", "f_max", "f_median", "f_q25", "f_q75",
    "f_mean_square", "f_positive_fraction", "f_skew",
    "f_excess_kurtosis", "f_hist_entropy", "f_zero_fraction", "f_abs_mean",
]
CLASSIFIERS = ("logistic_regression", "hist_gradient_boosting")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_classifier(name: str):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    raise ValueError(f"Unknown classifier: {name}")


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    # Fixed threshold, not selected from the held-out layer.
    predictions = (scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y_true, predictions)
        ),
    }


def main() -> int:
    start = time.time()

    if not INPUT.is_file():
        raise SystemExit(f"Input not found: {INPUT}")
    # Never silently overwrite a previous detector run.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(
            f"Output directory already contains files: {OUTPUT}\n"
            "Choose a new output directory before running again."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    required = {
        "model_id", "method", "layer_id", "block_index", "label", *FEATURES
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")

    observed_methods = set(df["method"].astype(str).unique())
    if observed_methods != set(METHODS):
        raise SystemExit(
            f"Method mismatch. Expected {sorted(METHODS)}, "
            f"found {sorted(observed_methods)}"
        )
    if set(df["label"].dropna().unique()) != {0, 1}:
        raise SystemExit("Expected binary labels exactly {0, 1}.")
    if df[["model_id"]].nunique().iloc[0] != 1:
        raise SystemExit("This script is for one model/run; multiple model IDs found.")
    if df[FEATURES].isna().any().any():
        raise SystemExit("Missing feature values found; refusing implicit imputation.")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite feature values found.")

    # Each method/layer/block must contain exactly one clean and one embedded row.
    pair_counts = df.groupby(
        ["method", "layer_id", "block_index"], sort=False
    )["label"].agg(rows="count", distinct_labels="nunique")
    valid_pairs = (pair_counts["rows"] == 2) & (
        pair_counts["distinct_labels"] == 2
    )
    if not valid_pairs.all():
        raise SystemExit(
            f"Pair integrity failed: {int((~valid_pairs).sum())} invalid groups."
        )

    layers = sorted(df["layer_id"].unique().tolist())
    if len(layers) < 3:
        raise SystemExit(f"Too few layer groups for evaluation: {layers}")

    fold_rows = []
    summary_rows = []
    prediction_frames = []
    logo = LeaveOneGroupOut()

    for method in METHODS:
        method_df = df.loc[df["method"] == method].copy().reset_index(drop=True)
        X = method_df[FEATURES].to_numpy(dtype=float)
        y = method_df["label"].to_numpy(dtype=int)
        groups = method_df["layer_id"].to_numpy()

        if len(np.unique(y)) != 2:
            raise SystemExit(f"{method}: both labels are required.")

        for classifier_name in CLASSIFIERS:
            oof_scores = np.full(len(method_df), np.nan, dtype=float)

            for fold_number, (train_idx, test_idx) in enumerate(
                logo.split(X, y, groups=groups), start=1
            ):
                train_layers = sorted(np.unique(groups[train_idx]).tolist())
                test_layers = sorted(np.unique(groups[test_idx]).tolist())
                if len(test_layers) != 1:
                    raise AssertionError("LOLO fold must hold out exactly one layer.")
                if set(train_layers) & set(test_layers):
                    raise AssertionError("Layer leakage detected.")

                model = make_classifier(classifier_name)
                model.fit(X[train_idx], y[train_idx])
                scores = model.predict_proba(X[test_idx])[:, 1]
                oof_scores[test_idx] = scores

                fold_metric = metrics(y[test_idx], scores)
                fold_rows.append({
                    "method": method,
                    "classifier": classifier_name,
                    "fold": fold_number,
                    "held_out_layer": int(test_layers[0]),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "test_clean_rows": int((y[test_idx] == 0).sum()),
                    "test_embedded_rows": int((y[test_idx] == 1).sum()),
                    **fold_metric,
                })

            if not np.isfinite(oof_scores).all():
                raise AssertionError("Some rows lack out-of-fold predictions.")

            pooled = metrics(y, oof_scores)
            prediction_frames.append(pd.DataFrame({
                "method": method,
                "classifier": classifier_name,
                "layer_id": method_df["layer_id"].to_numpy(),
                "block_index": method_df["block_index"].to_numpy(),
                "label": y,
                "score_embedded": oof_scores,
                "prediction_at_0_5": (oof_scores >= 0.5).astype(int),
            }))

            method_folds = [
                row for row in fold_rows
                if row["method"] == method
                and row["classifier"] == classifier_name
            ]
            summary_rows.append({
                "method": method,
                "classifier": classifier_name,
                "layers_evaluated": len(layers),
                "pooled_oof_roc_auc": pooled["roc_auc"],
                "pooled_oof_average_precision": pooled["average_precision"],
                "pooled_oof_balanced_accuracy_at_0_5":
                    pooled["balanced_accuracy_at_0_5"],
                "mean_layer_roc_auc": float(np.mean(
                    [r["roc_auc"] for r in method_folds]
                )),
                "std_layer_roc_auc": float(np.std(
                    [r["roc_auc"] for r in method_folds], ddof=1
                )),
                "mean_layer_average_precision": float(np.mean(
                    [r["average_precision"] for r in method_folds]
                )),
                "mean_layer_balanced_accuracy_at_0_5": float(np.mean(
                    [r["balanced_accuracy_at_0_5"] for r in method_folds]
                )),
            })
            print(
                f"{method:20s} | {classifier_name:24s} | "
                f"pooled OOF ROC-AUC={pooled['roc_auc']:.4f} | "
                f"AP={pooled['average_precision']:.4f}"
            )

    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    folds_path = OUTPUT / "detector_fold_metrics.csv"
    summary_path = OUTPUT / "detector_summary.csv"
    predictions_path = OUTPUT / "oof_predictions.csv"
    report_path = OUTPUT / "detectability_stage2_report.json"

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    report = {
        "schema": "nes.seven_method_detectability_stage2_lolo.v1",
        "purpose": (
            "Exploratory within-model residual detectability using "
            "leave-one-layer-out out-of-fold predictions."
        ),
        "input_path": str(INPUT),
        "input_sha256": sha256_file(INPUT),
        "output_dir": str(OUTPUT),
        "model_ids": sorted(df["model_id"].astype(str).unique().tolist()),
        "methods": METHODS,
        "feature_columns": FEATURES,
        "classifiers": list(CLASSIFIERS),
        "split_design": (
            "LeaveOneGroupOut grouped by layer_id, independently for each method; "
            "all clean and embedded blocks from a layer are held out together."
        ),
        "decision_threshold": 0.5,
        "threshold_note": "Fixed in advance; not tuned on held-out layers.",
        "integrity": {
            "input_rows": int(len(df)),
            "layer_count": int(len(layers)),
            "layers": [int(layer) for layer in layers],
            "pair_groups": int(len(pair_counts)),
            "valid_clean_embedded_pairs": int(valid_pairs.sum()),
            "invalid_pair_groups": int((~valid_pairs).sum()),
            "missing_feature_values": int(df[FEATURES].isna().sum().sum()),
        },
        "interpretation_guardrails": [
            "One TinyLlama model/run only; no cross-model generalization claim.",
            "Blocks within tensors are correlated; rows are not independent experimental replications.",
            "Layer folds are the held-out units; report fold variability alongside pooled OOF metrics.",
            "Pooled OOF metrics summarize predictions from different fold-trained models.",
            "Near-chance performance does not prove undetectability or universal stealth.",
            "Feature-based results cover only these 14 residual summary features and these two classifiers.",
            "No threshold is selected on the held-out layer.",
        ],
        "known_method_equivalence": {
            "methods": ["sign", "qae"],
            "finding": (
                "Prior audit found exactly equal embedded residual tensors and "
                "carrier positions across the 11 selected layers in this run."
            ),
            "explanation": (
                "Configuration-specific: the QAE margin was below the sign "
                "strategy's magnitude floor for all carriers, so both wrote the "
                "same values in this carrier distribution."
            ),
            "consequence": (
                "Report both named methods for traceability, but do not count "
                "their identical outcomes here as independent embedding evidence."
            ),
        },
        "outputs": {
            "fold_metrics_csv": str(folds_path),
            "summary_csv": str(summary_path),
            "oof_predictions_csv": str(predictions_path),
        },
        "software": {"scikit_learn": sklearn.__version__},
        "elapsed_seconds": round(time.time() - start, 3),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nReport:", report_path)
    print("Fold metrics:", folds_path)
    print("OOF predictions:", predictions_path)
    print("Elapsed seconds:", report["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
