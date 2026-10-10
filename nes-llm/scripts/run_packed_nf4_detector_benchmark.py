#!/usr/bin/env python3
"""Run one consolidated mixed-model packed-NF4 detector benchmark.

Primary: random split of matched clean/embedded block pairs, stratified by model.
Sensitivity: hold out complete source/run groups to expose artifact/run leakage.
This is an exploratory block-feature benchmark, not proof of universal stealth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = {
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index", "split",
}
SPLITS = ("train", "validation", "test")
PAIR_COLS = ("source_id", "run_id", "block_index")
GROUP_COLS = ("source_id", "run_id")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(df: pd.DataFrame) -> list[str]:
    required = META | set(PAIR_COLS) | set(GROUP_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    role_label = df["role"].map({"clean": 0, "embedded": 1})
    labels = pd.to_numeric(df["label"], errors="coerce")
    if role_label.isna().any() or labels.isna().any() or not np.array_equal(role_label.to_numpy(), labels.astype(int).to_numpy()):
        raise ValueError("Role/label mismatch or unknown role")
    features = [c for c in df.columns if c not in META]
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if not features or bad:
        raise ValueError(f"Features must be numeric and non-empty; invalid: {bad}")
    if df[["model_id", *PAIR_COLS]].isna().any().any():
        raise ValueError("Model and pair identifiers may not be null")
    pair = df.groupby(list(PAIR_COLS), dropna=False)
    bad_pairs = []
    for key, g in pair:
        if len(g) != 2 or set(g["label"].astype(int)) != {0, 1} or g["model_id"].nunique() != 1:
            bad_pairs.append(str(key))
            if len(bad_pairs) >= 5:
                break
    if bad_pairs:
        raise ValueError(f"Expected exactly one clean and one embedded row per matched pair; examples: {bad_pairs}")
    return features


def make_mixed_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign matched pairs to partitions; balance each model across partitions."""
    pairs = df[list(PAIR_COLS + ("model_id",))].drop_duplicates()
    if pairs.duplicated(list(PAIR_COLS)).any():
        raise ValueError("A matched pair maps to multiple model IDs")
    pairs["pair_key"] = pairs[list(PAIR_COLS)].astype(str).agg("::".join, axis=1)
    pairs["split"] = ""
    rng = np.random.RandomState(seed)
    for model, sub in pairs.groupby("model_id", sort=True):
        keys = sub["pair_key"].to_numpy().copy()
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            raise ValueError(f"Model {model!r} has only {n} pairs; need at least 3 for three partitions")
        n_train = max(1, int(round(n * 0.60)))
        n_val = max(1, int(round(n * 0.20)))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        assigned = ["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val)
        for key, split in zip(keys, assigned):
            pairs.loc[pairs["pair_key"] == key, "split"] = split
    lookup = pairs.set_index("pair_key")["split"].to_dict()
    row_keys = df[list(PAIR_COLS)].astype(str).agg("::".join, axis=1)
    result = df.copy()
    result["split"] = row_keys.map(lookup)
    if result["split"].isna().any():
        raise ValueError("Failed to assign split to one or more rows")
    return result


def metric_pack(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_rows": int(len(y)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "threshold": float(threshold),
    }


def model_zoo(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced", random_state=seed),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced",
                                   n_jobs=-1, random_state=seed),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=seed),
        ),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=120,
                          batch_size=512, random_state=seed),
        ),
    }


def evaluate_model(model, train, validation, test, features) -> dict:
    Xtr, ytr = train[features].to_numpy(float), train["label"].to_numpy(int)
    Xv, yv = validation[features].to_numpy(float), validation["label"].to_numpy(int)
    Xt, yt = test[features].to_numpy(float), test["label"].to_numpy(int)
    model.fit(Xtr, ytr)
    vscore = model.predict_proba(Xv)[:, 1]
    # Threshold chosen only on validation by Youden's J; test remains untouched.
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], vscore)))
    threshold = max(thresholds, key=lambda t: (
        np.mean(vscore[yv == 1] >= t) - np.mean(vscore[yv == 0] >= t),
        -abs(float(t) - 0.5),
    ))
    tscore = model.predict_proba(Xt)[:, 1]
    by_model = {}
    test_with_scores = test.copy()
    test_with_scores["score"] = tscore
    for model_id, sub in test_with_scores.groupby("model_id", sort=True):
        by_model[str(model_id)] = metric_pack(sub["label"].to_numpy(int), sub["score"].to_numpy(float), threshold)
    return {
        "validation": metric_pack(yv, vscore, threshold),
        "test_pooled": metric_pack(yt, tscore, threshold),
        "test_by_model": by_model,
        "validation_selected_threshold": float(threshold),
    }


def grouped_sensitivity(df, features, seed) -> dict:
    """Hold out complete source/run groups; report as sensitivity, not primary score."""
    group_ids = df[list(GROUP_COLS)].astype(str).agg("::".join, axis=1).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=group_ids))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    train_groups = set(group_ids[train_idx])
    # Keep model selection fixed to the primary benchmark's family; no hyperparameter search.
    results = {"held_out_groups": sorted(set(group_ids[test_idx])),
               "train_group_count": len(train_groups), "test_group_count": len(set(group_ids[test_idx])),
               "models": {}, "warning": "Small number of source/run groups; exploratory sensitivity only."}
    for name, model in model_zoo(seed).items():
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            results["status"] = "NOT_EVALUABLE"
            results["reason"] = "Grouped split did not contain both labels in train and test."
            return results
        fitted = clone(model).fit(train[features].to_numpy(float), train["label"].to_numpy(int))
        scores = fitted.predict_proba(test[features].to_numpy(float))[:, 1]
        results["models"][name] = metric_pack(test["label"].to_numpy(int), scores, 0.5)
    results["status"] = "COMPLETED"
    return results


def label_randomization(df, features, seed, repetitions=5) -> dict:
    """Sanity control: shuffle labels at matched-pair level and rerun a light baseline."""
    pair_keys = df[list(PAIR_COLS)].astype(str).agg("::".join, axis=1)
    pair_table = df[[*PAIR_COLS, "split"]].drop_duplicates()
    pair_table["pair_key"] = pair_table[list(PAIR_COLS)].astype(str).agg("::".join, axis=1)
    base = df.copy()
    base["pair_key"] = pair_keys
    train = base[base["split"] == "train"].copy()
    test = base[base["split"] == "test"].copy()
    out = []
    for i in range(repetitions):
        rng = np.random.RandomState(seed + 1000 + i)
        labels_by_pair = pair_table.set_index("pair_key")["split"].to_dict()
        # Permute pair labels only within each partition, so both rows of a pair remain together.
        shuffled = {}
        for split in SPLITS:
            keys = pair_table.loc[pair_table["split"] == split, "pair_key"].to_numpy().copy()
            vals = np.array([int(base.loc[base["pair_key"] == k, "label"].iloc[0]) for k in keys])
            rng.shuffle(vals)
            shuffled.update(dict(zip(keys, vals)))
        train_y = train["pair_key"].map(shuffled).to_numpy(int)
        test_y = test["pair_key"].map(shuffled).to_numpy(int)
        # If a tiny split loses a class after shuffling, report rather than crash.
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            out.append({"replicate": i, "status": "NOT_EVALUABLE", "reason": "Shuffled labels lack both classes"})
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed+i))
        model.fit(train[features].to_numpy(float), train_y)
        scores = model.predict_proba(test[features].to_numpy(float))[:, 1]
        out.append({"replicate": i, "status": "COMPLETED",
                    "metrics": metric_pack(test_y, scores, 0.5)})
    valid = [x["metrics"]["roc_auc"] for x in out if x["status"] == "COMPLETED"]
    return {"repetitions": repetitions, "results": out,
            "mean_roc_auc": float(np.mean(valid)) if valid else None,
            "interpretation": "Pipeline sanity control only; not a biological/stealth control."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()
    data_path, out = args.dataset.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    df = pd.read_csv(data_path)
    features = validate_dataset(df)
    mixed = make_mixed_split(df, args.seed)
    parts = {s: mixed[mixed["split"] == s].copy() for s in SPLITS}
    for split, part in parts.items():
        if part.empty or set(part["label"].astype(int).unique()) != {0, 1}:
            raise ValueError(f"{split} must contain both labels")
        if set(part["model_id"].unique()) != set(df["model_id"].unique()):
            raise ValueError(f"{split} does not contain every model family")
    # Persist a new derived split manifest; the original dataset and pilot remain untouched.
    manifest = {
        "schema": "nes.packed_nf4_mixed_model_split.v1",
        "seed": args.seed,
        "dataset_sha256": sha256_file(data_path),
        "pair_columns": list(PAIR_COLS),
        "group_columns": list(GROUP_COLS),
        "split_rows": {s: int(len(parts[s])) for s in SPLITS},
        "split_pair_counts": {s: int(parts[s][list(PAIR_COLS)].drop_duplicates().shape[0]) for s in SPLITS},
        "models_by_split": {s: sorted(parts[s]["model_id"].astype(str).unique().tolist()) for s in SPLITS},
        "split_rule": "Matched clean/embedded block pairs assigned together; pair groups stratified within model. Source/run groups may cross partitions in primary mixed-block evaluation.",
        "leakage_note": "Primary mixed-block result may be optimistic because blocks from the same source/run may occur in multiple partitions. See grouped sensitivity results.",
    }
    zoo = model_zoo(args.seed)
    model_results = {}
    for name, model in zoo.items():
        model_results[name] = evaluate_model(model, parts["train"], parts["validation"], parts["test"], features)
    # One label-randomization sanity control, plus one grouped sensitivity evaluation.
    randomized = label_randomization(mixed, features, args.seed, max(1, args.permutations))
    grouped = grouped_sensitivity(mixed, features, args.seed)
    report = {
        "schema": "nes.packed_nf4_detector_benchmark.v1",
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "rows": int(len(df)),
        "feature_columns": features,
        "metadata_excluded_from_features": sorted(META),
        "models": sorted(df["model_id"].astype(str).unique().tolist()),
        "split_manifest": manifest,
        "primary_mixed_block_evaluation": {
            "models": model_results,
            "metrics_note": "Validation selects threshold by Youden's J; test is evaluated once at that threshold. Pooled row metrics are descriptive because rows are correlated.",
        },
        "grouped_leakage_sensitivity": grouped,
        "label_randomization_control": randomized,
        "clean_vs_clean_control": {
            "status": "NOT_AVAILABLE_IN_CURRENT_DATASET",
            "reason": "The supplied dataset contains clean and embedded labels, but no independent clean-vs-clean artifact class with suitable provenance. Do not substitute duplicate clean files as independent negatives.",
        },
        "uncertainty": {
            "status": "NOT_ESTIMATED",
            "reason": "The available independent source/run group count is small; naive row bootstrap would overstate effective sample size. Reported row metrics are descriptive.",
        },
        "interpretation": {
            "scope": "Packed-NF4 block-feature detectability on this dataset and split.",
            "not_established": ["universal stealth", "undetectability", "generalization to unseen model families", "independent-run statistical significance"],
            "next_scientific_constraint": "Collect more independent source/run artifacts and valid clean-vs-clean controls before confirmatory claims.",
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "mixed_model_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = out / "packed_nf4_detector_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "report": str(report_path), "manifest": str(manifest_path),
        "split_rows": manifest["split_rows"], "models_by_split": manifest["models_by_split"],
        "primary_test_roc_auc": {k: v["test_pooled"]["roc_auc"] for k, v in model_results.items()},
        "grouped_sensitivity_status": grouped.get("status"),
        "label_randomization_mean_auc": randomized.get("mean_roc_auc"),
        "warning": "Mixed-block results are exploratory and may be optimistic due to source/run overlap.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
