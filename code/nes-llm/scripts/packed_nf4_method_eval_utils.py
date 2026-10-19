"""Validation helpers for method-specific packed-NF4 detector experiments."""
import pandas as pd

REQUIRED = {"run_id", "model_id", "role", "label", "split"}
SPLITS = {"train", "validation", "test"}


def validate_manifest(manifest):
    assignments = manifest.get("group_to_split")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Missing group_to_split mapping")
    if set(assignments.values()) - SPLITS:
        raise ValueError("Manifest contains an unknown split")
    counts = {s: sum(v == s for v in assignments.values()) for s in SPLITS}
    if counts["train"] < 2 or counts["validation"] < 1 or counts["test"] < 2:
        raise ValueError(f"Insufficient model groups per split: {counts}")
    for split in SPLITS:
        if not any(v == split for v in assignments.values()):
            raise ValueError(f"Empty model split: {split}")
    return counts


def infer_method(run_ids):
    method = run_ids.astype(str).str.extract(r"_(qse|dce)_20261011$", expand=False)
    if method.isna().any():
        raise ValueError("Unrecognized method/run_id suffix")
    return method


def validate_chunk(chunk, manifest):
    missing = REQUIRED - set(chunk.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if chunk.empty:
        return infer_method(chunk["run_id"])

    assignments = manifest["group_to_split"]
    if not chunk["model_id"].isin(assignments).all():
        raise ValueError("Dataset contains a model absent from the frozen manifest")
    expected_split = chunk["model_id"].map(assignments)
    if not expected_split.eq(chunk["split"]).all():
        raise ValueError("Dataset split disagrees with frozen manifest")

    expected_label = chunk["role"].map({"clean": 0, "embedded": 1})
    if expected_label.isna().any() or not expected_label.eq(
        pd.to_numeric(chunk["label"], errors="coerce")
    ).all():
        raise ValueError("Role/label mismatch or unknown role")

    return infer_method(chunk["run_id"])
