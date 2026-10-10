#!/usr/bin/env python3
"""Audit packed-NF4 detector data before any model fitting.

This is an evaluation gate, not a detector. It checks artifact identity,
clean/embedded pairing, split-manifest agreement, split class coverage, and
the number of independent model groups available per partition. Block rows
are never treated as independent evaluation samples.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED = {
    "artifact_sha256", "run_id", "source_id", "model_id", "tensor_key",
    "role", "label", "block_index", "split",
}
ROLES = {"clean": "0", "embedded": "1"}
SPLITS = {"train", "validation", "test"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Dataset CSV has no header")
        missing = REQUIRED - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset CSV has no rows")
    return rows


def audit_dataset(
    dataset_path: Path,
    split_manifest_path: Path,
    *,
    min_train_models: int = 2,
    min_validation_models: int = 1,
    min_test_models: int = 2,
) -> dict[str, Any]:
    """Return a machine-readable readiness report; never fits a classifier."""
    rows = read_rows(dataset_path)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "nes.packed_nf4_group_splits.v1":
        raise ValueError("Unsupported split-manifest schema")
    group_to_split = manifest.get("group_to_split")
    if not isinstance(group_to_split, dict) or not group_to_split:
        raise ValueError("Split manifest has no group_to_split mapping")

    errors: list[str] = []
    warnings: list[str] = []
    seen_blocks: set[tuple[str, str, str, str, str]] = set()
    artifact_identity: dict[str, tuple[str, str, str, str]] = {}
    pair_blocks: dict[tuple[str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    model_splits: dict[str, set[str]] = defaultdict(set)
    split_models: dict[str, set[str]] = defaultdict(set)
    split_labels: dict[str, set[str]] = defaultdict(set)
    model_labels: dict[str, set[str]] = defaultdict(set)
    role_hashes: dict[str, set[str]] = defaultdict(set)

    for row_number, row in enumerate(rows, start=2):
        missing_values = [field for field in REQUIRED if not row.get(field, "").strip()]
        if missing_values:
            errors.append(f"row {row_number}: blank required values {sorted(missing_values)}")
            continue
        role, label = row["role"], row["label"]
        if role not in ROLES:
            errors.append(f"row {row_number}: invalid role {role!r}")
            continue
        if label != ROLES[role]:
            errors.append(f"row {row_number}: role/label mismatch ({role!r}, {label!r})")
        if row["split"] not in SPLITS:
            errors.append(f"row {row_number}: invalid split {row['split']!r}")
        expected_split = group_to_split.get(row["model_id"])
        if expected_split is None:
            errors.append(
                f"row {row_number}: model_id {row['model_id']!r} absent from split manifest"
            )
        elif expected_split != row["split"]:
            errors.append(
                f"row {row_number}: row split {row['split']!r} disagrees with manifest "
                f"({expected_split!r})"
            )

        model_splits[row["model_id"]].add(row["split"])
        split_models[row["split"]].add(row["model_id"])
        split_labels[row["split"]].add(label)
        model_labels[row["model_id"]].add(label)
        role_hashes[role].add(row["artifact_sha256"])

        identity = (
            row["role"], row["model_id"], row["tensor_key"], row["split"]
        )
        previous = artifact_identity.setdefault(row["artifact_sha256"], identity)
        if previous != identity:
            errors.append(
                f"artifact hash {row['artifact_sha256']} has conflicting identities: "
                f"{previous!r} vs {identity!r}"
            )

        block_identity = (
            row["artifact_sha256"], row["tensor_key"], row["block_index"],
            row["role"], row["split"],
        )
        if block_identity in seen_blocks:
            errors.append(f"row {row_number}: duplicate artifact/tensor/block {block_identity}")
        seen_blocks.add(block_identity)

        pair_key = (row["source_id"], row["run_id"], row["model_id"], row["tensor_key"])
        pair_blocks[pair_key][role].add(row["block_index"])

    overlap = sorted(role_hashes["clean"] & role_hashes["embedded"])
    if overlap:
        errors.append(f"same artifact SHA appears as clean and embedded: {overlap[:3]}")

    for model_id, assigned in model_splits.items():
        if len(assigned) != 1:
            errors.append(f"model_id {model_id!r} crosses split boundaries: {sorted(assigned)}")

    for pair_key, roles in pair_blocks.items():
        if set(roles) != {"clean", "embedded"}:
            errors.append(f"unpaired clean/embedded group {pair_key!r}: {sorted(roles)}")
        elif roles["clean"] != roles["embedded"]:
            errors.append(f"block-index mismatch in clean/embedded pair {pair_key!r}")

    for split in sorted(SPLITS):
        if split_labels[split] != {"0", "1"}:
            errors.append(
                f"split {split!r} must contain both labels; found {sorted(split_labels[split])}"
            )

    # A model must contain both classes so the split cannot be solved by
    # model identity alone within this dataset.
    for model_id, labels in model_labels.items():
        if labels != {"0", "1"}:
            errors.append(
                f"model_id {model_id!r} does not contain both clean and embedded labels"
            )

    group_counts = {split: len(split_models[split]) for split in sorted(SPLITS)}
    minimums = {
        "train": min_train_models,
        "validation": min_validation_models,
        "test": min_test_models,
    }
    for split, minimum in minimums.items():
        if group_counts[split] < minimum:
            errors.append(
                f"insufficient independent model groups in {split}: "
                f"{group_counts[split]} found, {minimum} required"
            )

    if len(model_splits) < sum(minimums.values()):
        warnings.append(
            "Small source-group count makes generalization and uncertainty estimates fragile."
        )

    status = "PASS" if not errors else "BLOCKED"
    return {
        "schema": "nes.packed_nf4_detector_readiness.v1",
        "status": status,
        "dataset_csv": str(dataset_path),
        "split_manifest": str(split_manifest_path),
        "rows": len(rows),
        "unique_artifacts": len(artifact_identity),
        "model_groups": len(model_splits),
        "model_groups_per_split": group_counts,
        "minimum_model_groups_per_split": minimums,
        "errors": errors,
        "warnings": warnings,
        "scope_note": (
            "Readiness checks only. PASS does not establish statistical independence, "
            "detector validity, or stealth. This script never trains a classifier."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-models", type=int, default=2)
    parser.add_argument("--min-validation-models", type=int, default=1)
    parser.add_argument("--min-test-models", type=int, default=2)
    args = parser.parse_args()
    if min(args.min_train_models, args.min_validation_models, args.min_test_models) < 1:
        parser.error("minimum model-group counts must be >= 1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
    report = audit_dataset(
        args.dataset, args.splits,
        min_train_models=args.min_train_models,
        min_validation_models=args.min_validation_models,
        min_test_models=args.min_test_models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
